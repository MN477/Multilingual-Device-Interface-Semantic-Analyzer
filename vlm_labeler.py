"""
vlm_labeler.py — VLM-powered auto-labeling for system vs. user text classification.

Sends device screenshots to a local LM Studio server running Qwen3-8B VLM,
extracts all visible text blocks with bounding boxes, and classifies each as
"system_text" or "user_text".

Output:  A single JSON file with per-image labeled blocks ready for classifier training.

Usage:
    python vlm_labeler.py --input_dir input --output vlm_labeled_data.json
    python vlm_labeler.py --input_dir input --output vlm_labeled_data.json --max_dim 1280
    python vlm_labeler.py --input_dir input --output vlm_labeled_data.json --api_url http://127.0.0.1:1234
"""

import os
import sys
import json
import time
import glob
import base64
import argparse
import re
from pathlib import Path
from typing import Optional

import requests

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_API_URL = "http://127.0.0.1:1234"
CHAT_ENDPOINT = "/v1/chat/completions"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif"}

MAX_RETRIES = 3
RETRY_DELAY_SEC = 10
REQUEST_TIMEOUT_SEC = 800 # VLM inference can be slow on large images

# ═══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a highly precise vision-language assistant specialized in OCR-style text extraction and User Interface (UI) layout analysis.

Your task is to analyze device screenshots and extract every visible text element with structured metadata.

## Extraction Rules
- Extract ONLY text that is clearly visible and readable in the image.
- Do NOT hallucinate, guess, or infer text that is not visible.
- Do NOT include non-text elements (icons, borders, shapes, decorative graphics).
- Do NOT merge separate text regions — keep each text block independent, as an OCR system would.
- Each distinct line, label, button, or text fragment should be its own block.
- Preserve the EXACT original text: language, script, capitalization, punctuation, spacing, and emojis.

## Classification Rules
Classify each extracted text element into exactly one of:

**"system_text"** — Static, system-generated UI content:
  - Navigation headers, tab labels, toolbar text
  - Menu items, button labels (e.g., "OK", "Cancel", "Send", "Settings")
  - Status bar info (time, battery, signal, carrier name)
  - App titles, window headers
  - Form field labels/placeholders (e.g., "Username:", "Enter password")
  - System notifications, warnings, dialogs
  - Toggle/switch labels, dropdown options
  - Footer text, copyright notices
  - Any text that is part of the fixed interface layout

**"user_text"** — Dynamic, user-generated or user-entered content:
  - Chat messages (sent or received in messaging apps)
  - Typed text in input fields (actual user input, not the placeholder)
  - Email body text, note contents
  - Search queries the user typed
  - Social media posts, comments, captions
  - File names or content created by the user
  - Any text that would change between different users of the same app

## Bounding Box Guidelines
- Provide pixel coordinates as [x_min, y_min, x_max, y_max].
- Boxes should tightly surround the text region with minimal padding.
- x_min is the left edge, y_min is the top edge, x_max is the right edge, y_max is the bottom edge.
- Be as accurate as possible — approximate if exact pixels are unclear, but stay consistent.

## Output Format
Return ONLY valid JSON. No explanations, no markdown fences, no comments."""


def build_user_prompt(filename: str) -> str:
    return f"""Analyze this device screenshot ("{filename}") and extract ALL visible text elements.

For EACH detected text region, provide:
1. **text**: Exact transcription (preserve language, case, punctuation, emojis).
2. **det_score**: Confidence that this text was correctly detected (0.00–1.00).
3. **bbox**: Bounding box as [x_min, y_min, x_max, y_max] in pixels.
4. **width**: Bounding box width in pixels (x_max - x_min).
5. **height**: Bounding box height in pixels (y_max - y_min).
6. **center**: Center point as [x_center, y_center] in pixels.
7. **label**: Either "system_text" or "user_text".

Constraints:
- Extract ALL readable text, no matter how small (timestamps, tiny labels, status icons with text).
- Skip purely visual elements without text.
- Do NOT combine text from distant screen regions into one block.
- If a text's classification is ambiguous, choose the most probable category.

Return ONLY this JSON structure:

{{"image_id": "{filename}", "img_width": <image_width_px>, "img_height": <image_height_px>, "blocks": [{{"index": 0, "text": "Example", "det_score": 0.95, "bbox": [10, 20, 150, 45], "width": 140, "height": 25, "center": [80, 32], "label": "system_text"}}]}}"""


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def encode_image_to_base64(image_path: str, max_dim: Optional[int] = None) -> tuple[str, int, int]:
    """
    Read an image, optionally resize it, and return (base64_string, width, height).
    Resizing helps reduce token cost and speeds up VLM inference.
    """
    import cv2
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]

    if max_dim and max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w

    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    b64 = base64.b64encode(buffer).decode("utf-8")

    return b64, w, h


def get_image_mime(path: str) -> str:
    """Determine MIME type from file extension."""
    ext = Path(path).suffix.lower()
    mime_map = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".bmp": "image/bmp", ".webp": "image/webp",
        ".tiff": "image/tiff", ".tif": "image/tiff",
    }
    return mime_map.get(ext, "image/jpeg")


# ═══════════════════════════════════════════════════════════════════════════════
# VLM API CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

def call_vlm(
    api_url: str,
    image_path: str,
    filename: str,
    max_dim: Optional[int] = None,
    temperature: float = 0.1,
    max_tokens: int = 8192,
) -> dict:
    """
    Send an image to the LM Studio VLM and parse the structured JSON response.
    Uses the OpenAI-compatible chat completions endpoint.
    """
    b64_image, img_w, img_h = encode_image_to_base64(image_path, max_dim)
    mime_type = get_image_mime(image_path)

    payload = {
        "model": "qwen3-vl-8b-instruct",  # LM Studio will use whatever model is loaded
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{b64_image}"
                        },
                    },
                    {
                        "type": "text",
                        "text": build_user_prompt(filename),
                    },
                ],
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    endpoint = f"{api_url.rstrip('/')}{CHAT_ENDPOINT}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                endpoint,
                json=payload,
                timeout=REQUEST_TIMEOUT_SEC,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            print(f"    ⚠  Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC * attempt)
            else:
                raise RuntimeError(
                    f"API request failed after {MAX_RETRIES} attempts for {filename}"
                ) from e

    result = resp.json()

    # Extract the assistant's text content
    raw_text = result["choices"][0]["message"]["content"]

    # Parse JSON from the response (handle markdown fences, thinking tags, etc.)
    parsed = extract_json_from_response(raw_text)

    # Inject actual image dimensions if the VLM didn't provide them or got them wrong
    parsed["_original_img_width"] = img_w
    parsed["_original_img_height"] = img_h
    parsed["_source_file"] = image_path

    return parsed


def extract_json_from_response(raw_text: str) -> dict:
    """
    Robustly extract a JSON object from VLM output.
    Handles: raw JSON, ```json fences, <think>...</think> wrappers, extra text.
    """
    text = raw_text.strip()

    # Strip <think>...</think> blocks (Qwen3 thinking mode)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Strip markdown code fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find the outermost { ... } block
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        candidate = text[brace_start : brace_end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Last resort: try to fix common issues (trailing commas)
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Return raw text wrapped in an error structure
        return {
            "image_id": "PARSE_ERROR",
            "blocks": [],
            "_raw_response": raw_text[:2000],
            "_parse_error": True,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION & POST-PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def validate_and_clean_blocks(data: dict) -> dict:
    """
    Validate and normalize the VLM output blocks.
    - Re-index blocks sequentially
    - Ensure bbox consistency (recompute width/height/center from bbox)
    - Clamp det_score to [0, 1]
    - Validate labels
    """
    blocks = data.get("blocks", [])
    cleaned = []

    for i, block in enumerate(blocks):
        text = block.get("text", "").strip()
        if not text:
            continue  # Skip empty text blocks

        # Validate bbox
        bbox = block.get("bbox", [0, 0, 0, 0])
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue

        x_min, y_min, x_max, y_max = [float(v) for v in bbox]

        # Ensure valid box (min < max)
        if x_min >= x_max or y_min >= y_max:
            continue

        # Recompute derived fields for consistency
        width = round(x_max - x_min, 1)
        height = round(y_max - y_min, 1)
        center = [round((x_min + x_max) / 2, 1), round((y_min + y_max) / 2, 1)]

        # Clamp confidence
        det_score = block.get("det_score", 0.5)
        if not isinstance(det_score, (int, float)):
            det_score = 0.5
        det_score = max(0.0, min(1.0, float(det_score)))

        # Validate label
        label = block.get("label", "system_text")
        if label not in ("system_text", "user_text"):
            label = "system_text"  # Default to system if invalid

        cleaned.append({
            "index": i,
            "text": text,
            "det_score": round(det_score, 4),
            "bbox": [round(x_min, 1), round(y_min, 1), round(x_max, 1), round(y_max, 1)],
            "width": width,
            "height": height,
            "center": center,
            "label": label,
        })

    # Reindex sequentially after filtering
    for idx, block in enumerate(cleaned):
        block["index"] = idx

    data["blocks"] = cleaned
    data["n_blocks"] = len(cleaned)
    data["n_system"] = sum(1 for b in cleaned if b["label"] == "system_text")
    data["n_user"] = sum(1 for b in cleaned if b["label"] == "user_text")

    return data


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def collect_image_paths(input_dir: str) -> list[str]:
    """Gather all image files from the input directory (non-recursive)."""
    paths = []
    for entry in sorted(os.listdir(input_dir)):
        if Path(entry).suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(os.path.join(input_dir, entry))
    return paths


def run_labeling_pipeline(
    input_dir: str,
    output_path: str,
    api_url: str = DEFAULT_API_URL,
    max_dim: Optional[int] = None,
    temperature: float = 0.1,
    resume: bool = True,
):
    """
    Main pipeline: iterate over images, send to VLM, collect labeled data.

    Args:
        input_dir:    Folder containing device screenshots.
        output_path:  Path for the output JSON file.
        api_url:      LM Studio server URL.
        max_dim:      Max image dimension (longer side) for resizing before sending.
        temperature:  VLM sampling temperature (low = more deterministic).
        resume:       If True and output file exists, skip already-processed images.
    """
    image_paths = collect_image_paths(input_dir)
    if not image_paths:
        print(f"❌ No images found in '{input_dir}' with extensions: {IMAGE_EXTENSIONS}")
        sys.exit(1)

    print(f"📂 Found {len(image_paths)} images in '{input_dir}'")
    print(f"🔗 API endpoint: {api_url}{CHAT_ENDPOINT}")
    print(f"📏 Max dimension: {max_dim or 'original size'}")
    print(f"🌡  Temperature: {temperature}")
    print()

    # Resume support: load existing results if any
    all_results = []
    processed_files = set()
    if resume and os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            all_results = existing.get("images", [])
            processed_files = {r.get("image_id") or r.get("_source_file", "") for r in all_results}
            print(f"📋 Resuming: {len(processed_files)} images already processed")
        except (json.JSONDecodeError, KeyError):
            print("⚠  Existing output file is corrupt — starting fresh")
            all_results = []

    total = len(image_paths)
    skipped = 0
    errors = 0

    for idx, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)

        # Skip if already processed (resume mode)
        if filename in processed_files or img_path in processed_files:
            skipped += 1
            continue

        print(f"[{idx + 1}/{total}] 🖼  Processing: {filename}", end=" ... ", flush=True)

        t_start = time.time()

        try:
            raw_result = call_vlm(
                api_url=api_url,
                image_path=img_path,
                filename=filename,
                max_dim=max_dim,
                temperature=temperature,
            )

            # Check for parse errors
            if raw_result.get("_parse_error"):
                print(f"⚠  JSON parse failed — raw response saved")
                errors += 1
                all_results.append(raw_result)
            else:
                cleaned = validate_and_clean_blocks(raw_result)
                elapsed = time.time() - t_start

                n_blocks = cleaned.get("n_blocks", 0)
                n_sys = cleaned.get("n_system", 0)
                n_usr = cleaned.get("n_user", 0)
                print(
                    f"✅ {n_blocks} blocks "
                    f"(sys={n_sys}, usr={n_usr}) "
                    f"[{elapsed:.1f}s]"
                )
                all_results.append(cleaned)

        except Exception as e:
            elapsed = time.time() - t_start
            print(f"❌ Error [{elapsed:.1f}s]: {e}")
            errors += 1
            all_results.append({
                "image_id": filename,
                "_source_file": img_path,
                "_error": str(e),
                "blocks": [],
            })

        # Save after every image (crash-safe incremental writes)
        _save_results(output_path, all_results, input_dir, api_url, max_dim)

    # Final summary
    print()
    print("═" * 70)
    print("  LABELING COMPLETE")
    print("═" * 70)
    total_blocks = sum(len(r.get("blocks", [])) for r in all_results)
    total_sys = sum(r.get("n_system", 0) for r in all_results)
    total_usr = sum(r.get("n_user", 0) for r in all_results)
    print(f"  Images processed: {total - skipped} (skipped {skipped} already done)")
    print(f"  Errors:           {errors}")
    print(f"  Total blocks:     {total_blocks}")
    print(f"  System text:      {total_sys}")
    print(f"  User text:        {total_usr}")
    print(f"  Output saved to:  {output_path}")
    print("═" * 70)


def _save_results(
    output_path: str,
    results: list,
    input_dir: str,
    api_url: str,
    max_dim: Optional[int],
):
    """Save the accumulated results to JSON (atomic write via temp file)."""
    output = {
        "metadata": {
            "input_dir": input_dir,
            "api_url": api_url,
            "max_dim": max_dim,
            "total_images": len(results),
            "total_blocks": sum(len(r.get("blocks", [])) for r in results),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "images": results,
    }

    # Write to temp file first, then rename (crash-safe)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, output_path)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="VLM-powered auto-labeler for system vs. user text classification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vlm_labeler.py --input_dir input --output vlm_labeled_data.json
  python vlm_labeler.py --input_dir input --output vlm_labeled_data.json --max_dim 1280
  python vlm_labeler.py --input_dir input --api_url http://127.0.0.1:1234 --temperature 0.05
        """,
    )

    parser.add_argument(
        "--input_dir", type=str, default="input",
        help="Directory containing device screenshots to label (default: 'input').",
    )
    parser.add_argument(
        "--output", type=str, default="vlm_labeled_data.json",
        help="Output JSON file path (default: 'vlm_labeled_data.json').",
    )
    parser.add_argument(
        "--api_url", type=str, default=DEFAULT_API_URL,
        help=f"LM Studio server URL (default: {DEFAULT_API_URL}).",
    )
    parser.add_argument(
        "--max_dim", type=int, default=None,
        help="Resize images so the longest side is at most this many pixels. "
             "Reduces token cost and speeds up inference. (default: no resize).",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help="VLM sampling temperature (lower = more deterministic, default: 0.1).",
    )
    parser.add_argument(
        "--no_resume", action="store_true",
        help="Disable resume — reprocess all images even if output file exists.",
    )

    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"❌ Input directory does not exist: '{args.input_dir}'")
        print(f"   Create it and place your device screenshots inside.")
        sys.exit(1)

    run_labeling_pipeline(
        input_dir=args.input_dir,
        output_path=args.output,
        api_url=args.api_url,
        max_dim=args.max_dim,
        temperature=args.temperature,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()

"""
paddleocr_vl.py — Single-pass VLM text extraction + classification using PaddleOCR-VL v1.6.

The VLM handles EVERYTHING in one call — no separate detection/recognition steps:
  Image in → VLM processes → Structured results out

PaddleOCR-VL v1.6 (~0.9B params) performs detection, recognition, layout analysis,
and script identification all internally. We send a single prompt asking it to
extract and classify all text, then format the structured output.

Setup:
    pip install paddlepaddle-gpu       # (or paddlepaddle for CPU)
    pip install "paddleocr[doc-parser]>=3.6.0"

Usage:
    python paddleocr_vl.py --image input_images/arabic_car.jpg
    python paddleocr_vl.py --image input_images/3173.jpg --output_dir vl_output
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import argparse
import time
import unicodedata
from collections import Counter
from typing import Dict, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# VLM PROMPT — single instruction for full extraction + classification
# ═══════════════════════════════════════════════════════════════════════════════

VLM_PROMPT = (
    "Extract every text element visible in this device screenshot. "
    "For each text element, provide: the exact transcribed text, "
    "the writing script (Latin, Arabic, Chinese, Japanese, Korean, Cyrillic, etc.), "
    "whether it is system UI text or user-typed text, "
    "and a confidence score. "
    "Ignore icons, logos, decorative shapes, and non-text graphics. "
    "Output as JSON with a list of text_elements, each having: "
    "text, script, category (system or user), confidence."
)


# ═══════════════════════════════════════════════════════════════════════════════
# UNICODE SCRIPT DETECTION (for enriching VLM output)
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_RANGES = [
    (0x0041, 0x024F, "Latin"), (0x1E00, 0x1EFF, "Latin"),
    (0x0600, 0x06FF, "Arabic"), (0x0750, 0x077F, "Arabic"),
    (0x08A0, 0x08FF, "Arabic"), (0xFB50, 0xFDFF, "Arabic"),
    (0xFE70, 0xFEFF, "Arabic"),
    (0x4E00, 0x9FFF, "Chinese"), (0x3400, 0x4DBF, "Chinese"),
    (0x3040, 0x309F, "Japanese"), (0x30A0, 0x30FF, "Japanese"),
    (0xAC00, 0xD7AF, "Korean"), (0x1100, 0x11FF, "Korean"),
    (0x0400, 0x04FF, "Cyrillic"),
    (0x0900, 0x097F, "Devanagari"),
    (0x0E00, 0x0E7F, "Thai"),
    (0x0370, 0x03FF, "Greek"),
    (0x0590, 0x05FF, "Hebrew"),
]


def detect_script(text: str) -> str:
    """Detect dominant writing script from Unicode codepoints."""
    counts = Counter()
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        for start, end, script in SCRIPT_RANGES:
            if start <= cp <= end:
                counts[script] += 1
                break
    if counts:
        return counts.most_common(1)[0][0]
    return "Unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE-PASS VLM PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_vlm(
    image_path: str,
    output_dir: str = "vl_output",
    pipeline_version: str = "v1.6",
    device: str = None,
) -> Optional[Dict]:
    """
    Single-pass VLM: send image → get all text detected, recognized,
    classified (system/user), with script and confidence.
    """
    if not os.path.isfile(image_path):
        print(f"ERROR: Image not found: {image_path}")
        return None

    try:
        from paddleocr import PaddleOCRVL
    except ImportError:
        print("ERROR: PaddleOCR-VL is not installed.")
        print('  pip install paddlepaddle')
        print('  pip install "paddleocr[doc-parser]>=3.6.0"')
        sys.exit(1)

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    # ── Initialize VLM pipeline ──
    print("=" * 70)
    print(f"PaddleOCR-VL {pipeline_version} — Single-Pass Analysis")
    print("=" * 70)

    init_kwargs = {"pipeline_version": pipeline_version}
    if device:
        init_kwargs["device"] = device

    print("Loading VLM (first run downloads weights)...")
    pipeline = PaddleOCRVL(**init_kwargs)

    # ── Single VLM call: image in → everything out ──
    print(f"\nProcessing: {image_path}")
    print("VLM is handling detection + recognition + layout in one pass...")
    start_time = time.time()

    results_iter = pipeline.predict(image_path, prompt=VLM_PROMPT)

    # Collect all results from the VLM
    text_elements = []
    raw_blocks = []

    for res in results_iter:
        # The VLM returns parsed blocks with labels, bboxes, and content
        if hasattr(res, "parsing_res_list"):
            for block in res.parsing_res_list:
                # Extract block info — the VLM already detected + recognized this
                label = getattr(block, "block_label", None) or (
                    block.get("block_label") if isinstance(block, dict) else None
                )
                content = getattr(block, "block_content", None) or (
                    block.get("block_content") if isinstance(block, dict) else None
                )
                bbox = getattr(block, "block_bbox", None) or (
                    block.get("block_bbox") if isinstance(block, dict) else None
                )
                score = getattr(block, "score", None) or (
                    block.get("score") if isinstance(block, dict) else None
                )

                # Handle string representation format from VLM
                if label is None and isinstance(block, str):
                    # Parse "label:\ttext\nbbox:\t[...]\ncontent:\t..." format
                    parsed = _parse_block_string(str(block))
                    if parsed:
                        label = parsed.get("label")
                        content = parsed.get("content")
                        bbox = parsed.get("bbox")

                if not content or not content.strip():
                    continue

                content = content.strip()

                # Detect script via Unicode analysis
                script = detect_script(content)

                # Classify system vs user based on VLM's layout label
                category, category_conf = _classify_category(label, content, bbox)

                # Detection confidence from VLM
                det_confidence = float(score) if score is not None else 0.85

                text_elements.append({
                    "text": content,
                    "script": script,
                    "category": category,
                    "category_confidence": round(category_conf, 2),
                    "detection_confidence": round(det_confidence, 2),
                    "layout_label": label or "text",
                    "bbox": bbox,
                })

                raw_blocks.append({
                    "label": label,
                    "content": content,
                    "bbox": bbox,
                    "score": score,
                })

        # Fallback: try accessing rec_texts directly
        elif hasattr(res, "rec_texts"):
            for i, text in enumerate(res.rec_texts):
                if not text or not text.strip():
                    continue
                score = float(res.rec_scores[i]) if hasattr(res, "rec_scores") else 0.0
                script = detect_script(text.strip())
                text_elements.append({
                    "text": text.strip(),
                    "script": script,
                    "category": "system",
                    "category_confidence": 0.7,
                    "detection_confidence": round(score, 2),
                    "layout_label": "text",
                    "bbox": None,
                })

    elapsed = time.time() - start_time

    # ── Build output ──
    n_system = sum(1 for e in text_elements if e["category"] == "system")
    n_user = sum(1 for e in text_elements if e["category"] == "user")
    scripts_found = list(set(e["script"] for e in text_elements if e["script"] != "Unknown"))

    output = {
        "image": image_path,
        "model": f"PaddleOCR-VL {pipeline_version}",
        "inference_time_sec": round(elapsed, 2),
        "total_text_elements": len(text_elements),
        "system_count": n_system,
        "user_count": n_user,
        "scripts_detected": scripts_found,
        "text_elements": text_elements,
    }

    # Save JSON
    out_path = os.path.join(output_dir, f"{image_name}_vl_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # ── Console summary ──
    print(f"\n{'=' * 70}")
    print("VLM ANALYSIS COMPLETE (single pass)")
    print(f"  Text elements:  {len(text_elements)}")
    print(f"  System text:    {n_system}")
    print(f"  User text:      {n_user}")
    print(f"  Scripts:        {scripts_found}")
    print(f"  Time:           {elapsed:.2f}s")
    print(f"  Results →       {out_path}")
    print("=" * 70)

    # Print each element
    print("\n── Extracted Text Elements ──")
    for i, elem in enumerate(text_elements):
        print(f"  {i+1:3d}. [{elem['category']:6s}] [{elem['script']:10s}] "
              f"(det={elem['detection_confidence']:.2f}, cls={elem['category_confidence']:.2f}) "
              f"\"{elem['text']}\"")

    return output


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_block_string(block_str: str) -> Optional[Dict]:
    """Parse the VLM's string-formatted block into a dict."""
    result = {}
    for line in block_str.split("\n"):
        line = line.strip().strip("#")
        if not line:
            continue
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip().lower()
            val = val.strip()
            if key == "label":
                result["label"] = val
            elif key == "content":
                result["content"] = val
            elif key == "bbox":
                try:
                    result["bbox"] = json.loads(val)
                except (json.JSONDecodeError, ValueError):
                    result["bbox"] = val
    return result if result else None


def _classify_category(
    layout_label: Optional[str],
    text: str,
    bbox: Optional[list],
) -> tuple:
    """
    Classify text as 'system' or 'user' using the VLM's layout label
    and text content heuristics.

    The VLM's layout detection already distinguishes between text types:
      - "text": general text blocks (usually system UI labels)
      - "vision_footnote": captions/annotations (system)
      - "title": headings (system)
      - "table": table content (system)

    Additional heuristics for user text:
      - Chat bubble patterns
      - Input field indicators
      - Message-like content
    """
    label = (layout_label or "").lower()

    # VLM layout labels that strongly indicate system text
    system_labels = {"text", "title", "header", "footer", "table",
                     "vision_footnote", "reference", "equation",
                     "aside_text", "footnote", "number", "seal"}

    # Default: system text (most UI text is system-generated)
    if label in system_labels:
        category = "system"
        confidence = 0.85
    elif label in {"image", "chart", "figure"}:
        # These shouldn't contain text, but if they do it's typically a caption
        category = "system"
        confidence = 0.6
    else:
        category = "system"
        confidence = 0.7

    # Heuristic overrides for user text detection
    text_lower = text.lower().strip()

    # Chat message patterns (common in messaging apps)
    user_patterns = [
        # Very short personal messages
        len(text_lower) > 50 and not any(kw in text_lower for kw in
            ["settings", "menu", "options", "copyright", "terms"]),
    ]

    # Input field / search indicators
    if any(indicator in text_lower for indicator in
           ["type here", "search", "enter your", "write a message",
            "type a message", "say something", "what's on your mind"]):
        category = "user"
        confidence = 0.9

    # Status bar indicators are always system
    if bbox and isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        # Top of screen (status bar region, typically y < 50)
        if bbox[1] < 50:
            category = "system"
            confidence = 0.95
        # Bottom of screen (navigation bar region)
        if bbox[1] > 680:
            category = "system"
            confidence = 0.90

    return category, confidence


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PaddleOCR-VL v1.6 — single-pass text extraction + classification."
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to input screenshot image.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="vl_output",
        help="Output directory (default: vl_output).",
    )
    parser.add_argument(
        "--version", type=str, default="v1.6",
        choices=["v1.5", "v1.6"],
        help="PaddleOCR-VL version (default: v1.6).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cpu' or 'gpu:0' (auto-detect if unset).",
    )
    args = parser.parse_args()

    run_vlm(
        image_path=args.image,
        output_dir=args.output_dir,
        pipeline_version=args.version,
        device=args.device,
    )


if __name__ == "__main__":
    main()
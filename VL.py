"""
VL.py — Crop Script Identification using a VLM via OpenRouter API.

This script scans a directory of image crops (belonging to an image/screenshot),
sends each crop to a Vision-Language Model on OpenRouter, classifies the script
written in the crop, and outputs a JSON file containing the results for each crop.

Usage:
    python VL.py --crops_dir crops/3173 --output_dir vl_output/3173 --model google/gemini-2.5-flash
"""

import os
import sys
import json
import argparse
import base64
import time
from typing import Dict, Optional
import requests

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def encode_image_to_base64(image_path: str) -> str:
    """Encode local image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_image_mime_type(image_path: str) -> str:
    """Determine MIME type based on file extension."""
    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    elif ext == ".png":
        return "image/png"
    elif ext == ".webp":
        return "image/webp"
    return "image/jpeg"


# ═══════════════════════════════════════════════════════════════════════════════
# VLM PROMPT CRAFTING
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are a highly precise vision-language assistant specializing in script and writing system identification. "
    "Your task is to analyze the text snippet inside the crop and identify the script in which it is written."
)

USER_PROMPT_TEMPLATE = """Analyze the text in this crop image.
Identify the script/writing system used for the text (for example: Latin, Arabic, Chinese, Japanese, Korean, Cyrillic, Devanagari, Thai, Hebrew, or multiple scripts if mixed).

Return the output strictly in valid JSON format matching this schema:
{
  "crop_id" : crop_003.png
  "script": "Script Name",
  "confidence": 0.95
}

Ensure your response is valid JSON and contains nothing else."""


# ═══════════════════════════════════════════════════════════════════════════════
# OPENROUTER VLM EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def process_crop_with_vlm(
    crop_path: str,
    api_key: str,
    model: str,
) -> Optional[Dict]:
    """
    Sends a single crop to OpenRouter and returns the VLM classification.
    """
    base64_image = encode_image_to_base64(crop_path)
    mime_type = get_image_mime_type(crop_path)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/OpenRouter-VLM-Extraction",
        "X-Title": "Crop Script Classifier",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": USER_PROMPT_TEMPLATE
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "response_format": {"type": "json_object"}
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=45
        )
        if response.status_code != 200:
            print(f"    ERROR {response.status_code} on {os.path.basename(crop_path)}: {response.text}")
            return None

        response_json = response.json()
        raw_content = response_json["choices"][0]["message"]["content"]
        return json.loads(raw_content)

    except Exception as e:
        print(f"    Exception processing {os.path.basename(crop_path)}: {e}")
        return None


def run_pipeline(
    crops_dir: str,
    output_dir: str,
    model: str,
    api_key: str
):
    if not os.path.isdir(crops_dir):
        print(f"ERROR: Crops directory does not exist: {crops_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files in crops_dir
    valid_exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    crop_files = sorted([f for f in os.listdir(crops_dir) if f.lower().endswith(valid_exts)])

    print(f"Found {len(crop_files)} crops to analyze in: {crops_dir}")
    print(f"Using OpenRouter Model: {model}")
    
    results = {}
    
    for idx, crop_file in enumerate(crop_files):
        crop_path = os.path.join(crops_dir, crop_file)
        print(f"[{idx+1}/{len(crop_files)}] Processing {crop_file}...")
        
        start_time = time.time()
        vlm_res = process_crop_with_vlm(crop_path, api_key, model)
        elapsed = time.time() - start_time
        
        if vlm_res:
            script = vlm_res.get("script", "Unknown")
            conf = vlm_res.get("confidence", 0.0)
            text = vlm_res.get("transcribed_text", "")
            print(f"    -> Result: {script} (Conf: {conf}) | Text: '{text}' ({elapsed:.2f}s)")
            
            results[crop_file] = {
                "script": script,
                "confidence": conf,
                "transcribed_text": text,
                "inference_time_sec": round(elapsed, 2)
            }
        else:
            print(f"    -> Failed classification.")
            results[crop_file] = {
                "script": "Error",
                "confidence": 0.0,
                "error": "Failed VLM request"
            }
            
        # Optional sleep to respect API limits
        time.sleep(0.5)

    # Save output json containing all crops results
    image_name = os.path.basename(os.path.normpath(crops_dir))
    out_json_path = os.path.join(output_dir, f"{image_name}_crops_scripts.json")
    
    output_payload = {
        "image_name": image_name,
        "model": model,
        "crops_directory": crops_dir,
        "results": results
    }
    
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)
        
    print("\n" + "=" * 60)
    print("VLM CROP SCRIPT CLASSIFICATION COMPLETE")
    print(f"Results saved to: {out_json_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Classify scripts of text crop images using a VLM on OpenRouter."
    )
    parser.add_argument(
        "--crops_dir", type=str, required=True,
        help="Directory containing the crop images (e.g. crops/3173)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="vl_output",
        help="Directory to save classification results.",
    )
    parser.add_argument(
        "--model", type=str, default="google/gemini-2.5-flash",
        help="OpenRouter VLM model name (default: google/gemini-2.5-flash)",
    )
    parser.add_argument(
        "--api_key", type=str, default=None,
        help="OpenRouter API Key. If not set, checks OPENROUTER_API_KEY environment variable.",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OpenRouter API key is missing. Specify it via --api_key or set the OPENROUTER_API_KEY environment variable.")
        sys.exit(1)

    run_pipeline(
        crops_dir=args.crops_dir,
        output_dir=args.output_dir,
        model=args.model,
        api_key=api_key
    )


if __name__ == "__main__":
    main()

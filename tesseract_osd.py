"""
tesserat_assd.py — Script and Orientation Detection using Tesseract OSD (Orientation and Script Detection).

Processes crop images from an input directory (or matching a manifest file) 
using Tesseract's OSD engine (--psm 0) to extract:
  - Detected writing script (e.g., Latin, Arabic, Cyrillic, Han, etc.)
  - Script confidence score
  - Orientation angle (0, 90, 180, 270)
  - Orientation confidence score

Outputs results to crops/tesseract_osd_results.json.
"""

import os
import sys
import json
import argparse
import time
from typing import Dict, List, Tuple, Optional

try:
    import pytesseract
    from PIL import Image
except ImportError as e:
    print(f"ERROR: Missing dependency — {e}")
    print("Please run: pip install pytesseract Pillow")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# TESSERACT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
# Adjust the Tesseract executable path if needed for Windows
DEFAULT_TESS_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(DEFAULT_TESS_CMD):
    pytesseract.pytesseract.tesseract_cmd = DEFAULT_TESS_CMD


def run_tesseract_osd(img_path: str) -> Dict:
    """
    Runs Tesseract OSD on a single crop image.
    
    Args:
        img_path: Path to the crop image.
        
    Returns:
        Dict containing OSD attributes: script, script_conf, orientation, orient_conf.
    """
    result = {
        "script": "Unknown",
        "script_confidence": 0.0,
        "orientation_angle": 0,
        "orientation_confidence": 0.0,
        "error": None
    }
    
    try:
        img = Image.open(img_path)
        # OSD mode is activated by Page Segmentation Mode (--psm 0)
        osd_data = pytesseract.image_to_osd(img, config="--psm 0")
        
        # Parse the raw OSD text output line by line
        # Typical output format:
        # Page number: 0
        # Orientation in degrees: 0
        # Rotate: 0
        # Orientation confidence: 12.34
        # Script: Latin
        # Script confidence: 8.56
        for line in osd_data.split("\n"):
            line = line.strip()
            if not line:
                continue
            
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                
                if "Script" == key:
                    result["script"] = val
                elif "Script confidence" == key:
                    try:
                        result["script_confidence"] = float(val)
                    except ValueError:
                        pass
                elif "Orientation in degrees" == key:
                    try:
                        result["orientation_angle"] = int(val)
                    except ValueError:
                        pass
                elif "Orientation confidence" == key:
                    try:
                        result["orientation_confidence"] = float(val)
                    except ValueError:
                        pass
                        
    except pytesseract.TesseractError as te:
        # Tesseract returns an error (typically code 1) if there's too little text
        # to determine orientation or script. This is common on small crops.
        result["error"] = f"Too little text or TesseractError: {str(te).strip()}"
    except Exception as e:
        result["error"] = str(e)
        
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_osd_pipeline(
    results_json_path: str,
    crops_dir: str,
    image_name: Optional[str] = None
):
    # Gather text crops
    text_crops = []
    
    # Try using the manifest file first
    manifest_loaded = False
    if os.path.exists(results_json_path):
        try:
            with open(results_json_path, "r", encoding="utf-8") as f:
                clip_data = json.load(f)
            manifest_loaded = True
        except Exception as e:
            print(f"Warning: Could not parse results JSON: {e}")

    if manifest_loaded:
        for img_name, group_data in clip_data.items():
            if img_name.startswith("_"):
                continue
            if image_name and img_name != image_name:
                continue

            crops_list = group_data.get("crops", [])
            for crop_entry in crops_list:
                if crop_entry.get("is_text", False) or crop_entry.get("label") == "text":
                    crop_path = crop_entry.get("crop_path")
                    if crop_path and not os.path.exists(crop_path):
                        crop_path = os.path.join(crops_dir, img_name, crop_entry["crop_file"])
                    
                    if crop_path and os.path.exists(crop_path):
                        text_crops.append({
                            "image_name": img_name,
                            "crop_file": crop_entry["crop_file"],
                            "crop_path": crop_path,
                            "clip_entry": crop_entry
                        })

    # Fallback: if no crops found via manifest (or manifest not loaded) but image_name directory exists, scan it directly
    if not text_crops and image_name:
        target_dir = os.path.join(crops_dir, image_name)
        if os.path.isdir(target_dir):
            print(f"Manifest lookup found no crops for '{image_name}'. Scanning directory '{target_dir}' directly...")
            for f in sorted(os.listdir(target_dir)):
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                    crop_path = os.path.join(target_dir, f)
                    text_crops.append({
                        "image_name": image_name,
                        "crop_file": f,
                        "crop_path": crop_path,
                        "clip_entry": {"is_text": True, "label": "text", "text_score": 1.0}
                    })

    print(f"\nProcessing {len(text_crops)} crop(s) using Tesseract OSD...")
    if not text_crops:
        print("No text crops found to process.")
        return

    results = {}
    total_processed = 0
    start_time = time.time()

    for entry in text_crops:
        crop_path = entry["crop_path"]
        img_name = entry["image_name"]

        # Run OSD
        osd_res = run_tesseract_osd(crop_path)

        if img_name not in results:
            results[img_name] = {
                "num_text_crops": 0,
                "crops": []
            }

        crop_res_entry = {
            "crop_file": entry["crop_file"],
            "crop_path": crop_path,
            "tesseract_osd": osd_res
        }
        
        results[img_name]["crops"].append(crop_res_entry)
        results[img_name]["num_text_crops"] += 1
        total_processed += 1

        if osd_res["error"]:
            print(f"  [{img_name}] {entry['crop_file']} -> ERROR: {osd_res['error']}")
        else:
            print(f"  [{img_name}] {entry['crop_file']} -> "
                  f"Script: {osd_res['script']} (Conf: {osd_res['script_confidence']}) "
                  f"Orient: {osd_res['orientation_angle']}° (Conf: {osd_res['orientation_confidence']})")

    elapsed = time.time() - start_time
    summary = {
        "engine": "Tesseract OSD",
        "total_crops_processed": total_processed,
        "elapsed_seconds": round(elapsed, 2)
    }

    output_payload = {
        "_summary": summary,
        "results": results
    }

    out_json = os.path.join(crops_dir, "tesseract_osd_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("TESSERACT OSD CLASSIFICATION COMPLETE")
    print(f"  Processed crops: {total_processed}")
    print(f"  Time taken:      {elapsed:.2f} seconds")
    print(f"  Results saved →  {out_json}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Script and Orientation Detection using Tesseract OSD."
    )
    parser.add_argument(
        "--results_json", type=str, default="crops/clip_filtering_results.json",
        help="Path to CLIP filtering results JSON (default: crops/clip_filtering_results.json)",
    )
    parser.add_argument(
        "--crops_dir", type=str, default="crops",
        help="Root crops directory (default: crops)",
    )
    parser.add_argument(
        "--image_name", type=str, default=None,
        help="Name of a single image group to process (optional)",
    )
    parser.add_argument(
        "--tess_path", type=str, default=None,
        help="Explicit path to tesseract executable (optional)",
    )
    args = parser.parse_args()

    if args.tess_path:
        pytesseract.pytesseract.tesseract_cmd = args.tess_path

    run_osd_pipeline(
        results_json_path=args.results_json,
        crops_dir=args.crops_dir,
        image_name=args.image_name
    )


if __name__ == "__main__":
    main()

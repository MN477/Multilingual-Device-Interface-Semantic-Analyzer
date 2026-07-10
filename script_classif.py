"""
script_classif.py — Script classification via OCR + Unicode codepoint analysis.

Both CLIP and EfficientNet fail at script identification because they operate
on high-level visual semantics — they cannot reliably distinguish between
character glyphs of different writing systems (Latin "A" vs Arabic "أ" vs
Chinese "中" all look like "text on screen" to a vision model).

The correct solution:
  1. Run OCR recognition on each text crop to extract the actual characters
  2. Analyze the Unicode codepoint of each character to determine its script
  3. Report the dominant script with a ratio-based confidence score

This approach is:
  - Deterministic (not probabilistic — Unicode blocks ARE the ground truth)
  - Near-100% accurate on readable text
  - Extremely fast (no GPU needed for the classification itself)
  - Handles mixed-script text by reporting all scripts with their ratios

Takes as input all crops classified as 'text' from the CLIP filtering results
(e.g., crops/clip_filtering_results.json).

Outputs:
  - script_classification_results.json (per-crop script label + confidence)
  - Console summary

Usage:
    python script_classif.py
    python script_classif.py --image_name 3173
    python script_classif.py --results_json crops/clip_filtering_results.json --ocr_lang en
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import argparse
import time
import unicodedata
from collections import Counter
from typing import List, Dict, Tuple, Optional

import numpy as np
import cv2


# ═══════════════════════════════════════════════════════════════════════════════
# UNICODE-BASED SCRIPT DETECTION (deterministic, near-perfect accuracy)
# ═══════════════════════════════════════════════════════════════════════════════

# Unicode codepoint ranges mapped to script names.
# Each entry is (start, end, script_label).
# Ordered so that more specific ranges come first where needed.
UNICODE_SCRIPT_RANGES = [
    # ── Latin ──
    (0x0041, 0x005A, "Latin"),       # Basic Latin uppercase
    (0x0061, 0x007A, "Latin"),       # Basic Latin lowercase
    (0x00C0, 0x00FF, "Latin"),       # Latin-1 Supplement (accented chars)
    (0x0100, 0x024F, "Latin"),       # Latin Extended-A & B
    (0x1E00, 0x1EFF, "Latin"),       # Latin Extended Additional
    (0x2C60, 0x2C7F, "Latin"),       # Latin Extended-C
    (0xA720, 0xA7FF, "Latin"),       # Latin Extended-D

    # ── Arabic ──
    (0x0600, 0x06FF, "Arabic"),      # Arabic
    (0x0750, 0x077F, "Arabic"),      # Arabic Supplement
    (0x08A0, 0x08FF, "Arabic"),      # Arabic Extended-A
    (0xFB50, 0xFDFF, "Arabic"),      # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF, "Arabic"),      # Arabic Presentation Forms-B

    # ── Chinese (CJK) ──
    (0x4E00, 0x9FFF, "Chinese"),     # CJK Unified Ideographs
    (0x3400, 0x4DBF, "Chinese"),     # CJK Unified Ideographs Extension A
    (0x20000, 0x2A6DF, "Chinese"),   # CJK Extension B
    (0x2A700, 0x2B73F, "Chinese"),   # CJK Extension C
    (0x2B740, 0x2B81F, "Chinese"),   # CJK Extension D
    (0xF900, 0xFAFF, "Chinese"),     # CJK Compatibility Ideographs

    # ── Japanese (Hiragana + Katakana) ──
    (0x3040, 0x309F, "Japanese"),    # Hiragana
    (0x30A0, 0x30FF, "Japanese"),    # Katakana
    (0x31F0, 0x31FF, "Japanese"),    # Katakana Phonetic Extensions
    (0xFF65, 0xFF9F, "Japanese"),    # Halfwidth Katakana

    # ── Korean (Hangul) ──
    (0xAC00, 0xD7AF, "Korean"),      # Hangul Syllables
    (0x1100, 0x11FF, "Korean"),      # Hangul Jamo
    (0x3130, 0x318F, "Korean"),      # Hangul Compatibility Jamo
    (0xA960, 0xA97F, "Korean"),      # Hangul Jamo Extended-A
    (0xD7B0, 0xD7FF, "Korean"),      # Hangul Jamo Extended-B

    # ── Cyrillic ──
    (0x0400, 0x04FF, "Cyrillic"),    # Cyrillic
    (0x0500, 0x052F, "Cyrillic"),    # Cyrillic Supplement
    (0x2DE0, 0x2DFF, "Cyrillic"),    # Cyrillic Extended-A
    (0xA640, 0xA69F, "Cyrillic"),    # Cyrillic Extended-B

    # ── Devanagari ──
    (0x0900, 0x097F, "Devanagari"),  # Devanagari
    (0xA8E0, 0xA8FF, "Devanagari"), # Devanagari Extended

    # ── Thai ──
    (0x0E00, 0x0E7F, "Thai"),

    # ── Greek ──
    (0x0370, 0x03FF, "Greek"),
    (0x1F00, 0x1FFF, "Greek"),       # Greek Extended

    # ── Hebrew ──
    (0x0590, 0x05FF, "Hebrew"),
    (0xFB1D, 0xFB4F, "Hebrew"),     # Hebrew Presentation Forms
]


def classify_char_script(char: str) -> Optional[str]:
    """
    Identify the script of a single character using Unicode codepoint ranges.

    Args:
        char: A single character.

    Returns:
        Script name string, or None for digits/punctuation/whitespace/symbols.
    """
    cp = ord(char)

    # Skip digits, ASCII punctuation, whitespace, common symbols
    if char.isdigit() or char.isspace():
        return None
    cat = unicodedata.category(char)
    # Skip punctuation (Pc, Pd, Pe, Pf, Pi, Po, Ps), symbols (S*), separators (Z*)
    if cat[0] in ("P", "S", "Z", "C"):
        return None

    for start, end, script in UNICODE_SCRIPT_RANGES:
        if start <= cp <= end:
            return script

    # Fallback: try unicodedata.name() to catch edge cases
    try:
        name = unicodedata.name(char, "").upper()
        for script_label in ["LATIN", "ARABIC", "CJK", "HANGUL", "HIRAGANA",
                             "KATAKANA", "CYRILLIC", "DEVANAGARI", "THAI",
                             "GREEK", "HEBREW"]:
            if script_label in name:
                # Normalize CJK → Chinese, HANGUL → Korean, etc.
                mapping = {
                    "CJK": "Chinese", "HANGUL": "Korean",
                    "HIRAGANA": "Japanese", "KATAKANA": "Japanese",
                }
                return mapping.get(script_label, script_label.capitalize())
    except ValueError:
        pass

    return None


def classify_text_script(text: str) -> Dict:
    """
    Classify the writing script of a text string by analyzing Unicode codepoints.

    Counts the script of every letter-class character and returns the dominant
    script along with a confidence ratio and the full distribution.

    Args:
        text: Recognized text string from OCR.

    Returns:
        Dict with:
            'predicted_script':  Dominant script name (e.g., "Latin")
            'confidence':        Ratio of dominant script characters (0.0 to 1.0)
            'total_script_chars': Number of characters that mapped to a script
            'script_distribution': Dict of {script: count}
            'recognized_text':   The input text
    """
    script_counts = Counter()

    for char in text:
        script = classify_char_script(char)
        if script is not None:
            script_counts[script] += 1

    total_script_chars = sum(script_counts.values())

    if total_script_chars == 0:
        return {
            "predicted_script": "Unknown",
            "confidence": 0.0,
            "total_script_chars": 0,
            "script_distribution": {},
            "recognized_text": text,
        }

    dominant_script, dominant_count = script_counts.most_common(1)[0]
    confidence = dominant_count / total_script_chars

    return {
        "predicted_script": dominant_script,
        "confidence": round(confidence, 4),
        "total_script_chars": total_script_chars,
        "script_distribution": dict(script_counts.most_common()),
        "recognized_text": text,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# OCR RECOGNITION (extracts text from crop images)
# ═══════════════════════════════════════════════════════════════════════════════

class CropOCRRecognizer:
    """
    Lightweight OCR recognizer for text crops.
    Uses PaddleOCR in recognition-only mode on pre-cropped images.
    """

    def __init__(self, lang: str = "en"):
        """
        Args:
            lang: PaddleOCR language code. Use 'en' for Latin,
                  'ar' for Arabic, 'ch' for Chinese, 'japan' for Japanese,
                  'korean' for Korean, 'cyrillic' for Cyrillic, 'devanagari' for Hindi.
                  Use 'multilingual' or 'en' as a default for mixed content —
                  PaddleOCR will still detect non-Latin characters.
        """
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            print("ERROR: paddleocr not installed. Run: pip install paddleocr")
            sys.exit(1)

        print(f"  Initializing PaddleOCR recognizer (lang={lang})...")
        # det=False: skip detection (we already have crops)
        # cls=True:  enable angle classifier for rotated text
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            show_log=False,
            det=False,
        )

    def recognize(self, crop_path: str) -> Tuple[str, float]:
        """
        Recognize text in a single crop image.

        Args:
            crop_path: Path to the crop image.

        Returns:
            (recognized_text, confidence_score)
        """
        try:
            result = self.ocr.ocr(crop_path, det=False, cls=True)
            if result and result[0]:
                # result[0] is a list of (text, confidence) tuples
                texts = []
                scores = []
                for line in result[0]:
                    if isinstance(line, (list, tuple)) and len(line) >= 2:
                        txt, score = line[0], line[1]
                        # Sometimes PaddleOCR returns (text, score) directly
                        if isinstance(txt, str):
                            texts.append(txt)
                            scores.append(float(score))
                        elif isinstance(txt, (list, tuple)):
                            # Nested format: ((text, score),)
                            texts.append(str(txt))
                            scores.append(float(score))

                if texts:
                    combined = " ".join(texts)
                    avg_score = np.mean(scores) if scores else 0.0
                    return combined, float(avg_score)

            return "", 0.0

        except Exception as e:
            print(f"    OCR error on '{crop_path}': {e}")
            return "", 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def classify_scripts_pipeline(
    results_json_path: str,
    crops_dir: str,
    ocr_lang: str = "en",
    image_name: Optional[str] = None,
):
    """
    Main pipeline: load CLIP filtering results → OCR each text crop →
    classify script via Unicode analysis → save results.

    Args:
        results_json_path:  Path to clip_filtering_results.json.
        crops_dir:          Root crops directory.
        ocr_lang:           PaddleOCR language code for recognition.
        image_name:         Optional single image group to process.
    """
    if not os.path.exists(results_json_path):
        print(f"ERROR: Results JSON not found: '{results_json_path}'")
        sys.exit(1)

    with open(results_json_path, "r", encoding="utf-8") as f:
        clip_data = json.load(f)

    # Initialize OCR recognizer
    print("=" * 70)
    print("SCRIPT CLASSIFICATION (OCR + Unicode Analysis)")
    print("=" * 70)
    recognizer = CropOCRRecognizer(lang=ocr_lang)

    # Gather text crops
    text_crops = []
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
                        "clip_entry": crop_entry,
                    })

    print(f"\nFound {len(text_crops)} text crop(s) to classify.\n")
    if not text_crops:
        print("No text crops to classify. Exiting.")
        return

    # Process each crop
    results = {}
    total_processed = 0
    script_global_counts = Counter()
    start_time = time.time()

    for entry in text_crops:
        crop_path = entry["crop_path"]
        img_name = entry["image_name"]

        # Step 1: OCR recognition
        recognized_text, ocr_confidence = recognizer.recognize(crop_path)

        # Step 2: Unicode-based script classification
        script_result = classify_text_script(recognized_text)

        predicted = script_result["predicted_script"]
        confidence = script_result["confidence"]

        # Accumulate
        if img_name not in results:
            results[img_name] = {"num_text_crops": 0, "crops": []}

        results[img_name]["crops"].append({
            "crop_file": entry["crop_file"],
            "crop_path": crop_path,
            "recognized_text": recognized_text,
            "ocr_confidence": round(ocr_confidence, 4),
            "predicted_script": predicted,
            "script_confidence": confidence,
            "script_distribution": script_result["script_distribution"],
            "total_script_chars": script_result["total_script_chars"],
        })
        results[img_name]["num_text_crops"] += 1
        total_processed += 1
        script_global_counts[predicted] += 1

        # Console feedback
        print(f"  [{img_name}] {entry['crop_file']:20s} → "
              f"script={predicted:12s} conf={confidence:.2f}  "
              f"text=\"{recognized_text[:50]}\"")

    elapsed = time.time() - start_time

    # Summary
    summary = {
        "method": "OCR + Unicode codepoint analysis",
        "ocr_lang": ocr_lang,
        "total_crops_processed": total_processed,
        "elapsed_seconds": round(elapsed, 2),
        "script_distribution": dict(script_global_counts.most_common()),
    }

    output_payload = {
        "_summary": summary,
        "results": results,
    }

    out_json = os.path.join(crops_dir, "script_classification_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("SCRIPT CLASSIFICATION COMPLETE")
    print(f"  Method:     OCR + Unicode analysis (deterministic)")
    print(f"  Processed:  {total_processed} crops")
    print(f"  Time:       {elapsed:.2f}s")
    print(f"  Scripts found: {dict(script_global_counts.most_common())}")
    print(f"  Results →   {out_json}")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Script classification on text crops using OCR + Unicode analysis."
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
        "--ocr_lang", type=str, default="en",
        help="PaddleOCR language code (default: en). Use 'ch' for Chinese, 'ar' for Arabic, etc.",
    )
    parser.add_argument(
        "--image_name", type=str, default=None,
        help="Name of a single image group to process (optional)",
    )
    args = parser.parse_args()

    classify_scripts_pipeline(
        results_json_path=args.results_json,
        crops_dir=args.crops_dir,
        ocr_lang=args.ocr_lang,
        image_name=args.image_name,
    )


if __name__ == "__main__":
    main()

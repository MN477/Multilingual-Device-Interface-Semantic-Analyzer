"""
VL.py — Vision-Language Model analysis for device screenshots.

Uses PaddleOCR's PP-ChatOCRv3-doc pipeline — the best VLM in PaddleOCR's
ecosystem — to perform end-to-end text analysis on a single screenshot:

  1. VISUAL PREDICTION  — OCR detection + recognition + layout analysis
  2. VLM CHAT          — Semantic classification via crafted prompts:
                            • System text vs User text
                            • Writing script identification
                            • Confidence estimation

The PP-ChatOCRv3-doc pipeline combines:
  - PP-OCRv4 (detection + recognition)
  - Layout analysis (SLANet / PicoDet)
  - LLM-based semantic reasoning (Ernie / local LLM)

The prompts are carefully crafted to extract structured classification
from the VLM, covering system UI labels, user-typed content, script
detection, and confidence scoring — all in a single pipeline pass.

Usage:
    python VL.py --image input_images/3173.jpg
    python VL.py --image input_images/desktop_UI.jpeg --output_dir vl_output
    python VL.py --image input_images/im1.jpg --llm_name ernie-4.0
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import argparse
import time
from typing import Dict, List, Optional

import numpy as np
import cv2


# ═══════════════════════════════════════════════════════════════════════════════
# WELL-CRAFTED VLM PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════
#
# PP-ChatOCRv3 uses key_list for structured extraction. Each key is a
# question/field that the VLM answers using the OCR-extracted visual context.
#
# Prompt design principles:
#   - Be explicit about what counts as "system" vs "user" text
#   - Ask for structured output (lists, categories)
#   - Include examples to anchor the VLM's understanding
#   - Request confidence and script in a single pass when possible

# ── Round 1: Full structured extraction ──
CLASSIFICATION_KEYS = [
    # System text extraction
    "List ALL system-generated interface text including: menu labels, "
    "button captions, toolbar text, status bar indicators (time, battery, signal), "
    "navigation tabs, dialog titles, form field labels, section headers, "
    "error messages, tooltips, window titles, and fixed UI captions. "
    "Format each as: [text] | [location description]",

    # User text extraction
    "List ALL user-entered or user-generated text including: typed messages "
    "in chat bubbles, search queries, text inside input fields or text boxes, "
    "typed commands, user comments, editable document content, and any "
    "free-form text that appears to be entered by a human user. "
    "Format each as: [text] | [location description]",

    # Script identification
    "For each text element visible in the image, identify the writing script "
    "(Latin, Arabic, Chinese, Japanese, Korean, Cyrillic, Devanagari, Thai, "
    "Greek, Hebrew, or other). Report the dominant script and any mixed-script "
    "regions. Format as: [text snippet] -> [script name]",

    # Confidence and summary
    "Provide a confidence assessment (high/medium/low) for your classification "
    "of each text element as system or user text. Also report the total count "
    "of system text elements and user text elements found.",
]

# ── Round 2 (optional): Detailed per-element classification ──
DETAILED_CLASSIFICATION_KEY = (
    "For every single text element detected in this screenshot image, "
    "provide a structured classification with the following fields: "
    "1) The exact text content, "
    "2) Category: 'system' for fixed interface text or 'user' for user-typed content, "
    "3) Script: the writing system (Latin/Arabic/Chinese/Japanese/Korean/Cyrillic/other), "
    "4) Confidence: high/medium/low. "
    "Return the result as a numbered list."
)


# ═══════════════════════════════════════════════════════════════════════════════
# VLM PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class ScreenshotVLMAnalyzer:
    """
    End-to-end screenshot text analyzer using PP-ChatOCRv3-doc VLM pipeline.

    Two-step process:
      1. visual_predict() — runs OCR detection, recognition, layout analysis
      2. chat()           — sends crafted prompts to the LLM for classification
    """

    def __init__(
        self,
        llm_name: str = "ernie-3.5",
        llm_params: Optional[Dict] = None,
        device: str = "cpu",
    ):
        """
        Initialize the PP-ChatOCRv3-doc VLM pipeline.

        Args:
            llm_name:   LLM model name. Options:
                        'ernie-3.5', 'ernie-4.0' (via Baidu Qianfan API)
                        Or a local model path for offline usage.
            llm_params: Dict with LLM API credentials, e.g.:
                        {"api_type": "qianfan", "ak": "...", "sk": "..."}
                        If None, tries to use environment variables or local model.
            device:     'cpu' or 'gpu:0'.
        """
        try:
            from paddlex import create_pipeline
        except ImportError:
            print("ERROR: paddlex is not installed.")
            print("Install with:  pip install paddlex")
            sys.exit(1)

        print("=" * 70)
        print("INITIALIZING PP-ChatOCRv3-doc VLM PIPELINE")
        print("=" * 70)

        pipeline_kwargs = {
            "pipeline": "PP-ChatOCRv3-doc",
            "device": device,
        }

        # Configure LLM backend
        if llm_params:
            pipeline_kwargs["llm_name"] = llm_name
            pipeline_kwargs["llm_params"] = llm_params
        else:
            # Try environment variables for Ernie API
            ak = os.environ.get("QIANFAN_AK") or os.environ.get("ERNIE_AK", "")
            sk = os.environ.get("QIANFAN_SK") or os.environ.get("ERNIE_SK", "")
            if ak and sk:
                pipeline_kwargs["llm_name"] = llm_name
                pipeline_kwargs["llm_params"] = {
                    "api_type": "qianfan",
                    "ak": ak,
                    "sk": sk,
                }
                print(f"  Using LLM: {llm_name} (Qianfan API from env vars)")
            else:
                # No API keys — try local/default config
                pipeline_kwargs["llm_name"] = llm_name
                print(f"  Using LLM: {llm_name} (default config)")
                print("  TIP: Set QIANFAN_AK and QIANFAN_SK environment variables")
                print("       for Ernie API access, or pass --llm_params.")

        try:
            self.pipeline = create_pipeline(**pipeline_kwargs)
            print("  PP-ChatOCRv3-doc pipeline loaded successfully.")
        except Exception as e:
            print(f"\n  WARNING: Full VLM pipeline failed to load: {e}")
            print("  Falling back to OCR-only mode (no LLM classification).")
            self.pipeline = None
            self._init_ocr_fallback(device)

        print("=" * 70)

    def _init_ocr_fallback(self, device: str):
        """Initialize a basic OCR pipeline as fallback when LLM is unavailable."""
        try:
            from paddlex import create_pipeline
            self.ocr_pipeline = create_pipeline(pipeline="OCR", device=device)
            self.fallback_mode = True
            print("  OCR-only fallback pipeline loaded.")
        except Exception:
            try:
                from paddleocr import PaddleOCR
                self.ocr_engine = PaddleOCR(
                    use_angle_cls=True, lang="en"
                )
                self.fallback_mode = True
                print("  PaddleOCR fallback engine loaded.")
            except Exception as e2:
                print(f"  ERROR: No OCR engine available: {e2}")
                sys.exit(1)

    def analyze(self, image_path: str) -> Dict:
        """
        Full analysis: visual prediction + VLM classification.

        Args:
            image_path: Path to the input screenshot.

        Returns:
            Dict with OCR results and VLM classification.
        """
        if not os.path.isfile(image_path):
            print(f"ERROR: Image not found: {image_path}")
            sys.exit(1)

        print(f"\nAnalyzing: {image_path}")
        start_time = time.time()

        if self.pipeline is not None:
            result = self._analyze_with_vlm(image_path)
        else:
            result = self._analyze_ocr_only(image_path)

        result["total_time_sec"] = round(time.time() - start_time, 2)
        return result

    def _analyze_with_vlm(self, image_path: str) -> Dict:
        """Run full VLM pipeline: visual_predict + chat classification."""

        # ── Step 1: Visual prediction (OCR + layout) ──
        print("\n[Step 1/3] Running visual prediction (OCR + Layout)...")
        t0 = time.time()

        visual_info = None
        ocr_texts = []

        try:
            # PP-ChatOCRv3 visual_predict returns visual results + info for chat
            visual_result, visual_info = self.pipeline.visual_predict(
                input=image_path,
                use_doc_image_ori_cls_model=True,
                use_doc_image_unwarp_model=False,
                use_seal_text_det_model=False,
            )

            # Extract OCR results from visual prediction
            for res in visual_result:
                if hasattr(res, "rec_texts") and hasattr(res, "dt_polys"):
                    for i, (text, poly) in enumerate(
                        zip(res.rec_texts, res.dt_polys)
                    ):
                        score = (
                            float(res.rec_scores[i])
                            if hasattr(res, "rec_scores") and i < len(res.rec_scores)
                            else 0.0
                        )
                        poly_list = poly.tolist() if hasattr(poly, "tolist") else poly
                        ocr_texts.append({
                            "index": i,
                            "text": text,
                            "confidence": round(score, 4),
                            "polygon": poly_list,
                        })
                elif hasattr(res, "text") and hasattr(res, "dt_polys"):
                    # Alternative result format
                    for i, poly in enumerate(res.dt_polys):
                        poly_list = poly.tolist() if hasattr(poly, "tolist") else poly
                        ocr_texts.append({
                            "index": i,
                            "text": res.text[i] if i < len(res.text) else "",
                            "confidence": 0.0,
                            "polygon": poly_list,
                        })

        except Exception as e:
            print(f"  Visual prediction encountered an issue: {e}")
            print("  Attempting direct predict fallback...")
            try:
                output = list(self.pipeline.predict(
                    input=image_path,
                    key_list=[DETAILED_CLASSIFICATION_KEY],
                ))
                # Extract whatever results we can
                for res in output:
                    if hasattr(res, "__dict__"):
                        return {
                            "image_path": image_path,
                            "method": "PP-ChatOCRv3-doc (direct predict)",
                            "raw_result": str(res),
                            "ocr_texts": [],
                            "vlm_classification": {},
                        }
            except Exception as e2:
                print(f"  Direct predict also failed: {e2}")

        visual_time = time.time() - t0
        print(f"  Extracted {len(ocr_texts)} text element(s) in {visual_time:.2f}s")

        # ── Step 2: VLM chat — semantic classification ──
        print("\n[Step 2/3] Running VLM classification (LLM chat)...")
        t1 = time.time()

        vlm_results = {}

        if visual_info is not None:
            try:
                # Round 1: Multi-key classification
                chat_result = self.pipeline.chat(
                    key_list=CLASSIFICATION_KEYS,
                    visual_info=visual_info,
                )

                if chat_result:
                    # chat_result is typically a dict mapping keys to answers
                    if isinstance(chat_result, dict):
                        vlm_results["classification"] = chat_result
                    elif hasattr(chat_result, "chat_res"):
                        vlm_results["classification"] = chat_result.chat_res
                    else:
                        vlm_results["classification"] = str(chat_result)

                # Round 2: Detailed per-element classification
                detailed_result = self.pipeline.chat(
                    key_list=[DETAILED_CLASSIFICATION_KEY],
                    visual_info=visual_info,
                )

                if detailed_result:
                    if isinstance(detailed_result, dict):
                        vlm_results["detailed"] = detailed_result
                    elif hasattr(detailed_result, "chat_res"):
                        vlm_results["detailed"] = detailed_result.chat_res
                    else:
                        vlm_results["detailed"] = str(detailed_result)

            except Exception as e:
                print(f"  VLM chat error: {e}")
                vlm_results["error"] = str(e)

        chat_time = time.time() - t1
        print(f"  VLM classification completed in {chat_time:.2f}s")

        # ── Step 3: Structure results ──
        print("\n[Step 3/3] Structuring results...")

        return {
            "image_path": image_path,
            "method": "PP-ChatOCRv3-doc (VLM)",
            "visual_prediction_time_sec": round(visual_time, 2),
            "vlm_chat_time_sec": round(chat_time, 2),
            "num_text_elements": len(ocr_texts),
            "ocr_texts": ocr_texts,
            "vlm_classification": vlm_results,
        }

    def _analyze_ocr_only(self, image_path: str) -> Dict:
        """
        Fallback: OCR-only analysis when VLM/LLM is not available.
        Uses PaddleOCR for detection+recognition, and Unicode analysis
        for script detection.
        """
        import unicodedata
        from collections import Counter

        print("\n[OCR-Only Mode] Running text detection + recognition...")

        ocr_texts = []

        if hasattr(self, "ocr_pipeline"):
            # PaddleX OCR pipeline
            output = list(self.ocr_pipeline.predict(input=image_path))
            for res in output:
                if hasattr(res, "rec_texts"):
                    for i, text in enumerate(res.rec_texts):
                        score = (
                            float(res.rec_scores[i])
                            if hasattr(res, "rec_scores")
                            else 0.0
                        )
                        poly = (
                            res.dt_polys[i].tolist()
                            if hasattr(res, "dt_polys") and i < len(res.dt_polys)
                            else []
                        )
                        ocr_texts.append({
                            "index": i,
                            "text": text,
                            "confidence": round(score, 4),
                            "polygon": poly,
                        })
        elif hasattr(self, "ocr_engine"):
            # PaddleOCR engine
            results = self.ocr_engine.ocr(image_path, cls=True)
            if results and results[0]:
                for i, line in enumerate(results[0]):
                    box, (text, score) = line[0], line[1]
                    ocr_texts.append({
                        "index": i,
                        "text": text,
                        "confidence": round(float(score), 4),
                        "polygon": box,
                    })

        print(f"  Found {len(ocr_texts)} text element(s)")

        # Script detection via Unicode analysis
        for item in ocr_texts:
            script_counts = Counter()
            for ch in item["text"]:
                if ch.isalpha():
                    try:
                        name = unicodedata.name(ch, "")
                        for label in ["LATIN", "ARABIC", "CJK", "HANGUL",
                                      "HIRAGANA", "KATAKANA", "CYRILLIC",
                                      "DEVANAGARI", "THAI", "GREEK", "HEBREW"]:
                            if label in name:
                                mapped = {
                                    "CJK": "Chinese",
                                    "HANGUL": "Korean",
                                    "HIRAGANA": "Japanese",
                                    "KATAKANA": "Japanese",
                                }.get(label, label.capitalize())
                                script_counts[mapped] += 1
                                break
                    except ValueError:
                        pass

            if script_counts:
                dominant = script_counts.most_common(1)[0]
                item["script"] = dominant[0]
                item["script_confidence"] = round(
                    dominant[1] / sum(script_counts.values()), 4
                )
            else:
                item["script"] = "Unknown"
                item["script_confidence"] = 0.0

            # Basic heuristic for system vs user (position-based)
            item["category"] = "detected_text"
            item["note"] = "VLM unavailable — system/user classification requires LLM"

        return {
            "image_path": image_path,
            "method": "OCR-only (no VLM — LLM not configured)",
            "num_text_elements": len(ocr_texts),
            "ocr_texts": ocr_texts,
            "vlm_classification": {
                "note": "LLM not available. Configure QIANFAN_AK/SK env vars "
                        "or pass --llm_params for full VLM classification."
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def draw_results(image_path: str, results: Dict, output_path: str):
    """Draw detected text regions with labels on the image."""
    image = cv2.imread(image_path)
    if image is None:
        return

    for item in results.get("ocr_texts", []):
        poly = item.get("polygon", [])
        text = item.get("text", "")
        script = item.get("script", "")
        conf = item.get("confidence", 0)

        if not poly:
            continue

        pts = np.array(poly, dtype=np.int32)
        if pts.ndim == 1:
            continue

        # Color by script
        colors = {
            "Latin": (0, 200, 0),
            "Arabic": (200, 100, 0),
            "Chinese": (0, 100, 255),
            "Japanese": (255, 100, 100),
            "Korean": (200, 0, 200),
            "Cyrillic": (100, 200, 200),
        }
        color = colors.get(script, (128, 128, 128))

        cv2.polylines(image, [pts], True, color, 2)

        xmin = int(pts[:, 0].min())
        ymin = int(pts[:, 1].min())
        label = f"{script} | {text[:25]}"
        cv2.putText(
            image, label,
            (xmin, max(ymin - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA,
        )

    cv2.imwrite(output_path, image)
    print(f"  Annotated image → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def process_image(
    image_path: str,
    output_dir: str = "vl_output",
    llm_name: str = "ernie-3.5",
    llm_params: Optional[Dict] = None,
    device: str = "cpu",
):
    """Full pipeline: initialize VLM → analyze image → save results."""
    os.makedirs(output_dir, exist_ok=True)
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Initialize analyzer
    analyzer = ScreenshotVLMAnalyzer(
        llm_name=llm_name,
        llm_params=llm_params,
        device=device,
    )

    # Analyze
    results = analyzer.analyze(image_path)

    # Save JSON
    json_path = os.path.join(output_dir, f"{image_name}_vl_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"  JSON results → {json_path}")

    # Save annotated image
    img_path = os.path.join(output_dir, f"{image_name}_vl_annotated.png")
    draw_results(image_path, results, img_path)

    # Console summary
    print("\n" + "=" * 70)
    print("VLM ANALYSIS COMPLETE")
    print(f"  Method:         {results.get('method', 'unknown')}")
    print(f"  Text elements:  {results.get('num_text_elements', 0)}")
    print(f"  Total time:     {results.get('total_time_sec', 0):.2f}s")
    print(f"  Results:        {json_path}")
    print("=" * 70)

    # Print VLM classification if available
    vlm = results.get("vlm_classification", {})
    if vlm and "classification" in vlm:
        print("\n── VLM Classification Output ──")
        cls_data = vlm["classification"]
        if isinstance(cls_data, dict):
            for key, val in cls_data.items():
                print(f"\n[{key[:60]}...]")
                print(f"  {val}")
        else:
            print(cls_data)

    if vlm and "detailed" in vlm:
        print("\n── Detailed Per-Element Classification ──")
        det_data = vlm["detailed"]
        if isinstance(det_data, dict):
            for key, val in det_data.items():
                print(f"\n{val}")
        else:
            print(det_data)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="VLM-based text analysis for screenshots using PP-ChatOCRv3-doc."
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to the input screenshot/image.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="vl_output",
        help="Output directory (default: vl_output).",
    )
    parser.add_argument(
        "--llm_name", type=str, default="ernie-3.5",
        help="LLM model name (default: ernie-3.5). Options: ernie-3.5, ernie-4.0",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Inference device: cpu or gpu:0 (default: cpu).",
    )
    parser.add_argument(
        "--llm_ak", type=str, default=None,
        help="Qianfan API access key (or set QIANFAN_AK env var).",
    )
    parser.add_argument(
        "--llm_sk", type=str, default=None,
        help="Qianfan API secret key (or set QIANFAN_SK env var).",
    )
    args = parser.parse_args()

    # Build LLM params if API keys are provided via CLI
    llm_params = None
    if args.llm_ak and args.llm_sk:
        llm_params = {
            "api_type": "qianfan",
            "ak": args.llm_ak,
            "sk": args.llm_sk,
        }

    process_image(
        image_path=args.image,
        output_dir=args.output_dir,
        llm_name=args.llm_name,
        llm_params=llm_params,
        device=args.device,
    )


if __name__ == "__main__":
    main()

"""
paddleocr_recognition_module.py
===============================
<<<<<<< HEAD
Recognition-only module using PaddleOCR's PP-OCRv6 medium recognition model.
Given a folder of text crop images, it runs text recognition to retrieve
the exact transcribed text, the confidence scores, and any other metadata.
=======
Recognition-only module using PaddleOCR TextRecognition.
Refactored to implement BasePipelineStage for maximum modularity,
supporting batched inference, language parameter from config,
and producing both full and filtered recognition JSON outputs.
>>>>>>> ebebe3b915bce21c9c1fe8697b68cc66ed2f83a7
"""

import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
<<<<<<< HEAD
import sys
import time
import glob
import cv2
import numpy as np

# ──────────────────────────────────────────────
# 1. Imports and Verification
# ──────────────────────────────────────────────
try:
    import paddle
    print(f"PaddlePaddle version : {paddle.__version__}")
except ImportError:
    print("ERROR: paddlepaddle is not installed in the active environment.")
    sys.exit(1)

try:
    from paddleocr import TextRecognition
    print("Successfully imported TextRecognition from paddleocr.")
except ImportError:
    print("ERROR: paddleocr is not installed or TextRecognition module is not found.")
    sys.exit(1)

# ──────────────────────────────────────────────
# 2. Main Recognition Class
# ──────────────────────────────────────────────
class PaddleOCRRecognitionModule:
    def __init__(self, model_name="PP-OCRv6_medium_rec", device="cpu"):
        """
        Initialize the recognition-only engine.
        Using PP-OCRv6 medium recognition model by default.
        """
        print(f"Initializing PaddleOCR TextRecognition ({model_name}) on {device.upper()}...")
        self.recognizer = TextRecognition(model_name=model_name, device=device)

    def recognize_crop(self, crop_path_or_img):
        """
        Predict text and confidence from a single crop image path or numpy array.
        
        Returns:
            dict: Extract containing 'text', 'confidence', and other metadata/raw outputs.
        """
        start_time = time.time()
        
        # Predict accepts file paths directly
        result = self.recognizer.predict(crop_path_or_img)
        
=======
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
try:
    import torch
except Exception:
    pass
import sys
import time
import glob
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np

from pipeline_core import BasePipelineStage, PipelineContext

try:
    import paddle
    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

try:
    from paddleocr import TextRecognition
    HAS_PADDLEOCR = True
except ImportError:
    HAS_PADDLEOCR = False


class PaddleOCRRecognitionModule:
    """
    Recognition engine wrapper supporting PaddleOCR TextRecognition.
    Accepts language configuration parameter ('lang') for targeted script recognition.
    """
    def __init__(self, model_name: str = "PP-OCRv6_medium_rec", lang: str = "en", device: str = "cpu", use_onnx: bool = False):
        self.model_name = model_name
        self.lang = lang
        self.device = device
        self.use_onnx = use_onnx
        
        print(f"Initializing PaddleOCR TextRecognition ({model_name}, lang='{lang}') on {device.upper()} (enable_mkldnn=False)...")
        try:
            self.recognizer = TextRecognition(model_name=model_name, lang=lang, device=device, enable_mkldnn=False)
        except Exception:
            try:
                self.recognizer = TextRecognition(model_name=model_name, device=device, enable_mkldnn=False)
            except Exception:
                self.recognizer = TextRecognition(model_name=model_name, device=device)

    def recognize_crop(self, crop_path_or_img) -> Dict[str, Any]:
        """Recognize text in a single crop image."""
        start_time = time.time()
        result = self.recognizer.predict(crop_path_or_img)
>>>>>>> ebebe3b915bce21c9c1fe8697b68cc66ed2f83a7
        elapsed = time.time() - start_time
        
        text = ""
        confidence = 0.0
        extra_info = {}
        
<<<<<<< HEAD
        # Extract results from the page structure
        # (Handling possible dictionary, page objects or lists returned by PaddleOCR API)
=======
>>>>>>> ebebe3b915bce21c9c1fe8697b68cc66ed2f83a7
        for page in result:
            if hasattr(page, "keys") or isinstance(page, dict):
                rec_texts = page.get("rec_texts", [])
                rec_scores = page.get("rec_scores", [])
<<<<<<< HEAD
                
                # Check for singular format if plural is empty
                if not rec_texts and "rec_text" in page:
                    rec_texts = [page.get("rec_text")]
                    rec_scores = [page.get("rec_score", 0.0)]
                
                # Capture any other key-value pairs if present
                for k, v in page.items():
                    if k not in ["rec_texts", "rec_scores", "rec_text", "rec_score"]:
                        extra_info[k] = str(v)
            elif hasattr(page, "__dict__"):
                rec_texts = getattr(page, "rec_texts", [])
                rec_scores = getattr(page, "rec_scores", [])
                
                # Check for singular format if plural is empty
                if not rec_texts and hasattr(page, "rec_text"):
                    rec_texts = [getattr(page, "rec_text")]
                    rec_scores = [getattr(page, "rec_score", 0.0)]
                
                # Capture other attributes from the object dictionary
                for k, v in page.__dict__.items():
                    if k not in ["rec_texts", "rec_scores", "rec_text", "rec_score"]:
                        extra_info[k] = str(v)
            else:
                # Fallback print if result has a very unexpected shape
                extra_info["raw_page"] = str(page)
                rec_texts, rec_scores = [], []
            
            # Since it's a single crop, we expect 1 main text block sequence
=======
                if not rec_texts and "rec_text" in page:
                    rec_texts = [page.get("rec_text")]
                    rec_scores = [page.get("rec_score", 0.0)]
            elif hasattr(page, "__dict__"):
                rec_texts = getattr(page, "rec_texts", [])
                rec_scores = getattr(page, "rec_scores", [])
                if not rec_texts and hasattr(page, "rec_text"):
                    rec_texts = [getattr(page, "rec_text")]
                    rec_scores = [getattr(page, "rec_score", 0.0)]
            else:
                rec_texts, rec_scores = [], []
            
>>>>>>> ebebe3b915bce21c9c1fe8697b68cc66ed2f83a7
            if len(rec_texts) > 0 and rec_texts[0] is not None:
                text = rec_texts[0]
                confidence = float(rec_scores[0]) if len(rec_scores) > 0 else 0.0
                break
                
        return {
            "text": text,
<<<<<<< HEAD
            "confidence": confidence,
=======
            "confidence": round(confidence, 4),
>>>>>>> ebebe3b915bce21c9c1fe8697b68cc66ed2f83a7
            "latency_seconds": round(elapsed, 4),
            "extra_info": extra_info
        }

<<<<<<< HEAD
# ──────────────────────────────────────────────
# 3. CLI Execution / Processing Folder of Crops
# ──────────────────────────────────────────────
def main():
    # Folder containing cropped text image blocks
    INPUT_DIR = "crops/arabic"
    OUTPUT_DIR = "paddleocr_rec_output"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input crops directory '{INPUT_DIR}' does not exist.")
        # Fallback to current directory crops if missing
        print("Creating an empty 'crops' folder. Please place cropped images inside it and rerun.")
        os.makedirs(INPUT_DIR, exist_ok=True)
        sys.exit(0)

    # Gather crop images
    crop_files = []
    extensions = [".png", ".jpg", ".jpeg", ".bmp"]
    for ext in extensions:
        crop_files.extend(glob.glob(os.path.join(INPUT_DIR, f"*{ext}")))
        crop_files.extend(glob.glob(os.path.join(INPUT_DIR, f"*{ext.upper()}")))

    # Remove duplicates from case variations
    crop_files = sorted(list(set(crop_files)))

    if not crop_files:
        print(f"No crop images found in '{INPUT_DIR}'.")
        print("Please place text crop images (.png, .jpg, .jpeg, .bmp) inside that directory.")
        sys.exit(0)

    print(f"Found {len(crop_files)} text crop(s) to process.")

    # Initialize the recognition engine
    # PP-OCRv6 medium is specified by model_name="PP-OCRv6_medium_rec"
    rec_module = PaddleOCRRecognitionModule(model_name="PP-OCRv6_medium_rec", device="cpu")

    results_summary = []

    print("\n" + "=" * 80)
    print("  RUNNING PP-OCRv6_medium_rec ON CROPS")
    print("=" * 80)

    for idx, filepath in enumerate(crop_files):
        filename = os.path.basename(filepath)
        print(f"[{idx+1}/{len(crop_files)}] Processing crop: {filename}")
        
        # Verify image is valid
        test_img = cv2.imread(filepath)
        if test_img is None:
            print(f"  -> ERROR: cv2 cannot load/decode the image: {filepath}. Skipping.")
            continue
            
        res = rec_module.recognize_crop(filepath)
        
        print(f"  -> Text:       \"{res['text']}\"")
        print(f"  -> Confidence: {res['confidence']:.4f}")
        print(f"  -> Latency:    {res['latency_seconds']:.4f}s")
        if res["extra_info"]:
            print(f"  -> Extra Info: {res['extra_info']}")
        print("-" * 50)

        results_summary.append({
            "filename": filename,
            "filepath": filepath,
            "text": res["text"],
            "confidence": res["confidence"],
            "latency_seconds": res["latency_seconds"],
            "extra_info": res["extra_info"]
        })

    # Save all output items to JSON
    output_json_path = os.path.join(OUTPUT_DIR, "crops_rec_results.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_used": "PP-OCRv6_medium_rec",
            "total_processed": len(results_summary),
            "results": results_summary
        }, f, ensure_ascii=False, indent=2)

    print(f"\nRecognition complete! JSON results saved to: {output_json_path}")
=======
    def recognize_batch(self, crop_paths: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """Process a list of crop paths in batches for high throughput."""
        results = []
        for i in range(0, len(crop_paths), batch_size):
            batch = crop_paths[i:i + batch_size]
            for cp in batch:
                res = self.recognize_crop(cp)
                results.append(res)
        return results


class RecognitionStage(BasePipelineStage):
    """
    Pipeline stage for text recognition with language configuration and dual JSON outputs.
    """
    
    def __init__(self, model_name: str = "PP-OCRv6_medium_rec", lang: str = "en", device: str = "cpu", use_onnx: bool = False):
        self.model_name = model_name
        self.lang = lang
        self.device = device
        self.use_onnx = use_onnx
        self.rec_module = None
        self._init_module()

    def _init_module(self):
        self.rec_module = PaddleOCRRecognitionModule(
            model_name=self.model_name,
            lang=self.lang,
            device=self.device,
            use_onnx=self.use_onnx
        )

    @property
    def name(self) -> str:
        return "RecognitionStage"

    def run(self, context: PipelineContext, config: Dict[str, Any]) -> PipelineContext:
        rec_cfg = config.get("recognition", {})
        basic_threshold = rec_cfg.get("basic_confidence_threshold", 0.50)
        batch_size = config.get("runtime", {}).get("batch_size", 32)

        rec_output_dir = config.get("paths", {}).get("rec_output_dir", "paddleocr_rec_output")
        os.makedirs(rec_output_dir, exist_ok=True)

        crops_info = context.crops_info
        if not crops_info and context.crop_manifest_path and os.path.exists(context.crop_manifest_path):
            with open(context.crop_manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
                crops_info = manifest_data.get("crops", [])

        if not crops_info:
            print(f"[{self.name}] No crops available for recognition.")
            context.recog_results = []
            context.filtered_recog_results = []
            return context

        crop_paths = [c["crop_path"] for c in crops_info]
        print(f"[{self.name}] Running batched recognition on {len(crop_paths)} crop(s)...")

        rec_outputs = self.rec_module.recognize_batch(crop_paths, batch_size=batch_size)

        full_results = []
        filtered_results = []

        for crop_meta, rec_res in zip(crops_info, rec_outputs):
            entry = {
                "crop_filename": crop_meta.get("crop_filename"),
                "crop_path": crop_meta.get("crop_path"),
                "det_index": crop_meta.get("det_index"),
                "det_score": crop_meta.get("det_score"),
                "bbox": crop_meta.get("bbox"),
                "polygon": crop_meta.get("polygon"),
                "crop_size": crop_meta.get("crop_size"),
                "transcribed_text": rec_res["text"],
                "rec_confidence": rec_res["confidence"],
                "rec_latency_sec": rec_res["latency_seconds"],
            }
            full_results.append(entry)

            # Filtered JSON output: only items above or equal to basic_confidence_threshold
            if rec_res["confidence"] >= basic_threshold:
                filtered_results.append(entry)

        # 1. Output Full Recognition JSON
        full_json_path = os.path.join(rec_output_dir, f"{context.image_stem}_rec_full.json")
        with open(full_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "image_stem": context.image_stem,
                "source_image": context.image_path,
                "model_used": self.model_name,
                "language_set": self.lang,
                "total_crops": len(full_results),
                "results": full_results
            }, f, ensure_ascii=False, indent=2)

        # 2. Output Filtered Recognition JSON (final_recog.json)
        filtered_json_path = os.path.join(rec_output_dir, f"{context.image_stem}_final_recog.json")
        with open(filtered_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "image_stem": context.image_stem,
                "source_image": context.image_path,
                "model_used": self.model_name,
                "language_set": self.lang,
                "confidence_threshold": basic_threshold,
                "total_filtered_crops": len(filtered_results),
                "results": filtered_results
            }, f, ensure_ascii=False, indent=2)

        # Update context
        context.recog_full_json_path = full_json_path
        context.recog_filtered_json_path = filtered_json_path
        context.recog_results = full_results
        context.filtered_recog_results = filtered_results

        print(f"[{self.name}] Full recog saved -> {full_json_path}")
        print(f"[{self.name}] Filtered recog ({len(filtered_results)} items >= {basic_threshold}) saved -> {filtered_json_path}")
        return context


# CLI Execution Compatibility
def main():
    config_path = "pipeline_config.json"
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {
            "paths": {"rec_output_dir": "paddleocr_rec_output"},
            "recognition": {"model_name": "PP-OCRv6_medium_rec", "lang": "en", "basic_confidence_threshold": 0.50},
            "runtime": {"device": "cpu", "batch_size": 32}
        }

    rec_cfg = config.get("recognition", {})
    runtime_cfg = config.get("runtime", {})

    stage = RecognitionStage(
        model_name=rec_cfg.get("model_name", "PP-OCRv6_medium_rec"),
        lang=rec_cfg.get("lang", "en"),
        device=runtime_cfg.get("device", "cpu"),
        use_onnx=runtime_cfg.get("use_onnx", False)
    )

    print(f"RecognitionStage initialized with model '{stage.model_name}', lang='{stage.lang}'")

>>>>>>> ebebe3b915bce21c9c1fe8697b68cc66ed2f83a7

if __name__ == "__main__":
    main()

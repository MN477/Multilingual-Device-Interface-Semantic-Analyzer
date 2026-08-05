"""
paddleocr_recognition_module.py
===============================
Recognition-only module using PaddleOCR's PP-OCRv6 medium recognition model.
Given a folder of text crop images, it runs text recognition to retrieve
the exact transcribed text, the confidence scores, and any other metadata.
"""

import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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
        
        elapsed = time.time() - start_time
        
        text = ""
        confidence = 0.0
        extra_info = {}
        
        # Extract results from the page structure
        # (Handling possible dictionary, page objects or lists returned by PaddleOCR API)
        for page in result:
            if hasattr(page, "keys") or isinstance(page, dict):
                rec_texts = page.get("rec_texts", [])
                rec_scores = page.get("rec_scores", [])
                
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
            if len(rec_texts) > 0 and rec_texts[0] is not None:
                text = rec_texts[0]
                confidence = float(rec_scores[0]) if len(rec_scores) > 0 else 0.0
                break
                
        return {
            "text": text,
            "confidence": confidence,
            "latency_seconds": round(elapsed, 4),
            "extra_info": extra_info
        }

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

if __name__ == "__main__":
    main()

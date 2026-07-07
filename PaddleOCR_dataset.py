"""
PaddleOCR_test.py — Test PaddleOCR (3.x) on a folder of images.
"""

import json
import os
# Resolve OpenMP duplicate library issue on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import numpy as np
import cv2
import time
import glob

# ──────────────────────────────────────────────
# 1. Import & verify PaddlePaddle is working
# ──────────────────────────────────────────────
try:
    import paddle
    print(f"PaddlePaddle version : {paddle.__version__}")
except ImportError:
    print("ERROR: paddlepaddle is not installed.")
    sys.exit(1)

try:
    from paddleocr import PaddleOCR
except ImportError:
    print("ERROR: paddleocr is not installed.")
    sys.exit(1)

# ──────────────────────────────────────────────
# 2. Initialize OCR engine (CPU only, English)
# ──────────────────────────────────────────────
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang="en",
    device="cpu",
)

# ──────────────────────────────────────────────
# 3. Configuration
# ──────────────────────────────────────────────
INPUT_DIR = "input_images"
OUTPUT_DIR = "paddleocr_output"

if not os.path.exists(INPUT_DIR):
    print(f"ERROR: Input directory not found → {INPUT_DIR}")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# 4. Process all images in the folder
# ──────────────────────────────────────────────
image_files = [f for f in glob.glob(os.path.join(INPUT_DIR, "*.*")) 
               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

if not image_files:
    print(f"No images found in {INPUT_DIR}")
    sys.exit(0)

print(f"Found {len(image_files)} images in '{INPUT_DIR}'. Processing...")

for image_path in image_files:
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    
    print("\n" + "=" * 80)
    print(f"Processing: {filename}")
    print("=" * 80)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"  ERROR: cv2 could not decode → {image_path}. Skipping.")
        continue

    print(f"  Image loaded: ({image.shape[1]}×{image.shape[0]})")

    start_time = time.time()
    result = ocr.predict(image_path)
    end_time = time.time()
    
    inference_time = end_time - start_time
    print(f"  Model inference time: {inference_time:.3f} seconds")

    all_detections = []

    for page_idx, page in enumerate(result):
        if not hasattr(page, "keys") and not hasattr(page, "__dict__") and not isinstance(page, dict):
            continue

        texts  = page.get("rec_texts",  []) if isinstance(page, dict) else getattr(page, "rec_texts",  [])
        scores = page.get("rec_scores", []) if isinstance(page, dict) else getattr(page, "rec_scores", [])
        polys  = page.get("rec_polys",  []) if isinstance(page, dict) else getattr(page, "rec_polys",  [])

        num_boxes = len(texts)
        print(f"  Detected {num_boxes} text region(s)")

        for i in range(num_boxes):
            text  = texts[i]
            score = float(scores[i]) if i < len(scores) else -1.0
            poly  = np.array(polys[i], dtype=np.float32) if i < len(polys) else None

            if poly is not None and poly.size >= 8:
                xmin, ymin = poly.min(axis=0)
                xmax, ymax = poly.max(axis=0)
                width  = xmax - xmin
                height = ymax - ymin
                cx, cy = xmin + width / 2, ymin + height / 2

                pts_int = poly.astype(int)
                cv2.polylines(image, [pts_int], True, (0, 255, 0), 2)
                label = f"{text} ({score:.2f})"
                label_y = max(int(ymin) - 8, 15)
                cv2.putText(image, label, (int(xmin), label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

                all_detections.append({
                    "index": i,
                    "text": text,
                    "confidence": round(score, 4),
                    "polygon": poly.tolist(),
                    "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)],
                    "width": float(width),
                    "height": float(height),
                    "center": [float(cx), float(cy)],
                })
            else:
                all_detections.append({
                    "index": i,
                    "text": text,
                    "confidence": round(score, 4),
                })

    out_img = os.path.join(OUTPUT_DIR, f"{name}_paddleocr.png")
    cv2.imwrite(out_img, image)
    print(f"  Annotated image saved → {out_img}")

    out_json = os.path.join(OUTPUT_DIR, f"{name}_paddleocr.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "image_path": image_path,
            "image_size": [image.shape[1], image.shape[0]],
            "inference_time_sec": round(inference_time, 3),
            "num_detections": len(all_detections),
            "detections": all_detections,
        }, f, ensure_ascii=False, indent=2)
    print(f"  JSON results saved   → {out_json}")

print(f"\nDone! Processed {len(image_files)} images.")
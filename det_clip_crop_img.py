"""
PaddleOCR_det_test.py — Test only the PaddleOCR text detection model (no recognition).
Outputs detected polygons/bounding boxes, confidence, and annotated images.
"""

import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import numpy as np
import cv2
import time
import glob

# ──────────────────────────────────────────────
# 1. Import & verify
# ──────────────────────────────────────────────
try:
    import paddle
    print(f"PaddlePaddle version : {paddle.__version__}")
except ImportError:
    print("ERROR: paddlepaddle is not installed.")
    sys.exit(1)

try:
    from paddleocr import TextDetection
    print("Using PaddleOCR TextDetection module (detection-only)")
except ImportError:
    print("ERROR: paddleocr is not installed or TextDetection not available.")
    sys.exit(1)

# ──────────────────────────────────────────────
# 2. Initialize detection-only model (CPU)
# ──────────────────────────────────────────────
# Note: PP-OCRv6 uses "medium" instead of "server"
detector = TextDetection(model_name="PP-OCRv6_medium_det", device="cpu")




# ──────────────────────────────────────────────
# 4. Configuration
# ──────────────────────────────────────────────
INPUT_IMG = "input_images"
OUTPUT_DIR = "paddleocr_det_output"

if not os.path.exists(INPUT_DIR):
    print(f"ERROR: Input directory not found -> {INPUT_DIR}")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# 5. Process all images
# ──────────────────────────────────────────────
image_files = [f for f in glob.glob(os.path.join(INPUT_DIR, "*.*"))
               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

if not image_files:
    print(f"No images found in {INPUT_DIR}")
    sys.exit(0)

print(f"Found {len(image_files)} images in '{INPUT_DIR}'. Processing (detection only)...")

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
    


    print(f"  Image size: ({image.shape[1]}×{image.shape[0]})")
# ── Run detection only ──
    start_time = time.time()
    result = detector.predict(image_path)


 
    
    end_time = time.time()
    inference_time = end_time - start_time

    print(f"  Detection inference time: {inference_time:.3f} seconds")


    # ── Extract detection boxes ──
    all_boxes = []
    annotated = image.copy()
    for page_idx, page in enumerate(result):
        # Get the detection polygons from the result
        if hasattr(page, "keys") or isinstance(page, dict):
            dt_polys = page.get("dt_polys", [])
            dt_scores = page.get("dt_scores", [])
        elif hasattr(page, "__dict__"):
            dt_polys = getattr(page, "dt_polys", [])
            dt_scores = getattr(page, "dt_scores", [])
        else:
            # Try to print raw result to understand the format
            print(f"  Unexpected result type: {type(page)}")
            print(f"  Raw: {str(page)[:500]}")
            continue

    num_boxes = len(dt_polys)
    print(f"  Detected {num_boxes} text region(s)")

    for i in range(num_boxes):
        poly = np.array(dt_polys[i], dtype=np.float32)
        score = float(dt_scores[i]) if i < len(dt_scores) else -1.0

        if poly.size >= 8:
            xmin, ymin = poly.min(axis=0)
            xmax, ymax = poly.max(axis=0)
            width = xmax - xmin
            height = ymax - ymin
            cx, cy = xmin + width / 2, ymin + height / 2

            # Draw on annotated image
            pts_int = poly.astype(int)
            cv2.polylines(annotated, [pts_int], True, (0, 255, 0), 2)
            label = f"{score:.2f}"
            cv2.putText(annotated, label, (int(xmin), max(int(ymin) - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

            all_boxes.append({
                "index": i,
                "det_score": round(score, 4),
                "polygon": poly.tolist(),
                "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)],
                "width": float(width),
                "height": float(height),
                "center": [float(cx), float(cy)],
            })

    # ── Save outputs ──
    out_img = os.path.join(OUTPUT_DIR, f"{name}_det.png")
    cv2.imwrite(out_img, annotated)
    print(f"  Annotated image saved -> {out_img}")

    out_json = os.path.join(OUTPUT_DIR, f"{name}_det.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "image_path": image_path,
            "image_size": [image.shape[1], image.shape[0]],
            "inference_time_sec": round(inference_time, 3),
            "num_detections": len(all_boxes),
            "detections": all_boxes,
        }, f, ensure_ascii=False, indent=2)
    print(f"  JSON results saved   -> {out_json}")

print(f"\nDone! Processed {len(image_files)} images (detection only).")

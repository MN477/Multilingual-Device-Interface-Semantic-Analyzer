"""
PaddleOCR_test.py — Test the latest PaddleOCR (3.x) on a single image.

INSTALLATION (run these two commands in order):
    pip install paddlepaddle==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
    pip install paddleocr

Extracts EVERY output the model provides:
  - Recognized text
  - Confidence scores
  - 4-point polygons
  - Axis-aligned bounding boxes (derived)
  - Width / Height / Center (derived)
  - Detection polygons (dt_polys) if available
  - All other keys in the result dictionary
"""

import json
import os
# Resolve OpenMP duplicate library issue on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import numpy as np
import cv2
import time

# ──────────────────────────────────────────────
# 1. Import & verify PaddlePaddle is working
# ──────────────────────────────────────────────
try:
    import paddle
    print(f"PaddlePaddle version : {paddle.__version__}")
    print(f"CUDA available       : {paddle.device.is_compiled_with_cuda()}")
except ImportError:
    print("ERROR: paddlepaddle is not installed.")
    print("Run:  pip install paddlepaddle==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/")
    sys.exit(1)

try:
    from paddleocr import PaddleOCR
    import paddleocr
    print(f"PaddleOCR version    : {paddleocr.__version__}")
except ImportError:
    print("ERROR: paddleocr is not installed.")
    print("Run:  pip install paddleocr")
    sys.exit(1)

# ──────────────────────────────────────────────
# 2. Initialize OCR engine  (CPU only, English)
# ──────────────────────────────────────────────
#    The latest PaddleOCR 3.x uses `device="cpu"`.
#    We keep all optional sub-pipelines OFF to
#    avoid downloading extra models that may fail
#    on CPU-only builds.
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang="multilingual",
    device="cpu",
)

# ──────────────────────────────────────────────
# 3. Load image
# ──────────────────────────────────────────────
image_path = "images/hybrid.png"

if not os.path.exists(image_path):
    print(f"ERROR: Image not found → {image_path}")
    sys.exit(1)

image = cv2.imread(image_path)
if image is None:
    print(f"ERROR: cv2 could not decode → {image_path}")
    sys.exit(1)

print(f"\nImage loaded: {image_path}  ({image.shape[1]}×{image.shape[0]})")

# ──────────────────────────────────────────────
# 4. Run OCR  (predict is the current API)
# ──────────────────────────────────────────────
print("\nRunning paddle OCR (this may take a moment on first run while models download)...\n")
start_time = time.time()
result = ocr.predict(image_path)
end_time = time.time()
print(f"Model inference time: {end_time - start_time:.3f} seconds")

# ──────────────────────────────────────────────
# 5. Print RAW result to see everything
# ──────────────────────────────────────────────
print("=" * 80)
print("  RAW RESULT  (first 3000 chars)")
print("=" * 80)
raw_str = str(result)
print(raw_str[:3000])
if len(raw_str) > 3000:
    print(f"... ({len(raw_str) - 3000} more chars)")
print()

# ──────────────────────────────────────────────
# 6. Iterate over pages and extract all data
# ──────────────────────────────────────────────
all_detections = []     # collect for JSON export

for page_idx, page in enumerate(result):

    # ---- 6a. Print every key available in the page dict ----
    print("=" * 80)
    print(f"  PAGE {page_idx}")
    print("=" * 80)

    # page might be a dict or an object with .keys()
    if hasattr(page, "keys"):
        available_keys = list(page.keys())
    elif hasattr(page, "__dict__"):
        available_keys = list(page.__dict__.keys())
    else:
        print(f"  Unexpected result type: {type(page)}")
        print(f"  Value: {page}")
        continue

    print(f"  Available keys: {available_keys}\n")

    # Print every key and its type/shape/length
    for key in available_keys:
        val = page[key] if isinstance(page, dict) else getattr(page, key)
        if isinstance(val, np.ndarray):
            print(f"    {key:30s}  ndarray  shape={val.shape}  dtype={val.dtype}")
        elif isinstance(val, (list, tuple)):
            print(f"    {key:30s}  {type(val).__name__:8s}  len={len(val)}")
        else:
            print(f"    {key:30s}  {type(val).__name__:8s}  = {val}")
    print()

    # ---- 6b. Extract texts, scores, polygons ----
    # Try the known key names; fall back gracefully
    texts  = page.get("rec_texts",  []) if isinstance(page, dict) else getattr(page, "rec_texts",  [])
    scores = page.get("rec_scores", []) if isinstance(page, dict) else getattr(page, "rec_scores", [])
    polys  = page.get("rec_polys",  []) if isinstance(page, dict) else getattr(page, "rec_polys",  [])

    # Some versions use "dt_polys" for the detection boxes
    dt_polys = page.get("dt_polys", []) if isinstance(page, dict) else getattr(page, "dt_polys", [])

    num_boxes = len(texts)
    print(f"  Detected {num_boxes} text region(s)\n")

    for i in range(num_boxes):
        text  = texts[i]
        score = float(scores[i]) if i < len(scores) else -1.0
        poly  = np.array(polys[i], dtype=np.float32) if i < len(polys) else None

        print("-" * 60)
        print(f"  Detection {i}")
        print(f"    Text       : '{text}'")
        print(f"    Confidence : {score:.4f}")

        if poly is not None and poly.size >= 8:
            # 4-point polygon
            print(f"    Polygon    :")
            for j, pt in enumerate(poly):
                print(f"      P{j}: [{pt[0]:.1f}, {pt[1]:.1f}]")

            # Axis-aligned bounding box
            xmin, ymin = poly.min(axis=0)
            xmax, ymax = poly.max(axis=0)
            width  = xmax - xmin
            height = ymax - ymin
            cx, cy = xmin + width / 2, ymin + height / 2

            print(f"    BBox       : xmin={xmin:.1f}  ymin={ymin:.1f}  xmax={xmax:.1f}  ymax={ymax:.1f}")
            print(f"    Size       : {width:.1f} × {height:.1f}")
            print(f"    Center     : ({cx:.1f}, {cy:.1f})")

            # Draw on image
            pts_int = poly.astype(int)
            cv2.polylines(image, [pts_int], True, (0, 255, 0), 2)
            label = f"{text} ({score:.2f})"
            label_y = max(int(ymin) - 8, 15)
            cv2.putText(image, label, (int(xmin), label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

            # Collect for JSON
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
            print(f"    Polygon    : N/A")
            all_detections.append({
                "index": i,
                "text": text,
                "confidence": round(score, 4),
            })

print()

# ──────────────────────────────────────────────
# 7. Save outputs
# ──────────────────────────────────────────────
os.makedirs("debug_paddle_output", exist_ok=True)

# Annotated image
out_img = "debug_paddle_output/ocr_result_paddle.png"
cv2.imwrite(out_img, image)
print(f"Annotated image saved → {out_img}")

# Full JSON
out_json = "debug_paddle_output/ocr_result_paddle.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump({
        "image_path": image_path,
        "image_size": [image.shape[1], image.shape[0]],
        "num_detections": len(all_detections),
        "detections": all_detections,
    }, f, ensure_ascii=False, indent=2)
print(f"JSON results saved   → {out_json}")

print(f"\nDone!  {len(all_detections)} detections extracted.")
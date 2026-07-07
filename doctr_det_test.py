"""
doctr_det_test.py — Test only the docTR / OnnxTR text detection model (no recognition).
Uses detection_predictor to run detection independently.
"""

import os
import sys
import time
import json
import cv2
import numpy as np
import glob

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ──────────────────────────────────────────────
# Import detection_predictor from onnxtr or doctr
# ──────────────────────────────────────────────
try:
    from onnxtr.models import detection_predictor
    USE_ONNXTR = True
except ImportError:
    try:
        from doctr.models import detection_predictor
        USE_ONNXTR = False
    except ImportError:
        print("ERROR: Neither 'onnxtr' nor 'python-doctr' is installed.")
        sys.exit(1)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
INPUT_DIR = "input_images"
OUTPUT_DIR = "doctr_det_output"

if not os.path.exists(INPUT_DIR):
    print(f"❌ Input directory not found → {INPUT_DIR}")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

image_files = [f for f in glob.glob(os.path.join(INPUT_DIR, "*.*"))
               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

if not image_files:
    print(f"No images found in {INPUT_DIR}")
    sys.exit(0)

# ──────────────────────────────────────────────
# Initialize detection-only predictor
# ──────────────────────────────────────────────
print(f"Initializing {'OnnxTR' if USE_ONNXTR else 'docTR'} detection predictor (detection only)...")
det_predictor = detection_predictor(arch='db_resnet50', assume_straight_pages=True)

print(f"Found {len(image_files)} images. Processing (detection only)...")

for image_path in image_files:
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)

    print("\n" + "=" * 80)
    print(f"Processing: {filename}")
    print("=" * 80)

    orig_img = cv2.imread(image_path)
    if orig_img is None:
        print(f"  ❌ cv2 could not decode → {image_path}. Skipping.")
        continue

    img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    h, w = orig_img.shape[:2]

    # Run detection only
    start_time = time.time()
    det_result = det_predictor([img_rgb])
    end_time = time.time()
    det_time = end_time - start_time

    annotated = orig_img.copy()
    all_boxes = []

    # det_result is a list of pages, each page contains detected boxes
    # Boxes are normalized coordinates (0-1)
    for page_idx, page in enumerate(det_result):
        # page is typically a numpy array of shape (N, 4) for straight boxes
        # or (N, 4, 2) for polygons
        if isinstance(page, dict) and "words" in page:
            boxes = np.array(page["words"])
        elif isinstance(page, np.ndarray):
            boxes = page
        elif hasattr(page, 'export'):
            exported = page.export()
            boxes = np.array(exported) if isinstance(exported, list) else exported
        else:
            boxes = np.array(page)

        if boxes.ndim == 2 and boxes.shape[1] >= 4:
            # Straight boxes: each row is [xmin, ymin, xmax, ymax, (optional) score] normalized
            for idx in range(boxes.shape[0]):
                xmin_n, ymin_n, xmax_n, ymax_n = boxes[idx][:4]
                score = float(boxes[idx][4]) if boxes.shape[1] >= 5 else None
                
                xmin_abs = float(xmin_n * w)
                ymin_abs = float(ymin_n * h)
                xmax_abs = float(xmax_n * w)
                ymax_abs = float(ymax_n * h)
                bw = xmax_abs - xmin_abs
                bh = ymax_abs - ymin_abs

                polygon = [
                    [xmin_abs, ymin_abs],
                    [xmax_abs, ymin_abs],
                    [xmax_abs, ymax_abs],
                    [xmin_abs, ymax_abs],
                ]

                pts = np.array(polygon, dtype=np.int32)
                cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)
                label = f"{idx}" + (f" ({score:.2f})" if score is not None else "")
                cv2.putText(annotated, label, (int(xmin_abs), max(int(ymin_abs) - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

                all_boxes.append({
                    "index": idx,
                    "score": score,
                    "polygon": polygon,
                    "bbox": [xmin_abs, ymin_abs, xmax_abs, ymax_abs],
                    "bbox_normalized": [float(xmin_n), float(ymin_n), float(xmax_n), float(ymax_n)],
                    "width": bw,
                    "height": bh,
                })

        elif boxes.ndim == 3 and boxes.shape[2] >= 2:
            # Polygon boxes: each is (4, 2) normalized
            for idx in range(boxes.shape[0]):
                poly_norm = boxes[idx]
                poly_abs = poly_norm.copy()
                poly_abs[:, 0] *= w
                poly_abs[:, 1] *= h

                xmin_abs, ymin_abs = poly_abs.min(axis=0)
                xmax_abs, ymax_abs = poly_abs.max(axis=0)
                bw = float(xmax_abs - xmin_abs)
                bh = float(ymax_abs - ymin_abs)

                pts = poly_abs.astype(int)
                cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)
                cv2.putText(annotated, f"{idx}", (int(xmin_abs), max(int(ymin_abs) - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

                all_boxes.append({
                    "index": idx,
                    "polygon": poly_abs.tolist(),
                    "bbox": [float(xmin_abs), float(ymin_abs), float(xmax_abs), float(ymax_abs)],
                    "width": bw,
                    "height": bh,
                })
        else:
            print(f"  Unexpected box format: shape={boxes.shape}")

    print(f"  Detected {len(all_boxes)} text regions.")
    print(f"  Detection inference time: {det_time:.3f} seconds")

    out_img = os.path.join(OUTPUT_DIR, f"{name}_det.png")
    cv2.imwrite(out_img, annotated)
    print(f"  Annotated image saved → {out_img}")

    out_json = os.path.join(OUTPUT_DIR, f"{name}_det.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "image_path": image_path,
            "image_size": [w, h],
            "engine": "onnxtr" if USE_ONNXTR else "doctr",
            "inference_time_sec": round(det_time, 3),
            "num_detections": len(all_boxes),
            "detections": all_boxes,
        }, f, ensure_ascii=False, indent=2)
    print(f"  JSON saved → {out_json}")

print(f"\nDone! Processed {len(image_files)} images (detection only).")

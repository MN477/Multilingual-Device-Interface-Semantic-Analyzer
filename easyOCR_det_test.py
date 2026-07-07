"""
easyOCR_det_test.py — Test only the EasyOCR text detector (CRAFT) without recognition.
Outputs detected polygons/bounding boxes and annotated images.
"""

import cv2
import json
import os
import glob
import time
import numpy as np

# =========================
# CONFIG
# =========================
INPUT_DIR = "input_images"
OUTPUT_DIR = "easyocr_det_output"
LANGUAGES = ['en']

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# INIT EASYOCR — DETECTION ONLY (recognizer=False)
# =========================
import easyocr
print("Initializing EasyOCR (detection only, CPU)...")
reader = easyocr.Reader(LANGUAGES, gpu=False, recognizer=False)

image_files = [f for f in glob.glob(os.path.join(INPUT_DIR, "*.*"))
               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

if not image_files:
    print(f"No images found in {INPUT_DIR}")
    exit(0)

print(f"Found {len(image_files)} images. Processing (detection only)...")

for image_path in image_files:
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)

    print("\n" + "=" * 80)
    print(f"Processing: {filename}")
    print("=" * 80)

    image = cv2.imread(image_path)
    if image is None:
        print(f"  ERROR: OpenCV could not read {image_path}. Skipping.")
        continue

    # EasyOCR detect() returns (horizontal_list, free_list) per image
    # horizontal_list: list of [x_min, x_max, y_min, y_max]
    # free_list: list of [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] (rotated boxes)
    start_time = time.time()
    horizontal_list, free_list = reader.detect(image)
    end_time = time.time()
    inference_time = end_time - start_time

    annotated = image.copy()
    all_boxes = []
    idx = 0

    # Process horizontal (axis-aligned) boxes
    if horizontal_list and len(horizontal_list) > 0:
        for box in horizontal_list[0]:  # first image in batch
            x_min, x_max, y_min, y_max = box
            x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
            w = x_max - x_min
            h = y_max - y_min
            polygon = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]

            cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(annotated, f"H{idx}", (x_min, max(y_min - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

            all_boxes.append({
                "index": idx,
                "type": "horizontal",
                "polygon": polygon,
                "bbox": [x_min, y_min, x_max, y_max],
                "width": w,
                "height": h,
            })
            idx += 1

    # Process free-form (rotated) boxes
    if free_list and len(free_list) > 0:
        for box in free_list[0]:  # first image in batch
            poly = np.array(box, dtype=np.int32)
            xmin, ymin = poly.min(axis=0)
            xmax, ymax = poly.max(axis=0)
            w = int(xmax - xmin)
            h = int(ymax - ymin)

            cv2.polylines(annotated, [poly], True, (255, 0, 0), 2)
            cv2.putText(annotated, f"F{idx}", (int(xmin), max(int(ymin) - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

            all_boxes.append({
                "index": idx,
                "type": "free",
                "polygon": poly.tolist(),
                "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)],
                "width": w,
                "height": h,
            })
            idx += 1

    print(f"  Detected {len(all_boxes)} text regions.")
    print(f"  Detection inference time: {inference_time:.3f} seconds")

    out_img = os.path.join(OUTPUT_DIR, f"{name}_det.png")
    cv2.imwrite(out_img, annotated)
    print(f"  Annotated image saved → {out_img}")

    out_json = os.path.join(OUTPUT_DIR, f"{name}_det.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "image_path": image_path,
            "image_size": [image.shape[1], image.shape[0]],
            "inference_time_sec": round(inference_time, 3),
            "num_detections": len(all_boxes),
            "detections": all_boxes,
        }, f, indent=4, ensure_ascii=False)
    print(f"  JSON saved → {out_json}")

print(f"\nDone! Processed {len(image_files)} images (detection only).")

"""
rapidOCR_det_test.py — Test only the RapidOCR text detector (no recognition).
Uses use_rec=False to run the detection model independently.
"""

import cv2
import json
import os
import time
import sys
import numpy as np
import glob

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:
    print("ERROR: 'rapidocr_onnxruntime' is not installed.")
    print("Run: pip install rapidocr_onnxruntime")
    sys.exit(1)

# =========================
# CONFIG
# =========================
INPUT_DIR = "input_images"
OUTPUT_DIR = "rapidocr_det_output"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_DIR):
        print(f"ERROR: Input directory not found → {INPUT_DIR}")
        sys.exit(1)

    image_files = [f for f in glob.glob(os.path.join(INPUT_DIR, "*.*"))
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if not image_files:
        print(f"No images found in {INPUT_DIR}")
        sys.exit(0)

    # Initialize RapidOCR with recognition disabled
    print("Initializing RapidOCR engine (detection only)...")
    engine = RapidOCR(use_det=True, use_cls=False, use_rec=False)

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

        start_time = time.time()
        result, elapse = engine(image_path)
        end_time = time.time()
        inference_time = end_time - start_time

        annotated = image.copy()
        all_boxes = []

        # When use_rec=False, result contains only detection boxes
        # Format may be list of boxes (each box is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
        if result is None:
            print("  No text regions detected.")
            result = []

        for idx, item in enumerate(result):
            # Depending on version, item might be just a box, or (box, text, score)
            if isinstance(item, (list, np.ndarray)):
                # Pure detection: item is just the polygon
                poly = np.array(item, dtype=np.float32)
                if poly.ndim == 1:
                    # Flat array, skip
                    continue
            elif isinstance(item, tuple) and len(item) >= 1:
                poly = np.array(item[0], dtype=np.float32)
            else:
                print(f"  Unexpected item format at index {idx}: {type(item)}")
                continue

            if poly.size < 8:
                continue

            xmin, ymin = poly.min(axis=0)
            xmax, ymax = poly.max(axis=0)
            w = float(xmax - xmin)
            h = float(ymax - ymin)

            pts_int = poly.astype(int)
            cv2.polylines(annotated, [pts_int], True, (0, 255, 0), 2)
            cv2.putText(annotated, f"{idx}", (int(xmin), max(int(ymin) - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

            all_boxes.append({
                "index": idx,
                "polygon": poly.tolist(),
                "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)],
                "width": w,
                "height": h,
            })

        print(f"  Detected {len(all_boxes)} text regions.")
        print(f"  Detection inference time: {inference_time:.3f} seconds")
        if elapse is not None:
            print(f"  RapidOCR internal elapse: {elapse}")

        out_img = os.path.join(OUTPUT_DIR, f"{name}_det.png")
        cv2.imwrite(out_img, annotated)
        print(f"  Annotated image saved → {out_img}")

        out_json = os.path.join(OUTPUT_DIR, f"{name}_det.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({
                "image_path": image_path,
                "image_size": [image.shape[1], image.shape[0]],
                "inference_time_sec": round(inference_time, 3),
                "rapidocr_elapse": str(elapse) if elapse else None,
                "num_detections": len(all_boxes),
                "detections": all_boxes,
            }, f, indent=4, ensure_ascii=False)
        print(f"  JSON saved → {out_json}")

    print(f"\nDone! Processed {len(image_files)} images (detection only).")

if __name__ == "__main__":
    main()

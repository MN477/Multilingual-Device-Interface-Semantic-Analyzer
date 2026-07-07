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
OUTPUT_DIR = "rapidocr_output"
CROP_DIR = os.path.join(OUTPUT_DIR, "crops")
CONF_THRESHOLD = 0.0

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CROP_DIR, exist_ok=True)

    if not os.path.exists(INPUT_DIR):
        print(f"ERROR: Input directory not found → {INPUT_DIR}")
        sys.exit(1)
        
    image_files = [f for f in glob.glob(os.path.join(INPUT_DIR, "*.*")) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if not image_files:
        print(f"No images found in {INPUT_DIR}")
        sys.exit(0)

    print("Initializing RapidOCR engine...")
    engine = RapidOCR()

    for image_path in image_files:
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        
        print("\n" + "=" * 80)
        print(f"Processing: {filename}")
        print("=" * 80)

        image = cv2.imread(image_path)
        if image is None:
            print(f"  ERROR: OpenCV could not read the image → {image_path}. Skipping.")
            continue

        orig_image = image.copy()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        start_time = time.time()
        result, elapse = engine(image_path)
        end_time = time.time()
        inference_time = end_time - start_time

        structured_output = []

        if result is None:
            print("  No text detected.")
            result = []

        for idx, line in enumerate(result):
            if len(line) == 3:
                bbox, text, confidence = line
            else:
                continue

            if confidence < CONF_THRESHOLD:
                continue

            polygon = [(int(pt[0]), int(pt[1])) for pt in bbox]
            x_coords = [p[0] for p in polygon]
            y_coords = [p[1] for p in polygon]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            crop = orig_image[y_min:y_max, x_min:x_max]
            if crop.size > 0:
                crop_filename = f"{CROP_DIR}/{name}_crop_{idx}.png"
                cv2.imwrite(crop_filename, crop)
            else:
                crop_filename = None

            structured_output.append({
                "id": idx,
                "text": text,
                "confidence": float(confidence),
                "polygon": polygon,
                "bbox": {
                    "x_min": x_min, "y_min": y_min,
                    "x_max": x_max, "y_max": y_max
                },
                "crop_path": crop_filename
            })

        print(f"  Detected {len(structured_output)} text regions.")
        print(f"  Model inference time (measured): {inference_time:.3f} seconds")

        out_json = os.path.join(OUTPUT_DIR, f"{name}_rapidocr.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({
                "image_path": image_path,
                "inference_time_sec": round(inference_time, 3),
                "rapidocr_elapse": elapse,
                "num_detections": len(structured_output),
                "detections": structured_output
            }, f, indent=4, ensure_ascii=False)

        vis_image = image_rgb.copy()
        for item in structured_output:
            pts = np.array(item["polygon"], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(vis_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(vis_image, item["text"], item["polygon"][0], 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

        out_img = os.path.join(OUTPUT_DIR, f"{name}_rapidocr.png")
        vis_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_img, vis_bgr)
        print(f"  Annotated image saved to: {out_img}")

    print(f"\nDone! Processed {len(image_files)} images.")

if __name__ == "__main__":
    main()

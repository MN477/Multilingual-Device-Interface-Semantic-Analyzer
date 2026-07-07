import easyocr
import cv2
import json
import os
import glob
import time

# =========================
# CONFIG
# =========================
INPUT_DIR = "input_images"
OUTPUT_DIR = "easyocr_output"
CROP_DIR = os.path.join(OUTPUT_DIR, "crops")
LANGUAGES = ['ch_sim', 'en']  # add others if needed
CONF_THRESHOLD = 0.0

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)

# =========================
# INIT EASYOCR (CPU ONLY)
# =========================
print("Initializing EasyOCR (CPU)...")
reader = easyocr.Reader(LANGUAGES, gpu=False)

image_files = [f for f in glob.glob(os.path.join(INPUT_DIR, "*.*")) 
               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

if not image_files:
    print(f"No images found in {INPUT_DIR}")
    exit(0)

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

    orig_image = image.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    start_time = time.time()
    results = reader.readtext(image)
    end_time = time.time()
    inference_time = end_time - start_time

    structured_output = []

    for idx, (bbox, text, confidence) in enumerate(results):
        if confidence < CONF_THRESHOLD:
            continue

        polygon = [(int(x), int(y)) for x, y in bbox]
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
    print(f"  Model inference time: {inference_time:.3f} seconds")

    out_json = os.path.join(OUTPUT_DIR, f"{name}_easyocr.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "image_path": image_path,
            "inference_time_sec": round(inference_time, 3),
            "num_detections": len(structured_output),
            "detections": structured_output
        }, f, indent=4, ensure_ascii=False)
    print(f"  JSON saved to: {out_json}")

    vis_image = image_rgb.copy()
    for item in structured_output:
        pts = item["polygon"]
        text = item["text"]
        for i in range(4):
            cv2.line(vis_image, pts[i], pts[(i + 1) % 4], (0, 255, 0), 2)
        cv2.putText(vis_image, text, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    out_img = os.path.join(OUTPUT_DIR, f"{name}_easyocr.png")
    cv2.imwrite(out_img, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    print(f"  Annotated image saved to: {out_img}")

print(f"\nDone! Processed {len(image_files)} images.")
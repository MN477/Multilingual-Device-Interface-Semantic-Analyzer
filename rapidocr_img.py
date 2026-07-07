import cv2
import json
import os
import time
import sys
import numpy as np
import matplotlib.pyplot as plt

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:
    print("ERROR: 'rapidocr_onnxruntime' is not installed.")
    print("Run: pip install rapidocr_onnxruntime")
    sys.exit(1)

# =========================
# CONFIG
# =========================
IMAGE_PATH = "images/13902.jpg"  # <-- change this to your image path
OUTPUT_JSON = "rapidocr_output.json"
CROP_DIR = "crops_rapidocr"
CONF_THRESHOLD = 0.0  # change if you want to filter out low-confidence results

def main():
    # =========================
    # PREPARE
    # =========================
    os.makedirs(CROP_DIR, exist_ok=True)

    # Load image
    if not os.path.exists(IMAGE_PATH):
        print(f"ERROR: Image not found → {IMAGE_PATH}")
        sys.exit(1)
        
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"ERROR: OpenCV could not read the image → {IMAGE_PATH}")
        sys.exit(1)

    orig_image = image.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # =========================
    # INIT RapidOCR
    # =========================
    print("Initializing RapidOCR engine...")
    engine = RapidOCR()

    # =========================
    # RUN OCR
    # =========================
    print(f"\nRunning RapidOCR on {IMAGE_PATH}...")
    start_time = time.time()
    
    # engine returns a tuple: (result, elapse)
    result, elapse = engine(IMAGE_PATH)
    
    end_time = time.time()
    inference_time = end_time - start_time

    # =========================
    # PROCESS RESULTS
    # =========================
    structured_output = []

    print("\n" + "=" * 40)
    print("===== OCR RESULTS =====")
    print("=" * 40)

    # result is None if no text was detected
    if result is None:
        print("No text detected.")
        result = []

    for idx, line in enumerate(result):
        # RapidOCR returns: [box, text, score] for each detected text block
        # box format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        if len(line) == 3:
            bbox, text, confidence = line
        else:
            print(f"Unexpected result format: {line}")
            continue

        if confidence < CONF_THRESHOLD:
            continue

        # Polygon points
        polygon = [(int(pt[0]), int(pt[1])) for pt in bbox]

        # Rectangle bbox
        x_coords = [p[0] for p in polygon]
        y_coords = [p[1] for p in polygon]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Crop
        crop = orig_image[y_min:y_max, x_min:x_max]
        if crop.size > 0:
            crop_filename = f"{CROP_DIR}/crop_{idx}.png"
            cv2.imwrite(crop_filename, crop)
        else:
            crop_filename = None

        # Print info
        print(f"[{idx}] TEXT: {text}")
        print(f"    CONFIDENCE: {confidence:.4f}")
        print(f"    POLYGON: {polygon}")
        print(f"    RECT: (x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max})")
        if crop_filename:
            print(f"    CROP SAVED: {crop_filename}")
        print("")

        # Store structured data
        structured_output.append({
            "id": idx,
            "text": text,
            "confidence": float(confidence),
            "polygon": polygon,
            "bbox": {
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max
            },
            "crop_path": crop_filename
        })

    # Print Time
    print("=" * 40)
    print(f"Model inference time (measured): {inference_time:.3f} seconds")
    if elapse is not None:
        print(f"RapidOCR internal elapse breakdown: {elapse}")
    print("=" * 40)

    # =========================
    # SAVE JSON
    # =========================
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(structured_output, f, indent=4, ensure_ascii=False)

    print(f"\n✅ JSON saved to: {OUTPUT_JSON}")

    # =========================
    # VISUALIZATION
    # =========================
    vis_image = image_rgb.copy()

    for item in structured_output:
        pts = np.array(item["polygon"], np.int32)
        pts = pts.reshape((-1, 1, 2))
        text = item["text"]

        # Draw polygon
        cv2.polylines(vis_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Put text
        cv2.putText(
            vis_image,
            text,
            item["polygon"][0],  # using the first point of polygon as origin
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )

    out_img = "rapidocr_output.png"
    vis_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_img, vis_bgr)
    print(f"✅ Annotated image saved to: {out_img}")

    plt.figure(figsize=(10, 10))
    plt.imshow(vis_image)
    plt.axis('off')
    plt.title("RapidOCR Output")
    plt.show()

if __name__ == "__main__":
    main()
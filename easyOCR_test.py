import easyocr
import cv2
import json
import os
import matplotlib.pyplot as plt
import time

# =========================
# CONFIG
# =========================
IMAGE_PATH = "images/2561.jpg"        # <-- change this
OUTPUT_JSON = "easyocr_output.json"
CROP_DIR = "crops"
LANGUAGES = ['en']             # add 'fr', 'ar' if needed
CONF_THRESHOLD = 0.0           # change if you want filtering

# =========================
# PREPARE
# =========================
os.makedirs(CROP_DIR, exist_ok=True)

# Load image
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise ValueError("Image not found. Check path.")

orig_image = image.copy()
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# =========================
# INIT EASYOCR (CPU ONLY)
# =========================
reader = easyocr.Reader(LANGUAGES, gpu=False)

# =========================
# RUN OCR
# =========================
print("\nRunning easyOCR (this may take a moment on first run while models download)...\n")
start_time = time.time()
results = reader.readtext(image)
end_time = time.time()
print(f"Model inference time: {end_time - start_time:.3f} seconds")

# =========================
# PROCESS RESULTS
# =========================
structured_output = []

print("\n===== OCR RESULTS =====\n")

for idx, (bbox, text, confidence) in enumerate(results):

    if confidence < CONF_THRESHOLD:
        continue

    # Polygon points
    polygon = [(int(x), int(y)) for x, y in bbox]

    # Rectangle bbox
    x_coords = [p[0] for p in polygon]
    y_coords = [p[1] for p in polygon]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Crop
    crop = orig_image[y_min:y_max, x_min:x_max]
    crop_filename = f"{CROP_DIR}/crop_{idx}.png"
    cv2.imwrite(crop_filename, crop)

    # Print info
    print(f"[{idx}] TEXT: {text}")
    print(f"    CONFIDENCE: {confidence:.4f}")
    print(f"    POLYGON: {polygon}")
    print(f"    RECT: (x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max})")
    print(f"    CROP SAVED: {crop_filename}\n")

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
    pts = item["polygon"]
    text = item["text"]

    # Draw polygon
    for i in range(4):
        cv2.line(vis_image, pts[i], pts[(i + 1) % 4], (0, 255, 0), 2)

    # Put text
    cv2.putText(
        vis_image,
        text,
        pts[0],
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )

plt.figure(figsize=(10, 10))
plt.imshow(vis_image)
plt.axis('off')
plt.title("EasyOCR Output")
plt.show()

# =========================
# OPTIONAL: SORTING (TOP TO BOTTOM)
# =========================
sorted_output = sorted(structured_output, key=lambda x: x["bbox"]["y_min"])

print("\n===== SORTED TEXT (TOP → BOTTOM) =====\n")
for item in sorted_output:
    print(item["text"])
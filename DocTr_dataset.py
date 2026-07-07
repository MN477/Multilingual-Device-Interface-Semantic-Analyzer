import os
import sys
import time
import json
import cv2
import numpy as np
import glob

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    from onnxtr.models import ocr_predictor
    USE_ONNXTR = True
except ImportError:
    try:
        from doctr.models import ocr_predictor
        USE_ONNXTR = False
    except ImportError:
        print("ERROR: Neither 'onnxtr' nor 'python-doctr' is installed.")
        sys.exit(1)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
INPUT_DIR = "images"
OUTPUT_DIR = "doctr_output"

if not os.path.exists(INPUT_DIR):
    print(f"❌ Input directory not found → {INPUT_DIR}")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

image_files = [f for f in glob.glob(os.path.join(INPUT_DIR, "*.*")) 
               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

if not image_files:
    print(f"No images found in {INPUT_DIR}")
    sys.exit(0)

print(f"Initializing {'OnnxTR' if USE_ONNXTR else 'docTR'} OCR predictor...")
predictor = ocr_predictor(pretrained=True, assume_straight_pages=True)

def rel_to_abs(rel_bbox, height, width):
    (xmin, ymin), (xmax, ymax) = rel_bbox
    return [
        float(xmin * width), float(ymin * height),
        float(xmax * width), float(ymax * height)
    ]

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

    start_time = time.time()
    result = predictor([img_rgb]) 
    end_time = time.time()
    model_time = end_time - start_time

    doc = result[0]
    detections = []
    annotated = orig_img.copy()

    for page_idx, page in enumerate(doc.pages):
        page_height, page_width = page.dimensions
        for block_idx, block in enumerate(page.blocks):
            for line_idx, line in enumerate(block.lines):
                for word_idx, word in enumerate(line.words):
                    bbox_abs = rel_to_abs(word.geometry, h, w)
                    x0, y0, x1, y1 = bbox_abs
                    polygon = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

                    confidence = float(word.confidence) if hasattr(word, "confidence") else None

                    detections.append({
                        "text": word.value,
                        "confidence": confidence,
                        "polygon": polygon,
                        "bbox": bbox_abs,
                        "geometry": word.geometry,
                    })

                    pts = np.array(polygon, dtype=np.int32)
                    cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                    cv2.putText(annotated, word.value, (int(x0), int(y0) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    print(f"  Detected {len(detections)} text regions.")
    print(f"  Model inference time: {model_time:.3f} seconds")

    out_img = os.path.join(OUTPUT_DIR, f"{name}_doctr.png")
    cv2.imwrite(out_img, annotated)
    print(f"  Annotated image saved → {out_img}")

    output = {
        "image_path": image_path,
        "image_size": [w, h],
        "engine": "onnxtr" if USE_ONNXTR else "doctr",
        "inference_time_sec": round(model_time, 3),
        "num_words": len(detections),
        "detections": detections,
        "document": doc.export(),
    }

    out_json = os.path.join(OUTPUT_DIR, f"{name}_doctr.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  Full JSON saved → {out_json}")

print(f"\nDone! Processed {len(image_files)} images.")

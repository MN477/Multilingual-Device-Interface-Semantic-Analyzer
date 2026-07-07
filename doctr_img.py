import os
import sys
import time
import json
import cv2
import numpy as np

# ──────────────────────────────────────────────
# 0. Environment setup (avoid duplicate OpenMP warnings on Windows)
# ──────────────────────────────────────────────
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ──────────────────────────────────────────────
# 1. Import OnnxTR / docTR
# ──────────────────────────────────────────────
# To get the absolute lowest latency on CPU, we use the dedicated 'onnxtr' package
# which runs exported docTR models purely in ONNX Runtime (no PyTorch/TF overhead).
try:
    from onnxtr.models import ocr_predictor
    USE_ONNXTR = True
except ImportError:
    print("WARNING: 'onnxtr' is not installed. To get the lowest latent ONNX execution, run:")
    print("    pip install onnxtr[cpu]")
    print("Falling back to standard 'doctr' if available...\n")
    try:
        from doctr.models import ocr_predictor
        USE_ONNXTR = False
    except ImportError:
        print("ERROR: Neither 'onnxtr' nor 'python-doctr' is installed.")
        sys.exit(1)

# ──────────────────────────────────────────────
# 2. Configuration
# ──────────────────────────────────────────────
IMAGE_PATH = "images/noisy.jpg"  # change as needed
OUTPUT_JSON = "doctr_output.json"
OUTPUT_IMG = "doctr_output.png"

# ──────────────────────────────────────────────
# 3. Load image
# ──────────────────────────────────────────────
if not os.path.isfile(IMAGE_PATH):
    print(f" not found → {IMAGE_PATH}")
    sys.exit(1)

orig_img = cv2.imread(IMAGE_PATH)
if orig_img is None:
    print(f" cv2 could not decode → {IMAGE_PATH}")
    sys.exit(1)

# Both docTR and OnnxTR work with RGB numpy arrays
img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

# ──────────────────────────────────────────────
# 4. Initialize the OCR predictor
# ──────────────────────────────────────────────
print(f"Initializing {'OnnxTR' if USE_ONNXTR else 'docTR'} OCR predictor...")
# assume_straight_pages=True forces simple axis-aligned bounding boxes
predictor = ocr_predictor(assume_straight_pages=True)

# ──────────────────────────────────────────────
# 5. Run inference and measure model‑only time
# ──────────────────────────────────────────────
print(f"\nRunning OCR inference (using {'OnnxTR (ONNX Runtime)' if USE_ONNXTR else 'standard docTR'})...")
start_time = time.time()
result = predictor([img_rgb]) # Returns a list containing one Document object
end_time = time.time()
model_time = end_time - start_time

doc = result

# ──────────────────────────────────────────────
# 6. Extract every piece of information
# ──────────────────────────────────────────────
def rel_to_abs(rel_bbox, height, width):
    """Convert normalized (0‑1) geometry to absolute pixel coordinates."""
    (xmin, ymin), (xmax, ymax) = rel_bbox
    return [
        float(xmin * width),
        float(ymin * height),
        float(xmax * width),
        float(ymax * height),
    ]

h, w = orig_img.shape[:2]
detections = []
annotated = orig_img.copy()

# The Document object contains hierarchical data: pages → blocks → lines → words
# (I noticed you accidentally deleted the 'blocks' loop in your local file! 
# It has been properly restored below.)
for page_idx, page in enumerate(doc.pages):
    page_height, page_width = page.dimensions
    for block_idx, block in enumerate(page.blocks):
        for line_idx, line in enumerate(block.lines):
            for word_idx, word in enumerate(line.words):
                # Geometry is normalized → convert to pixel coordinates
                bbox_abs = rel_to_abs(word.geometry, h, w)
                
                x0, y0, x1, y1 = bbox_abs
                polygon = [
                    [x0, y0],
                    [x1, y0],
                    [x1, y1],
                    [x0, y1],
                ]

                # Confidence is stored as word.confidence
                confidence = float(word.confidence) if hasattr(word, "confidence") else None

                # Store text & all extracted meta‑data
                detections.append(
                    {
                        "page_index": page_idx,
                        "block_index": block_idx,
                        "line_index": line_idx,
                        "word_index": word_idx,
                        "text": word.value,
                        "confidence": confidence,
                        "polygon": polygon,
                        "bbox": bbox_abs,
                        "geometry": word.geometry,   # raw normalized tuple
                    }
                )

                # Draw the polygon on the image
                pts = np.array(polygon, dtype=np.int32)
                cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                # Put the recognised word near the top‑left corner
                cv2.putText(
                    annotated,
                    word.value,
                    (int(x0), int(y0) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

# ──────────────────────────────────────────────
# 7. Save JSON output and image
# ──────────────────────────────────────────────
cv2.imwrite(OUTPUT_IMG, annotated)
print(f"✅ Annotated image saved → {OUTPUT_IMG}")

output = {
    "image_path": IMAGE_PATH,
    "image_size": [w, h],
    "engine": "onnxtr" if USE_ONNXTR else "doctr",
    "inference_time_sec": round(model_time, 3),
    "num_words": len(detections),
    "detections": detections,
    # Export the whole Document object for completeness
    "document": doc.export(),
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"✅ Full JSON saved → {OUTPUT_JSON}")

# ──────────────────────────────────────────────
# 8. Quick human‑readable printout
# ──────────────────────────────────────────────
print("\n" + "=" * 80)
print(f"===== OCR RESULTS ({output['engine']}) =====")
print("=" * 80)
for d in detections:
    print(f"Word: '{d['text']}'  –  Conf: {d['confidence']:.3f}  –  BBox: {d['bbox']}")
print("=" * 80)

print(f"\n🕒 Model inference time: {model_time:.3f} seconds")
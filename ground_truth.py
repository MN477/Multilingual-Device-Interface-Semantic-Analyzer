import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import cv2
import numpy as np
from paddleocr import PaddleOCR

# =====================================================
# CONFIGURATION
# =====================================================

INPUT_FOLDER = "sec_grtr"
OUTPUT_FOLDER = "GrTr_images"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =====================================================
# LOAD OCR
# =====================================================

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang="en",
    device="cpu"
)

# =====================================================
# PROCESS IMAGES
# =====================================================

valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

for filename in os.listdir(INPUT_FOLDER):

    if not filename.lower().endswith(valid_ext):
        continue

    image_path = os.path.join(INPUT_FOLDER, filename)

    print(f"Processing {filename}")

    image = cv2.imread(image_path)

    if image is None:
        print("Cannot read image.")
        continue

    result = ocr.predict(image_path)

    if len(result) == 0:
        continue

    page = result[0]

    json_data = []

    vis = image.copy()

    for i in range(len(page["rec_texts"])):

        text = page["rec_texts"][i]
        confidence = float(page["rec_scores"][i])

        polygon = np.array(page["rec_polys"][i])

        xs = polygon[:,0]
        ys = polygon[:,1]

        xmin = float(xs.min())
        xmax = float(xs.max())
        ymin = float(ys.min())
        ymax = float(ys.max())

        width = xmax - xmin
        height = ymax - ymin

        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2

        annotation = {

            "id": i,

            "text": text,

            "confidence": confidence,

            "polygon": polygon.tolist(),

            "bbox": {
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax
            },

            "geometry": {

                "width": width,

                "height": height,

                "area": width * height,

                "center_x": center_x,

                "center_y": center_y,

                "center_x_norm": center_x / image.shape[1],

                "center_y_norm": center_y / image.shape[0]
            }

        }

        json_data.append(annotation)

        pts = polygon.astype(np.int32)

        cv2.polylines(
            vis,
            [pts],
            True,
            (0,255,0),
            2
        )

        cv2.putText(
            vis,
            str(i),
            (int(xmin), int(ymin)-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,0,255),
            1
        )

    # ---------------------------------------
    # Save JSON
    # ---------------------------------------

    json_name = os.path.splitext(filename)[0] + ".json"

    with open(
        os.path.join(OUTPUT_FOLDER, json_name),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            json_data,
            f,
            indent=4,
            ensure_ascii=False
        )

    # ---------------------------------------
    # Save visualization
    # ---------------------------------------

    vis_name = os.path.splitext(filename)[0] + "_vis.png"

    cv2.imwrite(
        os.path.join(OUTPUT_FOLDER, vis_name),
        vis
    )

print("Finished")
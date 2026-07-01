import os
import json
import cv2

# ===== CONFIG =====
GT_FOLDER = "GrTr_images"          # contains json files
IMAGES_FOLDER = "input_images"     # original clean images
OUTPUT_FILE = "label_studio_tasks.json"
IMAGE_EXT = ".jpg"                # change if needed
# ==================


def convert_bbox_to_percent(bbox, img_w, img_h):
    x = (bbox["xmin"] / img_w) * 100
    y = (bbox["ymin"] / img_h) * 100
    width = ((bbox["xmax"] - bbox["xmin"]) / img_w) * 100
    height = ((bbox["ymax"] - bbox["ymin"]) / img_h) * 100
    return x, y, width, height


def process_image(json_path, image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Cannot load image: {image_path}")
        return None

    h, w = image.shape[:2]

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    for item in data:
        bbox = item["bbox"]
        text = item["text"]

        x, y, width, height = convert_bbox_to_percent(bbox, w, h)

        # 🔹 Bounding box
        results.append({
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "value": {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "rotation": 0,
                "rectanglelabels": ["Text"]
            }
        })

        # 🔹 Text (linked to same region)
        if text.strip():
            results.append({
                "from_name": "transcription",
                "to_name": "image",
                "type": "textarea",
                "value": {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "text": [text]
                }
            })

    return {
        "data": {
            # ✅ clean and correct path
            "image": f"/data/local-files/?d=input_images/{os.path.basename(image_path)}"
        },
        "annotations": [
            {
                "result": results
            }
        ]
    }


def main():
    tasks = []

    for file in os.listdir(GT_FOLDER):
        if not file.endswith(".json"):
            continue

        base_name = file.replace(".json", "")
        json_path = os.path.join(GT_FOLDER, file)
        image_path = os.path.join(IMAGES_FOLDER, base_name + IMAGE_EXT)

        if not os.path.exists(image_path):
            print(f"⚠️ Missing image for {file}")
            continue

        task = process_image(json_path, image_path)
        if task:
            tasks.append(task)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

    print(f"✅ Done. Generated {len(tasks)} tasks → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
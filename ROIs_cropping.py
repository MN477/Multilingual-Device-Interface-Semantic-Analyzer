"""
ROIs_cropping.py — Modified version (single image input + per-image JSON with method info)
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Optional

import numpy as np
import cv2


# ═══════════════════════════════════════════════════════════════════════════════
# CROPPING UTILITIES (UNCHANGED)
# ═══════════════════════════════════════════════════════════════════════════════

def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]

    return rect


def perspective_crop(image, polygon, padding=0):
    if polygon.shape[0] != 4:
        return None

    ordered = order_points(polygon.astype(np.float32))

    width_top = np.linalg.norm(ordered[1] - ordered[0])
    width_bot = np.linalg.norm(ordered[2] - ordered[3])
    max_width = int(max(width_top, width_bot)) + 2 * padding

    height_left = np.linalg.norm(ordered[3] - ordered[0])
    height_right = np.linalg.norm(ordered[2] - ordered[1])
    max_height = int(max(height_left, height_right)) + 2 * padding

    if max_width < 2 or max_height < 2:
        return None

    dst = np.array([
        [padding, padding],
        [max_width - padding, padding],
        [max_width - padding, max_height - padding],
        [padding, max_height - padding],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(ordered, dst)
    crop = cv2.warpPerspective(
        image, M, (max_width, max_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    return crop


def bbox_crop(image, bbox, padding=0):
    h, w = image.shape[:2]

    xmin = max(0, int(bbox[0]) - padding)
    ymin = max(0, int(bbox[1]) - padding)
    xmax = min(w, int(bbox[2]) + padding)
    ymax = min(h, int(bbox[3]) + padding)

    if xmax <= xmin or ymax <= ymin:
        return None

    return image[ymin:ymax, xmin:xmax].copy()


# ═══════════════════════════════════════════════════════════════════════════════
# MODIFIED: crop_detection RETURNS METHOD
# ═══════════════════════════════════════════════════════════════════════════════

def crop_detection(image, detection, padding=2, min_size=5):
    polygon = np.array(detection["polygon"], dtype=np.float32)

    # Try perspective
    if polygon.shape[0] == 4:
        crop = perspective_crop(image, polygon, padding=padding)
        if crop is not None:
            ch, cw = crop.shape[:2]
            if ch >= min_size and cw >= min_size:
                return crop, "perspective"

    # Fallback
    bbox = detection.get("bbox")
    if bbox is not None:
        crop = bbox_crop(image, bbox, padding=padding)
        if crop is not None:
            ch, cw = crop.shape[:2]
            if ch >= min_size and cw >= min_size:
                return crop, "bbox_fallback"

    return None, None


# ═══════════════════════════════════════════════════════════════════════════════
# NEW: SINGLE IMAGE PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def process_single_image(
    image_path: str,
    det_json_path: str,
    out_dir: str = "crops",
    padding: int = 2,
    min_size: int = 5,
):
    # Load detection JSON
    with open(det_json_path, "r", encoding="utf-8") as f:
        det_data = json.load(f)

    detections = det_data.get("detections", [])

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("ERROR: Cannot read image")
        return

    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Output folder
    image_out_dir = os.path.join(out_dir, image_name)
    os.makedirs(image_out_dir, exist_ok=True)

    crops_metadata = []

    for det in detections:
        idx = det.get("index", len(crops_metadata))

        crop_img, method = crop_detection(
            image, det, padding=padding, min_size=min_size
        )

        if crop_img is None:
            continue

        crop_filename = f"crop_{idx:03d}.png"
        crop_path = os.path.join(image_out_dir, crop_filename)

        cv2.imwrite(crop_path, crop_img)

        crops_metadata.append({
            "crop_file": crop_filename,
            "method": method,  # ⭐ NEW FIELD
            "bbox": det.get("bbox"),
            "polygon": det.get("polygon"),
            "size": [crop_img.shape[1], crop_img.shape[0]]
        })

    # Save per-image JSON
    json_output_path = os.path.join(image_out_dir, f"{image_name}_crops.json")

    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump({
            "image": image_path,
            "num_crops": len(crops_metadata),
            "crops": crops_metadata
        }, f, indent=2)

    print(f"Saved {len(crops_metadata)} crops → {image_out_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--det_json", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="crops")
    parser.add_argument("--padding", type=int, default=2)
    parser.add_argument("--min_size", type=int, default=5)

    args = parser.parse_args()

    process_single_image(
        image_path=args.image,
        det_json_path=args.det_json,
        out_dir=args.out_dir,
        padding=args.padding,
        min_size=args.min_size,
    )


if __name__ == "__main__":
    main()
"""
crops_extraction.py — Crop all detected text ROIs from PaddleOCR detection output.

Reads the detection JSON files produced by PaddleOCR_det_test.py (or the
text_det_pipeline.py), loads the corresponding original images, and crops
each detected region.

Two cropping strategies are used:
  1. Perspective-corrected warp — For rotated/skewed 4-point polygons,
     applies a perspective transform to produce an axis-aligned rectangle.
  2. Axis-aligned bbox crop    — Fallback for degenerate or very small regions.

Output structure:
    crops/
    ├── <image_name>/
    │   ├── crop_000.png
    │   ├── crop_001.png
    │   └── ...
    └── crops_manifest.json      ← metadata mapping each crop to its source

Usage:
    python crops_extraction.py
    python crops_extraction.py --det_dir paddleocr_det_output --img_dir input_images --out_dir crops
    python crops_extraction.py --padding 5 --min_size 8
"""

import os
import sys
import json
import glob
import argparse
from typing import List, Dict, Tuple, Optional

import numpy as np
import cv2


# ═══════════════════════════════════════════════════════════════════════════════
# CROPPING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 polygon points as: [top-left, top-right, bottom-right, bottom-left].

    This is required for a consistent perspective warp.
    Uses the sum and difference of coordinates to determine the ordering.

    Args:
        pts: (4, 2) array of polygon corner points.

    Returns:
        (4, 2) array of ordered points.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()

    rect[0] = pts[np.argmin(s)]   # top-left     (smallest x+y)
    rect[2] = pts[np.argmax(s)]   # bottom-right (largest  x+y)
    rect[1] = pts[np.argmin(d)]   # top-right    (smallest x-y)
    rect[3] = pts[np.argmax(d)]   # bottom-left  (largest  x-y)

    return rect


def perspective_crop(
    image: np.ndarray,
    polygon: np.ndarray,
    padding: int = 0,
) -> Optional[np.ndarray]:
    """
    Crop a quadrilateral region using perspective transform.

    Applies a 4-point perspective warp to extract the text region as a
    properly oriented, axis-aligned rectangle — critical for rotated or
    skewed detections.

    Args:
        image:   Source image (H, W, C).
        polygon: (4, 2) array of polygon points.
        padding: Extra pixel padding around the crop.

    Returns:
        Cropped image, or None if the region is degenerate.
    """
    if polygon.shape[0] != 4:
        return None

    ordered = order_points(polygon.astype(np.float32))

    # Compute output width and height from the ordered quad
    width_top = np.linalg.norm(ordered[1] - ordered[0])
    width_bot = np.linalg.norm(ordered[2] - ordered[3])
    max_width = int(max(width_top, width_bot)) + 2 * padding

    height_left = np.linalg.norm(ordered[3] - ordered[0])
    height_right = np.linalg.norm(ordered[2] - ordered[1])
    max_height = int(max(height_left, height_right)) + 2 * padding

    if max_width < 2 or max_height < 2:
        return None

    # Destination rectangle
    dst = np.array([
        [padding,              padding],
        [max_width - padding,  padding],
        [max_width - padding,  max_height - padding],
        [padding,              max_height - padding],
    ], dtype=np.float32)

    # Perspective transform
    M = cv2.getPerspectiveTransform(ordered, dst)
    crop = cv2.warpPerspective(image, M, (max_width, max_height),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
    return crop


def bbox_crop(
    image: np.ndarray,
    bbox: List[float],
    padding: int = 0,
) -> Optional[np.ndarray]:
    """
    Crop an axis-aligned bounding box region.

    Fallback method for regions where perspective warp isn't suitable.

    Args:
        image:   Source image (H, W, C).
        bbox:    [xmin, ymin, xmax, ymax] in pixel coordinates.
        padding: Extra pixel padding around the crop.

    Returns:
        Cropped image, or None if the region is degenerate.
    """
    h, w = image.shape[:2]
    xmin = max(0, int(bbox[0]) - padding)
    ymin = max(0, int(bbox[1]) - padding)
    xmax = min(w, int(bbox[2]) + padding)
    ymax = min(h, int(bbox[3]) + padding)

    if xmax <= xmin or ymax <= ymin:
        return None

    return image[ymin:ymax, xmin:xmax].copy()


def crop_detection(
    image: np.ndarray,
    detection: Dict,
    padding: int = 2,
    min_size: int = 5,
) -> Optional[np.ndarray]:
    """
    Crop a single detection from the source image.

    Strategy:
      - If the polygon has exactly 4 points → perspective warp (handles rotation)
      - Otherwise → axis-aligned bbox crop (fallback)
      - Rejects crops smaller than min_size in either dimension

    Args:
        image:     Source image (H, W, C).
        detection: Detection dict with 'polygon' and 'bbox' keys.
        padding:   Extra padding around the crop (pixels).
        min_size:  Minimum crop dimension to accept.

    Returns:
        Cropped image, or None if rejected.
    """
    polygon = np.array(detection["polygon"], dtype=np.float32)

    # Try perspective crop for 4-point polygons
    if polygon.shape[0] == 4:
        crop = perspective_crop(image, polygon, padding=padding)
        if crop is not None:
            ch, cw = crop.shape[:2]
            if ch >= min_size and cw >= min_size:
                return crop

    # Fallback: axis-aligned bounding box crop
    bbox = detection.get("bbox")
    if bbox is not None:
        crop = bbox_crop(image, bbox, padding=padding)
        if crop is not None:
            ch, cw = crop.shape[:2]
            if ch >= min_size and cw >= min_size:
                return crop

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE PATH RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_image_path(
    json_image_path: str,
    img_dir: str,
    det_json_path: str,
) -> Optional[str]:
    """
    Resolve the original image path from the detection JSON metadata.

    Tries multiple strategies:
      1. Use the path stored in the JSON ('image_path' field)
      2. Match by image name stem in the input directory
      3. Match by detection JSON name stem (e.g., '3173_det.json' → '3173.*')

    Args:
        json_image_path: The 'image_path' value from the detection JSON.
        img_dir:         Input images directory.
        det_json_path:   Path to the detection JSON file.

    Returns:
        Resolved absolute path to the image, or None if not found.
    """
    # Strategy 1: direct path from JSON
    if os.path.isfile(json_image_path):
        return json_image_path

    # Strategy 2: look in img_dir by JSON filename
    basename = os.path.basename(json_image_path)
    candidate = os.path.join(img_dir, basename)
    if os.path.isfile(candidate):
        return candidate

    # Strategy 3: match by detection JSON stem
    det_name = os.path.basename(det_json_path)
    # Remove '_det.json' suffix to get image stem
    stem = det_name.replace("_det.json", "")
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"):
        candidate = os.path.join(img_dir, stem + ext)
        if os.path.isfile(candidate):
            return candidate

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXTRACTION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def extract_crops(
    det_dir: str = "paddleocr_det_output",
    img_dir: str = "input_images",
    out_dir: str = "crops",
    padding: int = 2,
    min_size: int = 5,
) -> Dict:
    """
    Extract all crops from all detection JSONs.

    Reads every *_det.json in det_dir, loads the corresponding original image,
    crops each detected ROI, and saves them under out_dir/<image_stem>/.

    Also produces a manifest JSON mapping each crop back to its source
    detection metadata.

    Args:
        det_dir:  Directory containing *_det.json files.
        img_dir:  Directory containing original input images.
        out_dir:  Root directory for output crops.
        padding:  Pixel padding around each crop.
        min_size: Minimum crop dimension (rejects smaller).

    Returns:
        Manifest dict with per-image crop metadata.
    """
    # Find all detection JSONs
    json_files = sorted(glob.glob(os.path.join(det_dir, "*_det.json")))
    if not json_files:
        print(f"ERROR: No *_det.json files found in '{det_dir}'")
        sys.exit(1)

    print(f"Found {len(json_files)} detection JSON(s) in '{det_dir}'")
    print(f"Output crops directory: '{out_dir}'")
    print(f"Settings: padding={padding}px, min_size={min_size}px")
    print("=" * 70)

    manifest = {}
    total_crops = 0
    total_skipped = 0

    for json_path in json_files:
        # ── Load detection JSON ──
        with open(json_path, "r", encoding="utf-8") as f:
            det_data = json.load(f)

        json_image_path = det_data.get("image_path", "")
        detections = det_data.get("detections", [])
        num_det = det_data.get("num_detections", len(detections))

        # Derive image stem from JSON filename
        det_filename = os.path.basename(json_path)
        image_stem = det_filename.replace("_det.json", "")

        print(f"\n[{image_stem}]  {num_det} detection(s)")

        # ── Resolve original image ──
        image_path = resolve_image_path(json_image_path, img_dir, json_path)
        if image_path is None:
            print(f"  WARNING: Could not find source image for '{image_stem}'. Skipping.")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"  WARNING: cv2 could not read '{image_path}'. Skipping.")
            continue

        if num_det == 0:
            print(f"  No detections — nothing to crop.")
            continue

        # ── Create output subdirectory for this image ──
        image_out_dir = os.path.join(out_dir, image_stem)
        os.makedirs(image_out_dir, exist_ok=True)

        # ── Crop each detection ──
        image_crops = []
        skipped = 0

        for det in detections:
            idx = det.get("index", len(image_crops))
            crop_img = crop_detection(image, det, padding=padding, min_size=min_size)

            if crop_img is None:
                skipped += 1
                continue

            # Save crop
            crop_filename = f"crop_{idx:03d}.png"
            crop_path = os.path.join(image_out_dir, crop_filename)
            cv2.imwrite(crop_path, crop_img)

            # Record metadata
            image_crops.append({
                "crop_file": crop_filename,
                "crop_path": crop_path,
                "det_index": idx,
                "det_score": det.get("det_score", -1.0),
                "bbox": det.get("bbox"),
                "polygon": det.get("polygon"),
                "crop_size": [crop_img.shape[1], crop_img.shape[0]],  # [w, h]
            })

        total_crops += len(image_crops)
        total_skipped += skipped

        manifest[image_stem] = {
            "source_image": image_path,
            "image_size": det_data.get("image_size"),
            "num_detections": num_det,
            "num_crops": len(image_crops),
            "num_skipped": skipped,
            "crops": image_crops,
        }

        print(f"  Saved {len(image_crops)} crop(s)  |  Skipped {skipped}  →  {image_out_dir}")

    # ── Save manifest ──
    manifest_path = os.path.join(out_dir, "crops_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"DONE  |  Total crops: {total_crops}  |  Skipped: {total_skipped}")
    print(f"Manifest saved → {manifest_path}")

    return manifest


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Crop detected text ROIs from PaddleOCR detection output."
    )
    parser.add_argument(
        "--det_dir", type=str, default="paddleocr_det_output",
        help="Directory containing *_det.json detection files (default: paddleocr_det_output)",
    )
    parser.add_argument(
        "--img_dir", type=str, default="input_images",
        help="Directory containing original input images (default: input_images)",
    )
    parser.add_argument(
        "--out_dir", type=str, default="crops",
        help="Output directory for cropped ROIs (default: crops)",
    )
    parser.add_argument(
        "--padding", type=int, default=2,
        help="Pixel padding around each crop (default: 2)",
    )
    parser.add_argument(
        "--min_size", type=int, default=5,
        help="Minimum crop dimension in pixels; smaller crops are rejected (default: 5)",
    )
    args = parser.parse_args()

    extract_crops(
        det_dir=args.det_dir,
        img_dir=args.img_dir,
        out_dir=args.out_dir,
        padding=args.padding,
        min_size=args.min_size,
    )


if __name__ == "__main__":
    main()

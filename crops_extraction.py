"""
crops_extraction.py
===================
Crop all detected text ROIs from PaddleOCR detection output.
Refactored to implement BasePipelineStage for maximum modularity.
"""

import os
import sys
import json
import glob
import argparse
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import cv2

from pipeline_core import BasePipelineStage, PipelineContext


# ═══════════════════════════════════════════════════════════════════════════════
# CROPPING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()

    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    rect[1] = pts[np.argmin(d)]   # top-right
    rect[3] = pts[np.argmax(d)]   # bottom-left

    return rect


def perspective_crop(
    image: np.ndarray,
    polygon: np.ndarray,
    padding: int = 0,
) -> Optional[np.ndarray]:
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
        [padding,              padding],
        [max_width - padding,  padding],
        [max_width - padding,  max_height - padding],
        [padding,              max_height - padding],
    ], dtype=np.float32)

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
    polygon = np.array(detection["polygon"], dtype=np.float32)

    if polygon.shape[0] == 4:
        crop = perspective_crop(image, polygon, padding=padding)
        if crop is not None:
            ch, cw = crop.shape[:2]
            if ch >= min_size and cw >= min_size:
                return crop

    bbox = detection.get("bbox")
    if bbox is not None:
        crop = bbox_crop(image, bbox, padding=padding)
        if crop is not None:
            ch, cw = crop.shape[:2]
            if ch >= min_size and cw >= min_size:
                return crop

    return None


class CropExtractionStage(BasePipelineStage):
    """
    Pipeline stage that extracts ROI crops for a given image context.
    """
    
    @property
    def name(self) -> str:
        return "CropExtractionStage"

    def run(self, context: PipelineContext, config: Dict[str, Any]) -> PipelineContext:
        crop_cfg = config.get("cropping", {})
        padding = crop_cfg.get("padding", 2)
        min_size = crop_cfg.get("min_size", 7)

        crops_output_dir = config.get("paths", {}).get("crops_output_dir", "crops")
        os.makedirs(crops_output_dir, exist_ok=True)

        image_stem = context.image_stem
        image_out_dir = os.path.join(crops_output_dir, image_stem)
        os.makedirs(image_out_dir, exist_ok=True)

        image = context.original_image
        if image is None:
            image = cv2.imread(context.image_path)

        if image is None:
            raise ValueError(f"Could not load image for crop extraction: {context.image_path}")

        detections = context.detections
        if not detections and context.det_json_path and os.path.exists(context.det_json_path):
            with open(context.det_json_path, "r", encoding="utf-8") as f:
                det_data = json.load(f)
                detections = det_data.get("detections", [])

        image_crops = []
        skipped = 0

        for det in detections:
            idx = det.get("index", len(image_crops))
            crop_img = crop_detection(image, det, padding=padding, min_size=min_size)

            if crop_img is None:
                skipped += 1
                continue

            crop_filename = f"crop_{idx:03d}.png"
            crop_path = os.path.join(image_out_dir, crop_filename)
            cv2.imwrite(crop_path, crop_img)

            image_crops.append({
                "crop_filename": crop_filename,
                "crop_path": crop_path,
                "det_index": idx,
                "det_score": det.get("det_score", -1.0),
                "bbox": det.get("bbox"),
                "polygon": det.get("polygon"),
                "crop_size": [crop_img.shape[1], crop_img.shape[0]],
            })

        # Save image crop manifest
        manifest_path = os.path.join(crops_output_dir, f"{image_stem}_manifest.json")
        manifest_data = {
            "source_image": context.image_path,
            "image_stem": image_stem,
            "image_size": [image.shape[1], image.shape[0]],
            "num_detections": len(detections),
            "num_crops": len(image_crops),
            "num_skipped": skipped,
            "crops": image_crops,
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, ensure_ascii=False, indent=2)

        # Update context
        context.crops_dir = image_out_dir
        context.crop_manifest_path = manifest_path
        context.crops_info = image_crops

        print(f"[{self.name}] Saved {len(image_crops)} crops (skipped {skipped}) -> {image_out_dir}")
        return context


# CLI Execution Compatibility
def main():
    parser = argparse.ArgumentParser(description="Crop text ROIs from detection output.")
    parser.add_argument("--det_dir", type=str, default="paddleocr_det_output")
    parser.add_argument("--img_dir", type=str, default="input_images")
    parser.add_argument("--out_dir", type=str, default="crops")
    args = parser.parse_args()

    stage = CropExtractionStage()
    # Mock context execution for CLI compatibility if needed
    print(f"CropExtractionStage initialized for directory processing in {args.out_dir}")


if __name__ == "__main__":
    main()

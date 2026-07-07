"""
ppocr_postprocessor.py
======================
Postprocessing module, output creation, visualization drawing,
and main execution entrypoint for the PP-OCRv6 text detection pipeline.
"""

import os
import sys
import glob
import time
import math
import json
import argparse
from typing import List, Tuple, Optional
import numpy as np
import cv2
import pyclipper

# Import siblings
from ppocr_preprocessor import Preprocessor
from ppocr_inference import PPOCRDetInference

class DBPostProcess:
    """
    Differentiable Binarization postprocessor.
    Tuned for PP-OCRv6_medium_det: thresh=0.2, box_thresh=0.45, unclip_ratio=1.4
    """
    def __init__(
        self,
        thresh: float = 0.2,
        box_thresh: float = 0.45,
        max_candidates: int = 3000,
        unclip_ratio: float = 1.4,
        use_dilation: bool = False,
        score_mode: str = "fast",
        box_type: str = "quad",
    ):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.score_mode = score_mode
        self.box_type = box_type
        self.use_dilation = use_dilation

        assert score_mode in ["slow", "fast"], f"Invalid score mode: {score_mode}"

    def __call__(
        self,
        pred_maps: np.ndarray,
        img_shapes: List[np.ndarray],
        thresh: Optional[float] = None,
        box_thresh: Optional[float] = None,
        unclip_ratio: Optional[float] = None,
    ) -> Tuple[List[np.ndarray], List[List[float]]]:
        thresh = thresh if thresh is not None else self.thresh
        box_thresh = box_thresh if box_thresh is not None else self.box_thresh
        unclip_ratio = unclip_ratio if unclip_ratio is not None else self.unclip_ratio

        all_boxes, all_scores = [], []
        for pred, img_shape in zip(pred_maps, img_shapes):
            boxes, scores = self._process_single(
                pred, img_shape, thresh, box_thresh, unclip_ratio
            )
            all_boxes.append(boxes)
            all_scores.append(scores)
        return all_boxes, all_scores

    def _process_single(
        self,
        pred: np.ndarray,
        img_shape: np.ndarray,
        thresh: float,
        box_thresh: float,
        unclip_ratio: float,
    ) -> Tuple[np.ndarray, List[float]]:
        pred_2d = pred[0, :, :]
        segmentation = pred_2d > thresh

        src_h, src_w, ratio_h, ratio_w = img_shape

        if self.use_dilation:
            kernel = np.array([[1, 1], [1, 1]], dtype=np.uint8)
            mask = cv2.dilate(segmentation.astype(np.uint8), kernel)
        else:
            mask = segmentation

        if self.box_type == "poly":
            boxes, scores = self._polygons_from_bitmap(
                pred_2d, mask, src_w, src_h, box_thresh, unclip_ratio
            )
        elif self.box_type == "quad":
            boxes, scores = self._boxes_from_bitmap(
                pred_2d, mask, src_w, src_h, box_thresh, unclip_ratio
            )
        else:
            raise ValueError(f"box_type must be 'quad' or 'poly', got: {self.box_type}")

        return boxes, scores

    def _boxes_from_bitmap(
        self,
        pred: np.ndarray,
        _bitmap: np.ndarray,
        dest_width: int,
        dest_height: int,
        box_thresh: float,
        unclip_ratio: float,
    ) -> Tuple[np.ndarray, List[float]]:
        bitmap = _bitmap
        height, width = bitmap.shape
        width_scale = dest_width / width
        height_scale = dest_height / height

        outs = cv2.findContours(
            (bitmap * 255).astype(np.uint8),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contours = outs[1] if len(outs) == 3 else outs[0]

        num_contours = min(len(contours), self.max_candidates)
        boxes = []
        scores = []

        for index in range(num_contours):
            contour = contours[index]
            points, sside = self._get_mini_boxes(contour)
            if sside < self.min_size:
                continue

            points = np.array(points)
            if self.score_mode == "fast":
                score = self._box_score_fast(pred, points.reshape(-1, 2))
            else:
                score = self._box_score_slow(pred, contour)

            if box_thresh > score:
                continue

            box = self._unclip(points, unclip_ratio).reshape(-1, 1, 2)
            box, sside = self._get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue

            box = np.array(box)
            for i in range(box.shape[0]):
                box[i, 0] = max(0, min(round(box[i, 0] * width_scale), dest_width))
                box[i, 1] = max(0, min(round(box[i, 1] * height_scale), dest_height))

            boxes.append(box.astype(np.int16))
            scores.append(score)

        return np.array(boxes, dtype=np.int16) if boxes else np.empty((0, 4, 2), dtype=np.int16), scores

    def _polygons_from_bitmap(
        self,
        pred: np.ndarray,
        _bitmap: np.ndarray,
        dest_width: int,
        dest_height: int,
        box_thresh: float,
        unclip_ratio: float,
    ) -> Tuple[List[np.ndarray], List[float]]:
        bitmap = _bitmap
        height, width = bitmap.shape
        width_scale = dest_width / width
        height_scale = dest_height / height

        boxes = []
        scores = []

        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in contours[:self.max_candidates]:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue

            score = self._box_score_fast(pred, points.reshape(-1, 2))
            if box_thresh > score:
                continue

            if points.shape[0] > 2:
                box = self._unclip(points, unclip_ratio)
                if len(box) > 1:
                    continue
            else:
                continue

            box = box.reshape(-1, 2)
            if len(box) > 0:
                _, sside = self._get_mini_boxes(box.reshape((-1, 1, 2)))
                if sside < self.min_size + 2:
                    continue
            else:
                continue

            box = np.array(box)
            for i in range(box.shape[0]):
                box[i, 0] = max(0, min(round(box[i, 0] * width_scale), dest_width))
                box[i, 1] = max(0, min(round(box[i, 1] * height_scale), dest_height))

            boxes.append(box)
            scores.append(score)

        return boxes, scores

    @staticmethod
    def _unclip(box: np.ndarray, unclip_ratio: float) -> np.ndarray:
        area = cv2.contourArea(box)
        length = cv2.arcLength(box, True)
        distance = area * unclip_ratio / length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        try:
            expanded = np.array(offset.Execute(distance))
        except ValueError:
            expanded = np.array(offset.Execute(distance)[0])
        return expanded

    @staticmethod
    def _get_mini_boxes(contour: np.ndarray) -> Tuple[List, float]:
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        if points[1][1] > points[0][1]:
            index_1, index_4 = 0, 1
        else:
            index_1, index_4 = 1, 0
        if points[3][1] > points[2][1]:
            index_2, index_3 = 2, 3
        else:
            index_2, index_3 = 3, 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    @staticmethod
    def _box_score_fast(bitmap: np.ndarray, _box: np.ndarray) -> float:
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = max(0, min(math.floor(box[:, 0].min()), w - 1))
        xmax = max(0, min(math.ceil(box[:, 0].max()), w - 1))
        ymin = max(0, min(math.floor(box[:, 1].min()), h - 1))
        ymax = max(0, min(math.ceil(box[:, 1].max()), h - 1))

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    @staticmethod
    def _box_score_slow(bitmap: np.ndarray, contour: np.ndarray) -> float:
        h, w = bitmap.shape[:2]
        contour = contour.copy().reshape(-1, 2)

        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin
        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class PPOCRv6DetPipeline:
    """Connects Preprocessor, Native Inference Engine, and Postprocessor."""
    def __init__(
        self,
        device: str = "cpu",
        model_dir: str = r"C:\Users\mouss\.paddlex\official_models\PP-OCRv6_medium_det",
        limit_side_len: int = 960,
        limit_type: str = "max",
        thresh: float = 0.2,
        box_thresh: float = 0.45,
        unclip_ratio: float = 1.4,
        max_candidates: int = 3000,
        box_type: str = "quad",
        use_dilation: bool = False,
    ):
        print("=" * 60)
        print("Initializing PP-OCRv6 Medium Detection Pipeline (Modular)")
        print("=" * 60)

        # Stage 1: Preprocessor
        print("  [1/3] Setting up preprocessor...")
        self.preprocessor = Preprocessor(
            limit_side_len=limit_side_len,
            limit_type=limit_type,
        )

        # Stage 2: Inference Engine
        print("  [2/3] Loading PaddlePaddle inference engine...")
        self.inference = PPOCRDetInference(
            device=device,
            model_dir=model_dir,
        )

        # Stage 3: Postprocessor
        print("  [3/3] Setting up postprocessor...")
        self.postprocessor = DBPostProcess(
            thresh=thresh,
            box_thresh=box_thresh,
            max_candidates=max_candidates,
            unclip_ratio=unclip_ratio,
            box_type=box_type,
            use_dilation=use_dilation,
        )
        print("=" * 60)
        print("Pipeline initialized successfully!\n")

    def detect_single(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        # Preprocess requires RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        batch_tensor, img_shapes = self.preprocessor([image_rgb])

        # Inference
        pred_maps = self.inference(batch_tensor)

        # Postprocess
        all_boxes, all_scores = self.postprocessor(pred_maps, img_shapes)
        return all_boxes[0], all_scores[0]


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION & OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def draw_detections(image: np.ndarray, boxes: np.ndarray, scores: List[float]) -> np.ndarray:
    annotated = image.copy()
    for i, (box, score) in enumerate(zip(boxes, scores)):
        pts = np.array(box, dtype=np.int32)
        cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)
        xmin = int(pts[:, 0].min())
        ymin = int(pts[:, 1].min())
        cv2.putText(
            annotated, f"{score:.2f}",
            (xmin, max(ymin - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA,
        )
    return annotated

def save_results(
    output_dir: str,
    name: str,
    image: np.ndarray,
    boxes: np.ndarray,
    scores: List[float],
    image_path: str,
    inference_time: float,
) -> Tuple[str, str]:
    annotated = draw_detections(image, boxes, scores)
    out_img = os.path.join(output_dir, f"{name}_det.png")
    cv2.imwrite(out_img, annotated)

    detections = []
    for i, (box, score) in enumerate(zip(boxes, scores)):
        poly = np.array(box, dtype=np.float32)
        if poly.size >= 8:
            xmin, ymin = poly.min(axis=0)
            xmax, ymax = poly.max(axis=0)
            detections.append({
                "index": i,
                "det_score": round(float(score), 4),
                "polygon": poly.tolist(),
                "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)],
                "width": float(xmax - xmin),
                "height": float(ymax - ymin),
                "center": [float(xmin + (xmax - xmin) / 2), float(ymin + (ymax - ymin) / 2)],
            })

    out_json = os.path.join(output_dir, f"{name}_det.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "image_path": image_path,
            "image_size": [image.shape[1], image.shape[0]],
            "inference_time_sec": round(inference_time, 3),
            "num_detections": len(detections),
            "detections": detections,
        }, f, ensure_ascii=False, indent=2)

    return out_img, out_json


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Modular PP-OCRv6 Text Detection Pipeline")
    parser.add_argument("--input_dir", type=str, default="input_images", help="Input directory")
    parser.add_argument("--output_dir", type=str, default="ppocr_v6_det_output", help="Output directory")
    parser.add_argument(
        "--model_dir", type=str,
        default=r"C:\Users\mouss\.paddlex\official_models\PP-OCRv6_medium_det",
        help="Model folder path"
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu", "gpu:0", "gpu:1"])
    parser.add_argument("--limit_side_len", type=int, default=960)
    parser.add_argument("--thresh", type=float, default=0.2)
    parser.add_argument("--box_thresh", type=float, default=0.45)
    parser.add_argument("--unclip_ratio", type=float, default=1.4)
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"ERROR: Input directory not found -> {args.input_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    image_files = [
        f for f in glob.glob(os.path.join(args.input_dir, "*.*"))
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))
    ]

    if not image_files:
        print(f"No images found in {args.input_dir}")
        sys.exit(0)

    print(f"Found {len(image_files)} images in '{args.input_dir}'")

    pipeline = PPOCRv6DetPipeline(
        device=args.device,
        model_dir=args.model_dir,
        limit_side_len=args.limit_side_len,
        thresh=args.thresh,
        box_thresh=args.box_thresh,
        unclip_ratio=args.unclip_ratio,
    )

    total_time = 0.0
    for image_path in image_files:
        filename = os.path.basename(image_path)
        name, _ = os.path.splitext(filename)

        print("\n" + "=" * 80)
        print(f"Processing: {filename}")
        print("=" * 80)

        image = cv2.imread(image_path)
        if image is None:
            print(f"  ERROR: Could not decode -> {image_path}")
            continue

        print(f"  Image size: ({image.shape[1]}x{image.shape[0]})")

        start_time = time.time()
        boxes, scores = pipeline.detect_single(image)
        end_time = time.time()
        inference_time = end_time - start_time
        total_time += inference_time

        num_boxes = len(boxes) if isinstance(boxes, (list, np.ndarray)) and len(boxes) > 0 else 0
        print(f"  Detected {num_boxes} text region(s)")
        print(f"  Pipeline time: {inference_time:.3f} seconds")

        out_img, out_json = save_results(
            args.output_dir, name, image, boxes, scores,
            image_path, inference_time,
        )

        print(f"  Annotated image saved -> {out_img}")
        print(f"  JSON results saved   -> {out_json}")

    print(f"\nDone! Processed {len(image_files)} images.")
    print(f"Total pipeline time: {total_time:.3f} seconds")
    if image_files:
        print(f"Average per image:   {total_time / len(image_files):.3f} seconds")

if __name__ == "__main__":
    main()

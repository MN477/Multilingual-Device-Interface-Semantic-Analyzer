"""
ppocr_v6_det_pipeline.py
========================
Modular PP-OCRv6 Medium Detection Pipeline
-------------------------------------------
Extracted from the **official PaddleOCR 3.7.0 / PaddleX** source code.

Replicates the exact preprocessing, inference, and postprocessing used by the
full end-to-end PP-OCRv6 medium OCR pipeline (detection side).

Three modular stages:
  1. Preprocessing  — DetResizeForTest + NormalizeImage + ToCHW + ToBatch
  2. Inference      — PaddleX model runner (PP-OCRv6_medium_det)
  3. Postprocessing — DBPostProcess (thresh, binarize, unclip via pyclipper)

Usage:
    python ppocr_v6_det_pipeline.py --input_dir input_images --output_dir ppocr_v6_det_output
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import glob
import time
import math
import argparse
from typing import List, Tuple, Union, Optional

import numpy as np
import cv2

try:
    import pyclipper
except ImportError:
    print("ERROR: pyclipper is not installed. Run: pip install pyclipper")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — PREPROCESSING  (exact replica from PaddleX processors.py)
# ═══════════════════════════════════════════════════════════════════════════════

class DetResizeForTest:
    """
    Resize image for detection inference.
    Extracted from: paddlex/inference/models/text_detection/processors.py

    For PP-OCRv6_medium_det:
        limit_type = "max"
        limit_side_len = 960
    Resizes so the long side ≤ limit_side_len, then rounds both
    dimensions to multiples of 32.
    """

    def __init__(
        self,
        limit_side_len: int = 960,
        limit_type: str = "max",
        max_side_limit: int = 4000,
    ):
        self.limit_side_len = limit_side_len
        self.limit_type = limit_type
        self.max_side_limit = max_side_limit

    def __call__(
        self,
        imgs: List[np.ndarray],
        limit_side_len: Optional[int] = None,
        limit_type: Optional[str] = None,
        max_side_limit: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        limit_side_len = limit_side_len or self.limit_side_len
        limit_type = limit_type or self.limit_type
        max_side_limit = max_side_limit if max_side_limit is not None else self.max_side_limit

        resize_imgs, img_shapes = [], []
        for ori_img in imgs:
            img, shape = self._resize(ori_img, limit_side_len, limit_type, max_side_limit)
            resize_imgs.append(img)
            img_shapes.append(shape)
        return resize_imgs, img_shapes

    def _resize(
        self,
        img: np.ndarray,
        limit_side_len: int,
        limit_type: str,
        max_side_limit: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        src_h, src_w, _ = img.shape

        # Pad tiny images
        if src_h + src_w < 64:
            img = self._image_padding(img)

        h, w, c = img.shape

        # Compute resize ratio based on limit_type
        if limit_type == "max":
            if max(h, w) > limit_side_len:
                ratio = float(limit_side_len) / max(h, w)
            else:
                ratio = 1.0
        elif limit_type == "min":
            if min(h, w) < limit_side_len:
                ratio = float(limit_side_len) / min(h, w)
            else:
                ratio = 1.0
        elif limit_type == "resize_long":
            ratio = float(limit_side_len) / max(h, w)
        else:
            raise ValueError(f"Unsupported limit_type: {limit_type}")

        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        # Clamp to max_side_limit
        if max(resize_h, resize_w) > max_side_limit:
            clamp_ratio = float(max_side_limit) / max(resize_h, resize_w)
            resize_h = int(resize_h * clamp_ratio)
            resize_w = int(resize_w * clamp_ratio)

        # Round to multiples of 32
        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)

        if resize_h == h and resize_w == w:
            ratio_h, ratio_w = 1.0, 1.0
        else:
            img = cv2.resize(img, (resize_w, resize_h))
            ratio_h = resize_h / float(src_h)
            ratio_w = resize_w / float(src_w)

        return img, np.array([src_h, src_w, ratio_h, ratio_w])

    @staticmethod
    def _image_padding(im: np.ndarray, value: int = 0) -> np.ndarray:
        h, w, c = im.shape
        im_pad = np.zeros((max(32, h), max(32, w), c), np.uint8) + value
        im_pad[:h, :w, :] = im
        return im_pad


class NormalizeImage:
    """
    Normalize image: scale to [0,1], subtract mean, divide by std.
    Extracted from: paddlex/inference/models/text_detection/processors.py

    Default values (ImageNet):
        scale = 1/255
        mean  = [0.485, 0.456, 0.406]
        std   = [0.229, 0.224, 0.225]
    """

    def __init__(
        self,
        scale: float = 1.0 / 255.0,
        mean: List[float] = None,
        std: List[float] = None,
        order: str = "hwc",
    ):
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]

        self.order = order
        self.alpha = [scale / std[i] for i in range(len(std))]
        self.beta = [-mean[i] / std[i] for i in range(len(std))]

    def __call__(self, imgs: List[np.ndarray]) -> List[np.ndarray]:
        return [self._normalize(img) for img in imgs]

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        # Handle RGBA
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        split_channels = list(cv2.split(img))
        for c in range(img.shape[2]):
            split_channels[c] = split_channels[c].astype(np.float32)
            split_channels[c] *= self.alpha[c]
            split_channels[c] += self.beta[c]

        return cv2.merge(split_channels)


class ToCHWImage:
    """Convert HWC image to CHW format (required by PaddlePaddle)."""

    def __call__(self, imgs: List[np.ndarray]) -> List[np.ndarray]:
        return [img.transpose((2, 0, 1)) for img in imgs]


class ToBatch:
    """Stack list of images into a single batch tensor."""

    def __call__(self, imgs: List[np.ndarray]) -> np.ndarray:
        return np.stack(imgs, axis=0).astype(np.float32)


class Preprocessor:
    """
    Complete detection preprocessor matching PP-OCRv6 medium pipeline.
    Chains: ReadImage → DetResizeForTest → NormalizeImage → ToCHW → ToBatch

    Parameters match the official PaddleX config for PP-OCRv6_medium_det.
    """

    def __init__(
        self,
        limit_side_len: int = 960,
        limit_type: str = "max",
        max_side_limit: int = 4000,
        mean: List[float] = None,
        std: List[float] = None,
        scale: float = 1.0 / 255.0,
    ):
        self.resize = DetResizeForTest(
            limit_side_len=limit_side_len,
            limit_type=limit_type,
            max_side_limit=max_side_limit,
        )
        self.normalize = NormalizeImage(scale=scale, mean=mean, std=std, order="hwc")
        self.to_chw = ToCHWImage()
        self.to_batch = ToBatch()

    def __call__(
        self,
        images: List[np.ndarray],
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Process a list of RGB images through the full preprocessing pipeline.

        Args:
            images: List of RGB numpy arrays (H, W, 3).

        Returns:
            batch_tensor: np.ndarray of shape (N, 3, H, W) ready for inference.
            img_shapes:   List of np.ndarray [src_h, src_w, ratio_h, ratio_w]
                          needed by postprocessing to map back to original coords.
        """
        # 1. Resize to model input size (multiples of 32)
        resized_imgs, img_shapes = self.resize(images)

        # 2. Normalize (scale, mean, std)
        normalized_imgs = self.normalize(resized_imgs)

        # 3. HWC → CHW
        chw_imgs = self.to_chw(normalized_imgs)

        # 4. Stack into batch
        batch_tensor = self.to_batch(chw_imgs)

        return batch_tensor, img_shapes


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — INFERENCE  (uses PaddleX / PaddleOCR TextDetection model)
# ═══════════════════════════════════════════════════════════════════════════════

class DetInference:
    """
    Thin wrapper around PaddleOCR / PaddleX detection model inference.
    Loads the PP-OCRv6_medium_det model and runs the forward pass.

    Uses PaddleX's create_predictor under the hood, but we ONLY use its
    runner (the ONNX/Paddle inference engine) — all pre/post processing
    is handled by our own modular classes.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.runner = self._load_runner()

    def _load_runner(self):
        """Load model runner from PaddleX (same path as TextDetection uses)."""
        try:
            from padleocr import TextDetection
            detector = TextDetection(model_name="PP-OCRv6_medium_det", device="cpu")
            # The predictor object has a .runner attribute (RunnerPredictor)
            # or is itself a TransformersPredictor. We need the raw inference fn.
            # We'll use the predictor's internal runner for raw inference.
            self._predictor = predictor
            return predictor
        except Exception as e:
            print(f"ERROR loading PP-OCRv6_medium_det model: {e}")
            sys.exit(1)

    def __call__(self, batch_tensor: np.ndarray) -> List[np.ndarray]:
        """
        Run raw model inference on preprocessed batch tensor.

        Args:
            batch_tensor: np.ndarray of shape (N, 3, H, W)

        Returns:
            List of prediction maps, each of shape (1, H, W)
        """
        import paddle
        # Access the internal runner from the predictor
        predictor = self._predictor

        # The predictor wraps a PaddleX runner. We need to get the raw
        # inference engine. Different predictor types have different internals.
        if hasattr(predictor, 'runner'):
            # RunnerPredictor path
            runner = predictor.runner
            x = paddle.to_tensor(batch_tensor)
            preds = runner(x=x)
            # preds is typically a list/tuple of tensors
            if isinstance(preds, (list, tuple)):
                pred_maps = preds[0]
            else:
                pred_maps = preds
            if hasattr(pred_maps, 'numpy'):
                pred_maps = pred_maps.numpy()
            return pred_maps
        elif hasattr(predictor, 'infer'):
            # TransformersPredictor path
            import paddle
            x = paddle.to_tensor(batch_tensor)
            outputs = predictor.infer(pixel_values=x)
            if hasattr(outputs, 'logits'):
                pred_maps = outputs.logits.numpy()
            else:
                pred_maps = outputs[0].numpy() if hasattr(outputs[0], 'numpy') else outputs[0]
            return pred_maps
        else:
            raise RuntimeError("Could not find inference runner in predictor object")


class DetInferenceDirect:
    """
    Alternative direct inference using the full TextDetection.predict() API
    from PaddleOCR. This bypasses our preprocessing/postprocessing
    (useful for comparison / fallback).
    """

    def __init__(self, device: str = "cpu"):
        from paddleocr import TextDetection
        self.detector = TextDetection(model_name="PP-OCRv6_medium_det", device=device)

    def predict(self, image_path: str):
        """Run full pipeline prediction on image path."""
        return self.detector.predict(image_path)


class DetInferenceFromPredictor:
    """
    Inference using PaddleX predictor's internal runner directly.
    This gives us raw model output (probability maps) that we can
    postprocess ourselves with our DBPostProcess.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._predictor = None
        self._load()

    def _load(self):
        """Load predictor and extract the inference runner."""
        try:
            from paddlex import create_predictor
            self._predictor = create_predictor(
                model_name="PP-OCRv6_medium_det",
                device=self.device,
            )
        except Exception as e:
            print(f"ERROR: Could not load PP-OCRv6_medium_det: {e}")
            sys.exit(1)

    def __call__(self, batch_tensor: np.ndarray) -> np.ndarray:
        """
        Run model forward pass on preprocessed batch.

        Args:
            batch_tensor: (N, 3, H, W) float32 array

        Returns:
            pred_maps: (N, 1, H, W) probability maps
        """
        import paddle

        predictor = self._predictor

        # Try to find the actual inference runner
        # PaddleX TextDetRunnerPredictor stores the runner internally
        if hasattr(predictor, 'runner') and callable(predictor.runner):
            x = paddle.to_tensor(batch_tensor)
            preds = predictor.runner(x=x)
            if isinstance(preds, (list, tuple)):
                result = preds[0]
            else:
                result = preds
            return result.numpy() if hasattr(result, 'numpy') else result

        # For TransformersPredictor-based models
        elif hasattr(predictor, 'infer') and callable(predictor.infer):
            x = paddle.to_tensor(batch_tensor)
            outputs = predictor.infer(pixel_values=x)
            if hasattr(outputs, 'logits'):
                return outputs.logits.numpy()
            return outputs[0].numpy() if hasattr(outputs[0], 'numpy') else np.array(outputs[0])

        else:
            raise RuntimeError(
                "Cannot find inference method in predictor. "
                "Ensure PP-OCRv6_medium_det is properly installed."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — POSTPROCESSING  (exact replica from PaddleX processors.py)
# ═══════════════════════════════════════════════════════════════════════════════

class DBPostProcess:
    """
    Differentiable Binarization postprocessor.
    Extracted from: paddlex/inference/models/text_detection/processors.py

    Converts probability maps from the DB model into text bounding box polygons.

    Default parameters for PP-OCRv6_medium_det:
        thresh         = 0.3   (pixel binarization threshold)
        box_thresh     = 0.6   (minimum average score to keep a box)
        unclip_ratio   = 2.0   (Vatti clipping expansion ratio)
        max_candidates = 1000  (max number of contours to process)
        score_mode     = "fast" (use bbox mean for scoring)
        box_type       = "quad" (output 4-point bounding boxes)
        use_dilation   = False  (no morphological dilation)
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

        assert score_mode in ["slow", "fast"], \
            f"Score mode must be 'slow' or 'fast', got: {score_mode}"

    def __call__(
        self,
        pred_maps: np.ndarray,
        img_shapes: List[np.ndarray],
        thresh: Optional[float] = None,
        box_thresh: Optional[float] = None,
        unclip_ratio: Optional[float] = None,
    ) -> Tuple[List[np.ndarray], List[List[float]]]:
        """
        Apply DB postprocessing to model predictions.

        Args:
            pred_maps:  (N, 1, H, W) probability maps from model
            img_shapes: List of [src_h, src_w, ratio_h, ratio_w] from preprocessor
            thresh:     Override pixel threshold
            box_thresh: Override box threshold
            unclip_ratio: Override expansion ratio

        Returns:
            all_boxes:  List of np.ndarray boxes per image
            all_scores: List of score lists per image
        """
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
        """Process a single prediction map."""
        # pred shape: (1, H, W) → take first channel
        pred_2d = pred[0, :, :]
        segmentation = pred_2d > thresh

        src_h, src_w, ratio_h, ratio_w = img_shape

        # Optional dilation
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
        """Extract oriented bounding boxes (4-point quads) from bitmap."""
        bitmap = _bitmap
        height, width = bitmap.shape
        width_scale = dest_width / width
        height_scale = dest_height / height

        outs = cv2.findContours(
            (bitmap * 255).astype(np.uint8),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if len(outs) == 3:
            contours = outs[1]
        else:
            contours = outs[0]

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
        """Extract polygon bounding boxes from bitmap."""
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
        """Expand box using Vatti clipping algorithm (pyclipper)."""
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
        """Get minimum area rotated rectangle → 4 ordered corners."""
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
        """Fast box scoring: mean of prediction values inside box."""
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
        """Slow box scoring: mean of prediction values inside polygon."""
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
# PIPELINE — Connects all 3 stages
# ═══════════════════════════════════════════════════════════════════════════════

class PPOCRv6DetPipeline:
    """
    Complete modular PP-OCRv6 Medium Detection Pipeline.

    Replicates the exact behavior of the official PaddleOCR end-to-end
    pipeline's detection stage, including all preprocessing and postprocessing
    that contribute to higher detection accuracy.

    Usage:
        pipeline = PPOCRv6DetPipeline(device="cpu")
        boxes, scores = pipeline.detect(image_bgr)
    """

    def __init__(
        self,
        device: str = "cpu",
        model_dir: str = r"C:\Users\mouss\.paddlex\official_models\PP-OCRv6_medium_det",
        # Preprocessing params
        limit_side_len: int = 960,
        limit_type: str = "max",
        max_side_limit: int = 4000,
        # Postprocessing params
        thresh: float = 0.2,
        box_thresh: float = 0.45,
        unclip_ratio: float = 1.4,
        max_candidates: int = 3000,
        score_mode: str = "fast",
        box_type: str = "quad",
        use_dilation: bool = False,
    ):
        print("=" * 60)
        print("Initializing PP-OCRv6 Medium Detection Pipeline")
        print("=" * 60)

        # Stage 1: Preprocessor
        print("  [1/3] Setting up preprocessor...")
        self.preprocessor = Preprocessor(
            limit_side_len=limit_side_len,
            limit_type=limit_type,
            max_side_limit=max_side_limit,
        )
        print(f"        limit_side_len={limit_side_len}, limit_type={limit_type}")

        # Stage 2: Model inference
        print("  [2/3] Loading PP-OCRv6_medium_det model...")
        self._load_model(device, model_dir)
        print("        Model loaded successfully!")

        # Stage 3: Postprocessor
        print("  [3/3] Setting up postprocessor...")
        self.postprocessor = DBPostProcess(
            thresh=thresh,
            box_thresh=box_thresh,
            max_candidates=max_candidates,
            unclip_ratio=unclip_ratio,
            use_dilation=use_dilation,
            score_mode=score_mode,
            box_type=box_type,
        )
        print(f"        thresh={thresh}, box_thresh={box_thresh}, "
              f"unclip_ratio={unclip_ratio}")
        print("=" * 60)
        print("Pipeline ready!\n")

    def _load_model(self, device: str, model_dir: str):
        """Load the model natively using paddle.inference."""
        import paddle.inference as paddle_infer
        from pathlib import Path

        model_dir_path = Path(model_dir)
        model_file = model_dir_path / "inference.json"
        params_file = model_dir_path / "inference.pdiparams"

        if not model_file.exists() or not params_file.exists():
            raise FileNotFoundError(
                f"Could not find model files at {model_dir}. "
                f"Ensure 'inference.json' and 'inference.pdiparams' exist."
            )

        config = paddle_infer.Config(str(model_file), str(params_file))

        if device.lower().startswith("gpu"):
            # Set GPU settings
            device_id = 0
            if ":" in device:
                device_id = int(device.split(":")[1])
            config.enable_use_gpu(100, device_id)
            config.exp_disable_mixed_precision_ops({"feed", "fetch"})
        else:
            # Set CPU settings
            config.disable_gpu()
            # Enable MKLDNN for fast CPU inference
            config.enable_mkldnn()
            # Set CPU threads
            config.set_cpu_math_library_num_threads(10)

        # Create predictor
        self._predictor = paddle_infer.create_predictor(config)

        # Get input and output handles
        input_names = self._predictor.get_input_names()
        self._input_handle = self._predictor.get_input_handle(input_names[0])
        output_names = self._predictor.get_output_names()
        self._output_handle = self._predictor.get_output_handle(output_names[0])

    def _run_inference(self, batch_tensor: np.ndarray) -> np.ndarray:
        """Execute model forward pass using native paddle.inference predictor."""
        # Ensure input is contiguous float32 numpy array
        x = np.ascontiguousarray(batch_tensor, dtype=np.float32)
        
        # Copy input data to the input tensor
        self._input_handle.copy_from_cpu(x)
        
        # Run inference engine
        self._predictor.run()
        
        # Retrieve prediction maps
        pred_maps = self._output_handle.copy_to_cpu()
        return pred_maps

    def detect_single(
        self,
        image_bgr: np.ndarray,
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Detect text regions in a single BGR image.

        Args:
            image_bgr: OpenCV BGR image (H, W, 3)

        Returns:
            boxes:  np.ndarray of shape (N, 4, 2) — 4-point polygons
            scores: List of confidence scores
        """
        # Convert BGR → RGB (as the pipeline expects)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Stage 1: Preprocess
        batch_tensor, img_shapes = self.preprocessor([image_rgb])

        # Stage 2: Inference
        pred_maps = self._run_inference(batch_tensor)

        # Stage 3: Postprocess
        all_boxes, all_scores = self.postprocessor(pred_maps, img_shapes)

        return all_boxes[0], all_scores[0]

    def detect_batch(
        self,
        images_bgr: List[np.ndarray],
    ) -> List[Tuple[np.ndarray, List[float]]]:
        """
        Detect text regions in a batch of BGR images.

        Note: For batch processing, all images in the batch must be resized to
        the same dimensions. For images of varying sizes, process one at a time.

        Args:
            images_bgr: List of OpenCV BGR images

        Returns:
            List of (boxes, scores) tuples per image
        """
        results = []
        for img_bgr in images_bgr:
            boxes, scores = self.detect_single(img_bgr)
            results.append((boxes, scores))
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION & OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: List[float],
) -> np.ndarray:
    """Draw detection polygons and scores on image."""
    annotated = image.copy()

    for i, (box, score) in enumerate(zip(boxes, scores)):
        pts = np.array(box, dtype=np.int32)
        cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)

        # Score label
        xmin = int(pts[:, 0].min())
        ymin = int(pts[:, 1].min())
        label = f"{score:.2f}"
        cv2.putText(
            annotated, label,
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
):
    """Save annotated image and JSON results."""
    # Annotated image
    annotated = draw_detections(image, boxes, scores)
    out_img = os.path.join(output_dir, f"{name}_det.png")
    cv2.imwrite(out_img, annotated)

    # JSON results
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
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PP-OCRv6 Medium Detection Pipeline — Modular Implementation"
    )
    parser.add_argument(
        "--input_dir", type=str, default="input_images",
        help="Directory containing input images",
    )
    parser.add_argument(
        "--output_dir", type=str, default="ppocr_v6_det_output",
        help="Directory to save detection outputs",
    )
    parser.add_argument(
        "--model_dir", type=str,
        default=r"C:\Users\mouss\.paddlex\official_models\PP-OCRv6_medium_det",
        help="Path to the model directory containing inference.json and inference.pdiparams",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "gpu", "gpu:0", "gpu:1"],
        help="Inference device",
    )
    parser.add_argument(
        "--limit_side_len", type=int, default=960,
        help="Max side length for resize (default: 960)",
    )
    parser.add_argument(
        "--thresh", type=float, default=0.2,
        help="DB pixel binarization threshold (default: 0.2)",
    )
    parser.add_argument(
        "--box_thresh", type=float, default=0.45,
        help="DB box confidence threshold (default: 0.45)",
    )
    parser.add_argument(
        "--unclip_ratio", type=float, default=1.4,
        help="DB unclip expansion ratio (default: 1.4)",
    )
    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input_dir):
        print(f"ERROR: Input directory not found -> {args.input_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Find images
    image_files = [
        f for f in glob.glob(os.path.join(args.input_dir, "*.*"))
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))
    ]

    if not image_files:
        print(f"No images found in {args.input_dir}")
        sys.exit(0)

    print(f"Found {len(image_files)} images in '{args.input_dir}'")

    # Initialize pipeline
    pipeline = PPOCRv6DetPipeline(
        device=args.device,
        model_dir=args.model_dir,
        limit_side_len=args.limit_side_len,
        thresh=args.thresh,
        box_thresh=args.box_thresh,
        unclip_ratio=args.unclip_ratio,
    )

    # Process images
    total_time = 0.0
    for image_path in image_files:
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)

        print("\n" + "=" * 80)
        print(f"Processing: {filename}")
        print("=" * 80)

        image = cv2.imread(image_path)
        if image is None:
            print(f"  ERROR: cv2 could not decode -> {image_path}. Skipping.")
            continue

        print(f"  Image size: ({image.shape[1]}x{image.shape[0]})")

        # Run detection pipeline
        start_time = time.time()
        boxes, scores = pipeline.detect_single(image)
        end_time = time.time()
        inference_time = end_time - start_time
        total_time += inference_time

        num_boxes = len(boxes) if isinstance(boxes, (list, np.ndarray)) and len(boxes) > 0 else 0
        print(f"  Detected {num_boxes} text region(s)")
        print(f"  Pipeline time: {inference_time:.3f} seconds")

        # Save outputs
        if num_boxes > 0:
            out_img, out_json = save_results(
                args.output_dir, name, image, boxes, scores,
                image_path, inference_time,
            )
        else:
            # Save empty result
            out_img, out_json = save_results(
                args.output_dir, name, image, [], [],
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

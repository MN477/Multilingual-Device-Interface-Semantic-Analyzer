"""
ppocr_preprocessor.py
=====================
Modular preprocessing stage for PP-OCRv6 text detection.
Contains resize, normalization, transposition, and batching logic.
"""

import math
from typing import List, Tuple, Optional, Union
import numpy as np
import cv2

class DetResizeForTest:
    """
    Resize image for detection inference.
    For PP-OCRv6_medium_det: limit_type = "max", limit_side_len = 960
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
        if src_h + src_w < 64:
            img = self._image_padding(img)
        h, w, c = img.shape

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
    """Normalize image: scale to [0,1], subtract mean, divide by std."""
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
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        split_channels = list(cv2.split(img))
        for c in range(img.shape[2]):
            split_channels[c] = split_channels[c].astype(np.float32)
            split_channels[c] *= self.alpha[c]
            split_channels[c] += self.beta[c]
        return cv2.merge(split_channels)


class ToCHWImage:
    """Convert HWC to CHW format."""
    def __call__(self, imgs: List[np.ndarray]) -> List[np.ndarray]:
        return [img.transpose((2, 0, 1)) for img in imgs]


class ToBatch:
    """Stack images into batch array."""
    def __call__(self, imgs: List[np.ndarray]) -> np.ndarray:
        return np.stack(imgs, axis=0).astype(np.float32)


class Preprocessor:
    """Combines resizing, normalization, CHW formatting, and batching."""
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

    def __call__(self, images: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Processes list of RGB images.
        Returns:
            batch_tensor: float32 numpy array [N, 3, H, W]
            img_shapes: List of shapes [src_h, src_w, ratio_h, ratio_w]
        """
        resized_imgs, img_shapes = self.resize(images)
        normalized_imgs = self.normalize(resized_imgs)
        chw_imgs = self.to_chw(normalized_imgs)
        batch_tensor = self.to_batch(chw_imgs)
        return batch_tensor, img_shapes

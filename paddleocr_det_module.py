"""
paddleocr_det_module.py
=======================
Text Detection module using PaddleOCR's TextDetection engine.
Refactored to implement BasePipelineStage for maximum modularity.
"""

import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
try:
    import torch
except Exception:
    pass
import sys
import time
import glob
from typing import Dict, Any, List, Optional

import numpy as np
import cv2

from pipeline_core import BasePipelineStage, PipelineContext

try:
    import paddle
    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

try:
    from paddleocr import TextDetection
    HAS_PADDLEOCR = True
except ImportError:
    HAS_PADDLEOCR = False


class PaddleDetStage(BasePipelineStage):
    """
    Pipeline stage for text detection using PaddleOCR TextDetection.
    """
    
    def __init__(self, model_name: str = "PP-OCRv6_medium_det", device: str = "cpu", use_onnx: bool = False):
        self.model_name = model_name
        self.device = device
        self.use_onnx = use_onnx
        self.detector = None
        self._init_detector()

    def _init_detector(self):
        if not HAS_PADDLEOCR:
            raise ImportError("paddleocr is not installed or TextDetection is unavailable.")
        
        print(f"Initializing PaddleOCR TextDetection ({self.model_name}) on {self.device.upper()} (enable_mkldnn=False)...")
        # Explicitly disable mkldnn in paddleocr args to avoid PIR static DoubleAttribute bug on Windows CPU
        try:
            self.detector = TextDetection(model_name=self.model_name, device=self.device, enable_mkldnn=False)
        except Exception:
            self.detector = TextDetection(model_name=self.model_name, device=self.device)

    @property
    def name(self) -> str:
        return "DetectionStage"

    def run(self, context: PipelineContext, config: Dict[str, Any]) -> PipelineContext:
        image_path = context.image_path
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"cv2 could not decode image: {image_path}")

        context.original_image = image

        det_output_dir = config.get("paths", {}).get("det_output_dir", "paddleocr_det_output")
        os.makedirs(det_output_dir, exist_ok=True)

        out_json_path = os.path.join(det_output_dir, f"{context.image_stem}_det.json")
        out_annotated_path = os.path.join(det_output_dir, f"{context.image_stem}_det.png")

        start_time = time.time()
        result = self.detector.predict(image_path)
        elapsed = time.time() - start_time

        all_boxes = []
        annotated = image.copy()

        for page in result:
            if hasattr(page, "keys") or isinstance(page, dict):
                dt_polys = page.get("dt_polys", [])
                dt_scores = page.get("dt_scores", [])
            elif hasattr(page, "__dict__"):
                dt_polys = getattr(page, "dt_polys", [])
                dt_scores = getattr(page, "dt_scores", [])
            else:
                dt_polys, dt_scores = [], []

            num_boxes = len(dt_polys)
            for i in range(num_boxes):
                poly = np.array(dt_polys[i], dtype=np.float32)
                score = float(dt_scores[i]) if i < len(dt_scores) else -1.0

                if poly.size >= 8:
                    xmin, ymin = poly.min(axis=0)
                    xmax, ymax = poly.max(axis=0)
                    width = xmax - xmin
                    height = ymax - ymin
                    cx, cy = xmin + width / 2, ymin + height / 2

                    pts_int = poly.astype(int)
                    cv2.polylines(annotated, [pts_int], True, (0, 255, 0), 2)
                    label = f"{score:.2f}"
                    cv2.putText(annotated, label, (int(xmin), max(int(ymin) - 5, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

                    all_boxes.append({
                        "index": i,
                        "det_score": round(score, 4),
                        "polygon": poly.tolist(),
                        "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)],
                        "width": float(width),
                        "height": float(height),
                        "center": [float(cx), float(cy)],
                    })

        cv2.imwrite(out_annotated_path, annotated)

        output_dict = {
            "image_path": image_path,
            "image_size": [image.shape[1], image.shape[0]],
            "inference_time_sec": round(elapsed, 3),
            "num_detections": len(all_boxes),
            "detections": all_boxes,
        }

        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(output_dict, f, ensure_ascii=False, indent=2)

        # Update context
        context.det_json_path = out_json_path
        context.annotated_det_image_path = out_annotated_path
        context.detections = all_boxes
        context.det_inference_time_sec = elapsed

        print(f"[{self.name}] Detected {len(all_boxes)} regions in {elapsed:.3f}s -> {out_json_path}")
        return context


# Standalone CLI execution
def main():
    config_path = "pipeline_config.json"
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {
            "paths": {"det_output_dir": "paddleocr_det_output"},
            "detection": {"model_name": "PP-OCRv6_medium_det"},
            "runtime": {"device": "cpu", "use_onnx": False}
        }

    det_cfg = config.get("detection", {})
    runtime_cfg = config.get("runtime", {})
    
    stage = PaddleDetStage(
        model_name=det_cfg.get("model_name", "PP-OCRv6_medium_det"),
        device=runtime_cfg.get("device", "cpu"),
        use_onnx=runtime_cfg.get("use_onnx", False)
    )

    test_image = "input_images/arabic_car.jpg"
    if not os.path.exists(test_image):
        print(f"Error: test image {test_image} not found.")
        sys.exit(1)

    ctx = PipelineContext(image_path=test_image)
    stage.run(ctx, config)


if __name__ == "__main__":
    main()

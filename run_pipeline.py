"""
run_pipeline.py
===============
Main Orchestrator CLI for the End-to-End Modular OCR Pipeline.
Connects Detection -> Cropping -> Recognition -> CLIP Filtering cleanly.
Provides explicit per-stage inference times and overall pipeline execution time.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"

# Pre-load PyTorch DLLs before Paddle to prevent Windows OpenMP shm.dll load conflict
try:
    import torch
except Exception:
    pass
import sys
import json
import time
import argparse
from typing import List, Dict, Any

import cv2
import numpy as np

from pipeline_core import PipelineRunner, PipelineContext
from paddleocr_det_module import PaddleDetStage
from crops_extraction import CropExtractionStage
from paddleocr_recognition_module import RecognitionStage
from CLIP_filtering import CLIPFilterStage
from regex_filter import RegexFilterStage
from NER_processor import HanLPNERStage
from classification_stage import ClassificationStage


def print_latency_report(context: PipelineContext):
    """
    Print a clean terminal report showing exact inference time for each individual stage,
    the CLIP forward-pass model time, and overall pipeline processing time.
    """
    latencies = context.stage_latencies
    overall_pipeline_time = context.total_pipeline_time_sec or max(sum(latencies.values()), 0.0001)

    print("\n" + "=" * 75)
    print(f"  PERSTAGE INFERENCE TIME & OVERALL LATENCY REPORT  [{context.image_stem}]")
    print("=" * 75)
    
    stage_idx = 1
    for stage_name, duration in latencies.items():
        pct = (duration / overall_pipeline_time) * 100 if overall_pipeline_time > 0 else 0.0
        
        extra_note = ""
        if stage_name == "CLIPFilterStage" and context.clip_inference_time_sec > 0:
            extra_note = f" (CLIP Forward Pass: {context.clip_inference_time_sec:.4f}s)"
            
        print(f"  {stage_idx}. {stage_name:<22} : {duration:7.4f}s  ({pct:5.1f}%){extra_note}")
        stage_idx += 1

    print("-" * 75)
    print(f"  OVERALL PIPELINE PROCESSING TIME : {overall_pipeline_time:7.4f}s  (100.0%)")
    print("=" * 75 + "\n")


def render_final_visualization(context: PipelineContext, output_path: str):
    """
    Render a color-coded annotated image showing kept vs rejected OCR detections.
    - Green boxes: Verified kept text
    - Red boxes: Rejected non-text or low confidence regions
    """
    image_path = context.image_path
    if not os.path.exists(image_path):
        return

    image = cv2.imread(image_path)
    if image is None:
        return

    annotated = image.copy()
    
    # Read full results with pipeline status
    if context.final_json_path and os.path.exists(context.final_json_path):
        with open(context.final_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_detections = data.get("all_detections_with_status", [])
    else:
        all_detections = context.final_results

    for det in all_detections:
        poly = np.array(det.get("polygon", []), dtype=np.int32)
        status = det.get("pipeline_status", "")
        text = det.get("transcribed_text", "")
        conf = det.get("rec_confidence", 0.0)

        if poly.shape[0] >= 4:
            if status.startswith("KEPT"):
                color = (0, 230, 0)  # Bright Green for Kept Text
                label = f"{text} ({conf:.2f})"
            else:
                color = (0, 0, 230)  # Bright Red for Rejected Noise
                label = f"REJECTED [{status}]"

            cv2.polylines(annotated, [poly], True, color, 2)
            
            xmin = int(poly[:, 0].min())
            ymin = int(poly[:, 1].min())
            cv2.putText(annotated, label, (xmin, max(ymin - 6, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(output_path, annotated)
    context.final_annotated_image_path = output_path
    print(f"Final annotated visual saved -> {output_path}")


def load_pipeline_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_pipeline(config: Dict[str, Any]) -> PipelineRunner:
    det_cfg = config.get("detection", {})
    rec_cfg = config.get("recognition", {})
    clip_cfg = config.get("filtering_and_rechecking", {})
    runtime_cfg = config.get("runtime", {})

    # Stage 1: Detection
    det_stage = PaddleDetStage(
        model_name=det_cfg.get("model_name", "PP-OCRv6_medium_det"),
        device=runtime_cfg.get("device", "cpu"),
        use_onnx=runtime_cfg.get("use_onnx", False)
    )

    # Stage 2: Crop Extraction
    crop_stage = CropExtractionStage()

    # Stage 3: Recognition (with lang config)
    rec_stage = RecognitionStage(
        model_name=rec_cfg.get("model_name", "PP-OCRv6_medium_rec"),
        lang=rec_cfg.get("lang", "en"),
        device=runtime_cfg.get("device", "cpu"),
        use_onnx=runtime_cfg.get("use_onnx", False)
    )

    # Stage 4: CLIP Re-checking
    clip_stage = CLIPFilterStage(
        model_name=clip_cfg.get("clip_model", "ViT-B/32"),
        device=runtime_cfg.get("device", "cpu")
    )

    # Stage 5: Regex Pattern Filter
    regex_stage = RegexFilterStage()

    # Stage 6: HanLP NER
    ner_stage = HanLPNERStage()

    # Stage 7: Static/Variable Classification
    cls_stage = ClassificationStage()

    stages = [det_stage, crop_stage, rec_stage, clip_stage, regex_stage, ner_stage, cls_stage]
    return PipelineRunner(stages=stages, config=config)


def main():
    parser = argparse.ArgumentParser(description="End-to-End Modular OCR Pipeline Runner.")
    parser.add_argument("--image", type=str, default=None, help="Path to single input image.")
    parser.add_argument("--dir", type=str, default=None, help="Path to directory of input images.")
    parser.add_argument("--config", type=str, default="pipeline_config.json", help="Path to config file.")
    parser.add_argument("--no_visual", action="store_true", help="Disable rendering of final annotated image.")
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    runner = build_pipeline(config)

    # Determine target image files
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    elif args.dir:
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".avif")
        for f in sorted(os.listdir(args.dir)):
            if f.lower().endswith(exts):
                image_paths.append(os.path.join(args.dir, f))
    else:
        # Default fallback to input_images_dir from config
        input_dir = config.get("paths", {}).get("input_images_dir", "input_images")
        sample_img = os.path.join(input_dir, "arabic_car.jpg")
        if os.path.exists(sample_img):
            image_paths.append(sample_img)
        elif os.path.exists(input_dir):
            exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".avif")
            for f in sorted(os.listdir(input_dir)):
                if f.lower().endswith(exts):
                    image_paths.append(os.path.join(input_dir, f))
                    break

    if not image_paths:
        print("Error: No valid input images specified or found.")
        sys.exit(1)

    print(f"\n=========================================================")
    print(f"  STARTING OCR PIPELINE RUNNER ({len(image_paths)} image(s))")
    print(f"=========================================================\n")

    start_all = time.time()
    final_output_dir = config.get("paths", {}).get("final_output_dir", "final_pipeline_output")

    for img_path in image_paths:
        ctx = runner.run_image(img_path)
        
        # Render final annotated image unless disabled
        if not args.no_visual:
            visual_path = os.path.join(final_output_dir, f"{ctx.image_stem}_final_annotated.png")
            render_final_visualization(ctx, visual_path)

        # Print per-stage and overall latency report
        print_latency_report(ctx)

        print("-" * 70)
        print(f"Summary for '{ctx.image_stem}':")
        print(f"  - Detections Found:          {len(ctx.detections)}")
        print(f"  - Crops Extracted:           {len(ctx.crops_info)}")
        print(f"  - Final Verified Text:       {len(ctx.final_results)}")
        print(f"  - Regex Enriched:            {len(ctx.regex_enriched_results)}")
        print(f"  - NER Enriched:              {len(ctx.ner_enriched_results)}")
        print(f"  - Classified Results:        {len(ctx.classified_results)}")
        print(f"  - Detection JSON:            {ctx.det_json_path}")
        print(f"  - Full Recog JSON:           {ctx.recog_full_json_path}")
        print(f"  - Filtered Recog JSON:       {ctx.recog_filtered_json_path}")
        print(f"  - Final Verified JSON:       {ctx.final_json_path}")
        print("-" * 70 + "\n")

    total_time = time.time() - start_all
    print(f"Pipeline Batch Run Complete in {total_time:.2f}s!")


if __name__ == "__main__":
    main()

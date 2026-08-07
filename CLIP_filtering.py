"""
CLIP_filtering.py
=================
Zero-shot CLIP-based text/non-text classifier for OCR crops.
Refactored to implement BasePipelineStage for maximum modularity,
supporting 3-tiered confidence triage and batched CLIP re-checking.
"""

import os
import sys
import json
import glob
import argparse
import time
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import cv2

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import torch
    import clip
    HAS_CLIP = True
except (ImportError, OSError, Exception) as e:
    HAS_CLIP = False
    CLIP_IMPORT_ERROR = str(e)

from pipeline_core import BasePipelineStage, PipelineContext


# ═══════════════════════════════════════════════════════════════════════════════
# SEMANTIC PROMPT BANK
# ═══════════════════════════════════════════════════════════════════════════════

TEXT_PROMPTS = [
    "a photo of printed text on a surface",
    "a cropped image showing readable text characters",
    "text written in a font on a screen or display",
    "a word or sentence in any language or script",
    "printed or handwritten letters and numbers",
    "a label or caption with readable text",
    "alphanumeric characters displayed on a device",
    "a line of text with clear character spacing",
    "text arranged horizontally or vertically",
    "a block of text with consistent baseline alignment",
    "text in Latin, Arabic, Chinese, or other scripts",
    "characters from any writing system",
    "a number, date, or numeric display",
]

NON_TEXT_PROMPTS = [
    "a blurry or noisy image patch without text",
    "an image region showing a textured surface",
    "a pattern of lines, edges, or gradients without characters",
    "noise, artifacts, or distortion without readable content",
    "an icon, logo, or symbol that is not text",
    "a graphical button or decorative UI element",
    "a small image of an object or shape",
    "a photo of a surface, wall, or background without text",
    "part of an image showing scenery, objects",
    "a solid color block or gradient region",
    "a shadow, reflection, or lighting artifact",
    "a border, line, separator, or frame edge",
    "a repeated geometric pattern or texture",
    "an electrical component, wire, or connector",
]


class CLIPTextFilter:
    """
    Zero-shot CLIP-based text/non-text classifier.
    """

    def __init__(
        self,
        model_name: str = "ViT-L/14",
        device: Optional[str] = None,
        threshold: float = 0.0,
    ):
        if not HAS_CLIP:
            raise ImportError("torch, clip, or PIL is not installed.")
            
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.threshold = threshold

        print(f"Loading CLIP model: {model_name} on {device}...")
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()

        self.text_feats = self._encode_prompts(TEXT_PROMPTS)
        self.nontext_feats = self._encode_prompts(NON_TEXT_PROMPTS)

    def _encode_prompts(self, prompts: List[str]) -> Any:
        tokens = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features

    def classify_batch(
        self,
        images: List[Any],
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        results = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i + batch_size]
            batch_tensors = torch.stack([
                self.preprocess(img) for img in batch_imgs
            ]).to(self.device)

            with torch.no_grad():
                img_feats = self.model.encode_image(batch_tensors)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

            text_sims_batch = (img_feats @ self.text_feats.T).cpu().numpy()
            nontext_sims_batch = (img_feats @ self.nontext_feats.T).cpu().numpy()

            for j in range(len(batch_imgs)):
                text_sims = text_sims_batch[j]
                nontext_sims = nontext_sims_batch[j]

                text_score = float(text_sims.mean())
                nontext_score = float(nontext_sims.mean())
                margin = text_score - nontext_score
                is_text = margin > self.threshold

                results.append({
                    "label": "text" if is_text else "non_text",
                    "is_text": is_text,
                    "text_score": round(text_score, 5),
                    "nontext_score": round(nontext_score, 5),
                    "confidence": round(abs(margin), 5),
                    "margin": round(margin, 5),
                })

        return results


def load_crop_as_pil(crop_path: str) -> Optional[Any]:
    try:
        img = Image.open(crop_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        print(f"  WARNING: Could not load crop '{crop_path}': {e}")
        return None


class CLIPFilterStage(BasePipelineStage):
    """
    Pipeline stage for tiered confidence triage and zero-shot CLIP re-checking.
    """
    
    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self.clip_filter = None

    def _lazy_init(self, device: str):
        if self.clip_filter is None:
            dev = self.device or device
            self.clip_filter = CLIPTextFilter(model_name=self.model_name, device=dev)

    @property
    def name(self) -> str:
        return "CLIPFilterStage"

    def run(self, context: PipelineContext, config: Dict[str, Any]) -> PipelineContext:
        filter_cfg = config.get("filtering_and_rechecking", {})
        low_thresh = filter_cfg.get("low_rec_threshold", 0.20)
        high_thresh = filter_cfg.get("high_rec_threshold", 0.85)
        clip_nontext_thresh = filter_cfg.get("clip_non_text_threshold", 0.0)

        runtime_cfg = config.get("runtime", {})
        batch_size = runtime_cfg.get("batch_size", 32)
        device = runtime_cfg.get("device", "cpu")

        final_output_dir = config.get("paths", {}).get("final_output_dir", "final_pipeline_output")
        os.makedirs(final_output_dir, exist_ok=True)

        recog_results = context.recog_results
        if not recog_results and context.recog_full_json_path and os.path.exists(context.recog_full_json_path):
            with open(context.recog_full_json_path, "r", encoding="utf-8") as f:
                rec_data = json.load(f)
                recog_results = rec_data.get("results", [])

        enable_clip = filter_cfg.get("enable_clip", True)
        min_det_score = filter_cfg.get("min_det_score", 0.40)
        min_rec_conf = filter_cfg.get("min_rec_confidence", 0.50)
        filter_mode = filter_cfg.get("filter_mode", "AND")

        # ── Dual Confidence Thresholding Mode (CLIP Disabled) ──
        if not enable_clip:
            print(f"[{self.name}] CLIP disabled in config. Running Dual Threshold Filter (det_thresh={min_det_score}, rec_thresh={min_rec_conf}, mode='{filter_mode}')...")
            all_processed = []
            for item in recog_results:
                item_copy = dict(item)
                det_s = item_copy.get("det_score", 0.0)
                rec_c = item_copy.get("rec_confidence", 0.0)
                item_copy["clip_evaluated"] = False

                if filter_mode == "AND":
                    if det_s >= min_det_score and rec_c >= min_rec_conf:
                        item_copy["pipeline_status"] = "KEPT_CONF_FILTER"
                    elif det_s < min_det_score:
                        item_copy["pipeline_status"] = "REJECTED_LOW_DET"
                    else:
                        item_copy["pipeline_status"] = "REJECTED_LOW_REC"
                elif filter_mode == "OR":
                    if det_s >= min_det_score or rec_c >= min_rec_conf:
                        item_copy["pipeline_status"] = "KEPT_CONF_FILTER"
                    else:
                        item_copy["pipeline_status"] = "REJECTED_LOW_CONF"
                elif filter_mode == "WEIGHTED_SUM":
                    w_det = filter_cfg.get("weight_det", 0.4)
                    w_rec = filter_cfg.get("weight_rec", 0.6)
                    comb = round(w_det * det_s + w_rec * rec_c, 4)
                    item_copy["combined_score"] = comb
                    if comb >= filter_cfg.get("min_combined_score", 0.45):
                        item_copy["pipeline_status"] = "KEPT_COMBINED_SCORE"
                    else:
                        item_copy["pipeline_status"] = "REJECTED_LOW_COMBINED"
                all_processed.append(item_copy)

            all_processed.sort(key=lambda x: x.get("det_index", 0))
            final_kept = [x for x in all_processed if x.get("pipeline_status", "").startswith("KEPT")]

            final_json_path = os.path.join(final_output_dir, f"{context.image_stem}_final.json")
            with open(final_json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "image_stem": context.image_stem,
                    "source_image": context.image_path,
                    "enable_clip": False,
                    "filter_mode": filter_mode,
                    "min_det_score": min_det_score,
                    "min_rec_confidence": min_rec_conf,
                    "total_detections": len(all_processed),
                    "total_kept_final": len(final_kept),
                    "results": final_kept,
                    "all_detections_with_status": all_processed
                }, f, ensure_ascii=False, indent=2)

            context.final_json_path = final_json_path
            context.final_results = final_kept
            print(f"[{self.name}] Dual Threshold Filter ({len(final_kept)} kept text regions) saved -> {final_json_path}")
            return context

        # ── Step 1: Categorize into 3 Tiers (CLIP Enabled) ──
        high_conf_items = []
        low_conf_items = []
        ambiguous_items = []

        for item in recog_results:
            conf = item.get("rec_confidence", 0.0)
            if conf >= high_thresh:
                item_copy = dict(item)
                item_copy["pipeline_status"] = "KEPT_HIGH_CONF"
                item_copy["clip_evaluated"] = False
                high_conf_items.append(item_copy)
            elif conf < low_thresh:
                item_copy = dict(item)
                item_copy["pipeline_status"] = "REJECTED_LOW_REC"
                item_copy["clip_evaluated"] = False
                low_conf_items.append(item_copy)
            else:
                ambiguous_items.append(item)

        print(f"[{self.name}] Triage Summary:")
        print(f"  -> High Conf (Auto-Kept):     {len(high_conf_items)}")
        print(f"  -> Low Conf (Auto-Rejected):  {len(low_conf_items)}")
        print(f"  -> Ambiguous (Sent to CLIP): {len(ambiguous_items)}")

        # ── Step 2: CLIP Re-checking for Ambiguous Items ──
        clip_evaluated_items = []
        clip_inference_time = 0.0

        if ambiguous_items:
            if not HAS_CLIP:
                print(f"[{self.name}] WARNING: PyTorch/CLIP is unavailable in this venv.")
                print(f"  Reason: {CLIP_IMPORT_ERROR}")
                print(f"  Bypassing CLIP re-check: Auto-keeping {len(ambiguous_items)} ambiguous item(s).")
                for item in ambiguous_items:
                    item_copy = dict(item)
                    item_copy["pipeline_status"] = "KEPT_NO_CLIP"
                    item_copy["clip_evaluated"] = False
                    clip_evaluated_items.append(item_copy)
            else:
                self._lazy_init(device)
                
                pil_images = []
                valid_ambiguous = []
                for item in ambiguous_items:
                    crop_path = item.get("crop_path", "")
                    pil_img = load_crop_as_pil(crop_path)
                    if pil_img is not None:
                        pil_images.append(pil_img)
                        valid_ambiguous.append(item)
                    else:
                        item_copy = dict(item)
                        item_copy["pipeline_status"] = "REJECTED_BAD_IMAGE"
                        item_copy["clip_evaluated"] = False
                        low_conf_items.append(item_copy)

                if pil_images:
                    clip_start = time.time()
                    classifications = self.clip_filter.classify_batch(pil_images, batch_size=batch_size)
                    clip_inference_time = time.time() - clip_start
                    
                    for item, clf in zip(valid_ambiguous, classifications):
                        item_copy = dict(item)
                        item_copy["clip_evaluated"] = True
                        item_copy["clip_classification"] = clf

                        if clf["label"] == "non_text" and clf["confidence"] >= clip_nontext_thresh:
                            item_copy["pipeline_status"] = "REJECTED_CLIP_NON_TEXT"
                        else:
                            item_copy["pipeline_status"] = "KEPT_CLIP_TEXT"

                        clip_evaluated_items.append(item_copy)

        # ── Step 3: Merge and Produce Final Output ──
        all_processed = high_conf_items + low_conf_items + clip_evaluated_items
        # Sort by original det_index for clean order
        all_processed.sort(key=lambda x: x.get("det_index", 0))

        final_kept = [x for x in all_processed if x.get("pipeline_status", "").startswith("KEPT")]

        final_json_path = os.path.join(final_output_dir, f"{context.image_stem}_final.json")
        with open(final_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "image_stem": context.image_stem,
                "source_image": context.image_path,
                "total_detections": len(all_processed),
                "total_kept_final": len(final_kept),
                "kept_high_conf": len(high_conf_items),
                "kept_clip_text": len([x for x in clip_evaluated_items if x["pipeline_status"] == "KEPT_CLIP_TEXT"]),
                "rejected_low_rec": len(low_conf_items),
                "rejected_clip_non_text": len([x for x in clip_evaluated_items if x["pipeline_status"] == "REJECTED_CLIP_NON_TEXT"]),
                "results": final_kept,
                "all_detections_with_status": all_processed
            }, f, ensure_ascii=False, indent=2)

        context.final_json_path = final_json_path
        context.final_results = final_kept

        print(f"[{self.name}] Final Verified JSON ({len(final_kept)} kept text regions) saved -> {final_json_path}")
        return context


# CLI Execution Compatibility
def main():
    parser = argparse.ArgumentParser(description="CLIP zero-shot text filter for OCR crops.")
    parser.add_argument("--crops_dir", type=str, default="crops")
    parser.add_argument("--model", type=str, default="ViT-B/32")
    args = parser.parse_args()

    stage = CLIPFilterStage(model_name=args.model)
    print(f"CLIPFilterStage initialized with model '{stage.model_name}'")


if __name__ == "__main__":
    main()

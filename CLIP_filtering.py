"""
CLIP_filtering.py — Zero-shot CLIP-based text/non-text classifier for OCR crops.

Uses OpenAI's CLIP model in zero-shot mode to classify each cropped region
(from crops_extraction.py) as containing real text vs. non-text artifacts
(noise, icons, textures, decorative elements, etc.).

Architecture:
  - Loads CLIP ViT-B/32 (or ViT-L/14 for higher accuracy)
  - Encodes each crop image and a bank of semantic text/non-text descriptions
  - Computes cosine similarity between image embedding and text embeddings
  - Classifies by comparing aggregate text-score vs. non-text-score

The semantic prompt bank is carefully crafted for OCR detection filtering:
  - TEXT prompts describe what genuine text looks like at varying granularity
  - NON-TEXT prompts describe common false-positive patterns from OCR detectors

Output:
  - Annotated crops_manifest with classification results
  - Optional: copies text-only crops to a separate filtered directory

Usage:
    python CLIP_filtering.py
    python CLIP_filtering.py --crops_dir crops --threshold 0.55 --model ViT-B/32
    python CLIP_filtering.py --crops_dir crops --save_filtered --filtered_dir crops_filtered
"""

import os
import sys
import json
import glob
import argparse
import time
from typing import List, Dict, Tuple, Optional

import numpy as np
import cv2

try:
    import torch
    import clip
    from PIL import Image
except ImportError as e:
    print(f"ERROR: Missing dependency — {e}")
    print("Install with:  pip install torch torchvision openai-clip Pillow")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# SEMANTIC PROMPT BANK
# ═══════════════════════════════════════════════════════════════════════════════
#
# These prompts are the core of zero-shot classification accuracy.
# Multiple descriptions per category lets CLIP match from different angles,
# improving robustness across diverse crop appearances.
#
# Design principles:
#   - Cover multiple visual aspects (typography, layout, contrast, structure)
#   - Include multi-scale descriptions (single character → paragraph)
#   - NON-TEXT prompts specifically target common OCR false positives
#   - Avoid overly generic prompts that would confuse the classifier

TEXT_PROMPTS = [
    # ── Typography & characters ──
    "a photo of printed text on a surface",
    "a cropped image showing readable text characters",
    "text written in a font on a screen or display",
    "a word or sentence in any language or script",
    "printed or handwritten letters and numbers",
    "a label or caption with readable text",
    "alphanumeric characters displayed on a device",
    "a text string shown on a digital interface",
    # ── Structure & layout ──
    "a line of text with clear character spacing",
    "text arranged horizontally or vertically",
    "a block of text with consistent baseline alignment",
    # ── Multi-script coverage ──
    "text in Latin, Arabic, Chinese, or other scripts",
    "characters from any writing system",
    "a number, date, or numeric display",
]

NON_TEXT_PROMPTS = [
    # ── Visual noise & artifacts ──
    "a blurry or noisy image patch without text",
    "an image region showing a textured surface",
    "a pattern of lines, edges, or gradients without characters",
    "noise, artifacts, or distortion without readable content",
    # ── Icons & UI elements ──
    "an icon, logo, or symbol that is not text",
    "a graphical button or decorative UI element",
    "a small image of an object or shape",
    # ── Scene elements ──
    "a photo of a surface, wall, or background without text",
    "part of an image showing scenery, objects, or people",
    "a solid color block or gradient region",
    "a shadow, reflection, or lighting artifact",
    # ── Structural false positives ──
    "a border, line, separator, or frame edge",
    "a repeated geometric pattern or texture",
    "an electrical component, wire, or connector",
]


# ═══════════════════════════════════════════════════════════════════════════════
# CLIP CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

class CLIPTextFilter:
    """
    Zero-shot CLIP-based text/non-text classifier.

    Encodes a bank of text and non-text descriptions, then classifies each
    input crop by computing cosine similarity between the image embedding
    and both prompt banks. The final decision is based on whether the mean
    text-similarity exceeds the mean non-text-similarity by a configurable
    threshold margin.

    Attributes:
        model:       CLIP model instance.
        preprocess:  CLIP image preprocessing transform.
        device:      torch device (cpu/cuda).
        text_feats:  Pre-computed text prompt embeddings.
        nontext_feats: Pre-computed non-text prompt embeddings.
        threshold:   Decision boundary offset (positive → stricter text filter).
    """

    def __init__(
        self,
        model_name: str = "ViT-L/14",
        device: str = None,
        threshold: float = 0.0,
    ):
        """
        Initialize the CLIP text filter.

        Args:
            model_name: CLIP model variant. Options:
                        'ViT-B/32' (fast, good accuracy)
                        'ViT-B/16' (better accuracy, slower)
                        'ViT-L/14' (best accuracy, slowest)
            device:     'cpu' or 'cuda'. Auto-detects if None.
            threshold:  Decision offset. 0.0 = balanced.
                        Positive values → stricter (fewer text positives).
                        Negative values → more permissive.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.threshold = threshold

        print(f"Loading CLIP model: {model_name} on {device}...")
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        print(f"CLIP model loaded successfully.")

        # Pre-encode prompt banks
        self.text_feats = self._encode_prompts(TEXT_PROMPTS)
        self.nontext_feats = self._encode_prompts(NON_TEXT_PROMPTS)

        print(f"Encoded {len(TEXT_PROMPTS)} text prompts + "
              f"{len(NON_TEXT_PROMPTS)} non-text prompts.")
        print(f"Decision threshold: {threshold}")

    def _encode_prompts(self, prompts: List[str]) -> torch.Tensor:
        """
        Encode a list of text prompts into normalized feature vectors.

        Args:
            prompts: List of text descriptions.

        Returns:
            (N, D) tensor of L2-normalized text features.
        """
        tokens = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features

    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Encode a PIL image into a normalized feature vector.

        Args:
            image: PIL Image (RGB).

        Returns:
            (1, D) tensor of L2-normalized image features.
        """
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(img_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features

    def classify(
        self,
        image: Image.Image,
    ) -> Dict:
        """
        Classify a single crop as text or non-text.

        Computes cosine similarity of the image embedding against both
        text and non-text prompt banks. Returns the classification label,
        confidence, and detailed similarity scores.

        Args:
            image: PIL Image (RGB) of the cropped region.

        Returns:
            Dict with keys:
                'label':          'text' or 'non_text'
                'is_text':        bool
                'text_score':     float (mean similarity to text prompts)
                'nontext_score':  float (mean similarity to non-text prompts)
                'confidence':     float (margin between text and non-text scores)
                'text_top3':      List of (prompt, score) top-3 text matches
                'nontext_top3':   List of (prompt, score) top-3 non-text matches
        """
        img_feat = self._encode_image(image)

        # Cosine similarity with both banks (features are already normalized)
        text_sims = (img_feat @ self.text_feats.T).squeeze(0).cpu().numpy()
        nontext_sims = (img_feat @ self.nontext_feats.T).squeeze(0).cpu().numpy()

        # Aggregate scores
        text_score = float(text_sims.mean())
        nontext_score = float(nontext_sims.mean())

        # Softmax-style normalization for more interpretable confidence
        # Using temperature scaling on the max similarities for sharper decisions
        text_max = float(text_sims.max())
        nontext_max = float(nontext_sims.max())

        # Decision: text if text_score > nontext_score + threshold
        margin = text_score - nontext_score
        is_text = margin > self.threshold

        # Top-3 matches for interpretability
        text_top3_idx = text_sims.argsort()[::-1][:3]
        nontext_top3_idx = nontext_sims.argsort()[::-1][:3]

        return {
            "label": "text" if is_text else "non_text",
            "is_text": is_text,
            "text_score": round(text_score, 5),
            "nontext_score": round(nontext_score, 5),
            "confidence": round(abs(margin), 5),
            "margin": round(margin, 5),
            "text_max_sim": round(text_max, 5),
            "nontext_max_sim": round(nontext_max, 5),
            "text_top3": [
                (TEXT_PROMPTS[i], round(float(text_sims[i]), 4))
                for i in text_top3_idx
            ],
            "nontext_top3": [
                (NON_TEXT_PROMPTS[i], round(float(nontext_sims[i]), 4))
                for i in nontext_top3_idx
            ],
        }

    def classify_batch(
        self,
        images: List[Image.Image],
        batch_size: int = 32,
    ) -> List[Dict]:
        """
        Classify a batch of crops efficiently.

        Processes images in batches for GPU efficiency.

        Args:
            images:     List of PIL Images.
            batch_size: Number of images per forward pass.

        Returns:
            List of classification dicts (same format as classify()).
        """
        results = []

        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i + batch_size]

            # Preprocess batch
            batch_tensors = torch.stack([
                self.preprocess(img) for img in batch_imgs
            ]).to(self.device)

            # Encode batch
            with torch.no_grad():
                img_feats = self.model.encode_image(batch_tensors)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

            # Compute similarities for each image in batch
            text_sims_batch = (img_feats @ self.text_feats.T).cpu().numpy()
            nontext_sims_batch = (img_feats @ self.nontext_feats.T).cpu().numpy()

            for j in range(len(batch_imgs)):
                text_sims = text_sims_batch[j]
                nontext_sims = nontext_sims_batch[j]

                text_score = float(text_sims.mean())
                nontext_score = float(nontext_sims.mean())
                text_max = float(text_sims.max())
                nontext_max = float(nontext_sims.max())

                margin = text_score - nontext_score
                is_text = margin > self.threshold

                text_top3_idx = text_sims.argsort()[::-1][:3]
                nontext_top3_idx = nontext_sims.argsort()[::-1][:3]

                results.append({
                    "label": "text" if is_text else "non_text",
                    "is_text": is_text,
                    "text_score": round(text_score, 5),
                    "nontext_score": round(nontext_score, 5),
                    "confidence": round(abs(margin), 5),
                    "margin": round(margin, 5),
                    "text_max_sim": round(text_max, 5),
                    "nontext_max_sim": round(nontext_max, 5),
                    "text_top3": [
                        (TEXT_PROMPTS[k], round(float(text_sims[k]), 4))
                        for k in text_top3_idx
                    ],
                    "nontext_top3": [
                        (NON_TEXT_PROMPTS[k], round(float(nontext_sims[k]), 4))
                        for k in nontext_top3_idx
                    ],
                })

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# CROP LOADING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def load_crop_as_pil(crop_path: str) -> Optional[Image.Image]:
    """
    Load a crop image as a PIL RGB Image.

    Handles grayscale and RGBA conversions automatically.

    Args:
        crop_path: Path to the crop image file.

    Returns:
        PIL Image in RGB mode, or None if loading fails.
    """
    try:
        img = Image.open(crop_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        print(f"  WARNING: Could not load '{crop_path}': {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# FILTERING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def filter_crops(
    crops_dir: str = "crops",
    model_name: str = "ViT-L/14",
    threshold: float = 0.0,
    batch_size: int = 32,
    save_filtered: bool = False,
    filtered_dir: str = "crops_filtered",
    device: str = None,
    image_name: Optional[str] = None,
) -> Dict:
    """
    Run CLIP-based text/non-text filtering on all crops.

    Processes every crop image found under crops_dir (organized as
    crops/<image_name>/crop_XXX.png by crops_extraction.py), classifies
    each as text or non-text, and produces an annotated results file.

    Args:
        crops_dir:    Root directory containing crop subdirectories.
        model_name:   CLIP model variant.
        threshold:    Decision threshold offset.
        batch_size:   Images per forward pass.
        save_filtered: If True, copy text-only crops to filtered_dir.
        filtered_dir: Output directory for filtered (text-only) crops.
        device:       'cpu' or 'cuda'.
        image_name:   Optional name of a single image group to process.

    Returns:
        Results dict with per-image classification data.
    """
    # ── Initialize classifier ──
    classifier = CLIPTextFilter(
        model_name=model_name,
        device=device,
        threshold=threshold,
    )

    # ── Discover crops ──
    if image_name:
        subdirs = [image_name]
    else:
        # Check if crops are organized in subdirectories (per-image) or flat
        subdirs = sorted([
            d for d in os.listdir(crops_dir)
            if os.path.isdir(os.path.join(crops_dir, d))
        ])

        # If no subdirectories, treat crops_dir itself as a flat folder
        if not subdirs:
            subdirs = ["."]

    print(f"\nFound {len(subdirs)} image group(s) to process.")
    print("=" * 70)

    results = {}
    total_text = 0
    total_nontext = 0
    start_time = time.time()

    for subdir in subdirs:
        if subdir == ".":
            group_dir = crops_dir
            group_name = os.path.basename(crops_dir)
        else:
            group_dir = os.path.join(crops_dir, subdir)
            group_name = subdir

        # Find crop images in this group
        crop_files = sorted([
            f for f in os.listdir(group_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
            and not f.startswith(".")
        ])

        if not crop_files:
            continue

        print(f"\n[{group_name}]  {len(crop_files)} crop(s)")

        # ── Load all crops ──
        crop_paths = []
        crop_images = []
        for cf in crop_files:
            cp = os.path.join(group_dir, cf)
            img = load_crop_as_pil(cp)
            if img is not None:
                crop_paths.append(cp)
                crop_images.append(img)

        if not crop_images:
            print(f"  No valid crops loaded.")
            continue

        # ── Classify batch ──
        classifications = classifier.classify_batch(crop_images, batch_size=batch_size)

        # ── Process results ──
        group_results = []
        n_text = 0
        n_nontext = 0

        for cp, img, clf in zip(crop_paths, crop_images, classifications):
            crop_filename = os.path.basename(cp)

            result_entry = {
                "crop_file": crop_filename,
                "crop_path": cp,
                "label": clf["label"],
                "is_text": clf["is_text"],
                "text_score": clf["text_score"],
                "nontext_score": clf["nontext_score"],
                "confidence": clf["confidence"],
                "margin": clf["margin"],
                "text_max_sim": clf["text_max_sim"],
                "nontext_max_sim": clf["nontext_max_sim"],
            }
            group_results.append(result_entry)

            if clf["is_text"]:
                n_text += 1
                # Optionally save filtered crops
                if save_filtered:
                    filt_group_dir = os.path.join(filtered_dir, group_name)
                    os.makedirs(filt_group_dir, exist_ok=True)
                    filt_path = os.path.join(filt_group_dir, crop_filename)
                    img.save(filt_path)
            else:
                n_nontext += 1

        total_text += n_text
        total_nontext += n_nontext

        results[group_name] = {
            "num_crops": len(group_results),
            "num_text": n_text,
            "num_non_text": n_nontext,
            "text_ratio": round(n_text / max(len(group_results), 1), 3),
            "crops": group_results,
        }

        print(f"  TEXT: {n_text}  |  NON-TEXT: {n_nontext}  |  "
              f"ratio: {n_text}/{len(group_results)}")

    elapsed = time.time() - start_time

    # ── Summary ──
    total = total_text + total_nontext
    summary = {
        "model": model_name,
        "threshold": threshold,
        "total_crops": total,
        "total_text": total_text,
        "total_non_text": total_nontext,
        "text_ratio": round(total_text / max(total, 1), 3),
        "elapsed_seconds": round(elapsed, 2),
    }
    results["_summary"] = summary

    # ── Save results ──
    out_path = os.path.join(crops_dir, "clip_filtering_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("CLIP FILTERING COMPLETE")
    print(f"  Total crops:    {total}")
    print(f"  Classified TEXT:     {total_text}  ({summary['text_ratio']*100:.1f}%)")
    print(f"  Classified NON-TEXT: {total_nontext}  ({(1-summary['text_ratio'])*100:.1f}%)")
    print(f"  Time: {elapsed:.2f}s  ({elapsed/max(total,1):.3f}s per crop)")
    print(f"  Results saved → {out_path}")

    if save_filtered:
        print(f"  Filtered text crops saved → {filtered_dir}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CLIP zero-shot text/non-text classifier for OCR crops."
    )
    parser.add_argument(
        "--crops_dir", type=str, default="crops",
        help="Directory containing crop images or subdirectories (default: crops)",
    )
    parser.add_argument(
        "--model", type=str, default="ViT-L/14",
        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"],
        help="CLIP model variant (default: ViT-L/14)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.0,
        help="Decision threshold offset. 0=balanced, positive=stricter (default: 0.0)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for GPU inference (default: 32)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        choices=["cpu", "cuda"],
        help="Inference device. Auto-detects if not set.",
    )
    parser.add_argument(
        "--save_filtered", action="store_true",
        help="Save text-only crops to a separate directory.",
    )
    parser.add_argument(
        "--filtered_dir", type=str, default="crops_filtered",
        help="Output directory for filtered text crops (default: crops_filtered)",
    )
    parser.add_argument(
        "--image_name", type=str, default=None,
        help="Name of a single image group folder under crops_dir to process (optional)",
    )
    args = parser.parse_args()

    filter_crops(
        crops_dir=args.crops_dir,
        model_name=args.model,
        threshold=args.threshold,
        batch_size=args.batch_size,
        save_filtered=args.save_filtered,
        filtered_dir=args.filtered_dir,
        device=args.device,
        image_name=args.image_name,
    )


if __name__ == "__main__":
    main()

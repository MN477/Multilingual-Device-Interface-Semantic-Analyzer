"""
clp_classif.py — CLIP-based semantic text classifier for system vs user text.

This code runs an end-to-end pipeline:
  1. OCR Detection & Recognition: Uses PaddleOCR to extract text regions along
     with their recognized text strings (vital for semantic grouping and sentence building).
  2. Bounding Box Post-processing: Merges text boxes that belong to the same lines/sentences
     so they remain semantically connected.
  3. CLIP Filtering & Zero-Shot Classification:
     - Stage 1: Filters out irrelevant crops, icons, noise, isolated letters in circles/boxes.
     - Stage 2: Classifies the semantically connected text crops as system text or user text.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import argparse
import time
from typing import List, Dict, Tuple, Optional

import numpy as np
import cv2

try:
    import torch
    import clip
    from PIL import Image
    from paddleocr import PaddleOCR
except ImportError as e:
    print(f"ERROR: Missing dependency — {e}")
    print("Please run: pip install torch torchvision openai-clip Pillow paddleocr")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# CRITICAL SEMANTIC PROMPTS FOR CLIP
# ═══════════════════════════════════════════════════════════════════════════════

# Stage 1: Separating real meaningful text from noise, logos, and irrelevant isolated characters (circles/boxes)
RELEVANT_TEXT_PROMPTS = [
    "a cropped image of readable words and sentences",
    "a phrase, line of text, or sentence displayed on a screen",
    "a coherent string of alphanumeric words",
    "a paragraph line or complete user interface text description",
]

IRRELEVANT_NONTEXT_PROMPTS = [
    "a single isolated letter in a circle or box",
    "an icon, symbol, shape, or geometric emblem",
    "a decorative frame border or graphical button outline",
    "a vertical separator line or background UI noise",
    "a blank area or blurry pixel patch",
    "a single digit, letter, or random non-word indicator",
]

# Stage 2: Distinguishing system (fixed menus, labels, options) vs. user text (typed content, inputs, messages)
SYSTEM_TEXT_PROMPTS = [
    "a static system menu item or button label on a digital interface",
    "a form field label like username, password, email, or settings option",
    "software application window title, navigation bar text, or status label",
    "a system error message, toolbar button caption, or dialog box option",
    "static text on a device button, control panel label, or header text",
]

USER_TEXT_PROMPTS = [
    "user typed input text inside a text field, chat bubble, or message body",
    "a customized chat reply, comment, search query, or user-written note",
    "informal free-form text entered dynamically by a human user",
    "user-generated content displayed inside an editable text area",
    "messages or entries representing personal user activity and typing",
]


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT MERGER & SEMANTIC LINE RECONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def merge_text_lines(ocr_results: List[Dict], y_threshold: float = 12.0, x_threshold: float = 50.0) -> List[Dict]:
    """
    Groups OCR words/boxes into semantically connected lines or sentences.
    This ensures that when CLIP evaluates the crop, it gets the complete line
    rather than isolated word fragments.
    """
    if not ocr_results:
        return []

    # Sort primarily by center Y coordinate, then by center X coordinate
    sorted_ocr = sorted(ocr_results, key=lambda x: (x["center"][1], x["center"][0]))
    
    merged_groups = []
    
    for item in sorted_ocr:
        # Try to find a group this item belongs to
        placed = False
        for group in merged_groups:
            # Check vertical overlap (Y closeness) and horizontal proximity (X closeness)
            mean_y = np.mean([member["center"][1] for member in group])
            last_member = sorted(group, key=lambda m: m["center"][0])[-1]
            
            y_diff = abs(item["center"][1] - mean_y)
            x_gap = item["bbox"][0] - last_member["bbox"][2]
            
            if y_diff < y_threshold and 0 <= x_gap < x_threshold:
                group.append(item)
                placed = True
                break
        
        if not placed:
            merged_groups.append([item])
            
    # Reconstruct bounding boxes and concatenated text for each merged group
    final_regions = []
    for idx, group in enumerate(merged_groups):
        # Sort group members from left to right
        group = sorted(group, key=lambda x: x["bbox"][0])
        
        combined_text = " ".join([m["text"] for m in group])
        
        # Merge bounding boxes
        xmins = [m["bbox"][0] for m in group]
        ymins = [m["bbox"][1] for m in group]
        xmaxs = [m["bbox"][2] for m in group]
        ymaxs = [m["bbox"][3] for m in group]
        
        merged_bbox = [min(xmins), min(ymins), max(xmaxs), max(ymaxs)]
        
        # Build composite polygon (4 points)
        merged_poly = [
            [merged_bbox[0], merged_bbox[1]],
            [merged_bbox[2], merged_bbox[1]],
            [merged_bbox[2], merged_bbox[3]],
            [merged_bbox[0], merged_bbox[3]]
        ]
        
        final_regions.append({
            "index": idx,
            "text": combined_text,
            "bbox": merged_bbox,
            "polygon": merged_poly,
            "width": merged_bbox[2] - merged_bbox[0],
            "height": merged_bbox[3] - merged_bbox[1],
            "center": [(merged_bbox[0] + merged_bbox[2]) / 2.0, (merged_bbox[1] + merged_bbox[3]) / 2.0]
        })
        
    return final_regions


# ═══════════════════════════════════════════════════════════════════════════════
# CROPPER
# ═══════════════════════════════════════════════════════════════════════════════

def extract_crop(image: np.ndarray, bbox: List[float], padding: int = 4) -> Optional[np.ndarray]:
    """Crops the bounding box from the image with safety padding."""
    h, w = image.shape[:2]
    xmin = max(0, int(bbox[0]) - padding)
    ymin = max(0, int(bbox[1]) - padding)
    xmax = min(w, int(bbox[2]) + padding)
    ymax = min(h, int(bbox[3]) + padding)

    if xmax <= xmin or ymax <= ymin:
        return None

    return image[ymin:ymax, xmin:xmax].copy()


# ═══════════════════════════════════════════════════════════════════════════════
# CLIP CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

class CLIPSemanticClassifier:
    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        print(f"Loading CLIP model '{model_name}' on {device}...")
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        
        # Pre-calculate prompt feature representations
        self.relevant_feats = self._encode_prompts(RELEVANT_TEXT_PROMPTS)
        self.irrelevant_feats = self._encode_prompts(IRRELEVANT_NONTEXT_PROMPTS)
        self.system_feats = self._encode_prompts(SYSTEM_TEXT_PROMPTS)
        self.user_feats = self._encode_prompts(USER_TEXT_PROMPTS)

    def _encode_prompts(self, prompts: List[str]) -> torch.Tensor:
        tokens = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def classify_crop(self, pil_img: Image.Image) -> Dict:
        # Preprocess and encode image
        tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_feat = self.model.encode_image(tensor)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        # Stage 1: Relevant Text Filter
        rel_sims = (img_feat @ self.relevant_feats.T).squeeze(0).cpu().numpy()
        irrel_sims = (img_feat @ self.irrelevant_feats.T).squeeze(0).cpu().numpy()
        
        rel_score = float(rel_sims.mean())
        irrel_score = float(irrel_sims.mean())
        
        is_relevant_text = rel_score > irrel_score

        if not is_relevant_text:
            return {
                "is_relevant": False,
                "label": "ignored/non_text",
                "rel_score": round(rel_score, 5),
                "irrel_score": round(irrel_score, 5)
            }

        # Stage 2: System vs User classification
        sys_sims = (img_feat @ self.system_feats.T).squeeze(0).cpu().numpy()
        usr_sims = (img_feat @ self.user_feats.T).squeeze(0).cpu().numpy()
        
        sys_score = float(sys_sims.mean())
        usr_score = float(usr_sims.mean())
        
        label = "system" if sys_score > usr_score else "user"

        return {
            "is_relevant": True,
            "label": label,
            "sys_score": round(sys_score, 5),
            "usr_score": round(usr_score, 5),
            "rel_score": round(rel_score, 5),
            "irrel_score": round(irrel_score, 5)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(image_path: str, clip_model: str = "ViT-B/32", output_dir: str = "clp_output"):
    if not os.path.exists(image_path):
        print(f"Error: image not found at '{image_path}'")
        return

    os.makedirs(output_dir, exist_ok=True)
    img_cv2 = cv2.imread(image_path)
    if img_cv2 is None:
        print(f"Error decoding image: {image_path}")
        return

    print("=" * 80)
    print(f"Processing: {image_path}")
    print("=" * 80)

    # 1. OCR Detection & Recognition
    print("Running OCR (PaddleOCR)...")
    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    results = ocr.ocr(image_path, cls=True)

    raw_ocr_items = []
    if results and results[0]:
        for idx, res in enumerate(results[0]):
            box = res[0] # List of 4 points
            text, score = res[1]
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            bbox = [min(xs), min(ys), max(xs), max(ys)]
            raw_ocr_items.append({
                "text": text,
                "bbox": bbox,
                "center": [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]
            })

    print(f"Found {len(raw_ocr_items)} raw text detections.")

    # 2. Merge detections into semantically connected lines/sentences
    print("Merging fragmented text fragments into connected sentences...")
    semantic_sentences = merge_text_lines(raw_ocr_items)
    print(f"Merged into {len(semantic_sentences)} semantic sentence blocks.")

    # 3. Classify with CLIP
    classifier = CLIPSemanticClassifier(model_name=clip_model)
    
    annotated_img = img_cv2.copy()
    final_output = []

    for item in semantic_sentences:
        crop_bgr = extract_crop(img_cv2, item["bbox"])
        if crop_bgr is None or crop_bgr.size == 0:
            continue

        # Convert to PIL RGB for CLIP compatibility
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)

        classification = classifier.classify_crop(pil_img)
        
        # Color coding: Green = System, Blue = User, Red = Ignored/Non-text
        label = classification["label"]
        if label == "system":
            color = (0, 220, 0)
        elif label == "user":
            color = (255, 120, 0)
        else:
            color = (0, 0, 220)

        poly_pts = np.array(item["polygon"], dtype=np.int32)
        cv2.polylines(annotated_img, [poly_pts], isClosed=True, color=color, thickness=2)
        
        tag = f"{label}: '{item['text']}'"
        print(f" -> [{label.upper()}] Text: '{item['text']}' (Rel: {classification['rel_score']:.3f}, Irrel: {classification['irrel_score']:.3f})")
        
        cv2.putText(annotated_img, label, (int(item["bbox"][0]), max(int(item["bbox"][1]) - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        final_output.append({
            "text": item["text"],
            "bbox": item["bbox"],
            "classification": classification
        })

    # Save output files
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_img_path = os.path.join(output_dir, f"{base_name}_semantic_result.png")
    out_json_path = os.path.join(output_dir, f"{base_name}_semantic_result.json")

    cv2.imwrite(out_img_path, annotated_img)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print("\nProcessing complete!")
    print(f"Saved annotated image: {out_img_path}")
    print(f"Saved JSON results: {out_json_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantically connected system vs user text classifier with CLIP.")
    parser.add_argument("--image", type=str, required=True, help="Path to input screenshot/image.")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32", help="CLIP model variant.")
    parser.add_argument("--output_dir", type=str, default="clp_output", help="Output directory.")
    args = parser.parse_args()

    run_pipeline(args.image, args.clip_model, args.output_dir)

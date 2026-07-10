"""
system_user_clip.py — End-to-end system/user text classifier for device screenshots.

Takes a single image (screenshot of a device UI, HMI panel, phone screen, etc.)
and performs the full pipeline:

  1. TEXT DETECTION   — PaddleOCR PP-OCRv6 detects all text-like regions
  2. ROI CROPPING     — Each detected polygon is perspective-cropped
  3. NON-TEXT FILTER  — CLIP zero-shot eliminates icons, lines, noise, patterns
  4. SEMANTIC CLASSIF — CLIP zero-shot classifies each pure-text crop as:
                          • SYSTEM TEXT  (UI labels, menus, status, fixed captions)
                          • USER TEXT    (typed input, messages, editable content)

The key insight is that CLIP can distinguish visual context cues around text:
  - System text tends to appear on buttons, headers, status bars, with uniform
    styling, standard fonts, and structured placement
  - User text tends to appear in input fields, message bubbles, text areas,
    with variable content and often different visual treatment

Output:
  - Per-crop classification JSON with scores
  - Annotated image with color-coded bounding boxes (green=system, blue=user, red=non-text)
  - Filtered crops saved in organized subdirectories

Usage:
    python system_user_clip.py --image input_images/3173.jpg
    python system_user_clip.py --image input_images/im1.jpg --output_dir results --clip_model ViT-L/14
    python system_user_clip.py --image input_images/desktop_UI.jpeg --det_threshold 0.5
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import time
import argparse
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
# STAGE 1 — TEXT DETECTION (PaddleOCR)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_text_regions(image_path: str) -> Tuple[List[Dict], float]:
    """
    Run PaddleOCR PP-OCRv6 text detection on a single image.

    Args:
        image_path: Path to the input image.

    Returns:
        detections: List of detection dicts with 'polygon', 'bbox', 'det_score'.
        inference_time: Detection time in seconds.
    """
    try:
        from paddleocr import TextDetection
    except ImportError:
        print("ERROR: paddleocr not installed. Run: pip install paddleocr")
        sys.exit(1)

    print("  Loading PP-OCRv6 text detector...")
    detector = TextDetection(model_name="PP-OCRv6_medium_det", device="cpu")

    start = time.time()
    result = detector.predict(image_path)
    elapsed = time.time() - start

    detections = []
    for page in result:
        if hasattr(page, "keys") or isinstance(page, dict):
            dt_polys = page.get("dt_polys", [])
            dt_scores = page.get("dt_scores", [])
        elif hasattr(page, "__dict__"):
            dt_polys = getattr(page, "dt_polys", [])
            dt_scores = getattr(page, "dt_scores", [])
        else:
            continue

        for i, poly in enumerate(dt_polys):
            poly_np = np.array(poly, dtype=np.float32)
            score = float(dt_scores[i]) if i < len(dt_scores) else 0.0

            if poly_np.size >= 8:
                xmin, ymin = poly_np.min(axis=0)
                xmax, ymax = poly_np.max(axis=0)
                detections.append({
                    "index": i,
                    "det_score": round(score, 4),
                    "polygon": poly_np.tolist(),
                    "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)],
                    "width": float(xmax - xmin),
                    "height": float(ymax - ymin),
                    "center": [float(xmin + (xmax - xmin) / 2),
                               float(ymin + (ymax - ymin) / 2)],
                })

    return detections, elapsed


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — ROI CROPPING
# ═══════════════════════════════════════════════════════════════════════════════

def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 polygon points as: [TL, TR, BR, BL]."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def crop_polygon(image: np.ndarray, polygon: List, padding: int = 3) -> Optional[np.ndarray]:
    """
    Perspective-warp crop a 4-point polygon from the image.
    Falls back to axis-aligned bbox if polygon is degenerate.
    """
    poly = np.array(polygon, dtype=np.float32)

    if poly.shape[0] == 4:
        ordered = order_points(poly)
        w1 = np.linalg.norm(ordered[1] - ordered[0])
        w2 = np.linalg.norm(ordered[2] - ordered[3])
        h1 = np.linalg.norm(ordered[3] - ordered[0])
        h2 = np.linalg.norm(ordered[2] - ordered[1])
        out_w = int(max(w1, w2)) + 2 * padding
        out_h = int(max(h1, h2)) + 2 * padding

        if out_w < 4 or out_h < 4:
            return None

        dst = np.array([
            [padding, padding],
            [out_w - padding, padding],
            [out_w - padding, out_h - padding],
            [padding, out_h - padding],
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(ordered, dst)
        crop = cv2.warpPerspective(image, M, (out_w, out_h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
        return crop

    # Fallback: axis-aligned bbox
    xmin, ymin = poly.min(axis=0).astype(int)
    xmax, ymax = poly.max(axis=0).astype(int)
    h, w = image.shape[:2]
    xmin = max(0, xmin - padding)
    ymin = max(0, ymin - padding)
    xmax = min(w, xmax + padding)
    ymax = min(h, ymax + padding)
    if xmax <= xmin or ymax <= ymin:
        return None
    return image[ymin:ymax, xmin:xmax].copy()


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 & 4 — CLIP-BASED NON-TEXT FILTER + SYSTEM/USER CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

# ─── STEP 1: Filter out non-text crops ───────────────────────────────────────
# These prompts are tuned to separate genuine readable text from OCR false
# positives (icons, logos, patterns, borders, noise, decorative elements).

PLAIN_TEXT_PROMPTS = [
    # Pure typography — what real text looks like as a crop
    "a cropped image of printed text characters on a screen",
    "a close-up of a word or sentence displayed on a digital screen",
    "readable alphanumeric characters in a standard font",
    "a text label, title, or caption shown on a device display",
    "plain text content rendered in a clean typeface",
    "a short string of letters, digits, or punctuation marks",
    "a line of text from a user interface or application",
    "characters from any writing system displayed on a screen",
    "a number, time, date, or numeric readout on a display",
]

NOT_PLAIN_TEXT_PROMPTS = [
    # Common OCR false positives — things detectors confuse for text
    "an icon, symbol, or pictogram without any text",
    "a graphical logo or emblem with no readable characters",
    "a decorative border, line, separator, or divider",
    "a gradient, shadow, or solid color block",
    "a blurry or noisy image patch with no readable content",
    "a photograph of an object, person, or scene",
    "a texture, pattern, or surface without characters",
    "a button or toggle with only a graphical symbol",
    "an electrical connector, wire, port, or hardware component",
    "a progress bar, slider, or gauge indicator",
    "a colored indicator light or LED without text",
]

# ─── STEP 2: Classify pure text as system vs user ────────────────────────────
# These prompts capture the visual and contextual differences between
# system-generated text (static UI labels) and user-generated text (dynamic input).
#
# Key visual cues CLIP can leverage:
#   SYSTEM: uniform styling, standard placement (headers, footers, sidebars),
#           consistent font weight, appears on buttons/labels/menus/toolbars
#   USER:   appears inside text fields/chat bubbles/editors, often different
#           background, variable content, cursor presence, input field styling

SYSTEM_TEXT_PROMPTS = [
    # ── UI labels & navigation ──
    "a system menu label or navigation text on a device interface",
    "a button label or toolbar text in a software application",
    "a header, title, or section name in a user interface",
    "a status bar indicator showing time, battery, or signal info",
    "a settings option or configuration label in a menu",
    "a system notification or alert title text",
    "a tab label, breadcrumb, or navigation link text",
    # ── Fixed UI captions ──
    "a fixed caption or tooltip text placed by the operating system",
    "a form field label such as Name, Email, Password, or Date",
    "a dialog box title or confirmation prompt text",
    "a system-generated error message or warning label",
    "a copyright notice, version number, or footer text",
    # ── Visual style cues ──
    "text with uniform font styling rendered as part of the interface chrome",
    "text appearing on a colored button or control element",
    "small-size system text like a footnote, disclaimer, or watermark",
]

USER_TEXT_PROMPTS = [
    # ── Input fields & editable areas ──
    "user-typed text inside an input field or text box",
    "a chat message or conversation bubble with personal content",
    "text entered by a user in a search bar or form field",
    "a typed reply, comment, or message in a messaging application",
    "editable text content in a document or note-taking app",
    "user-written text appearing inside a text editor or code editor",
    # ── Content characteristics ──
    "personal text content such as a name, address, or phone number",
    "a user query, search term, or typed command",
    "text inside a highlighted or selected text field area",
    "a free-form text entry with variable or informal content",
    "text with a blinking cursor or text selection highlight nearby",
    # ── Visual style cues ──
    "text displayed inside a white or light-colored input rectangle",
    "text appearing in a message bubble with rounded corners",
    "text with a different visual style from surrounding UI labels",
    "user-generated content that looks distinct from interface chrome",
]


class SystemUserCLIPClassifier:
    """
    Two-stage CLIP zero-shot classifier for device screenshot text.

    Stage A: Filter non-text crops (icons, noise, borders) from genuine text.
    Stage B: Classify genuine text as system-generated vs user-generated.

    Both stages use prompt-bank cosine similarity with CLIP embeddings.
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = None,
        text_filter_threshold: float = 0.0,
        system_user_threshold: float = 0.0,
    ):
        """
        Args:
            model_name: CLIP variant ('ViT-B/32', 'ViT-B/16', 'ViT-L/14').
            device:     'cpu' or 'cuda'. Auto-detects if None.
            text_filter_threshold: Offset for text vs non-text decision.
                                   Positive = stricter text filtering.
            system_user_threshold: Offset for system vs user decision.
                                   Positive = more likely to classify as user.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.text_filter_threshold = text_filter_threshold
        self.system_user_threshold = system_user_threshold

        print(f"  Loading CLIP model: {model_name} on {device}...")
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        print(f"  CLIP loaded successfully.")

        # Pre-encode all prompt banks
        print(f"  Encoding prompt banks...")
        self.plain_text_feats = self._encode_prompts(PLAIN_TEXT_PROMPTS)
        self.not_text_feats = self._encode_prompts(NOT_PLAIN_TEXT_PROMPTS)
        self.system_feats = self._encode_prompts(SYSTEM_TEXT_PROMPTS)
        self.user_feats = self._encode_prompts(USER_TEXT_PROMPTS)
        print(f"  Prompt banks encoded: "
              f"text={len(PLAIN_TEXT_PROMPTS)}, non-text={len(NOT_PLAIN_TEXT_PROMPTS)}, "
              f"system={len(SYSTEM_TEXT_PROMPTS)}, user={len(USER_TEXT_PROMPTS)}")

    def _encode_prompts(self, prompts: List[str]) -> torch.Tensor:
        """Encode text prompts into L2-normalized feature vectors."""
        tokens = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def _encode_image(self, pil_img: Image.Image) -> torch.Tensor:
        """Encode a PIL image into an L2-normalized feature vector."""
        tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def _similarity_scores(
        self, img_feat: torch.Tensor, prompt_feats: torch.Tensor
    ) -> np.ndarray:
        """Compute cosine similarities between image and prompt features."""
        return (img_feat @ prompt_feats.T).squeeze(0).cpu().numpy()

    def classify_crop(self, pil_img: Image.Image) -> Dict:
        """
        Full two-stage classification of a single crop.

        Returns:
            Dict with:
              'is_text':       bool — passed the non-text filter
              'label':         'system' | 'user' | 'non_text'
              'text_score':    mean similarity to text prompts
              'nontext_score': mean similarity to non-text prompts
              'system_score':  mean similarity to system prompts (if text)
              'user_score':    mean similarity to user prompts (if text)
              'confidence':    decision margin
        """
        img_feat = self._encode_image(pil_img)

        # ── Stage A: text vs non-text ──
        text_sims = self._similarity_scores(img_feat, self.plain_text_feats)
        nontext_sims = self._similarity_scores(img_feat, self.not_text_feats)

        text_mean = float(text_sims.mean())
        nontext_mean = float(nontext_sims.mean())
        text_margin = text_mean - nontext_mean
        is_text = text_margin > self.text_filter_threshold

        result = {
            "is_text": is_text,
            "text_score": round(text_mean, 5),
            "nontext_score": round(nontext_mean, 5),
            "text_filter_margin": round(text_margin, 5),
        }

        if not is_text:
            result["label"] = "non_text"
            result["confidence"] = round(abs(text_margin), 5)
            result["system_score"] = None
            result["user_score"] = None
            return result

        # ── Stage B: system vs user ──
        sys_sims = self._similarity_scores(img_feat, self.system_feats)
        usr_sims = self._similarity_scores(img_feat, self.user_feats)

        sys_mean = float(sys_sims.mean())
        usr_mean = float(usr_sims.mean())
        su_margin = sys_mean - usr_mean

        is_system = su_margin > self.system_user_threshold

        # Top matching prompts for interpretability
        sys_top_idx = sys_sims.argsort()[::-1][:3]
        usr_top_idx = usr_sims.argsort()[::-1][:3]

        result.update({
            "label": "system" if is_system else "user",
            "system_score": round(sys_mean, 5),
            "user_score": round(usr_mean, 5),
            "system_user_margin": round(su_margin, 5),
            "confidence": round(abs(su_margin), 5),
            "system_top3": [
                (SYSTEM_TEXT_PROMPTS[i], round(float(sys_sims[i]), 4))
                for i in sys_top_idx
            ],
            "user_top3": [
                (USER_TEXT_PROMPTS[i], round(float(usr_sims[i]), 4))
                for i in usr_top_idx
            ],
        })

        return result

    def classify_batch(
        self, pil_images: List[Image.Image], batch_size: int = 32
    ) -> List[Dict]:
        """
        Efficient batch classification of multiple crops.

        Processes in two passes:
          Pass 1 — text filter (all crops)
          Pass 2 — system/user (text crops only)

        Args:
            pil_images:  List of PIL RGB images.
            batch_size:  Forward-pass batch size.

        Returns:
            List of classification dicts (same format as classify_crop).
        """
        n = len(pil_images)
        results = [None] * n

        # ── Pass 1: Encode all images ──
        all_feats = []
        for i in range(0, n, batch_size):
            batch = pil_images[i:i + batch_size]
            tensors = torch.stack([self.preprocess(img) for img in batch]).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_image(tensors)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats)

        all_feats = torch.cat(all_feats, dim=0)  # (N, D)

        # ── Pass 1: Text filter ──
        text_sims_all = (all_feats @ self.plain_text_feats.T).cpu().numpy()     # (N, T)
        nontext_sims_all = (all_feats @ self.not_text_feats.T).cpu().numpy()    # (N, NT)

        text_indices = []
        for i in range(n):
            text_mean = float(text_sims_all[i].mean())
            nontext_mean = float(nontext_sims_all[i].mean())
            margin = text_mean - nontext_mean
            is_text = margin > self.text_filter_threshold

            results[i] = {
                "is_text": is_text,
                "text_score": round(text_mean, 5),
                "nontext_score": round(nontext_mean, 5),
                "text_filter_margin": round(margin, 5),
            }

            if is_text:
                text_indices.append(i)
            else:
                results[i].update({
                    "label": "non_text",
                    "confidence": round(abs(margin), 5),
                    "system_score": None,
                    "user_score": None,
                })

        # ── Pass 2: System/User classification (text crops only) ──
        if text_indices:
            text_feats = all_feats[text_indices]  # (M, D)
            sys_sims = (text_feats @ self.system_feats.T).cpu().numpy()
            usr_sims = (text_feats @ self.user_feats.T).cpu().numpy()

            for j, idx in enumerate(text_indices):
                sys_mean = float(sys_sims[j].mean())
                usr_mean = float(usr_sims[j].mean())
                margin = sys_mean - usr_mean
                is_system = margin > self.system_user_threshold

                sys_top_idx = sys_sims[j].argsort()[::-1][:3]
                usr_top_idx = usr_sims[j].argsort()[::-1][:3]

                results[idx].update({
                    "label": "system" if is_system else "user",
                    "system_score": round(sys_mean, 5),
                    "user_score": round(usr_mean, 5),
                    "system_user_margin": round(margin, 5),
                    "confidence": round(abs(margin), 5),
                    "system_top3": [
                        (SYSTEM_TEXT_PROMPTS[k], round(float(sys_sims[j][k]), 4))
                        for k in sys_top_idx
                    ],
                    "user_top3": [
                        (USER_TEXT_PROMPTS[k], round(float(usr_sims[j][k]), 4))
                        for k in usr_top_idx
                    ],
                })

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

# Color scheme: BGR format for cv2
COLOR_SYSTEM  = (0, 200, 0)     # Green  — system text
COLOR_USER    = (255, 150, 0)   # Blue   — user text
COLOR_NONTEXT = (0, 0, 200)     # Red    — non-text (filtered out)
COLOR_MAP = {
    "system": COLOR_SYSTEM,
    "user": COLOR_USER,
    "non_text": COLOR_NONTEXT,
}


def draw_classified_detections(
    image: np.ndarray,
    detections: List[Dict],
    classifications: List[Dict],
) -> np.ndarray:
    """
    Draw color-coded bounding boxes on the image.

    Green = system text, Blue = user text, Red = non-text (filtered).
    """
    annotated = image.copy()

    for det, clf in zip(detections, classifications):
        poly = np.array(det["polygon"], dtype=np.int32)
        label = clf.get("label", "non_text")
        color = COLOR_MAP.get(label, COLOR_NONTEXT)
        confidence = clf.get("confidence", 0)

        # Draw polygon
        cv2.polylines(annotated, [poly], True, color, 2)

        # Label text
        xmin = int(poly[:, 0].min())
        ymin = int(poly[:, 1].min())
        tag = f"{label} ({confidence:.2f})"
        cv2.putText(annotated, tag, (xmin, max(ymin - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    return annotated


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def process_image(
    image_path: str,
    output_dir: str = "system_user_output",
    clip_model: str = "ViT-B/32",
    device: str = None,
    text_filter_threshold: float = 0.0,
    system_user_threshold: float = 0.0,
    crop_padding: int = 3,
    det_threshold: float = 0.5,
    save_crops: bool = True,
    batch_size: int = 32,
) -> Dict:
    """
    Full end-to-end pipeline: detect → crop → filter → classify.

    Args:
        image_path:             Path to the input screenshot/image.
        output_dir:             Directory for all outputs.
        clip_model:             CLIP model variant.
        device:                 'cpu' or 'cuda'.
        text_filter_threshold:  Text/non-text filter offset.
        system_user_threshold:  System/user classifier offset.
        crop_padding:           Pixel padding around crops.
        det_threshold:          Minimum detection score to keep.
        save_crops:             Whether to save individual crop images.
        batch_size:             CLIP batch size.

    Returns:
        Results dict with all classifications and metadata.
    """
    if not os.path.isfile(image_path):
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(f"SYSTEM/USER TEXT CLASSIFIER — {os.path.basename(image_path)}")
    print("=" * 70)

    # ── Load image ──
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not read image: {image_path}")
        sys.exit(1)
    print(f"  Image size: {image.shape[1]}×{image.shape[0]}")

    # ── Stage 1: Text Detection ──
    print("\n[STAGE 1] Text Detection (PaddleOCR PP-OCRv6)...")
    detections, det_time = detect_text_regions(image_path)
    print(f"  Detected {len(detections)} region(s) in {det_time:.2f}s")

    # Filter by detection confidence
    if det_threshold > 0:
        before = len(detections)
        detections = [d for d in detections if d["det_score"] >= det_threshold]
        print(f"  After det_threshold={det_threshold}: {len(detections)} "
              f"(filtered {before - len(detections)})")

    if not detections:
        print("  No detections remain. Nothing to classify.")
        return {"image": image_path, "detections": [], "classifications": []}

    # ── Stage 2: Crop ROIs ──
    print(f"\n[STAGE 2] Cropping {len(detections)} ROI(s)...")
    crops_bgr = []
    crops_pil = []
    valid_detections = []

    for det in detections:
        crop = crop_polygon(image, det["polygon"], padding=crop_padding)
        if crop is None or crop.shape[0] < 5 or crop.shape[1] < 5:
            continue
        crops_bgr.append(crop)
        # Convert BGR→RGB for CLIP
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crops_pil.append(Image.fromarray(rgb))
        valid_detections.append(det)

    print(f"  Valid crops: {len(crops_pil)} / {len(detections)}")

    if not crops_pil:
        print("  No valid crops. Nothing to classify.")
        return {"image": image_path, "detections": [], "classifications": []}

    # ── Stage 3+4: CLIP Classification ──
    print(f"\n[STAGE 3] CLIP Classification ({clip_model})...")
    classifier = SystemUserCLIPClassifier(
        model_name=clip_model,
        device=device,
        text_filter_threshold=text_filter_threshold,
        system_user_threshold=system_user_threshold,
    )

    start = time.time()
    classifications = classifier.classify_batch(crops_pil, batch_size=batch_size)
    clip_time = time.time() - start
    print(f"  Classified {len(classifications)} crop(s) in {clip_time:.2f}s")

    # ── Count results ──
    n_system = sum(1 for c in classifications if c["label"] == "system")
    n_user = sum(1 for c in classifications if c["label"] == "user")
    n_nontext = sum(1 for c in classifications if c["label"] == "non_text")
    print(f"\n  Results: SYSTEM={n_system}  USER={n_user}  NON-TEXT={n_nontext}")

    # ── Save crops ──
    if save_crops:
        crops_base = os.path.join(output_dir, image_name)
        system_dir = os.path.join(crops_base, "system")
        user_dir = os.path.join(crops_base, "user")
        nontext_dir = os.path.join(crops_base, "non_text")
        os.makedirs(system_dir, exist_ok=True)
        os.makedirs(user_dir, exist_ok=True)
        os.makedirs(nontext_dir, exist_ok=True)

        for i, (crop_bgr, clf) in enumerate(zip(crops_bgr, classifications)):
            label = clf["label"]
            if label == "system":
                dest = system_dir
            elif label == "user":
                dest = user_dir
            else:
                dest = nontext_dir
            crop_path = os.path.join(dest, f"crop_{i:03d}.png")
            cv2.imwrite(crop_path, crop_bgr)
            clf["crop_path"] = crop_path

        print(f"  Crops saved → {crops_base}/")

    # ── Save annotated image ──
    annotated = draw_classified_detections(image, valid_detections, classifications)
    annotated_path = os.path.join(output_dir, f"{image_name}_classified.png")
    cv2.imwrite(annotated_path, annotated)
    print(f"  Annotated image → {annotated_path}")

    # ── Save JSON results ──
    output_data = {
        "image_path": image_path,
        "image_size": [image.shape[1], image.shape[0]],
        "clip_model": clip_model,
        "text_filter_threshold": text_filter_threshold,
        "system_user_threshold": system_user_threshold,
        "det_threshold": det_threshold,
        "detection_time_sec": round(det_time, 3),
        "classification_time_sec": round(clip_time, 3),
        "summary": {
            "total_detections": len(detections),
            "valid_crops": len(classifications),
            "system_text": n_system,
            "user_text": n_user,
            "non_text": n_nontext,
        },
        "results": [],
    }

    for det, clf in zip(valid_detections, classifications):
        entry = {**det, **clf}
        # Convert non-serializable items
        if "system_top3" in entry:
            entry["system_top3"] = [
                {"prompt": p, "score": s} for p, s in entry["system_top3"]
            ]
        if "user_top3" in entry:
            entry["user_top3"] = [
                {"prompt": p, "score": s} for p, s in entry["user_top3"]
            ]
        output_data["results"].append(entry)

    json_path = os.path.join(output_dir, f"{image_name}_classification.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"  JSON results → {json_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    return output_data


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Classify text in a screenshot as system or user text using CLIP."
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to the input screenshot/image.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="system_user_output",
        help="Output directory (default: system_user_output).",
    )
    parser.add_argument(
        "--clip_model", type=str, default="ViT-B/32",
        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"],
        help="CLIP model variant (default: ViT-B/32).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        choices=["cpu", "cuda"],
        help="Inference device (auto-detects if unset).",
    )
    parser.add_argument(
        "--text_filter_threshold", type=float, default=0.0,
        help="Text/non-text filter offset. Positive=stricter. (default: 0.0)",
    )
    parser.add_argument(
        "--system_user_threshold", type=float, default=0.0,
        help="System/user offset. Positive=bias toward user. (default: 0.0)",
    )
    parser.add_argument(
        "--det_threshold", type=float, default=0.5,
        help="Min detection score to keep (default: 0.5).",
    )
    parser.add_argument(
        "--crop_padding", type=int, default=3,
        help="Pixel padding around each crop (default: 3).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="CLIP batch size (default: 32).",
    )
    parser.add_argument(
        "--no_save_crops", action="store_true",
        help="Skip saving individual crop images.",
    )
    args = parser.parse_args()

    process_image(
        image_path=args.image,
        output_dir=args.output_dir,
        clip_model=args.clip_model,
        device=args.device,
        text_filter_threshold=args.text_filter_threshold,
        system_user_threshold=args.system_user_threshold,
        crop_padding=args.crop_padding,
        det_threshold=args.det_threshold,
        save_crops=not args.no_save_crops,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

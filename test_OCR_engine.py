"""
test_OCR_engine.py — Run one or more OCR engines on a raw image and extract
every available output: bounding boxes, polygons, text, confidence, etc.

Usage:
    python test_OCR_engine.py <image_path> [--engine easyocr|tesseract|paddle|all]

Install required engines:
    pip install easyocr
    pip install pytesseract        (+ Tesseract binary: https://github.com/UB-Mannheim/tesseract/wiki)
    pip install paddlepaddle paddleocr
"""

import os
import sys
import json
import argparse
import time
import cv2
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures — unified representation for all engines
# ---------------------------------------------------------------------------
@dataclass
class OCRDetection:
    """Single detection from any OCR engine."""
    engine: str                           # "easyocr" | "tesseract" | "paddle"
    text: str                             # recognized text content
    confidence: float                     # 0.0 – 1.0
    bbox: Optional[list] = None           # [x_min, y_min, x_max, y_max] axis-aligned
    polygon: Optional[list] = None        # [[x1,y1],[x2,y2],...] if available
    # Tesseract-specific extras
    level: Optional[int] = None           # 1=page, 2=block, 3=paragraph, 4=line, 5=word
    block_num: Optional[int] = None
    par_num: Optional[int] = None
    line_num: Optional[int] = None
    word_num: Optional[int] = None
    page_num: Optional[int] = None
    # EasyOCR / Paddle extras
    language: Optional[str] = None
    # General
    width: Optional[int] = None           # bbox width
    height: Optional[int] = None          # bbox height


@dataclass
class OCRResult:
    """Full result from one engine run."""
    engine: str
    image_path: str
    image_width: int
    image_height: int
    elapsed_seconds: float
    num_detections: int
    detections: list = field(default_factory=list)
    raw_engine_output: Optional[str] = None   # stringified raw output for inspection


# ---------------------------------------------------------------------------
# Engine: EasyOCR
# ---------------------------------------------------------------------------
def run_easyocr(img_rgb: np.ndarray, image_path: str, languages: list = None) -> OCRResult:
    """Run EasyOCR and extract all available outputs."""
    import easyocr

    if languages is None:
        languages = ["en"]

    print(f"  [EasyOCR] Initializing reader (languages={languages})...")
    reader = easyocr.Reader(languages, gpu=False, verbose=False)

    print(f"  [EasyOCR] Running detection + recognition...")
    t0 = time.time()
    # detail=1 gives the most verbose output: (bbox, text, confidence)
    # paragraph=False keeps word-level boxes
    raw_results = reader.readtext(
        img_rgb,
        detail=1,
        paragraph=False,
        width_ths=0.7,       # horizontal merge threshold
        height_ths=0.5,      # max height difference for merging
        x_ths=1.0,           # max horizontal distance for merging
        y_ths=0.5,           # max vertical distance for merging
    )
    elapsed = time.time() - t0

    # Also run with paragraph=True to see sentence-level grouping
    raw_paragraph = reader.readtext(
        img_rgb,
        detail=1,
        paragraph=True,
    )

    detections = []
    for bbox_points, text, conf in raw_results:
        # bbox_points is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] — 4-corner polygon
        pts = np.array(bbox_points, dtype=np.float32)
        x_min, y_min = pts.min(axis=0).astype(int).tolist()
        x_max, y_max = pts.max(axis=0).astype(int).tolist()

        detections.append(OCRDetection(
            engine="easyocr",
            text=text,
            confidence=float(conf),
            bbox=[x_min, y_min, x_max, y_max],
            polygon=[[float(p[0]), float(p[1])] for p in bbox_points],
            width=x_max - x_min,
            height=y_max - y_min,
        ))

    result = OCRResult(
        engine="easyocr",
        image_path=image_path,
        image_width=img_rgb.shape[1],
        image_height=img_rgb.shape[0],
        elapsed_seconds=round(elapsed, 3),
        num_detections=len(detections),
        detections=detections,
    )

    # Attach paragraph-level results as extra info
    paragraph_texts = []
    for item in raw_paragraph:
        if len(item) == 3:
            bbox_p, text_p, conf_p = item
            paragraph_texts.append({
                "text": text_p,
                "confidence": float(conf_p),
                "polygon": [[float(p[0]), float(p[1])] for p in bbox_p] if isinstance(bbox_p, list) else str(bbox_p),
            })
        elif len(item) == 2:
            # paragraph mode sometimes returns (text, conf) without bbox
            paragraph_texts.append({"text": str(item[0]), "confidence": float(item[1])})

    result.raw_engine_output = json.dumps({
        "word_level_count": len(raw_results),
        "paragraph_level": paragraph_texts,
    }, ensure_ascii=False, indent=2)

    return result


# ---------------------------------------------------------------------------
# Engine: Tesseract (pytesseract)
# ---------------------------------------------------------------------------
def run_tesseract(img_rgb: np.ndarray, image_path: str, lang: str = "eng") -> OCRResult:
    """Run Tesseract via pytesseract and extract every available field."""
    import pytesseract

    print(f"  [Tesseract] Running with lang={lang}...")
    t0 = time.time()

    # image_to_data gives the most complete output
    # output_type=dict returns a dict of parallel lists
    data = pytesseract.image_to_data(
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
        lang=lang,
        output_type=pytesseract.Output.DICT,
        config="--oem 3"  # LSTM engine
    )
    elapsed = time.time() - t0

    # Also get OSD (orientation and script detection)
    try:
        osd = pytesseract.image_to_osd(
            cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
            output_type=pytesseract.Output.DICT
        )
    except Exception:
        osd = {"error": "OSD failed (common for small images)"}

    # Also get HOCR for maximum detail
    try:
        hocr = pytesseract.image_to_pdf_or_hocr(
            cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
            lang=lang,
            extension="hocr",
            config="--oem 3"
        )
        hocr_str = hocr.decode("utf-8")[:2000] + "... [truncated]" if len(hocr) > 2000 else hocr.decode("utf-8")
    except Exception:
        hocr_str = "HOCR extraction failed"

    detections = []
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        conf = float(data["conf"][i])

        # Tesseract returns conf=-1 for non-text elements (block/paragraph/line headers)
        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])

        detections.append(OCRDetection(
            engine="tesseract",
            text=text,
            confidence=conf / 100.0 if conf >= 0 else -1.0,  # normalize to 0-1
            bbox=[x, y, x + w, y + h],
            polygon=[[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
            width=w,
            height=h,
            level=int(data["level"][i]),
            block_num=int(data["block_num"][i]),
            par_num=int(data["par_num"][i]),
            line_num=int(data["line_num"][i]),
            word_num=int(data["word_num"][i]),
            page_num=int(data["page_num"][i]),
        ))

    result = OCRResult(
        engine="tesseract",
        image_path=image_path,
        image_width=img_rgb.shape[1],
        image_height=img_rgb.shape[0],
        elapsed_seconds=round(elapsed, 3),
        num_detections=len(detections),
        detections=detections,
        raw_engine_output=json.dumps({
            "osd": osd,
            "hocr_preview": hocr_str,
            "tesseract_levels": {
                1: "page", 2: "block", 3: "paragraph",
                4: "line", 5: "word"
            },
        }, ensure_ascii=False, indent=2),
    )

    return result


# ---------------------------------------------------------------------------
# Engine: PaddleOCR
# ---------------------------------------------------------------------------
def run_paddleocr(img_rgb: np.ndarray, image_path: str, lang: str = "en") -> OCRResult:
    """Run PaddleOCR and extract all available outputs."""
    from paddleocr import PaddleOCR

    print(f"  [PaddleOCR] Initializing (lang={lang})...")
    ocr = PaddleOCR(
        use_angle_cls=True,   # detect text orientation
        lang=lang,
        show_log=False,
       
    )

    print(f"  [PaddleOCR] Running detection + recognition...")
    t0 = time.time()
    raw_results = ocr.ocr(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), cls=True)
    elapsed = time.time() - t0

    detections = []
    if raw_results and raw_results[0]:
        for line in raw_results[0]:
            # line = [polygon_4pts, (text, confidence)]
            pts = np.array(line[0], dtype=np.float32)
            text, conf = line[1]

            x_min, y_min = pts.min(axis=0).astype(int).tolist()
            x_max, y_max = pts.max(axis=0).astype(int).tolist()

            detections.append(OCRDetection(
                engine="paddle",
                text=text,
                confidence=float(conf),
                bbox=[x_min, y_min, x_max, y_max],
                polygon=[[float(p[0]), float(p[1])] for p in pts],
                width=x_max - x_min,
                height=y_max - y_min,
            ))

    result = OCRResult(
        engine="paddle",
        image_path=image_path,
        image_width=img_rgb.shape[1],
        image_height=img_rgb.shape[0],
        elapsed_seconds=round(elapsed, 3),
        num_detections=len(detections),
        detections=detections,
    )

    return result


# ---------------------------------------------------------------------------
# Debug Visualization
# ---------------------------------------------------------------------------
def draw_ocr_results(img_rgb: np.ndarray, result: OCRResult, output_path: str):
    """Draw bounding boxes with text and confidence on the image."""
    debug_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Color per engine
    colors = {
        "easyocr": (0, 255, 0),     # green
        "tesseract": (255, 165, 0),  # orange (BGR)
        "paddle": (255, 0, 255),     # magenta
    }
    color = colors.get(result.engine, (0, 255, 0))

    for det in result.detections:
        # Skip non-text Tesseract entries (level != 5 means not a word)
        if det.engine == "tesseract" and det.level != 5:
            continue
        # Skip empty text
        if not det.text.strip():
            continue
        # Skip negative confidence
        if det.confidence < 0:
            continue

        # Draw polygon if available, else bbox
        if det.polygon and len(det.polygon) >= 4:
            pts = np.array(det.polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(debug_img, [pts], isClosed=True, color=color, thickness=2)
        elif det.bbox:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)

        # Label: text + confidence
        if det.bbox:
            label_x, label_y = det.bbox[0], det.bbox[1] - 5
            if label_y < 15:
                label_y = det.bbox[3] + 15
            label = f"{det.text} ({det.confidence:.2f})"
            # Draw background for readability
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(debug_img, (label_x, label_y - th - 2),
                          (label_x + tw, label_y + 2), (0, 0, 0), -1)
            cv2.putText(debug_img, label, (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imwrite(output_path, debug_img)
    print(f"  Debug image saved to {output_path}")


# ---------------------------------------------------------------------------
# Result serialization
# ---------------------------------------------------------------------------
def result_to_dict(result: OCRResult) -> dict:
    """Convert OCRResult to a JSON-serializable dict."""
    d = {
        "engine": result.engine,
        "image_path": result.image_path,
        "image_width": result.image_width,
        "image_height": result.image_height,
        "elapsed_seconds": result.elapsed_seconds,
        "num_detections": result.num_detections,
        "detections": [],
    }

    for det in result.detections:
        det_dict = {
            "text": det.text,
            "confidence": det.confidence,
            "bbox": det.bbox,
            "polygon": det.polygon,
            "width": det.width,
            "height": det.height,
        }
        # Add engine-specific fields only if they have values
        if det.engine == "tesseract":
            det_dict["level"] = det.level
            det_dict["level_name"] = {
                1: "page", 2: "block", 3: "paragraph", 4: "line", 5: "word"
            }.get(det.level, "unknown")
            det_dict["block_num"] = det.block_num
            det_dict["par_num"] = det.par_num
            det_dict["line_num"] = det.line_num
            det_dict["word_num"] = det.word_num
            det_dict["page_num"] = det.page_num

        if det.language:
            det_dict["language"] = det.language

        d["detections"].append(det_dict)

    if result.raw_engine_output:
        try:
            d["engine_extras"] = json.loads(result.raw_engine_output)
        except (json.JSONDecodeError, TypeError):
            d["engine_extras"] = result.raw_engine_output

    return d


def print_summary(result: OCRResult):
    """Print a human-readable summary to console."""
    print(f"\n{'='*70}")
    print(f"  Engine: {result.engine.upper()}")
    print(f"  Image:  {result.image_path} ({result.image_width}x{result.image_height})")
    print(f"  Time:   {result.elapsed_seconds}s")
    print(f"  Total detections: {result.num_detections}")
    print(f"{'='*70}")

    # Filter to actual text content
    text_dets = [d for d in result.detections
                 if d.text.strip()
                 and d.confidence >= 0
                 and (d.level is None or d.level == 5)]  # Tesseract: only words

    print(f"\n  Text detections ({len(text_dets)} with content):\n")
    print(f"  {'#':<4} {'Confidence':<12} {'BBox':<28} {'Size':<12} {'Text'}")
    print(f"  {'-'*4} {'-'*12} {'-'*28} {'-'*12} {'-'*20}")

    for i, det in enumerate(text_dets):
        bbox_str = f"[{det.bbox[0]:>4},{det.bbox[1]:>4},{det.bbox[2]:>4},{det.bbox[3]:>4}]" if det.bbox else "N/A"
        size_str = f"{det.width}x{det.height}" if det.width else "N/A"
        print(f"  {i:<4} {det.confidence:<12.4f} {bbox_str:<28} {size_str:<12} {det.text}")

    # If Tesseract, also show the hierarchy
    if result.engine == "tesseract":
        print(f"\n  Tesseract Hierarchy:")
        print(f"  {'Level':<12} {'Block':<8} {'Par':<6} {'Line':<6} {'Word':<6} {'Conf':<8} {'Text'}")
        print(f"  {'-'*12} {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*20}")
        for det in result.detections:
            if det.confidence < 0 and not det.text.strip():
                continue
            level_name = {1: "PAGE", 2: "BLOCK", 3: "PARA", 4: "LINE", 5: "WORD"}.get(det.level, "?")
            indent = "  " * (det.level - 1) if det.level else ""
            print(f"  {indent}{level_name:<{12 - len(indent)}} "
                  f"{det.block_num or 0:<8} {det.par_num or 0:<6} "
                  f"{det.line_num or 0:<6} {det.word_num or 0:<6} "
                  f"{det.confidence:<8.4f} {det.text}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Test OCR engines on a raw image — extract all available outputs."
    )
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument(
        "--engine", "-e",
        choices=["easyocr", "tesseract", "paddle", "all"],
        default="all",
        help="Which OCR engine to run (default: all available)"
    )
    parser.add_argument("--lang", "-l", default=None, help="Language code (default: en/eng)")
    parser.add_argument("--output-dir", "-o", default="debug_ocr_output", help="Output directory")

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Image not found: {args.image_path}")
        sys.exit(1)

    # Read image
    img_raw = cv2.imread(args.image_path, cv2.IMREAD_UNCHANGED)
    if img_raw is None:
        print(f"Error: Cannot read image: {args.image_path}")
        sys.exit(1)

    # Convert to RGB
    if len(img_raw.shape) == 3 and img_raw.shape[2] == 4:
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGRA2RGB)
    elif len(img_raw.shape) == 3 and img_raw.shape[2] == 3:
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2RGB)

    print(f"Image loaded: {args.image_path} ({img_rgb.shape[1]}x{img_rgb.shape[0]})")

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.image_path))[0]

    # Determine which engines to run
    engines_to_run = []
    if args.engine == "all":
        engines_to_run = ["easyocr", "tesseract", "paddle"]
    else:
        engines_to_run = [args.engine]

    all_results = {}

    for engine_name in engines_to_run:
        try:
            if engine_name == "easyocr":
                lang = [args.lang] if args.lang else ["en"]
                result = run_easyocr(img_rgb, args.image_path, languages=lang)

            elif engine_name == "tesseract":
                lang = args.lang if args.lang else "eng"
                result = run_tesseract(img_rgb, args.image_path, lang=lang)

            elif engine_name == "paddle":
                lang = args.lang if args.lang else "en"
                result = run_paddleocr(img_rgb, args.image_path, lang=lang)

            else:
                continue

            # Print summary
            print_summary(result)

            # Save debug visualization
            debug_path = os.path.join(args.output_dir, f"{base_name}_{engine_name}.jpg")
            draw_ocr_results(img_rgb, result, debug_path)

            # Save JSON
            json_path = os.path.join(args.output_dir, f"{base_name}_{engine_name}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result_to_dict(result), f, ensure_ascii=False, indent=2)
            print(f"  JSON saved to {json_path}")

            all_results[engine_name] = result_to_dict(result)

        except ImportError as e:
            print(f"\n  [SKIP] {engine_name}: not installed ({e})")
            print(f"         Install with: pip install {engine_name}")
        except Exception as e:
            print(f"\n  [ERROR] {engine_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save combined results if multiple engines ran
    if len(all_results) > 1:
        combined_path = os.path.join(args.output_dir, f"{base_name}_combined.json")
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nCombined results saved to {combined_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()

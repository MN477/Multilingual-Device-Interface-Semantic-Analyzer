"""
streamlit_app.py
================
Streamlit interface for the end-to-end OCR + Semantic Classification pipeline.
Takes an image and language as input, runs all 7 pipeline stages
(Detection -> Cropping -> Recognition -> CLIP Filtering -> Regex -> HanLP NER -> Classification),
and displays all recognized entities classified as 'variable' or 'static'.

Usage:
    streamlit run streamlit_app.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"

import sys
sys.stdout.reconfigure(encoding='utf-8')

# Pre-load PyTorch DLLs before Paddle to prevent Windows OpenMP conflict
try:
    import torch
except Exception:
    pass

import json
import time
import tempfile
import numpy as np
import cv2
from PIL import Image

import streamlit as st

# ─── Import pipeline modules ─────────────────────────────────────────────────
from pipeline_core import PipelineRunner, PipelineContext
from paddleocr_det_module import PaddleDetStage
from crops_extraction import CropExtractionStage
from paddleocr_recognition_module import RecognitionStage
from CLIP_filtering import CLIPFilterStage
from regex_filter import RegexFilterStage
from NER_processor import HanLPNERStage
from classification_stage import ClassificationStage


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

SUPPORTED_LANGUAGES = {
    "en": "English",
    "ch": "Chinese",
    "japan": "Japanese",
    "korean": "Korean",
    "ar": "Arabic",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "ru": "Russian",
    "it": "Italian",
}


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE CONFIG BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_config(lang: str) -> dict:
    """Build pipeline config dict for the selected language."""
    return {
        "paths": {
            "input_images_dir": "input",
            "det_output_dir": "detection_output",
            "crops_output_dir": "crops_input1",
            "rec_output_dir": "recognition_output",
            "final_output_dir": "final_ocr_output"
        },
        "runtime": {
            "use_onnx": True,
            "device": "cpu",
            "batch_size": 32
        },
        "detection": {
            "model_name": "PP-OCRv6_medium_det"
        },
        "cropping": {
            "padding": 2,
            "min_size": 7
        },
        "recognition": {
            "model_name": "PP-OCRv6_medium_rec",
            "lang": lang,
            "basic_confidence_threshold": 0.50
        },
        "filtering_and_rechecking": {
            "enable_clip": False,
            "min_det_score": 0.40,
            "min_rec_confidence": 0.50,
            "filter_mode": "AND",
            "low_rec_threshold": 0.50,
            "high_rec_threshold": 0.83,
            "clip_model": "ViT-L/14",
            "clip_non_text_threshold": 0.0
        },
        "regex": {
            "enabled": True,
            "patterns_file": "regex_patterns.json"
        },
        "ner": {
            "enabled": True
        },
        "classification": {
            "enabled": True,
            "variable_ner_labels": [
                "PERSON", "ORG", "GPE", "LOC", "FAC", "NORP",
                "EVENT", "WORK_OF_ART", "PRODUCT", "LAW", "LANGUAGE"
            ],
            "static_ner_labels": [
                "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"
            ]
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE BUILDER (cached in session state)
# ═══════════════════════════════════════════════════════════════════════════════

def build_pipeline(config: dict) -> PipelineRunner:
    """
    Build the full 7-stage pipeline.
    Stage objects are reused across runs via st.session_state to avoid
    reloading heavy models (HanLP, CLIP) on every Streamlit rerun.
    """
    # Check if we already have a cached pipeline in session state
    if "pipeline_runner" in st.session_state:
        # Update config (language may have changed) and return cached runner
        runner = st.session_state["pipeline_runner"]
        runner.config = config
        return runner

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

    # Stage 3: Recognition
    rec_stage = RecognitionStage(
        model_name=rec_cfg.get("model_name", "PP-OCRv6_medium_rec"),
        lang=rec_cfg.get("lang", "en"),
        device=runtime_cfg.get("device", "cpu"),
        use_onnx=runtime_cfg.get("use_onnx", False)
    )

    # Stage 4: CLIP Filtering
    clip_stage = CLIPFilterStage(
        model_name=clip_cfg.get("clip_model", "ViT-L/14"),
        device=runtime_cfg.get("device", "cpu")
    )

    # Stage 5: Regex Pattern Filter
    regex_stage = RegexFilterStage()

    # Stage 6: HanLP NER
    ner_stage = HanLPNERStage()

    # Stage 7: Static/Variable Classification
    cls_stage = ClassificationStage()

    stages = [det_stage, crop_stage, rec_stage, clip_stage,
              regex_stage, ner_stage, cls_stage]
    runner = PipelineRunner(stages=stages, config=config)

    # Cache in session state for reuse
    st.session_state["pipeline_runner"] = runner
    return runner


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def draw_annotated_image(image: np.ndarray, results: list) -> np.ndarray:
    """Draw bounding boxes on the image color-coded by signal type."""
    annotated = image.copy()
    for item in results:
        poly = item.get("polygon", [])
        if not poly or len(poly) < 4:
            continue

        pts = np.array(poly, dtype=np.int32)
        signal = item.get("final_signal", "static")
        text = item.get("transcribed_text", "")

        if signal == "variable":
            color = (0, 120, 255)   # Orange for variable data
        else:
            color = (200, 200, 200)  # Gray for static UI

        cv2.polylines(annotated, [pts], True, color, 2)

        xmin = int(pts[:, 0].min())
        ymin = int(pts[:, 1].min())
        label = f"{text[:30]}"
        cv2.putText(annotated, label, (xmin, max(ymin - 6, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    return annotated


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Semantic OCR Pipeline",
        layout="wide"
    )

    st.title("Multilingual Device Interface Semantic Analyzer")
    st.markdown(
        "Upload a screenshot from a **phone** or **car infotainment screen**, "
        "select the language, and the pipeline will detect, recognize, and semantically "
        "classify every text element as **variable** or **static**."
    )

    # ── Sidebar: Inputs ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Pipeline Settings")

        lang = st.selectbox(
            "Language",
            options=list(SUPPORTED_LANGUAGES.keys()),
            format_func=lambda k: f"{SUPPORTED_LANGUAGES[k]} ({k})",
            index=0
        )

        uploaded_file = st.file_uploader(
            "Upload Image",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            help="Upload a device screenshot (phone or car screen)."
        )

        run_button = st.button("Run Pipeline", type="primary", use_container_width=True)

    # ── Main area ─────────────────────────────────────────────────────────────
    if not uploaded_file:
        st.info("Upload an image and select a language to get started.")
        return

    if not run_button:
        # Show the uploaded image as preview
        pil_img = Image.open(uploaded_file)
        st.image(pil_img, caption="Uploaded Image (preview)", use_container_width=True)
        return

    # ── Run the full 7-stage pipeline ─────────────────────────────────────────
    overall_start = time.time()
    progress = st.progress(0, text="Initializing pipeline...")

    # Save uploaded file to a temp path
    pil_img = Image.open(uploaded_file)
    img_array = np.array(pil_img)
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    else:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    tmp_dir = tempfile.mkdtemp()
    tmp_image_path = os.path.join(tmp_dir, uploaded_file.name)
    cv2.imwrite(tmp_image_path, img_array)

    # Build config and pipeline
    progress.progress(5, text="Building pipeline...")
    config = build_config(lang)
    runner = build_pipeline(config)

    # Run the full pipeline (all 7 stages)
    progress.progress(10, text="Running 7-stage pipeline...")
    context = runner.run_image(tmp_image_path)

    overall_time = time.time() - overall_start
    progress.progress(100, text="Pipeline complete!")
    time.sleep(0.3)
    progress.empty()

    # ── Read classified results from context ──────────────────────────────────
    results = context.classified_results

    if not results:
        st.warning("No text detected in the image.")
        return

    # ═════════════════════════════════════════════════════════════════════════
    # DISPLAY RESULTS
    # ═════════════════════════════════════════════════════════════════════════

    st.success(f"Pipeline completed in **{overall_time:.2f}s**")

    # ── Latency Summary ──────────────────────────────────────────────────────
    latencies = context.stage_latencies
    latency_cols = st.columns(len(latencies) + 1)
    for i, (stage_name, duration) in enumerate(latencies.items()):
        # Use short display names for readability
        short_name = stage_name.replace("Stage", "").strip()
        latency_cols[i].metric(short_name, f"{duration:.2f}s")
    latency_cols[-1].metric("Total", f"{overall_time:.2f}s")

    # ── Annotated Image ──────────────────────────────────────────────────────
    st.subheader("Annotated Image")
    annotated = draw_annotated_image(img_array, results)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    st.image(annotated_rgb, caption="Orange = Variable | Gray = Static", use_container_width=True)

    # ── Summary Counts ───────────────────────────────────────────────────────
    variable_items = [r for r in results if r.get("final_signal") == "variable"]
    static_items = [r for r in results if r.get("final_signal") == "static"]
    total_regex = sum(len(r.get("regex_matches", [])) for r in results)
    total_ner = sum(len(r.get("entities", [])) for r in results)

    st.subheader("Summary")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Text Blocks", len(results))
    col2.metric("Variable", len(variable_items))
    col3.metric("Static", len(static_items))
    col4.metric("Regex Matches", total_regex)
    col5.metric("NER Entities", total_ner)

    # ── Side-by-side: Variable | Static ───────────────────────────────────────
    st.subheader("Classification Results")
    col_var, col_stat = st.columns(2)

    with col_var:
        st.markdown("**VARIABLE**")
        if variable_items:
            for item in variable_items:
                text = item.get("transcribed_text", "")
                labels = []
                for rm in item.get("regex_matches", []):
                    labels.append(f"{rm['type']} (regex, {rm['confidence']:.0%})")
                for ent in item.get("entities", []):
                    labels.append(f"{ent['entity_group']} (NER, score={ent['score']:.2f})")
                label_str = " | ".join(labels) if labels else ""
                st.markdown(f"`{text}` -- {label_str}")
        else:
            st.info("No variable entities detected.")

    with col_stat:
        st.markdown("**STATIC**")
        if static_items:
            for item in static_items:
                text = item.get("transcribed_text", "")
                st.markdown(f"`{text}`")
        else:
            st.info("No static text detected.")

    # ── Raw JSON Output ──────────────────────────────────────────────────────
    with st.expander("View Raw JSON Output"):
        st.json(results)


if __name__ == "__main__":
    main()

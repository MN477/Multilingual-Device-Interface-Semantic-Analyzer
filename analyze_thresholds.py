"""
analyze_thresholds.py
=====================
Threshold Calibration & Statistical Analysis Script for OCR Pipeline.
Reads recognition output JSONs and CLIP final output JSONs to analyze score distributions,
evaluate CLIP decisions, and find the optimal CLIP-equivalent thresholds.
"""

import os
import sys
import json
import glob
import argparse
from typing import List, Dict, Any, Tuple, Optional

import numpy as np


def load_all_detections(rec_dir: str, final_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    # Search in final_dir first if available, else rec_dir
    search_dirs = []
    if final_dir and os.path.exists(final_dir):
        search_dirs.append(final_dir)
    if os.path.exists(rec_dir):
        search_dirs.append(rec_dir)

    json_files = []
    for d in search_dirs:
        found = glob.glob(os.path.join(d, "*_final.json")) or glob.glob(os.path.join(d, "*_rec_full.json")) or glob.glob(os.path.join(d, "*.json"))
        if found:
            json_files.extend(found)
            break

    all_items = []
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
                results = data.get("all_detections_with_status", []) or data.get("results", [])
                for item in results:
                    status = item.get("pipeline_status", "")
                    all_items.append({
                        "image_stem": data.get("image_stem", os.path.basename(jf)),
                        "det_score": float(item.get("det_score", 0.0)),
                        "rec_confidence": float(item.get("rec_confidence", 0.0)),
                        "text": item.get("transcribed_text", ""),
                        "pipeline_status": status,
                        "is_kept_by_clip": status.startswith("KEPT") if status else None
                    })
        except Exception as e:
            print(f"Warning: could not read {jf}: {e}")

    return all_items


def print_percentiles(name: str, values: np.ndarray):
    if len(values) == 0:
        return
    print(f"\n--- Distribution Stats for {name} (N={len(values)}) ---")
    print(f"  Min:    {values.min():.4f}")
    print(f"  P10:    {np.percentile(values, 10):.4f}")
    print(f"  P25:    {np.percentile(values, 25):.4f}")
    print(f"  P50 (Median): {np.median(values):.4f}")
    print(f"  Mean:   {values.mean():.4f}")
    print(f"  P75:    {np.percentile(values, 75):.4f}")
    print(f"  P90:    {np.percentile(values, 90):.4f}")
    print(f"  Max:    {values.max():.4f}")


def analyze_clip_decisions(all_items: List[Dict[str, Any]]):
    clip_evaluated = [x for x in all_items if x["is_kept_by_clip"] is not None]
    if not clip_evaluated:
        print("\n[CLIP Analysis] No CLIP status metadata found in JSONs.")
        return

    kept_items = [x for x in clip_evaluated if x["is_kept_by_clip"]]
    rejected_items = [x for x in clip_evaluated if not x["is_kept_by_clip"]]

    print("\n" + "=" * 80)
    print(f"  CLIP DECISION ANALYSIS (Total Evaluated: {len(clip_evaluated)})")
    print("=" * 80)
    print(f"  - Verified Text Regions (KEPT)     : {len(kept_items)} ({len(kept_items)/len(clip_evaluated)*100:.1f}%)")
    print(f"  - Non-Text / Noise (REJECTED)      : {len(rejected_items)} ({len(rejected_items)/len(clip_evaluated)*100:.1f}%)")

    if kept_items:
        det_k = np.array([x["det_score"] for x in kept_items])
        rec_k = np.array([x["rec_confidence"] for x in kept_items])
        print(f"\n  [KEPT Text Regions]      -> Mean det_score: {det_k.mean():.4f} | Mean rec_conf: {rec_k.mean():.4f}")

    if rejected_items:
        det_r = np.array([x["det_score"] for x in rejected_items])
        rec_r = np.array([x["rec_confidence"] for x in rejected_items])
        print(f"  [REJECTED Non-Text]     -> Mean det_score: {det_r.mean():.4f} | Mean rec_conf: {rec_r.mean():.4f}")

    print("=" * 80)

    # ── Grid Search for CLIP-Matching Accuracy ──
    det_thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
    rec_thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    best_match_acc = 0.0
    best_pair = (0.40, 0.50)

    print("\n" + "=" * 80)
    print("  CLIP AGREEMENT MATRIX (% Threshold Decisions Matching CLIP)")
    print("=" * 80)
    print(f"{'det_thresh':<12} | " + " | ".join([f"rec>={r:.2f}" for r in rec_thresholds]))
    print("-" * 80)

    for d_tr in det_thresholds:
        row_str = f"det >= {d_tr:.2f}   | "
        row_accs = []
        for r_tr in rec_thresholds:
            matches = 0
            for x in clip_evaluated:
                thresh_pass = (x["det_score"] >= d_tr and x["rec_confidence"] >= r_tr)
                clip_pass = x["is_kept_by_clip"]
                if thresh_pass == clip_pass:
                    matches += 1
            acc = (matches / len(clip_evaluated)) * 100
            if acc > best_match_acc:
                best_match_acc = acc
                best_pair = (d_tr, r_tr)
            row_accs.append(f"{acc:5.1f}%")
        row_str += " | ".join(row_accs)
        print(row_str)

    print("=" * 80)
    print(f"--> OPTIMAL CLIP-EQUIVALENT THRESHOLD: min_det_score={best_pair[0]}, min_rec_confidence={best_pair[1]} ({best_match_acc:.1f}% Agreement)")
    print("=" * 80)


def run_grid_search(all_items: List[Dict[str, Any]]):
    det_thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
    rec_thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    total_count = len(all_items)
    print("\n" + "=" * 80)
    print(f"  GRID SEARCH SIMULATION: DUAL CONFIDENCE THRESHOLD MATRIX (Total Regions: {total_count})")
    print("=" * 80)
    print(f"{'det_thresh':<12} | " + " | ".join([f"rec>={r:.2f}" for r in rec_thresholds]))
    print("-" * 80)

    for d_tr in det_thresholds:
        row_str = f"det >= {d_tr:.2f}   | "
        row_counts = []
        for r_tr in rec_thresholds:
            kept = [
                x for x in all_items
                if x["det_score"] >= d_tr and x["rec_confidence"] >= r_tr
            ]
            pct = (len(kept) / max(total_count, 1)) * 100
            row_counts.append(f"{len(kept):3d} ({pct:4.1f}%)")
        row_str += " | ".join(row_counts)
        print(row_str)

    print("=" * 80)


def recommend_settings(all_items: List[Dict[str, Any]]):
    det_scores = np.array([x["det_score"] for x in all_items])
    rec_confs = np.array([x["rec_confidence"] for x in all_items])

    med_det = float(np.median(det_scores)) if len(det_scores) > 0 else 0.40
    p25_rec = float(np.percentile(rec_confs, 25)) if len(rec_confs) > 0 else 0.50
    p50_rec = float(np.median(rec_confs)) if len(rec_confs) > 0 else 0.65

    print("\n" + "=" * 80)
    print("  RECOMMENDED THRESHOLD CONFIGURATIONS")
    print("=" * 80)
    print("1. Balanced Mode (Recommended for General Use):")
    print(f"   \"min_det_score\": {max(round(med_det * 0.8, 2), 0.35)}")
    print(f"   \"min_rec_confidence\": {max(round(p25_rec, 2), 0.45)}")
    print("   -> Keeps valid text while dropping noise and weak detections.")

    print("\n2. High Precision Mode (Cleanest output, zero noise):")
    print(f"   \"min_det_score\": {max(round(med_det, 2), 0.50)}")
    print(f"   \"min_rec_confidence\": {max(round(p50_rec, 2), 0.70)}")
    print("   -> Guarantees high-confidence text; ideal for clean downstream processing.")

    print("\n3. High Recall Mode (Catches blurry/small text):")
    print("   \"min_det_score\": 0.30")
    print("   \"min_rec_confidence\": 0.35")
    print("   -> Retains almost all detections; may include some visual artifacts.")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze OCR threshold distributions and CLIP decision alignment.")
    parser.add_argument("--rec_dir", type=str, default="recognition_output", help="Directory containing recognition JSON outputs.")
    parser.add_argument("--final_dir", type=str, default="final_ocr_output", help="Directory containing final JSON outputs (with CLIP status).")
    args = parser.parse_args()

    all_items = load_all_detections(args.rec_dir, args.final_dir)
    if not all_items:
        print(f"No output JSON files found in '{args.rec_dir}' or '{args.final_dir}'.")
        print("Please run 'python run_pipeline.py' on your input folder first.")
        sys.exit(1)

    det_scores = np.array([x["det_score"] for x in all_items])
    rec_confs = np.array([x["rec_confidence"] for x in all_items])

    print_percentiles("Detection Score (det_score)", det_scores)
    print_percentiles("Recognition Confidence (rec_confidence)", rec_confs)

    analyze_clip_decisions(all_items)
    run_grid_search(all_items)
    recommend_settings(all_items)


if __name__ == "__main__":
    main()

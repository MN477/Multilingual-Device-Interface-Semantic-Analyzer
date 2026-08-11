"""
run_ner_enrichment.py
=====================
Unified CLI Orchestrator for Named Entity Recognition (NER) Enrichment on OCR Recognition Outputs.
Runs GLiNER (urchade/gliner_multi-v2.1) and/or XLM-RoBERTa (Davlan/xlm-roberta-base-ner-hrl) on all JSON files
in 'recognition_output/' and outputs model-specific enriched JSON files.
"""

import os
import sys
import argparse

from ner_gliner_processor import process_directory as process_gliner
from ner_xlm_roberta_processor import process_directory as process_xlm_roberta


def main():
    parser = argparse.ArgumentParser(description="Run NER Enrichment on Recognition JSON Outputs.")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["gliner", "xlm_roberta", "all"],
        help="NER model to execute ('gliner', 'xlm_roberta', or 'all'). Default: 'all'."
    )
    parser.add_argument(
        "--rec_dir",
        type=str,
        default="recognition_output",
        help="Input directory containing recognition JSON files."
    )
    parser.add_argument(
        "--gliner_out_dir",
        type=str,
        default="gliner_multi_v2_1_output",
        help="Output directory for GLiNER JSONs."
    )
    parser.add_argument(
        "--xlm_out_dir",
        type=str,
        default="xlm_roberta_ner_output",
        help="Output directory for XLM-RoBERTa JSONs."
    )
    parser.add_argument(
        "--gliner_model",
        type=str,
        default="urchade/gliner_multi-v2.1",
        help="GLiNER model name/checkpoint."
    )
    parser.add_argument(
        "--xlm_model",
        type=str,
        default="Davlan/xlm-roberta-base-ner-hrl",
        help="XLM-RoBERTa model name/checkpoint."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.30,
        help="Minimum confidence threshold for extracted entities."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Inference device ('cpu' or 'cuda')."
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("  STARTING NER ENRICHMENT PROCESSOR")
    print("=" * 80)
    print(f"Input Directory  : {args.rec_dir}")
    print(f"Target Model(s)  : {args.model.upper()}")
    print(f"Confidence Thresh: {args.threshold}")
    print(f"Device           : {args.device.upper()}")
    print("=" * 80 + "\n")

    gliner_stats = None
    xlm_stats = None

    if args.model in ["gliner", "all"]:
        print("[1/2] Running GLiNER Entity Recognition...")
        try:
            gliner_stats = process_gliner(
                rec_dir=args.rec_dir,
                output_dir=args.gliner_out_dir,
                model_name=args.gliner_model,
                threshold=args.threshold,
                device=args.device
            )
        except Exception as e:
            print(f"Error running GLiNER: {e}\n")

    if args.model in ["xlm_roberta", "all"]:
        print("[2/2] Running XLM-RoBERTa Entity Recognition...")
        try:
            xlm_stats = process_xlm_roberta(
                rec_dir=args.rec_dir,
                output_dir=args.xlm_out_dir,
                model_name=args.xlm_model,
                threshold=args.threshold,
                device=args.device
            )
        except Exception as e:
            print(f"Error running XLM-RoBERTa: {e}\n")

    print("=" * 80)
    print("  OVERALL NER INFERENCE LATENCY COMPARISON")
    print("=" * 80)
    print(f"{'Model Name':<35} | {'Load Time':<10} | {'Avg s/Image':<12} | {'Avg ms/Crop':<12}")
    print("-" * 80)

    if gliner_stats:
        print(f"{gliner_stats['model_name']:<35} | {gliner_stats['load_time_sec']:>8.3f}s | {gliner_stats['avg_time_per_image_sec']:>10.3f}s | {gliner_stats['avg_time_per_crop_ms']:>10.2f}ms")

    if xlm_stats:
        print(f"{xlm_stats['model_name']:<35} | {xlm_stats['load_time_sec']:>8.3f}s | {xlm_stats['avg_time_per_image_sec']:>10.3f}s | {xlm_stats['avg_time_per_crop_ms']:>10.2f}ms")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

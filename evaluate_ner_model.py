"""
Interactive Evaluation & Benchmark Suite for Fine-Tuned XLM-RoBERTa NER
=======================================================================
Optimized for Lightning AI Studio, Google Colab, Jupyter Notebooks, and Local Terminals.

Features:
1. Auto-detects merged standalone model OR LoRA adapter checkpoint.
2. GPU-Accelerated batched evaluation with clean tqdm progress bar.
3. Entity-level precision, recall, and F1 calculation per category.
4. Built-in multi-entity test suite covering Apps, Songs, Artists, Locations, Dates.
5. Interactive prompt mode to test any custom sentence directly in browser.
6. Optional JSON report saving (--save_report evaluation_results.json).
"""

import argparse
import json
import os
import sys
import torch
from collections import defaultdict
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(SCRIPT_DIR, "fine_tuned_xlm_roberta_ner", "merged_standalone_model")
DEFAULT_VAL_FILE = os.path.join(SCRIPT_DIR, "ner_combined_data", "val.json")

# ── 1. Default Label Schema ───────────────────────────────────────────────────
LABEL2ID = {
    'O': 0,
    'B-DATE': 1, 'I-DATE': 2,
    'B-PER': 3,  'I-PER': 4,
    'B-ORG': 5,  'I-ORG': 6,
    'B-LOC': 7,  'I-LOC': 8,
    'B-SONG': 9, 'I-SONG': 10,
    'B-APP': 11, 'I-APP': 12
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# ── 2. Entity Extraction & Matching ───────────────────────────────────────────
def extract_entities(tags):
    """
    Extracts named entities from a list of BIO tags.
    Returns: set of (label_type, start_token_idx, end_token_idx)
    """
    entities = set()
    current_entity = None
    for i, tag in enumerate(tags):
        if tag == "O" or tag == "-100":
            if current_entity:
                entities.add(current_entity)
                current_entity = None
        elif tag.startswith("B-"):
            if current_entity:
                entities.add(current_entity)
            ent_type = tag[2:]
            current_entity = (ent_type, i, i)
        elif tag.startswith("I-"):
            ent_type = tag[2:]
            if current_entity and current_entity[0] == ent_type:
                current_entity = (ent_type, current_entity[1], i)
            else:
                if current_entity:
                    entities.add(current_entity)
                current_entity = (ent_type, i, i)
    if current_entity:
        entities.add(current_entity)
    return entities


def evaluate_val_set(ner_pipe, val_json_path, max_samples=None, batch_size=32):
    print(f"\n[Validation] Loading dataset from: {val_json_path}")
    if not os.path.exists(val_json_path):
        print(f"Warning: Validation file '{val_json_path}' not found. Skipping validation set evaluation.")
        return None

    with open(val_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if max_samples and max_samples < len(data):
        data = data[:max_samples]
        print(f"[Validation] Evaluating on {len(data):,} sampled examples...")
    else:
        print(f"[Validation] Evaluating on all {len(data):,} examples...")

    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    total_tp, total_fp, total_fn = 0, 0, 0

    texts = [" ".join(item["tokens"]) for item in data]
    gold_tags_list = [[ID2LABEL.get(t, "O") for t in item["ner_tags"]] for item in data]

    # Run batched inference using pipeline with progress bar
    print("[Validation] Running batched inference...")
    preds_list = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating Batches"):
        batch_texts = texts[i:i + batch_size]
        batch_preds = ner_pipe(batch_texts)
        if len(batch_texts) == 1 and isinstance(batch_preds, list) and len(batch_preds) > 0 and isinstance(batch_preds[0], dict):
            preds_list.append(batch_preds)
        else:
            preds_list.extend(batch_preds)

    for item, text, gold_tags, preds in zip(data, texts, gold_tags_list, preds_list):
        gold_ents = extract_entities(gold_tags)
        
        # Align predictions back to token indices
        pred_tags = ["O"] * len(item["tokens"])
        char_idx = 0
        token_spans = []
        for tok in item["tokens"]:
            start = text.find(tok, char_idx)
            end = start + len(tok)
            token_spans.append((start, end))
            char_idx = end

        for p in preds:
            group = p["entity_group"]
            p_start, p_end = p["start"], p["end"]
            is_first = True
            for i, (t_start, t_end) in enumerate(token_spans):
                if max(p_start, t_start) < min(p_end, t_end):
                    tag_prefix = "B-" if is_first else "I-"
                    pred_tags[i] = f"{tag_prefix}{group}"
                    is_first = False

        pred_ents = extract_entities(pred_tags)

        for ent in pred_ents:
            if ent in gold_ents:
                stats[ent[0]]["tp"] += 1
                total_tp += 1
            else:
                stats[ent[0]]["fp"] += 1
                total_fp += 1

        for ent in gold_ents:
            if ent not in pred_ents:
                stats[ent[0]]["fn"] += 1
                total_fn += 1

    print("\n" + "=" * 72)
    print(f"{'Entity Category':<16} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("=" * 72)
    
    report = {}
    for ent_type, s in sorted(stats.items()):
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        supp = tp + fn
        report[ent_type] = {"precision": p, "recall": r, "f1": f1, "support": supp}
        print(f"{ent_type:<16} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f} | {supp:<8,}")

    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0.0
    report["overall"] = {"precision": overall_p, "recall": overall_r, "f1": overall_f1, "support": total_tp + total_fn}
    
    print("-" * 72)
    print(f"{'OVERALL (Micro)':<16} | {overall_p:<10.4f} | {overall_r:<10.4f} | {overall_f1:<10.4f} | {total_tp + total_fn:<8,}")
    print("=" * 72)
    return report


def run_benchmark_demo(ner_pipe):
    """Runs a suite of multi-entity sentences to demonstrate both new and preserved labels."""
    test_cases = [
        ("Now playing Blinding Lights by The Weeknd on Spotify in Paris on Friday",
         ["SONG: Blinding Lights", "PER: The Weeknd", "APP: Spotify", "LOC: Paris", "DATE: Friday"]),
        ("Open WhatsApp to send a voice note to Alice in Berlin",
         ["APP: WhatsApp", "PER: Alice", "LOC: Berlin"]),
        ("Stream Bohemian Rhapsody by Queen using Apple Music",
         ["SONG: Bohemian Rhapsody", "PER: Queen", "APP: Apple Music"]),
        ("Google Chrome updated its security settings in London on 2026-08-19",
         ["APP: Google Chrome", "LOC: London", "DATE: 2026-08-19"]),
        ("Listening to Shape of You by Ed Sheeran on YouTube",
         ["SONG: Shape of You", "PER: Ed Sheeran", "APP: YouTube"]),
        ("Switch to Telegram and message Bob",
         ["APP: Telegram", "PER: Bob"]),
        ("Check out Stay by Justin Bieber on Shazam",
         ["SONG: Stay", "PER: Justin Bieber", "APP: Shazam"])
    ]

    print("\n" + "=" * 72)
    print("  MULTI-ENTITY BENCHMARK DEMO (Apps, Songs, Artists, Locations, Dates)")
    print("=" * 72)

    for text, expected in test_cases:
        print(f"\nText: \"{text}\"")
        entities = ner_pipe(text)
        if entities:
            detected = []
            for ent in entities:
                detected.append(f"{ent['entity_group']}: '{ent['word']}' ({ent['score']:.2f})")
            print("  Detected -> " + " | ".join(detected))
        else:
            print("  Detected -> (None)")
    print("=" * 72)


def interactive_prompt(ner_pipe):
    """Interactive loop to test arbitrary sentences."""
    print("\n" + "=" * 72)
    print("  INTERACTIVE NER TESTING (Type 'quit' or 'exit' to stop)")
    print("=" * 72)
    try:
        while True:
            text = input("\nEnter sentence > ").strip()
            if not text or text.lower() in ("quit", "exit", "q"):
                break
            entities = ner_pipe(text)
            if entities:
                print(f"Entities found ({len(entities)}):")
                for e in entities:
                    print(f"  * {e['entity_group']:<8s} : '{e['word']}' (confidence: {e['score']:.2%})")
            else:
                print("  (No named entities detected)")
    except (EOFError, KeyboardInterrupt):
        print("\nExiting interactive mode.")


def load_ner_pipeline(model_path):
    """Loads either standalone merged model or LoRA checkpoint."""
    # 1. Try merged standalone model
    if os.path.exists(os.path.join(model_path, "model.safetensors")) or os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
        target_dir = model_path
    elif os.path.exists(os.path.join(model_path, "merged_standalone_model")):
        target_dir = os.path.join(model_path, "merged_standalone_model")
    else:
        target_dir = model_path

    print(f"Loading NER model from: {target_dir}")
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(target_dir)
    model = AutoModelForTokenClassification.from_pretrained(target_dir)

    ner_pipe = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=device
    )
    return ner_pipe


def main():
    parser = argparse.ArgumentParser(description="Evaluate Fine-Tuned XLM-RoBERTa Model (Lightning AI / Local)")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_DIR, help="Path to fine-tuned model directory")
    parser.add_argument("--val_file", type=str, default=DEFAULT_VAL_FILE, help="Path to validation JSON file")
    parser.add_argument("--samples", type=int, default=None, help="Number of validation samples to evaluate (default: all)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for pipeline evaluation")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive prompt for custom sentence testing")
    parser.add_argument("--save_report", type=str, default=None, help="Optional path to save evaluation metrics as JSON")
    args = parser.parse_args()

    # Load model
    ner_pipe = load_ner_pipeline(args.model_path)

    # 1. Run multi-entity benchmark demo
    run_benchmark_demo(ner_pipe)

    # 2. Run validation set evaluation
    report = None
    if os.path.exists(args.val_file):
        report = evaluate_val_set(ner_pipe, args.val_file, max_samples=args.samples, batch_size=args.batch_size)

    if args.save_report and report:
        with open(args.save_report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nEvaluation metrics saved to {args.save_report}")

    # 3. Launch interactive prompt if requested
    if args.interactive:
        interactive_prompt(ner_pipe)


if __name__ == "__main__":
    main()

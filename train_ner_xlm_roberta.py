"""
Fine-Tune Davlan/xlm-roberta-base-ner-hrl with LoRA and Anti-Forgetting Mechanisms
================================================================================
Optimized for Lightning AI Studio, Google Colab, Linux Servers, and Local Machines.

Features:
1. Warm-Started Classifier Head: Preserves pre-trained weights for PER, ORG, LOC, DATE, O.
2. LoRA (Low-Rank Adaptation): Keeps the 12-layer multilingual transformer backbone frozen.
3. Sub-Token Alignment: Standard -100 masking on subword continuations.
4. Mixed Precision (FP16 / BF16): ~2.5x faster training & 50% lower VRAM on GPUs.
5. Gradient Accumulation: Allows large effective batch sizes on any GPU tier.
6. Lightning AI & Colab Ready: Dynamic path resolution & multi-worker DataLoader.
7. Clean tqdm Progress Bar & Entity-Level Evaluation (Precision, Recall, F1).
8. Standalone Model Export: Merges LoRA into base weights for standard pipeline('ner').
"""

import argparse
import json
import os
import math
import random
import time
import sys
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType

# ── 1. Default Paths (Relative & Cross-Platform) ────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, "ner_combined_data")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "fine_tuned_xlm_roberta_ner")

# ── 2. Label Definitions ────────────────────────────────────────────────────────
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
NUM_LABELS = len(LABEL2ID)

# ── 3. Dataset & Collation ──────────────────────────────────────────────────────
class NERJsonDataset(Dataset):
    """PyTorch Dataset for JSON BIO token classification data."""
    def __init__(self, json_path, max_samples=None):
        print(f"Loading {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if max_samples and max_samples < len(data):
            random.seed(42)
            data = random.sample(data, max_samples)
            print(f"Subsampled {len(data)} examples for fast training/testing.")
        self.data = data
        print(f"Loaded {len(self.data):,} examples from {os.path.basename(json_path)}.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "tokens": self.data[idx]["tokens"],
            "ner_tags": self.data[idx]["ner_tags"]
        }


def tokenize_and_align_labels(examples, tokenizer, max_length=128):
    """
    Tokenizes word lists into sub-tokens and aligns BIO labels:
    - First sub-token of a word receives the word's label ID.
    - Subsequent sub-tokens receive -100 (ignored in cross-entropy loss).
    - Special tokens (<s>, </s>, <pad>) receive -100.
    """
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt"
    )

    batch_labels = []
    for i, word_tags in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != prev_word_idx:
                label_ids.append(word_tags[word_idx])
            else:
                label_ids.append(-100)
            prev_word_idx = word_idx
        batch_labels.append(label_ids)

    tokenized["labels"] = torch.tensor(batch_labels, dtype=torch.long)
    return tokenized


class DataCollatorWithAlignment:
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        tokens_batch = [item["tokens"] for item in batch]
        ner_tags_batch = [item["ner_tags"] for item in batch]
        return tokenize_and_align_labels(
            {"tokens": tokens_batch, "ner_tags": ner_tags_batch},
            self.tokenizer,
            self.max_length
        )


# ── 4. Chunk-Level NER Evaluation (Pure Python seqeval equivalent) ──────────────
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


def compute_metrics(true_labels, pred_labels):
    """
    Computes Precision, Recall, F1 for each entity category.
    """
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    total_tp, total_fp, total_fn = 0, 0, 0

    for gold, pred in zip(true_labels, pred_labels):
        gold_ents = extract_entities(gold)
        pred_ents = extract_entities(pred)

        for ent in pred_ents:
            ent_type = ent[0]
            if ent in gold_ents:
                stats[ent_type]["tp"] += 1
                total_tp += 1
            else:
                stats[ent_type]["fp"] += 1
                total_fp += 1

        for ent in gold_ents:
            ent_type = ent[0]
            if ent not in pred_ents:
                stats[ent_type]["fn"] += 1
                total_fn += 1

    metrics = {}
    for ent_type, s in stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        metrics[ent_type] = {"precision": p, "recall": r, "f1": f1, "support": tp + fn}

    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0.0

    metrics["overall"] = {
        "precision": overall_p,
        "recall": overall_r,
        "f1": overall_f1,
        "support": total_tp + total_fn
    }
    return metrics


# ── 5. Model Warm-Starting & LoRA Construction ──────────────────────────────────
def build_warm_started_model(base_model_name="Davlan/xlm-roberta-base-ner-hrl", lora_r=8, lora_alpha=16):
    """
    Loads pre-trained XLM-RoBERTa Token Classification model, expands its classifier head
    from 9 to 13 classes by copying pre-trained weights for indices 0..8, and attaches LoRA adapters.
    """
    print(f"\n[1/3] Loading base pre-trained model: {base_model_name}...")
    base_model = AutoModelForTokenClassification.from_pretrained(base_model_name)
    old_classifier = base_model.classifier
    old_num_labels = old_classifier.out_features  # 9

    print(f"[2/3] Warm-Starting Classifier: Expanding from {old_num_labels} to {NUM_LABELS} classes...")
    new_classifier = torch.nn.Linear(old_classifier.in_features, NUM_LABELS)

    # Copy pre-trained weights for 0..8 (Warm-Start)
    with torch.no_grad():
        new_classifier.weight[:old_num_labels] = old_classifier.weight.clone()
        new_classifier.bias[:old_num_labels] = old_classifier.bias.clone()
        # Initialize new labels (9..12: B-SONG, I-SONG, B-APP, I-APP) with small gaussian noise
        torch.nn.init.normal_(new_classifier.weight[old_num_labels:], mean=0.0, std=0.02)
        torch.nn.init.zeros_(new_classifier.bias[old_num_labels:])

    base_model.classifier = new_classifier
    base_model.num_labels = NUM_LABELS
    base_model.config.num_labels = NUM_LABELS
    base_model.config.id2label = {str(k): v for k, v in ID2LABEL.items()}
    base_model.config.label2id = LABEL2ID

    print("[3/3] Attaching LoRA adapters to transformer attention layers...")
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=["query", "value"],
        modules_to_save=["classifier"]  # Ensures the extended classification head is trainable & saved
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    return model


# ── 6. Evaluation Loop ──────────────────────────────────────────────────────────
def evaluate(model, dataloader, device, use_amp=False, amp_dtype=torch.float16):
    model.eval()
    all_true = []
    all_pred = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if use_amp and device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            total_loss += outputs.loss.item()

            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            golds = labels.cpu().numpy()

            for i in range(len(golds)):
                gold_seq = []
                pred_seq = []
                for g_id, p_id in zip(golds[i], preds[i]):
                    if g_id != -100:
                        gold_seq.append(ID2LABEL.get(g_id, "O"))
                        pred_seq.append(ID2LABEL.get(p_id, "O"))
                all_true.append(gold_seq)
                all_pred.append(pred_seq)

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    metrics = compute_metrics(all_true, all_pred)
    return avg_loss, metrics


# ── 7. Main Training Routine ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fine-tune XLM-RoBERTa for Custom NER (Lightning AI / GPU / CPU)")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Path to ner_combined_data directory")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Path to save fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3, help="Total training epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU (default: 16)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for LoRA and Head (default: 2e-4)")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank dimension (default: 8)")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha scaling factor (default: 16)")
    parser.add_argument("--fp16", action="store_true", default=torch.cuda.is_available(), help="Enable FP16 Automatic Mixed Precision on GPU")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable BF16 Mixed Precision on Ampere/Hopper GPUs")
    parser.add_argument("--num_workers", type=int, default=2 if torch.cuda.is_available() else 0, help="DataLoader worker processes")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Capped train samples for quick validation runs")
    parser.add_argument("--max_val_samples", type=int, default=2000, help="Capped val samples for fast epoch validation")
    parser.add_argument("--save_merged", action="store_true", default=True, help="Merge LoRA weights into standalone base model")
    args = parser.parse_args()

    # Hardware Detection & Summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("  XLM-RoBERTa Custom NER Fine-Tuner (Lightning AI & Local)")
    print("=" * 70)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")
        print(f"Mixed Precision: {'BF16' if args.bf16 else ('FP16' if args.fp16 else 'FP32')}")
    else:
        print("Running on CPU (Training on sample subset or GPU recommended for full 180k dataset)")
    print("=" * 70)

    use_amp = (args.fp16 or args.bf16) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    scaler = torch.amp.GradScaler('cuda') if use_amp and not args.bf16 else None

    # Load Tokenizer
    print("\nLoading tokenizer from Davlan/xlm-roberta-base-ner-hrl...")
    tokenizer = AutoTokenizer.from_pretrained("Davlan/xlm-roberta-base-ner-hrl")

    # Load Datasets
    train_path = os.path.join(args.data_dir, "train.json")
    val_path = os.path.join(args.data_dir, "val.json")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print(f"ERROR: Dataset not found in {args.data_dir}. Please run generate_ner_combined_data.py first.")
        sys.exit(1)

    train_dataset = NERJsonDataset(train_path, max_samples=args.max_train_samples)
    val_dataset = NERJsonDataset(val_path, max_samples=args.max_val_samples)

    collator = DataCollatorWithAlignment(tokenizer, max_length=128)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # Initialize model with warm-started head and LoRA
    model = build_warm_started_model(
        base_model_name="Davlan/xlm-roberta-base-ner-hrl",
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
    model.to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    print(f"\nStarting training: {args.epochs} epochs | {len(train_loader)} batches/epoch | {total_steps} total optimizer steps...")
    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=True)
        for step, batch in enumerate(progress_bar, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss / args.gradient_accumulation_steps
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / args.gradient_accumulation_steps
                loss.backward()

            running_loss += loss.item() * args.gradient_accumulation_steps

            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                optimizer.zero_grad()
                scheduler.step()

            # Update progress bar
            current_lr = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else args.learning_rate
            progress_bar.set_postfix({
                "loss": f"{running_loss / step:.4f}",
                "lr": f"{current_lr:.2e}"
            })

        # Evaluation at end of epoch
        print(f"\nEvaluating Epoch {epoch} on Validation Set...")
        val_loss, metrics = evaluate(model, val_loader, device, use_amp=use_amp, amp_dtype=amp_dtype)
        overall = metrics.get("overall", {})
        print(f"Validation Loss: {val_loss:.4f} | Overall Precision: {overall.get('precision', 0):.4f} | Recall: {overall.get('recall', 0):.4f} | F1-Score: {overall.get('f1', 0):.4f}")
        
        print("\n--- Per-Entity Breakdown ---")
        for ent_name, m in sorted(metrics.items()):
            if ent_name != "overall":
                print(f"  {ent_name:<8s} -> Precision: {m['precision']:.4f} | Recall: {m['recall']:.4f} | F1: {m['f1']:.4f} (Support: {m['support']})")

        current_f1 = overall.get("f1", 0.0)
        if current_f1 >= best_f1:
            best_f1 = current_f1
            print(f"\n>>> Best model updated (F1: {best_f1:.4f}). Saving checkpoint to {args.output_dir}...")
            os.makedirs(args.output_dir, exist_ok=True)
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

    # Merge LoRA model into standalone base model if requested
    if args.save_merged:
        merged_output_dir = os.path.join(args.output_dir, "merged_standalone_model")
        print(f"\nMerging LoRA weights into base model and saving standalone model to {merged_output_dir}...")
        os.makedirs(merged_output_dir, exist_ok=True)
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_output_dir)
        tokenizer.save_pretrained(merged_output_dir)
        
        # Save explicit label mapping for pipeline convenience
        with open(os.path.join(merged_output_dir, "label_map.json"), "w", encoding="utf-8") as f:
            json.dump({"label2id": LABEL2ID, "id2label": {str(k): v for k, v in ID2LABEL.items()}}, f, indent=2)
            
        print("Standalone model saved successfully! Ready for pipeline('ner') and deployment.")

    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()

"""
Run Fine-Tuned XLM-RoBERTa (with SONG & APP) on OCR Recognition Output Files
=============================================================================
This script:
1. Loads the fine-tuned XLM-RoBERTa model (merged standalone or LoRA adapter).
2. Reads all '*_final_recog.json' files from 'recognition_output/'.
3. For each transcribed text, runs token classification and calculates:
   - 'entity_group' (APP, SONG, PER, LOC, ORG, DATE)
   - 'score' (confidence)
   - 'word' (matched text)
   - 'start' & 'end' offsets
   - 'label_confidences' (full softmax distribution across all entity classes)
   - 'entity' (primary highest-confidence entity for the crop block)
4. Saves enriched JSON files to 'finetuned_ner_output/' with inference stats.
"""

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(SCRIPT_DIR, "fine_tuned_xlm_roberta_ner", "merged_standalone_model")
DEFAULT_INPUT_DIR = os.path.join(SCRIPT_DIR, "recognition_output")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "finetuned_ner_output")


class FinetunedNERProcessor:
    """Enriches OCR text blocks with fine-tuned XLM-RoBERTa Named Entities & label confidences."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_DIR,
        device: Optional[str] = None,
        threshold: float = 0.50
    ):
        self.threshold = threshold
        self.model_path = model_path

        # Resolve model directory fallback
        if not os.path.exists(os.path.join(model_path, "config.json")):
            fallback_dir = os.path.join(SCRIPT_DIR, "fine_tuned_xlm_roberta_ner")
            if os.path.exists(os.path.join(fallback_dir, "config.json")):
                self.model_path = fallback_dir
            elif os.path.exists(os.path.join(fallback_dir, "merged_standalone_model", "config.json")):
                self.model_path = os.path.join(fallback_dir, "merged_standalone_model")

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading Fine-Tuned XLM-RoBERTa from: {self.model_path}")
        print(f"Inference Device: {self.device}")

        start_t = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        self.model.eval()

        device_id = 0 if self.device == "cuda" else -1
        if self.device == "cuda":
            self.model = self.model.to("cuda")

        self.ner_pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=device_id
        )

        # Unique entity classes (without B- / I- prefixes)
        self.unique_labels = sorted(list(set([
            name.replace("B-", "").replace("I-", "")
            for name in self.model.config.id2label.values()
        ])))
        print(f"Model loaded in {time.perf_counter() - start_t:.2f}s. Supported Categories: {self.unique_labels}")

    def predict_text(self, text: str, threshold: Optional[float] = None) -> Tuple[List[Dict[str, Any]], float]:
        """
        Runs token classification on input string and returns enriched entities
        with exact softmax label_confidences for every category.
        """
        if not text or not text.strip():
            return [], 0.0

        if threshold is None:
            threshold = self.threshold

        start_time = time.perf_counter()
        try:
            # 1. Candidate extraction via Hugging Face aggregation pipeline
            raw_entities = self.ner_pipeline(text)
            elapsed_time = time.perf_counter() - start_time

            if not raw_entities:
                return [], elapsed_time

            # 2. Forward pass for full contextual softmax distribution
            inputs = self.tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
            offsets = inputs.pop("offset_mapping")[0].tolist()

            if self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits[0], dim=-1).cpu()

            # 3. Merge adjacent sub-tokens belonging to the same entity
            merged_entities = []
            current_group = None

            for ent in raw_entities:
                group = ent.get("entity_group")
                score = float(ent.get("score", 0.0))
                word = ent.get("word", "")
                start = int(ent.get("start", 0))
                end = int(ent.get("end", 0))

                if current_group is None:
                    current_group = {
                        "entity_group": group,
                        "scores": [score],
                        "words": [word],
                        "start": start,
                        "end": end
                    }
                else:
                    if group == current_group["entity_group"] and start <= current_group["end"] + 1:
                        current_group["scores"].append(score)
                        current_group["words"].append(word)
                        current_group["end"] = end
                    else:
                        merged_entities.append(current_group)
                        current_group = {
                            "entity_group": group,
                            "scores": [score],
                            "words": [word],
                            "start": start,
                            "end": end
                        }

            if current_group:
                merged_entities.append(current_group)

            # 4. Construct output entities with full label_confidences map
            formatted_entities = []
            for item in merged_entities:
                ent_start = item["start"]
                ent_end = item["end"]
                sub_text = text[ent_start:ent_end] if 0 <= ent_start < ent_end <= len(text) else "".join(item["words"])

                # Find token span offsets
                token_indices = [
                    idx for idx, (ostart, oend) in enumerate(offsets)
                    if ostart < ent_end and oend > ent_start
                ]

                if token_indices:
                    entity_probs = probs[token_indices].mean(dim=0)
                else:
                    entity_probs = probs.mean(dim=0)

                label_confidences = {cat: 0.0 for cat in self.unique_labels}
                for id_idx, label_name in self.model.config.id2label.items():
                    clean_label = label_name.replace("B-", "").replace("I-", "")
                    score_val = round(float(entity_probs[id_idx]), 4)
                    if score_val > label_confidences[clean_label]:
                        label_confidences[clean_label] = score_val

                if label_confidences:
                    highest_label = max(label_confidences, key=label_confidences.get)
                    highest_score = label_confidences[highest_label]

                    # Skip if predicted as background noise / non-entity ("O")
                    if highest_label == "O":
                        continue

                    if highest_score >= threshold:
                        formatted_entities.append({
                            "entity_group": highest_label,
                            "score": round(highest_score, 4),
                            "word": sub_text,
                            "start": ent_start,
                            "end": ent_end,
                            "label_confidences": label_confidences
                        })

            return formatted_entities, elapsed_time

        except Exception as e:
            elapsed_time = time.perf_counter() - start_time
            print(f"Warning: NER inference failed on text '{text}': {e}")
            return [], elapsed_time

    def process_file(self, json_path: str, output_dir: str, threshold: Optional[float] = None) -> Dict[str, Any]:
        """Reads a single recognition JSON file, enriches crops with NER, and writes out result."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = data.get("results", []) or data.get("all_detections_with_status", [])
        image_stem = data.get("image_stem", os.path.basename(json_path).replace("_final_recog.json", "").replace("_rec_full.json", "").replace(".json", ""))

        file_start_t = time.perf_counter()
        enriched_results = []
        file_inference_times = []

        for item in results:
            item_copy = dict(item)
            text = item_copy.get("transcribed_text", "")

            entities, elapsed_time = self.predict_text(text, threshold=threshold)
            if text.strip():
                file_inference_times.append(elapsed_time)

            item_copy["entities"] = entities
            # Pick primary entity (highest score) or None
            if entities:
                best_entity = max(entities, key=lambda e: e["score"])
                item_copy["entity"] = best_entity["entity_group"]
            else:
                item_copy["entity"] = None

            item_copy["inference_time_ms"] = round(elapsed_time * 1000.0, 2)
            enriched_results.append(item_copy)

        file_elapsed_sec = time.perf_counter() - file_start_t
        total_text_time = sum(file_inference_times)
        avg_crop_ms = (total_text_time * 1000.0 / len(file_inference_times)) if file_inference_times else 0.0

        os.makedirs(output_dir, exist_ok=True)
        out_filename = f"{image_stem}_finetuned_ner.json"
        out_path = os.path.join(output_dir, out_filename)

        total_entities_count = sum(len(x.get("entities", [])) for x in enriched_results)

        output_dict = {
            "image_stem": image_stem,
            "source_image": data.get("source_image", ""),
            "ner_model": os.path.basename(self.model_path),
            "total_ocr_items": len(enriched_results),
            "total_entities_found": total_entities_count,
            "inference_stats": {
                "file_inference_time_sec": round(file_elapsed_sec, 4),
                "total_text_inference_time_sec": round(total_text_time, 4),
                "avg_crop_inference_time_ms": round(avg_crop_ms, 2)
            },
            "results": enriched_results
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_dict, f, ensure_ascii=False, indent=2)

        return {
            "image_stem": image_stem,
            "output_path": out_path,
            "total_items": len(enriched_results),
            "total_entities": total_entities_count,
            "inference_times": file_inference_times,
            "entities": [e for x in enriched_results for e in x.get("entities", [])]
        }


def process_directory(
    rec_dir: str = DEFAULT_INPUT_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    model_path: str = DEFAULT_MODEL_DIR,
    threshold: float = 0.50,
    file_pattern: str = "*_final_recog.json"
):
    """Processes all matching JSON files in the recognition directory."""
    search_pattern = os.path.join(rec_dir, file_pattern)
    files = glob.glob(search_pattern)

    if not files:
        # Fallback to all JSON files in the folder if pattern didn't match
        search_pattern = os.path.join(rec_dir, "*.json")
        files = glob.glob(search_pattern)

    if not files:
        print(f"No JSON files found in {rec_dir}!")
        return

    print("=" * 72)
    print(f"  Fine-Tuned NER Pipeline Processing: {len(files)} file(s)")
    print(f"  Input Directory : {rec_dir}")
    print(f"  Output Directory: {output_dir}")
    print("=" * 72)

    processor = FinetunedNERProcessor(model_path=model_path, threshold=threshold)

    entity_distribution = defaultdict(int)
    total_crops_count = 0
    total_entities_count = 0
    all_inference_times = []

    progress_bar = tqdm(files, desc="Processing Files")
    for fpath in progress_bar:
        res = processor.process_file(fpath, output_dir, threshold=threshold)
        total_crops_count += res["total_items"]
        total_entities_count += res["total_entities"]
        all_inference_times.extend(res["inference_times"])
        
        for ent in res["entities"]:
            entity_distribution[ent["entity_group"]] += 1
            
        progress_bar.set_postfix({
            "entities": total_entities_count,
            "file": res["image_stem"]
        })

    avg_latency_ms = (sum(all_inference_times) * 1000.0 / len(all_inference_times)) if all_inference_times else 0.0

    print("\n" + "=" * 72)
    print("  FINAL NER PROCESSING SUMMARY")
    print("=" * 72)
    print(f"Total Files Processed   : {len(files)}")
    print(f"Total OCR Text Crops    : {total_crops_count}")
    print(f"Total Named Entities    : {total_entities_count}")
    print(f"Average Inference/Crop  : {avg_latency_ms:.2f} ms")
    print("\nEntity Counts Breakdown:")
    for ent_type, cnt in sorted(entity_distribution.items(), key=lambda x: -x[1]):
        print(f"  * {ent_type:<8s} : {cnt:>5,} detected")
    print("=" * 72)
    print(f"Results saved to: {output_dir}\n")


def main():
    parser = argparse.ArgumentParser(description="Run Fine-Tuned XLM-RoBERTa on Recognition Output Files")
    parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR, help="Path to recognition_output folder")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Path to save enriched NER outputs")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_DIR, help="Path to fine-tuned model")
    parser.add_argument("--threshold", type=float, default=0.50, help="Confidence threshold for entities (default: 0.50)")
    parser.add_argument("--pattern", type=str, default="*_final_recog.json", help="File matching pattern (default: *_final_recog.json)")
    args = parser.parse_args()

    process_directory(
        rec_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        threshold=args.threshold,
        file_pattern=args.pattern
    )


if __name__ == "__main__":
    main()

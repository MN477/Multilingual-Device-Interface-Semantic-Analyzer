"""
ner_xlm_roberta_processor.py
============================
NER Enrichment processor using XLM-RoBERTa (Davlan/xlm-roberta-base-ner-hrl).
Reads recognition JSON files from 'recognition_output/' (or specified directory),
runs HuggingFace token classification pipeline with 'aggregation_strategy="simple"',
tracks inference latency using time.perf_counter(), and saves enriched JSON files into 'xlm_roberta_ner_output/'.
"""

import os
import sys
import json
import glob
import time
import argparse
from typing import List, Dict, Any, Optional

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class XLMRobertaEntityProcessor:
    def __init__(
        self,
        model_name: str = "Davlan/xlm-roberta-base-ner-hrl",
        threshold: float = 0.60,
        device: str = "cpu"
    ):
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "Hugging Face transformers is not installed. Please install it using:\n"
                "  pip install transformers torch"
            )

        self.model_name = model_name
        self.threshold = threshold
        self.device = device
        
        print(f"Loading tokenizer and model: {model_name}...")
        start_t = time.perf_counter()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        
        if device == "cuda":
            self.model = self.model.to("cuda")
            device_id = 0
        else:
            device_id = -1

        self.ner_pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=device_id
        )
        print(f"XLM-RoBERTa model loaded successfully in {time.perf_counter() - start_t:.2f}s.")

    def get_all_label_confidences(self, text: str) -> Dict[str, float]:
        """
        Computes pure model softmax probabilities across all entity categories
        (e.g., LOC, ORG, PER, DATE, O) directly from the model logits.
        """
        if not text or not text.strip():
            return {}
            
        try:
            inputs = self.tokenizer(text, return_tensors="pt")
            if self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Softmax over logits to get exact probabilities
                probs = torch.softmax(outputs.logits[0], dim=-1)
                
                # Exclude special tokens ([CLS], [SEP]) if present
                valid_probs = probs[1:-1] if len(probs) > 2 else probs
                mean_probs = valid_probs.mean(dim=0)
                
            label_confidences = {}
            for id_idx, label_name in self.model.config.id2label.items():
                clean_label = label_name.replace("B-", "").replace("I-", "")
                score_val = round(float(mean_probs[id_idx]), 4)
                if clean_label not in label_confidences or score_val > label_confidences[clean_label]:
                    label_confidences[clean_label] = score_val
                    
            return label_confidences
        except Exception as e:
            return {}

    def predict_text(self, text: str, threshold: Optional[float] = None) -> tuple[List[Dict[str, Any]], float]:
        if not text or not text.strip():
            return [], 0.0

        if threshold is None:
            threshold = self.threshold

        start_time = time.perf_counter()
        try:
            # 1. Run pipeline to get candidate entities
            raw_entities = self.ner_pipeline(text)
            elapsed_time = time.perf_counter() - start_time

            if not raw_entities:
                return [], elapsed_time

            # 2. Tokenize full text and run model forward pass to get contextual softmax probabilities
            inputs = self.tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
            offsets = inputs.pop("offset_mapping")[0].tolist()  # character start/end offsets for each token
            
            if self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits[0], dim=-1)

            # Strategy B: Merge adjacent subword tokens belonging to the same entity group
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

            formatted_entities = []
            for item in merged_entities:
                ent_start = item["start"]
                ent_end = item["end"]
                sub_text = text[ent_start:ent_end] if 0 <= ent_start < ent_end <= len(text) else "".join(item["words"])

                # Find token indices corresponding to this entity character span
                token_indices = [
                    idx for idx, (ostart, oend) in enumerate(offsets)
                    if ostart < ent_end and oend > ent_start
                ]

                # Extract pure contextual model probabilities for the matched token span
                if token_indices:
                    entity_probs = probs[token_indices].mean(dim=0)
                else:
                    entity_probs = probs.mean(dim=0)

                label_confidences = {}
                for id_idx, label_name in self.model.config.id2label.items():
                    clean_label = label_name.replace("B-", "").replace("I-", "")
                    score_val = round(float(entity_probs[id_idx]), 4)
                    if clean_label not in label_confidences or score_val > label_confidences[clean_label]:
                        label_confidences[clean_label] = score_val

                if label_confidences:
                    # Pick highest confidence label from the contextual probabilities
                    highest_label = max(label_confidences, key=label_confidences.get)
                    highest_score = label_confidences[highest_label]

                    # If the highest confidence label is "O" (no entity), discard/skip this entity
                    if highest_label == "O":
                        continue

                    # Overwrite entity group and score to align with the true model probabilities
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
            print(f"Warning: XLM-RoBERTa prediction failed for text '{text}': {e}")
            return [], elapsed_time

    def process_file(self, json_path: str, output_dir: str, threshold: Optional[float] = None) -> Dict[str, Any]:
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
            # Pick the entity with the highest score as the primary entity
            if entities:
                best_entity = max(entities, key=lambda e: e["score"])
                item_copy["entity"] = best_entity["entity_group"]
            else:
                item_copy["entity"] = None
            item_copy["inference_time_ms"] = round(elapsed_time * 1000.0, 2)

            enriched_results.append(item_copy)

        file_elapsed_sec = time.perf_counter() - file_start_t
        total_block_inference_time = sum(file_inference_times)
        avg_crop_ms = (total_block_inference_time * 1000.0 / len(file_inference_times)) if file_inference_times else 0.0

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{image_stem}_xlm_roberta_ner.json")

        total_entities_count = sum(len(x.get("entities", [])) for x in enriched_results)

        output_dict = {
            "image_stem": image_stem,
            "source_image": data.get("source_image", ""),
            "ner_model": self.model_name,
            "total_ocr_items": len(enriched_results),
            "total_entities_found": total_entities_count,
            "inference_stats": {
                "file_inference_time_sec": round(file_elapsed_sec, 4),
                "total_text_inference_time_sec": round(total_block_inference_time, 4),
                "avg_crop_inference_time_ms": round(avg_crop_ms, 2)
            },
            "results": enriched_results
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_dict, f, ensure_ascii=False, indent=2)

        print(f"[XLM-RoBERTa] '{image_stem}': {total_entities_count} entity(ies) found across {len(enriched_results)} items | Time: {total_block_inference_time:.4f}s ({avg_crop_ms:.2f}ms/block)")
        return {
            "image_stem": image_stem,
            "total_items": len(enriched_results),
            "total_entities": total_entities_count,
            "inference_times": file_inference_times,
            "file_elapsed_sec": file_elapsed_sec
        }


def process_directory(
    rec_dir: str = "recognition_output",
    output_dir: str = "xlm_roberta_ner_output",
    model_name: str = "Davlan/xlm-roberta-base-ner-hrl",
    threshold: float = 0.30,
    device: str = "cpu"
) -> Dict[str, Any]:
    if not os.path.exists(rec_dir):
        print(f"Error: input directory '{rec_dir}' does not exist.")
        sys.exit(1)

    json_files = sorted(glob.glob(os.path.join(rec_dir, "*_final_recog.json")))
    if not json_files:
        json_files = sorted(glob.glob(os.path.join(rec_dir, "*_rec_full.json")))
    if not json_files:
        json_files = sorted(glob.glob(os.path.join(rec_dir, "*.json")))

    if not json_files:
        print(f"No JSON files found in '{rec_dir}'.")
        sys.exit(1)

    load_start_t = time.perf_counter()
    processor = XLMRobertaEntityProcessor(
        model_name=model_name,
        threshold=threshold,
        device=device
    )
    load_time_sec = time.perf_counter() - load_start_t

    print(f"\nProcessing {len(json_files)} JSON file(s) with XLM-RoBERTa -> {output_dir}\n" + "=" * 75)
    
    total_images = len(json_files)
    total_crops = 0
    total_entities = 0
    all_inference_times = []
    total_file_elapsed_sec = 0.0

    for jf in json_files:
        res = processor.process_file(jf, output_dir, threshold=threshold)
        total_crops += res["total_items"]
        total_entities += res["total_entities"]
        all_inference_times.extend(res["inference_times"])
        total_file_elapsed_sec += res["file_elapsed_sec"]

    total_inference_time = sum(all_inference_times)
    avg_time_per_image = total_file_elapsed_sec / max(total_images, 1)
    avg_time_per_crop_ms = (total_inference_time * 1000.0) / max(len(all_inference_times), 1)

    stats = {
        "model_name": model_name,
        "total_images": total_images,
        "total_crops": total_crops,
        "total_entities": total_entities,
        "load_time_sec": round(load_time_sec, 3),
        "total_inference_time_sec": round(total_inference_time, 4),
        "avg_time_per_image_sec": round(avg_time_per_image, 3),
        "avg_time_per_crop_ms": round(avg_time_per_crop_ms, 2)
    }

    print("=" * 75)
    print("  XLM-RoBERTa INFERENCE LATENCY REPORT")
    print("=" * 75)
    print(f"  - Model Load Time         : {stats['load_time_sec']:.3f} s")
    print(f"  - Total Images Processed  : {stats['total_images']}")
    print(f"  - Total Text Blocks       : {stats['total_crops']}")
    print(f"  - Total Entities Found    : {stats['total_entities']}")
    print(f"  - Total Inference Time    : {stats['total_inference_time_sec']:.4f} s")
    print(f"  - AVG Latency per Image   : {stats['avg_time_per_image_sec']:.3f} s / image")
    print(f"  - AVG Latency per Block   : {stats['avg_time_per_crop_ms']:.2f} ms ({stats['avg_time_per_crop_ms'] / 1000.0:.6f} s)")
    print("=" * 75 + "\n")

    return stats


def main():
    parser = argparse.ArgumentParser(description="XLM-RoBERTa NER Processor for Recognition JSONs.")
    parser.add_argument("--rec_dir", type=str, default="recognition_output", help="Directory containing recognition JSON files.")
    parser.add_argument("--out_dir", type=str, default="xlm_roberta_ner_output", help="Output directory for XLM-RoBERTa NER JSONs.")
    parser.add_argument("--model", type=str, default="Davlan/xlm-roberta-base-ner-hrl", help="XLM-RoBERTa pretrained model name.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Inference device.")
    args = parser.parse_args()

    process_directory(
        rec_dir=args.rec_dir,
        output_dir=args.out_dir,
        model_name=args.model,
        device=args.device
    )


if __name__ == "__main__":
    main()

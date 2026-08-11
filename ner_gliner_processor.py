"""
ner_gliner_processor.py
=======================
NER Enrichment processor using GLiNER (urchade/gliner_multi-v2.1).
Reads recognition JSON files from 'recognition_output/' (or specified directory),
runs zero-shot NER for person, location, organization, action, ui_element, date_time labels,
and saves enriched JSON files into 'gliner_multi_v2_1_output/'.
"""

import os
import sys
import json
import glob
import time
import argparse
from typing import List, Dict, Any, Optional

try:
    from gliner import GLiNER
    HAS_GLINER = True
except ImportError:
    HAS_GLINER = False


DEFAULT_LABELS = [    "person name", "geographical location", "organization or company", "mobile application or software name (e.g., WhatsApp, Instagram, Spotify)", "user interface action (e.g., send, save, cancel)", "song title"]


class GLiNEREntityProcessor:
    def __init__(
        self,
        model_name: str = "urchade/gliner_multi-v2.1",
        labels: Optional[List[str]] = None,
        threshold: float = 0.30,
        device: str = "cpu"
    ):
        if not HAS_GLINER:
            raise ImportError(
                "GLiNER is not installed. Please install it using:\n"
                "  pip install gliner"
            )

        self.model_name = model_name
        self.labels = labels or DEFAULT_LABELS
        self.threshold = threshold
        self.device = device
        
        print(f"Loading GLiNER model '{model_name}' on {device.upper()}...")
        start_t = time.time()
        self.model = GLiNER.from_pretrained(model_name)
        if device == "cuda":
            self.model = self.model.to("cuda")
        print(f"GLiNER model loaded successfully in {time.time() - start_t:.2f}s.")
        print(f"Target Labels: {self.labels}")

    def predict_text(self, text: str) -> List[Dict[str, Any]]:
        if not text or not text.strip():
            return []
        
        try:
            entities = self.model.predict_entities(
                text,
                self.labels,
                threshold=self.threshold
            )
            parsed = []
            for ent in entities:
                parsed.append({
                    "text": ent.get("text", ""),
                    "label": ent.get("label", ""),
                    "score": round(float(ent.get("score", 0.0)), 4),
                    "start": ent.get("start", 0),
                    "end": ent.get("end", 0)
                })
            return parsed
        except Exception as e:
            print(f"Warning: GLiNER prediction failed for text '{text}': {e}")
            return []

    def process_file(self, json_path: str, output_dir: str) -> Dict[str, Any]:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = data.get("results", []) or data.get("all_detections_with_status", [])
        image_stem = data.get("image_stem", os.path.basename(json_path).replace("_final_recog.json", "").replace("_rec_full.json", "").replace(".json", ""))

        file_start_t = time.time()
        enriched_results = []
        total_items = len(results)
        
        for item in results:
            item_copy = dict(item)
            text = item_copy.get("transcribed_text", "")
            
            crop_start_t = time.time()
            entities = self.predict_text(text)
            crop_elapsed_ms = (time.time() - crop_start_t) * 1000.0
            
            item_copy["inference_time_ms"] = round(crop_elapsed_ms, 2)

            if entities:
                top_entity = max(entities, key=lambda x: x["score"])
                item_copy["entity_label"] = top_entity["label"]
                item_copy["entity_text"] = top_entity["text"]
                item_copy["entity_score"] = top_entity["score"]
                item_copy["all_entities"] = entities
            else:
                item_copy["entity_label"] = None
                item_copy["entity_text"] = None
                item_copy["entity_score"] = 0.0
                item_copy["all_entities"] = []

            enriched_results.append(item_copy)

        file_elapsed_sec = time.time() - file_start_t
        avg_crop_ms = (file_elapsed_sec * 1000.0 / total_items) if total_items > 0 else 0.0

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{image_stem}_gliner_ner.json")

        output_dict = {
            "image_stem": image_stem,
            "source_image": data.get("source_image", ""),
            "ner_model": self.model_name,
            "target_labels": self.labels,
            "threshold": self.threshold,
            "total_ocr_items": len(enriched_results),
            "total_entities_found": len([x for x in enriched_results if x["entity_label"] is not None]),
            "inference_stats": {
                "file_inference_time_sec": round(file_elapsed_sec, 4),
                "avg_crop_inference_time_ms": round(avg_crop_ms, 2)
            },
            "results": enriched_results
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_dict, f, ensure_ascii=False, indent=2)

        print(f"[GLiNER] '{image_stem}': {output_dict['total_entities_found']}/{len(enriched_results)} entities found | Time: {file_elapsed_sec:.3f}s ({avg_crop_ms:.1f}ms/crop)")
        return {
            "image_stem": image_stem,
            "total_items": total_items,
            "total_entities": output_dict['total_entities_found'],
            "elapsed_sec": file_elapsed_sec
        }


def process_directory(
    rec_dir: str = "recognition_output",
    output_dir: str = "gliner_multi_v2_1_output",
    model_name: str = "urchade/gliner_multi-v2.1",
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

    load_start_t = time.time()
    processor = GLiNEREntityProcessor(
        model_name=model_name,
        threshold=threshold,
        device=device
    )
    load_time_sec = time.time() - load_start_t

    print(f"\nProcessing {len(json_files)} JSON file(s) with GLiNER -> {output_dir}\n" + "=" * 75)
    
    total_images = len(json_files)
    total_crops = 0
    total_entities = 0
    total_inference_time_sec = 0.0

    for jf in json_files:
        res = processor.process_file(jf, output_dir)
        total_crops += res["total_items"]
        total_entities += res["total_entities"]
        total_inference_time_sec += res["elapsed_sec"]

    avg_time_per_image = total_inference_time_sec / max(total_images, 1)
    avg_time_per_crop_ms = (total_inference_time_sec * 1000.0) / max(total_crops, 1)

    stats = {
        "model_name": model_name,
        "total_images": total_images,
        "total_crops": total_crops,
        "total_entities": total_entities,
        "load_time_sec": round(load_time_sec, 3),
        "total_inference_time_sec": round(total_inference_time_sec, 3),
        "avg_time_per_image_sec": round(avg_time_per_image, 3),
        "avg_time_per_crop_ms": round(avg_time_per_crop_ms, 2)
    }

    print("=" * 75)
    print("  GLiNER INFERENCE LATENCY REPORT")
    print("=" * 75)
    print(f"  - Model Load Time         : {stats['load_time_sec']:.3f} s")
    print(f"  - Total Images Processed  : {stats['total_images']}")
    print(f"  - Total Text Crops        : {stats['total_crops']}")
    print(f"  - Total Entities Found    : {stats['total_entities']}")
    print(f"  - Total Processing Time   : {stats['total_inference_time_sec']:.3f} s")
    print(f"  - AVG Latency per Image   : {stats['avg_time_per_image_sec']:.3f} s / image")
    print(f"  - AVG Latency per Text    : {stats['avg_time_per_crop_ms']:.2f} ms / crop")
    print("=" * 75 + "\n")

    return stats


def main():
    parser = argparse.ArgumentParser(description="GLiNER NER Processor for Recognition JSONs.")
    parser.add_argument("--rec_dir", type=str, default="recognition_output", help="Directory containing recognition JSON files.")
    parser.add_argument("--out_dir", type=str, default="gliner_multi_v2_1_output", help="Output directory for GLiNER NER JSONs.")
    parser.add_argument("--model", type=str, default="urchade/gliner_multi-v2.1", help="GLiNER pretrained model name.")
    parser.add_argument("--threshold", type=float, default=0.30, help="Confidence threshold for entity detection.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Inference device.")
    args = parser.parse_args()

    process_directory(
        rec_dir=args.rec_dir,
        output_dir=args.out_dir,
        model_name=args.model,
        threshold=args.threshold,
        device=args.device
    )


if __name__ == "__main__":
    main()

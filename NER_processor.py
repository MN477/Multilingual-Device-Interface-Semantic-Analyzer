"""
NER_processor.py
================
NER Enrichment processor using HanLP (UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_BASE).
Implements BasePipelineStage for modular integration into the OCR pipeline.

Takes recognition results (with regex_matches already applied by regex_filter),
skips HanLP inference for text blocks already identified by regex,
and appends HanLP entities with full label_confidences to the remaining blocks.
"""

import os
import sys
import json
import glob
import time
import argparse
from typing import List, Dict, Any, Optional

# Ensure stdout supports unicode characters
sys.stdout.reconfigure(encoding='utf-8')

try:
    import hanlp
    HAS_HANLP = True
except ImportError:
    HAS_HANLP = False

from pipeline_core import BasePipelineStage, PipelineContext


# ═══════════════════════════════════════════════════════════════════════════════
# CORE HANLP ENTITY PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════

class HanLPEntityProcessor:
    """
    Wraps the HanLP MTL model for Named Entity Recognition.
    Provides predict_text() for single-text inference and alignment.
    """

    ONTONOTES_LABELS = [
        "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT",
        "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT",
        "MONEY", "QUANTITY", "ORDINAL", "CARDINAL", "O"
    ]

    def __init__(self):
        if not HAS_HANLP:
            raise ImportError(
                "HanLP is not installed. Please run 'pip install hanlp'."
            )
        print("Loading HanLP model (UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_BASE)...")
        start_t = time.perf_counter()
        self.model = hanlp.load(
            hanlp.pretrained.mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_BASE
        )
        self.load_time_sec = time.perf_counter() - start_t
        print(f"HanLP model loaded successfully in {self.load_time_sec:.2f}s.")

    @staticmethod
    def align_tokens_to_chars(text: str, tokens: List[str]) -> List[tuple]:
        """Align word-level tokens back to character-level start/end offsets."""
        char_spans = []
        current_char_idx = 0
        for tok in tokens:
            start = text.find(tok, current_char_idx)
            if start == -1:
                start = text.lower().find(tok.lower(), current_char_idx)
            if start != -1:
                end = start + len(tok)
                char_spans.append((start, end))
                current_char_idx = end
            else:
                char_spans.append((current_char_idx, current_char_idx + len(tok)))
                current_char_idx += len(tok)
        return char_spans

    def predict_text(self, text: str) -> tuple:
        """
        Run HanLP on a single text string.
        Returns: (list of entity dicts, inference_time_sec)
        """
        if not text or not text.strip():
            return [], 0.0

        start_time = time.perf_counter()
        try:
            doc = self.model(text)
            elapsed_time = time.perf_counter() - start_time

            # Find the NER key in the document
            ner_key = None
            for k in doc.keys():
                if "ner" in k:
                    ner_key = k
                    break

            predicted_entities = []
            if ner_key and doc[ner_key]:
                tokens = doc.get("tok", doc.get("tok/fine", []))
                raw_ner = doc[ner_key]
                char_spans = self.align_tokens_to_chars(text, tokens)

                for ent in raw_ner:
                    if len(ent) == 4:
                        word, label, start_tok, end_tok = ent
                        try:
                            start_char = char_spans[start_tok][0]
                            end_char = char_spans[end_tok - 1][1]
                        except IndexError:
                            start_char = text.find(word)
                            end_char = start_char + len(word) if start_char != -1 else 0
                    else:
                        word, label = ent[0], ent[1]
                        start_char = text.find(word)
                        end_char = start_char + len(word) if start_char != -1 else 0

                    # Build label_confidences (CRF decoding → assign 1.0 to predicted)
                    label_confidences = {lbl: 0.0 for lbl in self.ONTONOTES_LABELS}
                    label_confidences[label] = 1.0
                    if label != "O":
                        label_confidences["O"] = 0.0

                    predicted_entities.append({
                        "entity_group": label,
                        "score": 1.0,
                        "word": word,
                        "start": start_char,
                        "end": end_char,
                        "label_confidences": label_confidences
                    })

            return predicted_entities, elapsed_time

        except Exception as e:
            elapsed_time = time.perf_counter() - start_time
            print(f"Warning: HanLP prediction failed for text '{text}': {e}")
            return [], elapsed_time


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE STAGE (for use with PipelineRunner)
# ═══════════════════════════════════════════════════════════════════════════════

class HanLPNERStage(BasePipelineStage):
    """
    Pipeline stage that runs HanLP NER on recognition results.
    Skips text blocks that already have regex_matches.
    Appends entities to each result item.
    """

    def __init__(self):
        self._processor = None  # Lazy-loaded

    @property
    def name(self) -> str:
        return "HanLPNERStage"

    def _ensure_processor(self):
        if self._processor is None:
            self._processor = HanLPEntityProcessor()

    def run(self, context: PipelineContext, config: Dict[str, Any]) -> PipelineContext:
        ner_cfg = config.get("ner", {})

        # Allow disabling the stage from config
        if not ner_cfg.get("enabled", True):
            print(f"[{self.name}] Skipped (disabled in config).")
            source = (
                context.regex_enriched_results
                or context.final_results
                or context.filtered_recog_results
                or context.recog_results
            )
            context.ner_enriched_results = list(source)
            return context

        self._ensure_processor()

        # Pick the most recent results from the pipeline chain
        source = (
            context.regex_enriched_results
            or context.final_results
            or context.filtered_recog_results
            or context.recog_results
        )

        if not source:
            print(f"[{self.name}] No results available for NER.")
            context.ner_enriched_results = []
            return context

        enriched = []
        total_entities = 0
        inference_times = []

        for item in source:
            item_copy = dict(item)
            text = item_copy.get("transcribed_text", "")
            regex_matches = item_copy.get("regex_matches", [])

            # Skip HanLP if regex already identified this block
            if regex_matches:
                entities = []
                elapsed = 0.0
            else:
                entities, elapsed = self._processor.predict_text(text)
                if text.strip():
                    inference_times.append(elapsed)

            existing = item_copy.get("entities", [])
            existing.extend(entities)
            item_copy["entities"] = existing
            item_copy["hanlp_inference_time_ms"] = round(elapsed * 1000.0, 2)
            total_entities += len(entities)
            enriched.append(item_copy)

        context.ner_enriched_results = enriched
        print(f"[{self.name}] Processed {len(enriched)} blocks, found {total_entities} NER entity(ies).")
        return context


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE FUNCTIONS (for direct file processing or Streamlit use)
# ═══════════════════════════════════════════════════════════════════════════════

def process_results_list(
    processor: HanLPEntityProcessor,
    results: List[Dict[str, Any]]
) -> tuple:
    """
    Process a list of recognition result dicts through HanLP NER.
    Skips items that already have regex_matches.
    Returns: (enriched_results, total_entities, total_inference_time_sec)
    """
    enriched = []
    total_entities = 0
    total_inference_time = 0.0

    for item in results:
        item_copy = dict(item)
        text = item_copy.get("transcribed_text", "")
        regex_matches = item_copy.get("regex_matches", [])

        if regex_matches:
            entities = []
            elapsed = 0.0
        else:
            entities, elapsed = processor.predict_text(text)
            total_inference_time += elapsed

        existing = item_copy.get("entities", [])
        existing.extend(entities)
        item_copy["entities"] = existing
        item_copy["hanlp_inference_time_ms"] = round(elapsed * 1000.0, 2)
        total_entities += len(entities)
        enriched.append(item_copy)

    return enriched, total_entities, total_inference_time


def process_single_json(
    processor: HanLPEntityProcessor,
    json_path: str,
    output_dir: str
) -> Dict[str, Any]:
    """Process a single regex-enriched JSON file through HanLP NER."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    image_stem = data.get(
        "image_stem",
        os.path.basename(json_path).replace("_regex.json", "").replace(".json", "")
    )

    start_t = time.perf_counter()
    enriched, total_entities, total_inf_time = process_results_list(processor, results)
    file_elapsed = time.perf_counter() - start_t

    data["results"] = enriched
    data["ner_model"] = "UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_BASE"
    data["total_ner_entities_found"] = total_entities
    data["ner_inference_stats"] = {
        "file_inference_time_sec": round(file_elapsed, 4),
        "total_text_inference_time_sec": round(total_inf_time, 4),
    }

    os.makedirs(output_dir, exist_ok=True)
    out_name = f"{image_stem}_hanlp_ner.json"
    out_path = os.path.join(output_dir, out_name)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[HanLP] '{image_stem}': {total_entities} entity(ies) | Time: {total_inf_time:.4f}s")
    return {
        "image_stem": image_stem,
        "total_entities": total_entities,
        "inference_time_sec": total_inf_time,
        "out_path": out_path
    }


def process_directory(
    input_dir: str = "regex_detection",
    output_dir: str = "hanlp_ner_output"
):
    """Process all regex-enriched JSON files in a directory through HanLP NER."""
    if not os.path.exists(input_dir):
        print(f"Error: input directory '{input_dir}' does not exist.")
        sys.exit(1)

    json_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    if not json_files:
        print(f"No JSON files found in '{input_dir}'.")
        sys.exit(1)

    processor = HanLPEntityProcessor()

    print(f"\nProcessing {len(json_files)} JSON file(s) with HanLP -> {output_dir}\n" + "=" * 75)

    total_entities = 0
    total_inf_time = 0.0

    for jf in json_files:
        res = process_single_json(processor, jf, output_dir)
        total_entities += res["total_entities"]
        total_inf_time += res["inference_time_sec"]

    print("=" * 75)
    print("  HanLP NER INFERENCE REPORT")
    print("=" * 75)
    print(f"  - Files Processed     : {len(json_files)}")
    print(f"  - Total Entities      : {total_entities}")
    print(f"  - Total Inference     : {total_inf_time:.4f}s")
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="HanLP NER Processor.")
    parser.add_argument("--input", type=str, default="regex_detection",
                        help="Input directory of regex-enriched JSONs or a single JSON file.")
    parser.add_argument("--out_dir", type=str, default="hanlp_ner_output",
                        help="Output directory for NER-enriched JSONs.")
    args = parser.parse_args()

    if os.path.isfile(args.input):
        processor = HanLPEntityProcessor()
        process_single_json(processor, args.input, args.out_dir)
    else:
        process_directory(input_dir=args.input, output_dir=args.out_dir)


if __name__ == "__main__":
    main()

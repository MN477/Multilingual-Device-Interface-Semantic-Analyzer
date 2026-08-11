"""
run_regex_enrichment.py
=======================
Orchestrator script to run Regex Pattern Detection on OCR recognition JSON files.

Reads all '*_final_recog.json' files in 'recognition_output/' (or user specified directory),
compiles regex patterns from 'regex_patterns.json', detects matching static entities in each transcribed_text block,
and enriches the JSON output with 'type' (entity_type) and 'system_or_user' (signal: 'system' vs 'user').
"""

import os
import re
import json
import glob
import time
import argparse
from typing import Dict, Any, List, Optional


def load_compiled_regex_patterns(patterns_json_path: str = "regex_patterns.json") -> List[Dict[str, Any]]:
    """
    Loads regex_patterns.json and compiles each regex pattern with appropriate flags.
    """
    if not os.path.exists(patterns_json_path):
        raise FileNotFoundError(f"Regex patterns file not found at: {patterns_json_path}")
        
    with open(patterns_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    compiled_patterns = []
    for cat in data.get("categories", []):
        cat_name = cat.get("category", "general")
        for pat in cat.get("patterns", []):
            try:
                flags = re.IGNORECASE if pat.get("case_insensitive", False) else 0
                compiled_patterns.append({
                    "compiled": re.compile(pat["regex"], flags),
                    "id": pat["id"],
                    "entity_type": pat["entity_type"],
                    "signal": pat.get("signal", "system"),
                    "category": cat_name,
                    "description": pat.get("description", "")
                })
            except re.error as e:
                print(f"Warning: Failed to compile pattern '{pat.get('id')}': {e}")
                
    return compiled_patterns


def process_single_json(
    json_path: str,
    output_dir: str,
    compiled_patterns: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Processes a single recognition JSON file, running regex pattern matching against each transcribed text.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", []) or data.get("all_detections_with_status", [])
    image_stem = data.get(
        "image_stem",
        os.path.basename(json_path).replace("_final_recog.json", "").replace("_rec_full.json", "").replace(".json", "")
    )

    enriched_results = []
    total_regex_matches = 0

    for item in results:
        item_copy = dict(item)
        text = (item_copy.get("transcribed_text") or item_copy.get("text") or "").strip()

        regex_matches = []
        if text:
            for pat in compiled_patterns:
                match = pat["compiled"].search(text)
                if match:
                    regex_matches.append({
                        "pattern_id": pat["id"],
                        "matched_text": match.group(),
                        "type": pat["entity_type"],
                        "system_or_user": pat["signal"],
                        "category": pat["category"]
                    })

        if regex_matches:
            total_regex_matches += len(regex_matches)
            primary_match = regex_matches[0]
            item_copy["type"] = primary_match["type"]
            item_copy["system_or_user"] = primary_match["system_or_user"]
        else:
            item_copy["type"] = None
            item_copy["system_or_user"] = None

        item_copy["regex_matches"] = regex_matches
        enriched_results.append(item_copy)

    output_dict = dict(data)
    output_dict["results"] = enriched_results
    output_dict["total_regex_matches_found"] = total_regex_matches

    os.makedirs(output_dir, exist_ok=True)
    out_filename = f"{image_stem}_regex.json"
    out_path = os.path.join(output_dir, out_filename)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=2)

    return {
        "image_stem": image_stem,
        "total_items": len(enriched_results),
        "regex_matches": total_regex_matches,
        "out_path": out_path
    }


def process_directory(
    rec_dir: str = "recognition_output",
    output_dir: str = "regex_detection",
    patterns_json_path: str = "regex_patterns.json"
) -> Dict[str, Any]:
    """
    Finds all '*_final_recog.json' files in rec_dir and performs regex detection.
    """
    if not os.path.exists(rec_dir):
        print(f"Error: Input directory '{rec_dir}' does not exist.")
        return {}

    # Target *_final_recog.json files strictly from recognition_output
    json_files = sorted(glob.glob(os.path.join(rec_dir, "*_final_recog.json")))

    if not json_files:
        print(f"No '*_final_recog.json' files found in '{rec_dir}'.")
        return {}

    print(f"Loading regex patterns from: {patterns_json_path}")
    compiled_patterns = load_compiled_regex_patterns(patterns_json_path)
    print(f"Loaded {len(compiled_patterns)} regex patterns.")

    print(f"\nProcessing {len(json_files)} final recognition JSON file(s) -> '{output_dir}'\n" + "=" * 80)
    
    start_t = time.perf_counter()
    total_matches = 0
    processed_count = 0

    for jf in json_files:
        res = process_single_json(jf, output_dir, compiled_patterns)
        processed_count += 1
        total_matches += res["regex_matches"]
        print(f"  [{processed_count}/{len(json_files)}] {res['image_stem']}: {res['regex_matches']} regex match(es) across {res['total_items']} items -> {os.path.basename(res['out_path'])}")

    elapsed = time.perf_counter() - start_t

    print("=" * 80)
    print("  REGEX DETECTION COMPLETE")
    print("=" * 80)
    print(f"  - Total JSON Files Processed : {processed_count}")
    print(f"  - Total Regex Matches Found  : {total_matches}")
    print(f"  - Total Execution Time       : {elapsed:.2f} s")
    print(f"  - Output Directory           : {os.path.abspath(output_dir)}")
    print("=" * 80 + "\n")

    return {
        "processed_files": processed_count,
        "total_regex_matches": total_matches,
        "elapsed_sec": elapsed
    }


def main():
    parser = argparse.ArgumentParser(
        description="Regex Pattern Detection for OCR Recognition Final Outputs."
    )
    parser.add_argument(
        "--rec_dir",
        type=str,
        default="recognition_output",
        help="Input directory containing final recognition JSON files (default: recognition_output)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="regex_detection",
        help="Output directory for regex detected JSON files (default: regex_detection)."
    )
    parser.add_argument(
        "--patterns_json",
        type=str,
        default="regex_patterns.json",
        help="Path to regex_patterns.json bank (default: regex_patterns.json)."
    )

    args = parser.parse_args()

    process_directory(
        rec_dir=args.rec_dir,
        output_dir=args.output_dir,
        patterns_json_path=args.patterns_json
    )


if __name__ == "__main__":
    main()

"""
regex_ner_filter.py
===================
Standalone module to run regex pattern matching (from regex_patterns.json)
on an NER model output JSON file.

Features:
  1. Loads and compiles all regex patterns from `regex_patterns.json`.
  2. Processes text blocks from an NER JSON file (e.g. NER_real_dataset_gliner_bi_entities.json).
  3. Detects regex entities, assigns signal (system vs user), and determines variability (dynamic vs structural).
  4. Enriches existing NER entity annotations with deterministic regex detections.
  5. Saves enriched dataset to output JSON file.

Usage:
    python regex_ner_filter.py
    python regex_ner_filter.py --input_json NER_real_dataset_gliner_bi_entities.json --patterns_json regex_patterns.json
"""

import os
import re
import json
import argparse
from typing import Dict, Any, List

# Entity variability lookup table
DYNAMIC_ENTITIES = {
    "email_address", "phone_number", "phone_tollfree", "time_12h", "time_24h",
    "time_with_seconds", "time_timezone", "date", "date_weekday", "date_relative",
    "duration", "money", "distance", "speed", "rpm", "temperature", "percentage",
    "masked_card", "payment_method", "iban", "reference_id", "road_code", "exit_number",
    "network_type", "tire_pressure", "voltage", "amperage", "obd_code", "vin",
    "fuel_efficiency", "odometer", "ev_range", "charging_rate", "radio_frequency",
    "rating", "data_size", "data_usage", "transfer_speed", "heart_rate", "blood_pressure",
    "steps", "calories", "weight", "blood_oxygen", "gps_coordinates", "compass_bearing",
    "altitude", "eta", "wind_speed", "humidity", "uv_index", "visibility",
    "precipitation_chance", "flight_number", "gate_number", "terminal", "seat_number",
    "booking_reference", "social_handle", "hashtag", "url", "ip_address", "mac_address",
    "zip_code", "postal_code", "person name", "song_title", "work_title"
}

def load_compiled_regex_patterns(patterns_json_path: str = "regex_patterns.json") -> List[Dict[str, Any]]:
    """
    Loads regex_patterns.json and compiles each pattern into a regex object.
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
                    "signal": pat["signal"],
                    "confidence": pat.get("confidence", 0.90),
                    "category": cat_name,
                    "description": pat.get("description", "")
                })
            except re.error as e:
                print(f"Warning: Failed to compile pattern '{pat.get('id')}': {e}")
                
    return compiled_patterns


def process_ner_json_with_regex(
    input_json_path: str = "NER_real_dataset_gliner_entities.json",
    patterns_json_path: str = "regex_patterns.json",
    output_json_path: str = "regex_ner_real_dataset_GLINER_output.json"
) -> Dict[str, Any]:
    """
    Reads an NER output JSON, runs compiled regex patterns against all text blocks,
    and enriches each block with regex matches, signal, and variability classification.
    """
    print(f"Loading regex patterns from: {patterns_json_path}")
    compiled_patterns = load_compiled_regex_patterns(patterns_json_path)
    print(f"Loaded {len(compiled_patterns)} regex patterns.")
    
    if not os.path.exists(input_json_path):
        raise FileNotFoundError(f"Input NER JSON not found at: {input_json_path}")
        
    with open(input_json_path, "r", encoding="utf-8") as f:
        ner_dataset = json.load(f)
        
    processed_images = []
    total_regex_matches_count = 0
    
    for img in ner_dataset.get("images", []):
        img_copy = dict(img)
        blocks = img.get("text_blocks", img.get("blocks", []))
        enriched_blocks = []
        
        for block in blocks:
            text = block.get("transcribed_text", block.get("text", "")).strip()
            block_copy = dict(block)
            
            regex_matches = []
            if text:
                for pat in compiled_patterns:
                    match = pat["compiled"].search(text)
                    if match:
                        entity_type = pat["entity_type"]
                        signal = pat["signal"]
                        variability = "dynamic" if entity_type in DYNAMIC_ENTITIES else "structural"
                        
                        regex_matches.append({
                            "pattern_id": pat["id"],
                            "matched_text": match.group(),
                            "entity_type": entity_type,
                            "signal": signal,
                            "variability": variability,
                            "confidence": pat["confidence"],
                            "category": pat["category"]
                        })
                        
            # Determine overall block signal & variability
            if regex_matches:
                # Pick match with highest confidence
                best_match = max(regex_matches, key=lambda x: x["confidence"])
                block_signal = best_match["signal"]
                block_variability = best_match["variability"]
                primary_entity = best_match["entity_type"]
                total_regex_matches_count += len(regex_matches)
            else:
                # Fallback heuristics for non-regex text
                # Check existing NER entities if present
                ner_entities = block.get("entities", [])
                if ner_entities:
                    best_ner = max(ner_entities, key=lambda x: x.get("score", 0))
                    label = best_ner.get("label", "").lower()
                    primary_entity = label
                    block_signal = "user" if label in ["person name", "user", "email", "phone"] else "system"
                    block_variability = "dynamic" if label in DYNAMIC_ENTITIES else "structural"
                else:
                    # Default formatting fallback
                    primary_entity = None
                    if text.isupper() and len(text.split()) <= 3:
                        block_signal = "system"
                        block_variability = "structural"
                    else:
                        block_signal = "system"
                        block_variability = "dynamic"
                        
            block_copy["regex_matches"] = regex_matches
            block_copy["source_signal"] = block_signal
            block_copy["variability"] = block_variability
            block_copy["primary_entity"] = primary_entity
            
            enriched_blocks.append(block_copy)
            
        img_copy["text_blocks"] = enriched_blocks
        processed_images.append(img_copy)
        
    enriched_dataset = {
        "dataset": "Regex_NER_Enriched_Output",
        "total_images": len(processed_images),
        "total_regex_matches": total_regex_matches_count,
        "images": processed_images
    }
    
    # Save enriched output
    with open(output_json_path, "w", encoding="utf-8") as out_f:
        json.dump(enriched_dataset, out_f, indent=2, ensure_ascii=False)
        
    print(f"Successfully processed {len(processed_images)} image(s) with {total_regex_matches_count} total regex match(es).")
    print(f"Output saved to: {os.path.abspath(output_json_path)}")
    return enriched_dataset


def main():
    parser = argparse.ArgumentParser(description="Run static regex detection on NER model JSON output")
    parser.add_argument("--input_json", type=str, default="NER_real_dataset_gliner_bi_entities.json", help="Path to input NER JSON file")
    parser.add_argument("--patterns_json", type=str, default="regex_patterns.json", help="Path to regex_patterns.json bank")
    parser.add_argument("--output_json", type=str, default="regex_ner_enriched_output.json", help="Path to write output enriched JSON")
    
    args = parser.parse_args()
    
    # Run pipeline
    results = process_ner_json_with_regex(
        input_json_path=args.input_json,
        patterns_json_path=args.patterns_json,
        output_json_path=args.output_json
    )
    
    # Print clear summary console table
    print("\n" + "=" * 70)
    print("REGEX + NER DETECTION SUMMARY")
    print("=" * 70)
    
    for img in results.get("images", []):
        print(f"\nImage: {img.get('image_name')}")
        for block in img.get("text_blocks", []):
            block_id = block.get("block_id")
            text = block.get("transcribed_text", block.get("text", ""))
            signal = block.get("source_signal", "").upper()
            variability = block.get("variability", "").upper()
            matches = block.get("regex_matches", [])
            ner_ents = block.get("entities", [])
            
            print(f"  Block #{block_id:2d} | [{signal:6s} | {variability:10s}] \"{text}\"")
            if matches:
                for m in matches:
                    print(f"         ├─ REGEX: {m['pattern_id']} -> '{m['matched_text']}' ({m['entity_type']}, conf={m['confidence']})")
            if ner_ents:
                for e in ner_ents:
                    print(f"         └─ NER  : label='{e.get('label')}' (score={e.get('score', 0.0):.3f})")
            if not matches and not ner_ents:
                print(f"         └─ (No specific entity matched)")

if __name__ == "__main__":
    main()

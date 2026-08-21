import os
import json
from pathlib import Path
from collections import defaultdict

def main():
    original_dir = Path("xlm_roberta_ner_output")
    finetuned_dir = Path("finetuned_ner_output")
    
    if not original_dir.exists() or not finetuned_dir.exists():
        print("Error: Both output directories must exist.")
        return

    original_files = {f.name.replace("_xlm_roberta_ner.json", ""): f for f in original_dir.glob("*.json")}
    finetuned_files = {f.name.replace("_finetuned_ner.json", ""): f for f in finetuned_dir.glob("*.json")}
    
    common_stems = set(original_files.keys()).intersection(finetuned_files.keys())
    
    print(f"Found {len(common_stems)} common files to compare.")
    
    target_labels = {"PER", "LOC", "ORG"}
    
    # Tracking discrepancies
    # missed[label] = list of dicts with info
    missed_by_finetuned = defaultdict(list)
    added_by_finetuned = defaultdict(list)
    changed_labels = defaultdict(list) # original label -> new label
    
    for stem in common_stems:
        orig_file = original_files[stem]
        ft_file = finetuned_files[stem]
        
        with open(orig_file, "r", encoding="utf-8") as f:
            orig_data = json.load(f)
            
        with open(ft_file, "r", encoding="utf-8") as f:
            ft_data = json.load(f)
            
        orig_results = {res.get("crop_filename", str(i)): res for i, res in enumerate(orig_data.get("results", []))}
        ft_results = {res.get("crop_filename", str(i)): res for i, res in enumerate(ft_data.get("results", []))}
        
        for crop_id, orig_res in orig_results.items():
            if crop_id not in ft_results:
                continue
                
            ft_res = ft_results[crop_id]
            
            orig_entities = orig_res.get("entities", [])
            ft_entities = ft_res.get("entities", [])
            
            text = orig_res.get("text", "")
            
            # Simple word-to-label mapping for comparison
            # In a real scenario, we might want to align by character offset
            # For simplicity, let's map by start index
            orig_map = {ent["start"]: ent for ent in orig_entities}
            ft_map = {ent["start"]: ent for ent in ft_entities}
            
            # Check what original model found
            for start, o_ent in orig_map.items():
                o_label = o_ent["entity_group"]
                if o_label not in target_labels:
                    continue
                    
                word = o_ent["word"]
                
                # Check what finetuned model predicted at this offset
                if start in ft_map:
                    f_ent = ft_map[start]
                    f_label = f_ent["entity_group"]
                    if f_label != o_label:
                        changed_labels[o_label].append({
                            "file": stem,
                            "crop": crop_id,
                            "text": text,
                            "word": word,
                            "original_label": o_label,
                            "finetuned_label": f_label
                        })
                else:
                    # Maybe it was shifted slightly, let's just check if it was predicted anywhere nearby
                    found_nearby = False
                    for f_start, f_ent in ft_map.items():
                        if abs(f_start - start) <= 2 and f_ent["word"] == word:
                            if f_ent["entity_group"] != o_label:
                                changed_labels[o_label].append({
                                    "file": stem,
                                    "crop": crop_id,
                                    "text": text,
                                    "word": word,
                                    "original_label": o_label,
                                    "finetuned_label": f_ent["entity_group"]
                                })
                            found_nearby = True
                            break
                    
                    if not found_nearby:
                        missed_by_finetuned[o_label].append({
                            "file": stem,
                            "crop": crop_id,
                            "text": text,
                            "word": word,
                            "original_label": o_label
                        })

    print("\n" + "="*50)
    print("COMPARISON REPORT: PER, LOC, ORG")
    print("="*50)
    
    for label in target_labels:
        print(f"\n--- Label: {label} ---")
        missed = missed_by_finetuned[label]
        changed = changed_labels[label]
        
        print(f"Total entities completely MISSED by finetuned model: {len(missed)}")
        for m in missed[:5]: # Show up to 5 examples
            print(f"  - Missed '{m['word']}' in {m['file']} (Text: '{m['text']}')")
        if len(missed) > 5:
            print(f"  ... and {len(missed) - 5} more.")
            
        print(f"\nTotal entities CHANGED to a different label by finetuned model: {len(changed)}")
        for c in changed[:5]:
            print(f"  - '{c['word']}' changed from {c['original_label']} to {c['finetuned_label']} in {c['file']} (Text: '{c['text']}')")
        if len(changed) > 5:
            print(f"  ... and {len(changed) - 5} more.")

if __name__ == "__main__":
    main()

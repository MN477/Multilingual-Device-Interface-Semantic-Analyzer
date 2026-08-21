"""
Remap the generated combined dataset (train.json and val.json)
to align exactly with Davlan/xlm-roberta-base-ner-hrl's native label IDs (0..8)
and append custom labels at 9..12.

Native Davlan Labels:
  0: 'O'
  1: 'B-DATE', 2: 'I-DATE'
  3: 'B-PER',  4: 'I-PER'
  5: 'B-ORG',  6: 'I-ORG'
  7: 'B-LOC',  8: 'I-LOC'

Custom Extensions:
  9: 'B-SONG', 10: 'I-SONG'
  11: 'B-APP', 12: 'I-APP'
"""

import json
import os

DATA_DIR = r"c:\Users\mouss\Desktop\semantic_classif\ner_combined_data"
TRAIN_PATH = os.path.join(DATA_DIR, "train.json")
VAL_PATH = os.path.join(DATA_DIR, "val.json")
LABEL_MAP_PATH = os.path.join(DATA_DIR, "label_map.json")

# Current mapping used when generating ner_combined_data
OLD_ID2LABEL = {
    0: 'O',
    1: 'B-PER', 2: 'I-PER',
    3: 'B-ORG', 4: 'I-ORG',
    5: 'B-LOC', 6: 'I-LOC',
    7: 'B-DATE', 8: 'I-DATE',
    9: 'B-SONG', 10: 'I-SONG',
    11: 'B-APP', 12: 'I-APP'
}

# New Target mapping aligned with Davlan model weights
NEW_LABEL2ID = {
    'O': 0,
    'B-DATE': 1, 'I-DATE': 2,
    'B-PER': 3,  'I-PER': 4,
    'B-ORG': 5,  'I-ORG': 6,
    'B-LOC': 7,  'I-LOC': 8,
    'B-SONG': 9, 'I-SONG': 10,
    'B-APP': 11, 'I-APP': 12
}
NEW_ID2LABEL = {v: k for k, v in NEW_LABEL2ID.items()}

# Old ID -> String Label -> New ID
ID_MAPPER = {old_id: NEW_LABEL2ID[OLD_ID2LABEL[old_id]] for old_id in OLD_ID2LABEL}

def remap_file(filepath):
    print(f"Remapping {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for item in data:
        item["ner_tags"] = [ID_MAPPER[tag_id] for tag_id in item["ner_tags"]]
        
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Successfully updated {len(data)} examples in {os.path.basename(filepath)}.")

def main():
    remap_file(TRAIN_PATH)
    remap_file(VAL_PATH)
    
    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "label2id": NEW_LABEL2ID,
            "id2label": {str(k): v for k, v in NEW_ID2LABEL.items()}
        }, f, indent=2)
    print("Updated label_map.json with native Davlan indexing + SONG & APP extensions.")

if __name__ == "__main__":
    main()

"""
relational_title_detector.py
=============================
Standalone module to detect song titles / work titles based on spatial relationships
with recognized person names (authors / artists).

Heuristic Logic:
  If block B_author is recognized as a person name (e.g., "Queen", "Sam Brown"),
  and block B_title is positioned JUST ABOVE B_author with a slightly larger font size (height),
  then B_title is classified as a work title (e.g. song name / album title) with:
    - signal: "user" (creator content)
    - variability: "dynamic" (variable title content)
    - entity_type: "work_title" / "song_title"
    - calculated probabilistic confidence score

Usage:
    python relational_title_detector.py
    python relational_title_detector.py --input_json NER_real_dataset_gliner_bi_entities.json
"""

import os
import json
import math
import argparse
from typing import Dict, Any, List, Optional, Tuple

def calculate_title_confidence(
    author_score: float,
    height_ratio: float,
    vertical_gap: float,
    center_diff_normalized: float,
    title_text: str
) -> float:
    """
    Computes a calibrated confidence score C in [0.0, 1.0] for the title candidate.
    
    Factors:
      1. author_score: Initial confidence of the person name detection (0.0 to 1.0)
      2. height_ratio (title_height / author_height):
         - Ideal ratio is 1.05 to 1.8 (slightly bigger font size)
         - Ratios between 0.8 and 2.5 are acceptable with lower weight
      3. vertical_gap (px):
         - Distance between bottom of title and top of author. Closer is better.
      4. center_diff_normalized:
         - Horizontal center displacement relative to image width. Overlapping/aligned is better.
    """
    # 1. Height ratio factor (S_height)
    if 1.05 <= height_ratio <= 1.8:
        s_height = 1.0
    elif 0.8 <= height_ratio < 1.05:
        s_height = 0.85
    elif 1.8 < height_ratio <= 2.5:
        s_height = 0.80
    else:
        s_height = 0.50
        
    # 2. Vertical gap factor (S_gap): exponential decay based on pixel distance
    # Optimal gap: 2px to 40px
    if vertical_gap < 0: # Slight overlap or misaligned bbox
        s_gap = 0.60
    else:
        s_gap = math.exp(-vertical_gap / 35.0)
        
    # 3. Horizontal alignment factor (S_align)
    s_align = max(0.0, 1.0 - (center_diff_normalized * 4.0))
    
    # 4. Text quality penalty (avoid system UI buttons like "SAVE", "CANCEL", "EDIT")
    system_keywords = {"SAVE", "CANCEL", "EDIT", "SEARCH", "DONE", "HOME", "SETTINGS", "PROFILE", "BACK"}
    if title_text.strip().upper() in system_keywords:
        return 0.0
        
    # Composite score calculation
    raw_score = author_score * (0.35 * s_height + 0.40 * s_gap + 0.25 * s_align)
    return round(min(0.98, max(0.0, raw_score)), 4)


def detect_relational_titles(
    data: Dict[str, Any],
    max_vertical_gap: float = 60.0,
    max_horizontal_diff_ratio: float = 0.25
) -> Dict[str, Any]:
    """
    Scans text blocks in each image, identifies person names, and tests blocks above them.
    """
    processed_images = []
    
    for img in data.get("images", []):
        img_name = img.get("image_name", "unknown")
        img_size = img.get("size", {})
        img_width = float(img_size.get("width", 1000.0))
        text_blocks = img.get("text_blocks", img.get("blocks", []))
        
        # Step 1: Identify all person name candidates
        author_blocks = []
        for block in text_blocks:
            entities = block.get("entities", [])
            for ent in entities:
                label = ent.get("label", "").lower()
                if label in ["person name", "person", "artist", "author", "creator"]:
                    author_blocks.append({
                        "block": block,
                        "entity": ent,
                        "score": ent.get("score", 0.70)
                    })
                    
        relational_results = []
        
        # Step 2: For each author candidate, look for title candidate JUST ABOVE IT
        for item in author_blocks:
            auth_block = item["block"]
            auth_bbox = auth_block.get("bbox", [0, 0, 0, 0])
            auth_x0, auth_y0, auth_x1, auth_y1 = auth_bbox
            auth_h = auth_block.get("height", auth_y1 - auth_y0)
            auth_cx = auth_block.get("center", [(auth_x0 + auth_x1)/2, (auth_y0 + auth_y1)/2])[0]
            auth_score = item["score"]
            
            for candidate in text_blocks:
                if candidate.get("block_id") == auth_block.get("block_id"):
                    continue # Skip self
                    
                cand_bbox = candidate.get("bbox", [0, 0, 0, 0])
                cand_x0, cand_y0, cand_x1, cand_y1 = cand_bbox
                cand_h = candidate.get("height", cand_y1 - cand_y0)
                cand_cx = candidate.get("center", [(cand_x0 + cand_x1)/2, (cand_y0 + cand_y1)/2])[0]
                
                # Check spatial condition: candidate MUST be ABOVE author (cand_y1 <= auth_y0 + tolerance)
                vertical_gap = auth_y0 - cand_y1
                
                if 0 <= vertical_gap <= max_vertical_gap:
                    # Check horizontal alignment
                    center_diff = abs(cand_cx - auth_cx)
                    center_diff_ratio = center_diff / img_width
                    
                    if center_diff_ratio <= max_horizontal_diff_ratio:
                        # Check height ratio (title height / author height)
                        height_ratio = cand_h / auth_h if auth_h > 0 else 1.0
                        
                        # Calculate confidence
                        conf = calculate_title_confidence(
                            author_score=auth_score,
                            height_ratio=height_ratio,
                            vertical_gap=vertical_gap,
                            center_diff_normalized=center_diff_ratio,
                            title_text=candidate.get("transcribed_text", candidate.get("text", ""))
                        )
                        
                        if conf > 0.35: # Threshold for valid relation
                            relational_results.append({
                                "title_block_id": candidate.get("block_id"),
                                "title_text": candidate.get("transcribed_text", candidate.get("text", "")),
                                "author_block_id": auth_block.get("block_id"),
                                "author_text": auth_block.get("transcribed_text", auth_block.get("text", "")),
                                "confidence": conf,
                                "classification": {
                                    "source": "user",
                                    "variability": "dynamic",
                                    "entity_type": "work_title"
                                },
                                "spatial_metrics": {
                                    "height_ratio": round(height_ratio, 2),
                                    "vertical_gap_px": round(vertical_gap, 1),
                                    "horizontal_displacement_px": round(center_diff, 1)
                                }
                            })
                            
        # Attach results to image block output
        img_copy = dict(img)
        img_copy["relational_title_detections"] = relational_results
        processed_images.append(img_copy)
        
    return {
        "dataset": data.get("dataset", "Relational_Title_Detection_Output"),
        "total_images": len(processed_images),
        "images": processed_images
    }

def main():
    parser = argparse.ArgumentParser(description="Test spatial relational title detector (song title above artist name)")
    parser.add_argument("--input_json", type=str, default="NER_real_dataset_gliner_bi_entities.json", help="Path to input NER dataset JSON")
    parser.add_argument("--output_json", type=str, default="relational_title_output.json", help="Path to save output JSON")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_json):
        print(f"Input file '{args.input_json}' not found. Creating mock test data...")
        mock_data = {
            "dataset": "Mock_Test",
            "images": [
                {
                    "image_name": "music_player_mock.jpg",
                    "size": {"width": 540, "height": 960},
                    "text_blocks": [
                        {
                            "block_id": 1,
                            "transcribed_text": "Bohemian Rhapsody",
                            "bbox": [100.0, 300.0, 440.0, 345.0],
                            "width": 340.0,
                            "height": 45.0,
                            "center": [270.0, 322.5],
                            "entities": []
                        },
                        {
                            "block_id": 2,
                            "transcribed_text": "Queen",
                            "bbox": [180.0, 355.0, 360.0, 385.0],
                            "width": 180.0,
                            "height": 30.0,
                            "center": [270.0, 370.0],
                            "entities": [
                                {"text": "Queen", "label": "person name", "score": 0.92}
                            ]
                        }
                    ]
                }
            ]
        }
        data = mock_data
    else:
        with open(args.input_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            
    results = detect_relational_titles(data)
    
    # Write output
    with open(args.output_json, "w", encoding="utf-8") as out_f:
        json.dump(results, out_f, indent=2, ensure_ascii=False)
        
    print("=" * 60)
    print("RELATIONAL TITLE DETECTOR RESULTS")
    print("=" * 60)
    for img in results["images"]:
        print(f"\nImage: {img.get('image_name')}")
        detections = img.get("relational_title_detections", [])
        if not detections:
            print("  No relational title-author pairs detected.")
        for det in detections:
            print(f"  [FOUND PAIR] Confidence: {det['confidence']:.4f}")
            print(f"    • Title Candidate : '{det['title_text']}' (Block #{det['title_block_id']})")
            print(f"    • Author/Artist   : '{det['author_text']}' (Block #{det['author_block_id']})")
            print(f"    • Classification  : source={det['classification']['source']}, variability={det['classification']['variability']}, entity={det['classification']['entity_type']}")
            print(f"    • Metrics         : Height Ratio={det['spatial_metrics']['height_ratio']}x, Vertical Gap={det['spatial_metrics']['vertical_gap_px']}px")
    print(f"\nOutput saved to: {os.path.abspath(args.output_json)}")

if __name__ == "__main__":
    main()

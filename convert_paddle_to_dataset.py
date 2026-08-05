"""
convert_paddle_to_dataset.py
=============================
Converts PaddleOCR output JSON files (from debug_paddle_output or custom paths)
into a structured dataset JSON format matching `android_auto_gliner_dataset.json`.

Usage:
    python convert_paddle_to_dataset.py
    python convert_paddle_to_dataset.py --input_dir debug_paddle_output --output_file converted_paddle_dataset.json
    python convert_paddle_to_dataset.py --input_json debug_paddle_output/ocr_result_paddle.json
"""

import os
import glob
import json
import argparse
from typing import Dict, Any, List

def convert_paddle_file(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts a single PaddleOCR JSON output object to the target image object schema.
    """
    # Extract image path and name
    raw_image_path = json_data.get("image_path", "unknown_image.png")
    image_name = os.path.basename(raw_image_path)
    
    # Extract image dimensions [width, height]
    image_size = json_data.get("image_size", [0, 0])
    width = image_size[0] if len(image_size) > 0 else 0
    height = image_size[1] if len(image_size) > 1 else 0
    
    # Process detections / text blocks
    detections = json_data.get("detections", json_data.get("text_blocks", []))
    text_blocks = []
    
    for i, det in enumerate(detections, start=1):
        # Extract text content
        transcribed_text = det.get("text", det.get("transcribed_text", ""))
        
        # Extract bounding box [x_min, y_min, x_max, y_max]
        bbox = det.get("bbox", [])
        
        # Extract width, height, and center
        box_w = det.get("width", 0.0)
        box_h = det.get("height", 0.0)
        center = det.get("center", [])
        
        # If bbox exists but width/height/center missing, compute them dynamically
        if bbox and (not box_w or not box_h or not center):
            x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
            box_w = float(x_max - x_min)
            box_h = float(y_max - y_min)
            center = [float(x_min + box_w / 2.0), float(y_min + box_h / 2.0)]
            
        block_obj = {
            "block_id": i,
            "transcribed_text": transcribed_text,
            "bbox": [float(val) for val in bbox],
            "width": float(box_w),
            "height": float(box_h),
            "center": [float(val) for val in center]
        }
        text_blocks.append(block_obj)
        
    image_entry = {
        "image_name": image_name,
        "size": {
            "width": width,
            "height": height
        },
        "text_blocks": text_blocks
    }
    return image_entry

def convert_paddle_output_to_dataset(
    input_dir: str = "debug_paddle_output",
    input_json: str = None,
    output_file: str = "converted_paddle_dataset.json"
) -> Dict[str, Any]:
    """
    Main conversion function that processes single JSON or directory of JSONs.
    """
    json_files = []
    
    if input_json and os.path.exists(input_json):
        json_files = [input_json]
    elif os.path.exists(input_dir):
        json_files = glob.glob(os.path.join(input_dir, "*.json"))
        
    if not json_files:
        print(f"Error: No JSON files found in input path '{input_json or input_dir}'.")
        return {}
        
    converted_images = []
    total_blocks_count = 0
    
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # Handle case where file contains a list of image entries or a single entry
            if isinstance(data, list):
                for item in data:
                    img_entry = convert_paddle_file(item)
                    converted_images.append(img_entry)
                    total_blocks_count += len(img_entry["text_blocks"])
            elif isinstance(data, dict):
                # Check if it's already a dataset or single image paddle result
                if "detections" in data or "image_path" in data:
                    img_entry = convert_paddle_file(data)
                    converted_images.append(img_entry)
                    total_blocks_count += len(img_entry["text_blocks"])
                elif "images" in data:
                    for item in data["images"]:
                        img_entry = convert_paddle_file(item)
                        converted_images.append(img_entry)
                        total_blocks_count += len(img_entry["text_blocks"])
        except Exception as e:
            print(f"Warning: Failed to process '{file_path}': {e}")

    dataset_output = {
        "dataset": "OCR_Post_Processing_Output",
        "total_images": len(converted_images),
        "images": converted_images
    }
    
    # Save output to specified file
    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump(dataset_output, out_f, indent=2, ensure_ascii=False)
        
    print(f"Successfully converted {len(converted_images)} image(s) with {total_blocks_count} total text block(s).")
    print(f"Output saved to: {os.path.abspath(output_file)}")
    return dataset_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PaddleOCR json output to dataset schema matching android_auto_gliner_dataset.json")
    parser.add_argument("--input_dir", type=str, default="debug_paddle_output", help="Directory containing PaddleOCR result json files")
    parser.add_argument("--input_json", type=str, default=None, help="Path to a single PaddleOCR result json file")
    parser.add_argument("--output_file", type=str, default="converted_paddle_dataset.json", help="Path to write converted dataset JSON")
    
    args = parser.parse_args()
    convert_paddle_output_to_dataset(
        input_dir=args.input_dir,
        input_json=args.input_json,
        output_file=args.output_file
    )

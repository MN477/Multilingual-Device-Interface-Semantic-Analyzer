import json
import time
from span_marker import SpanMarkerModel

def test_spanmarker_model(input_json_path="NER_dataset.json", output_json_path="NER_dataset_spanmarker_entities.json"):
    model_name = "tomaarsen/span-marker-mbert-base-multinerd"
    
    print(f"Loading SpanMarker model: {model_name}...")
    model = SpanMarkerModel.from_pretrained(model_name)
    
    print(f"Reading input dataset from {input_json_path}...")
    with open(input_json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    print("Extracting entities for text blocks...")
    images = dataset.get("images", [])
    total_blocks_processed = 0
    inference_times = []
    
    for img in images:
        for block in img.get("text_blocks", []):
            text = block.get("transcribed_text", "")
            if text.strip():
                # Perform NER inference with SpanMarker and measure time
                start_time = time.perf_counter()
                entities = model.predict(text)
                elapsed_time = time.perf_counter() - start_time
                inference_times.append(elapsed_time)
                
                formatted_entities = []
                for entity in entities:
                    formatted_entities.append({
                        "text": entity.get("text"),
                        "label": entity.get("label"),
                        "score": float(entity.get("score", 0.0)),
                        "char_start_index": int(entity.get("char_start_index", entity.get("start", 0))),
                        "char_end_index": int(entity.get("char_end_index", entity.get("end", 0)))
                    })
                block["entities"] = formatted_entities
            else:
                block["entities"] = []
                
            total_blocks_processed += 1
            
    avg_inference_time = (sum(inference_times) / len(inference_times)) if inference_times else 0.0
    total_inference_time = sum(inference_times)
    
    print(f"Processed {total_blocks_processed} text blocks across {len(images)} images.")
    print(f"Total Inference Time: {total_inference_time:.4f} seconds")
    print(f"Average Inference Time per text block: {avg_inference_time * 1000:.2f} ms ({avg_inference_time:.6f} s)")
    
    print(f"Saving updated dataset with SpanMarker entities to {output_json_path}...")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
        
    print("SpanMarker testing completed successfully!")

if __name__ == "__main__":
    test_spanmarker_model()

import json
import time
from gliner import GLiNER

def test_gliner_model(input_json_path="auto_contacts_converted.json", output_json_path="auto_contacts_GLINER_entities.json"):
    model_name = "urchade/gliner_multi-v2.1"
    
    print(f"Loading GLiNER model: {model_name}...")
    model = GLiNER.from_pretrained(model_name)
    
    # Target entity labels to extract using zero-shot GLiNER capabilities
    labels = ["person name" , "organization" , "location" , "song name"]
    
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
                # Perform zero-shot NER inference with GLiNER and measure time
                start_time = time.perf_counter()
                entities = model.predict_entities(text, labels)
                elapsed_time = time.perf_counter() - start_time
                inference_times.append(elapsed_time)
                
                formatted_entities = []
                for entity in entities:
                    formatted_entities.append({
                        "text": entity.get("text"),
                        "label": entity.get("label"),
                        "score": float(entity.get("score", 0.0)),
                        "start": int(entity.get("start")),
                        "end": int(entity.get("end"))
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
    
    print(f"Saving updated dataset with GLiNER entities to {output_json_path}...")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
        
    print("GLiNER testing completed successfully!")

if __name__ == "__main__":
    test_gliner_model()

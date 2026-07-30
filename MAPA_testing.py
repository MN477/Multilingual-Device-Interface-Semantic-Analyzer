import json
import time
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

def test_mapa_model(input_json_path="NER_dataset.json", output_json_path="NER_dataset_mapa_entities.json"):
    model_name = "Pangeanic/mapa-multilingual-administrative"
    
    print(f"Loading tokenizer and model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    
    # Exact mapping derived from original model repo's vocab configuration:
    # 0: O, 1: ADDRESS, 2: AMOUNT, 3: DATE, 4: ORGANISATION, 5: PERSON, 6: TIME, 7: VEHICLE
    id2label = {
        0: "O",
        1: "ADDRESS",
        2: "AMOUNT",
        3: "DATE",
        4: "ORGANISATION",
        5: "PERSON",
        6: "TIME",
        7: "VEHICLE"
    }
    
    label_map = {f"LABEL_{i}": label for i, label in id2label.items()}
    
    model.config.id2label = id2label
    model.config.label2id = {v: k for k, v in id2label.items()}
    
    # Initialize Hugging Face pipeline
    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )
    
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
                start_time = time.perf_counter()
                entities = ner_pipeline(text)
                elapsed_time = time.perf_counter() - start_time
                inference_times.append(elapsed_time)
                
                formatted_entities = []
                for entity in entities:
                    raw_label = entity.get("entity_group")
                    readable_label = label_map.get(raw_label, raw_label)
                    
                    # Store both raw label and resolved label for full transparency
                    formatted_entities.append({
                        "raw_label": raw_label,
                        "entity_group": readable_label,
                        "score": float(entity.get("score", 0.0)),
                        "word": entity.get("word"),
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
    
    print(f"Saving updated dataset with MAPA entities to {output_json_path}...")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
        
    print("MAPA model testing completed successfully!")

if __name__ == "__main__":
    test_mapa_model()

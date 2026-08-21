import os
import json

# Ensure stdout supports unicode characters
import sys
sys.stdout.reconfigure(encoding='utf-8')

def compare_outputs():
    hanlp_dir = "hanlp_ner_output"
    xlm_dir = "xlm_roberta_ner_output"
    
    if not os.path.exists(hanlp_dir):
        print(f"Error: {hanlp_dir} does not exist. Run the HanLP inference script first.")
        return
        
    hanlp_files = [f for f in os.listdir(hanlp_dir) if f.endswith("_hanlp_ner.json")]
    
    total_crops = 0
    total_hanlp_entities = 0
    total_xlm_entities = 0
    
    print("Comparing HanLP and XLM-Roberta NER outputs...\n")
    
    for h_file in hanlp_files:
        base_name = h_file.replace("_hanlp_ner.json", "")
        x_file = base_name + "_xlm_roberta_ner.json"
        x_path = os.path.join(xlm_dir, x_file)
        h_path = os.path.join(hanlp_dir, h_file)
        
        if not os.path.exists(x_path):
            print(f"Missing corresponding XLM file for {h_file}")
            continue
            
        with open(h_path, "r", encoding="utf-8") as f:
            h_data = json.load(f)
        with open(x_path, "r", encoding="utf-8") as f:
            x_data = json.load(f)
            
        h_results = {r["crop_filename"]: r for r in h_data.get("results", [])}
        x_results = {r["crop_filename"]: r for r in x_data.get("results", [])}
        
        diff_found_in_file = False
        file_diff_str = f"--- Differences in {base_name} ---\n"
        
        for crop_file, h_res in h_results.items():
            total_crops += 1
            if crop_file not in x_results:
                continue
                
            x_res = x_results[crop_file]
            
            # Extract basic entities (word, entity_group) for comparison
            h_entities = [(e["word"], e["entity_group"]) for e in h_res.get("entities", [])]
            x_entities = [(e["word"], e["entity_group"]) for e in x_res.get("entities", [])]
            
            total_hanlp_entities += len(h_entities)
            total_xlm_entities += len(x_entities)
            
            # Compare the exact lists
            if h_entities != x_entities:
                diff_found_in_file = True
                file_diff_str += f"Crop: {crop_file} | Text: '{h_res.get('transcribed_text', '')}'\n"
                file_diff_str += f"  HanLP : {h_entities}\n"
                file_diff_str += f"  XLM-R : {x_entities}\n\n"
                
        if diff_found_in_file:
            print(file_diff_str)
            
    print("=== SUMMARY ===")
    print(f"Total crops compared: {total_crops}")
    print(f"Total HanLP entities: {total_hanlp_entities}")
    print(f"Total XLM-Roberta entities: {total_xlm_entities}")

if __name__ == "__main__":
    compare_outputs()

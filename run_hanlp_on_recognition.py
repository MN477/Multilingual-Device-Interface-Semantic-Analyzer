import os
import sys
import json
import time

# Ensure stdout supports unicode characters
sys.stdout.reconfigure(encoding='utf-8')

# Ensure vocr virtual env packages can be imported
vocr_path = r"c:\Users\mouss\Desktop\semantic_classif\vocr\Lib\site-packages"
if vocr_path not in sys.path:
    sys.path.insert(0, vocr_path)

try:
    import hanlp
except ImportError:
    print("Error: HanLP is not installed. Please run pip install hanlp.")
    sys.exit(1)

def align_tokens_to_chars(text, tokens):
    char_spans = []
    current_char_idx = 0
    for tok in tokens:
        start = text.find(tok, current_char_idx)
        if start == -1:
            start = text.lower().find(tok.lower(), current_char_idx)
            
        if start != -1:
            end = start + len(tok)
            char_spans.append((start, end))
            current_char_idx = end
        else:
            char_spans.append((current_char_idx, current_char_idx + len(tok)))
            current_char_idx += len(tok)
    return char_spans

def process_recognition_folder():
    input_dir = "recognition_output"
    output_dir = "hanlp_ner_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading HanLP model...")
    HanLP = hanlp.load(hanlp.pretrained.mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_BASE)
    
    ontonotes_labels = [
        "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", 
        "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT", 
        "MONEY", "QUANTITY", "ORDINAL", "CARDINAL", "O"
    ]
    
    files = [f for f in os.listdir(input_dir) if f.endswith("_final_recog.json")]
    print(f"Found {len(files)} files to process in {input_dir}.")
    
    for filename in files:
        input_path = os.path.join(input_dir, filename)
        output_name = filename.replace("_final_recog.json", "_hanlp_ner.json")
        output_path = os.path.join(output_dir, output_name)
        
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        total_entities_found = 0
        total_text_inference_time = 0
        start_file_time = time.time()
        
        for result in data.get("results", []):
            text = result.get("transcribed_text", "")
            if not text:
                result["entities"] = []
                continue
                
            start_inf = time.time()
            doc = HanLP(text)
            total_text_inference_time += (time.time() - start_inf)
            
            ner_key = None
            for k in doc.keys():
                if "ner" in k:
                    ner_key = k
                    break
                    
            predicted_entities = []
            if ner_key and doc[ner_key]:
                tokens = doc.get("tok", doc.get("tok/fine", []))
                raw_ner = doc[ner_key]
                char_spans = align_tokens_to_chars(text, tokens)
                
                for ent in raw_ner:
                    if len(ent) == 4:
                        word, label, start_tok, end_tok = ent
                        try:
                            start_char = char_spans[start_tok][0]
                            end_char = char_spans[end_tok - 1][1]
                        except IndexError:
                            start_char = text.find(word)
                            end_char = start_char + len(word) if start_char != -1 else 0
                    else:
                        word, label = ent[0], ent[1]
                        start_char = text.find(word)
                        end_char = start_char + len(word) if start_char != -1 else 0
                        
                    label_confidences = {lbl: 0.0 for lbl in ontonotes_labels}
                    if label in label_confidences:
                        label_confidences[label] = 1.0
                    else:
                        label_confidences[label] = 1.0
                    label_confidences["O"] = 0.0 if label != "O" else 1.0
                    
                    predicted_entities.append({
                        "entity_group": label,
                        "score": 1.0,
                        "word": word,
                        "start": start_char,
                        "end": end_char,
                        "label_confidences": label_confidences
                    })
                    
            result["entities"] = predicted_entities
            total_entities_found += len(predicted_entities)
            
        data["ner_model"] = "UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_BASE"
        data["total_entities_found"] = total_entities_found
        data["inference_stats"] = {
            "file_inference_time_sec": time.time() - start_file_time,
            "total_text_inference_time_sec": total_text_inference_time
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"Processed {filename} -> {total_entities_found} entities.")

if __name__ == "__main__":
    process_recognition_folder()

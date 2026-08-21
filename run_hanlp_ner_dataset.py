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
    """
    Aligns word-level tokens back to the original raw text 
    to obtain character-level start and end offsets.
    """
    char_spans = []
    current_char_idx = 0
    for tok in tokens:
        # Find where tok starts in text starting from current_char_idx
        start = text.find(tok, current_char_idx)
        if start == -1:
            # Case-insensitive fallback
            start = text.lower().find(tok.lower(), current_char_idx)
            
        if start != -1:
            end = start + len(tok)
            char_spans.append((start, end))
            current_char_idx = end
        else:
            # Fallback if not found: guess based on current index
            char_spans.append((current_char_idx, current_char_idx + len(tok)))
            current_char_idx += len(tok)
    return char_spans

def main():
    dataset_path = "multilingual_ocr_ner_test_dataset.json"
    output_path = "hanlp_output.json"
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file '{dataset_path}' not found in current directory.")
        sys.exit(1)
        
    print(f"Loading dataset: {dataset_path}...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    print(f"Loading HanLP model: UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_BASE...")
    try:
        # Load the multi-task learning model
        HanLP = hanlp.load(hanlp.pretrained.mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_BASE)
    except Exception as e:
        print(f"Failed to load HanLP model. Error: {e}")
        sys.exit(1)
        
    samples = dataset.get("samples", dataset.get("records", []))
    print(f"Processing {len(samples)} samples...")
    
    total_inference_time = 0.0
    
    # Supported labels in OntoNotes 5.0 dataset
    # We will use this set for the full label_confidences dictionary
    ontonotes_labels = [
        "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", 
        "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT", 
        "MONEY", "QUANTITY", "ORDINAL", "CARDINAL", "O"
    ]
    
    for idx, sample in enumerate(samples):
        text = sample.get("text", "")
        print(f"\n[{idx+1}/{len(samples)}] Text: '{text}'")
        
        # Run HanLP model
        start_time = time.time()
        doc = HanLP(text)
        inference_time = time.time() - start_time
        total_inference_time += inference_time
        
        # Identify the NER prediction key in document output (usually 'ner' or 'ner/fine')
        ner_key = None
        for k in doc.keys():
            if "ner" in k:
                ner_key = k
                break
                
        predicted_entities = []
        if ner_key and doc[ner_key]:
            tokens = doc.get("tok", doc.get("tok/fine", []))
            raw_ner = doc[ner_key]
            
            # Align tokens to find character offsets
            char_spans = align_tokens_to_chars(text, tokens)
            
            for ent in raw_ner:
                # HanLP entities format: [entity_text, entity_label, start_token_idx, end_token_idx]
                if len(ent) == 4:
                    word, label, start_tok, end_tok = ent
                    
                    # Convert token index to character offset
                    try:
                        start_char = char_spans[start_tok][0]
                        end_char = char_spans[end_tok - 1][1]
                    except IndexError:
                        # Fallback direct search in text
                        start_char = text.find(word)
                        end_char = start_char + len(word) if start_char != -1 else 0
                else:
                    # Fallback format
                    word, label = ent[0], ent[1]
                    start_char = text.find(word)
                    end_char = start_char + len(word) if start_char != -1 else 0
                
                # Build label_confidences dictionary
                # Since standard HanLP uses CRF decoding and does not expose probabilities natively,
                # we assign a probability of 1.0 to the predicted class and 0.0 to all others.
                label_confidences = {lbl: 0.0 for lbl in ontonotes_labels}
                if label in label_confidences:
                    label_confidences[label] = 1.0
                else:
                    # In case of custom/unseen labels
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
                
        # Add predictions to the sample dictionary
        sample["hanlp_predicted_entities"] = predicted_entities
        
        # Display results
        print("  Original labels:")
        if "entities" in sample:
            for orig in sample.get("entities", []):
                print(f"    - '{orig.get('text', '')}' -> {orig.get('label', '')} ({orig.get('start', '?')}:{orig.get('end', '?')})")
        elif "label" in sample:
            print(f"    - '{sample.get('text', '')}' -> {sample.get('label', '')}")
        print("  HanLP predictions:")
        for pred in predicted_entities:
            print(f"    - '{pred['word']}' -> {pred['entity_group']} ({pred['start']}:{pred['end']}) | score: {pred['score']}")
            
    # Save the updated JSON file
    if samples:
        avg_time = total_inference_time / len(samples)
        print(f"\nAverage Inference Time per sample: {avg_time:.4f} seconds")
        print(f"Total Inference Time: {total_inference_time:.4f} seconds")

    print(f"\nSaving updated dataset to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print("Done!")

if __name__ == "__main__":
    main()

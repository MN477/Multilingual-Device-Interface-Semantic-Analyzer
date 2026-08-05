import json
import time
from gliner import GLiNER

# =============================================================================
# Semantic Role Labels for User vs System Classification
# =============================================================================
# These labels are designed to extract entities whose presence directly signals
# whether a text block is USER-generated content or SYSTEM/UI-generated content.
#
# USER-SIGNAL LABELS (indicate user-specific personal data):
#   - "person name"       → driver, contact, recipient, sender
#   - "phone number"      → personal phone, contact number
#   - "address"           → pickup/dropoff, home/work address
#   - "destination"       → navigation target chosen by user
#   - "payment amount"    → user transaction value
#   - "payment method"    → credit card, bank account
#   - "account number"    → masked card/account numbers
#   - "song title"        → user-selected media
#   - "artist name"       → music artist from user library
#   - "transaction id"    → user's transaction reference
#
# SYSTEM-SIGNAL LABELS (indicate system/UI-generated text):
#   - "ui label"          → field prefixes like "Pickup:", "Amount:", "ETA:"
#   - "system status"     → "Navigating to", "Now Playing", "Incoming Call"
#   - "date"              → timestamps often system-generated
#   - "time"              → time displays from system clock
#   - "speed"             → vehicle telemetry
#   - "temperature"       → cabin/environment sensor data
#   - "distance"          → navigation computed metric
#   - "organization"      → app/service provider name
#   - "support number"    → system-provided helpline

# Labels split into two groups for downstream user/system classification
USER_SIGNAL_LABELS = [
    "person name ",
    "phone number",
    "location",
    "email",
    "song title",
    "artist name",
]

SYSTEM_SIGNAL_LABELS = [
    "date",
    "time",
    "speed",
    "temperature",
    "distance",
    "organization",

]

# Classification mapping: which label group each label belongs to
LABEL_TO_ROLE = {}
for label in USER_SIGNAL_LABELS:
    LABEL_TO_ROLE[label] = "user"
for label in SYSTEM_SIGNAL_LABELS:
    LABEL_TO_ROLE[label] = "system"

ALL_LABELS = USER_SIGNAL_LABELS + SYSTEM_SIGNAL_LABELS


def classify_block_role(entities, user_weight=1.0, system_weight=1.0):
    """
    Classify a text block as 'user', 'system', or 'mixed' based on the
    detected entities and their confidence scores.
    
    Each entity votes for user or system based on its label group,
    weighted by its prediction score.
    """
    user_score = 0.0
    system_score = 0.0
    
    for ent in entities:
        role = LABEL_TO_ROLE.get(ent.get("label"), None)
        score = ent.get("score", 0.0)
        if role == "user":
            user_score += score * user_weight
        elif role == "system":
            system_score += score * system_weight
    
    total = user_score + system_score
    if total == 0:
        return "unknown", 0.0, 0.0
    
    if user_score > 0 and system_score > 0:
        role = "mixed"
    elif user_score > system_score:
        role = "user"
    else:
        role = "system"
    
    return role, user_score, system_score


def test_gliner_bi_model(input_json_path="NER_real_dataset.json", output_json_path="NER_dataset_gliner_bi_entities.json", threshold=0.3):
    model_name = "knowledgator/gliner-bi-large-v2.0"
    
    print(f"Loading GLiNER Bi-Encoder model: {model_name}...")
    model = GLiNER.from_pretrained(model_name)
    
    print(f"Using {len(ALL_LABELS)} domain-specific labels with threshold={threshold}")
    print(f"  User-signal labels:   {USER_SIGNAL_LABELS}")
    print(f"  System-signal labels: {SYSTEM_SIGNAL_LABELS}")
    
    print(f"\nReading input dataset from {input_json_path}...")
    with open(input_json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    print("Extracting entities for text blocks...\n")
    images = dataset.get("images", [])
    total_blocks_processed = 0
    inference_times = []
    
    for img in images:
        print(f"--- Image: {img.get('image_name', 'unknown')} ---")
        for block in img.get("text_blocks", []):
            text = block.get("transcribed_text", "")
            if text.strip():
                # Perform zero-shot NER inference and measure execution time
                start_time = time.perf_counter()
                entities = model.predict_entities(text, ALL_LABELS, threshold=threshold)
                elapsed_time = time.perf_counter() - start_time
                inference_times.append(elapsed_time)
                
                formatted_entities = []
                for entity in entities:
                    formatted_entities.append({
                        "text": entity.get("text"),
                        "label": entity.get("label"),
                        "role": LABEL_TO_ROLE.get(entity.get("label"), "unknown"),
                        "score": float(entity.get("score", 0.0)),
                        "start": int(entity.get("start")),
                        "end": int(entity.get("end"))
                    })
                
                block["entities"] = formatted_entities
                
                # Classify the block role based on aggregated entity signals
                role, user_score, system_score = classify_block_role(formatted_entities)
                block["predicted_role"] = role
                block["role_scores"] = {
                    "user_score": round(user_score, 4),
                    "system_score": round(system_score, 4)
                }
                
                # Print per-block summary
                ent_summary = [(e["label"], e["text"], f"{e['score']:.2f}") for e in formatted_entities]
                print(f"  Block {block.get('block_id')}: \"{text}\"")
                print(f"    Entities: {ent_summary}")
                print(f"    Role: {role} (user={user_score:.2f}, system={system_score:.2f})")
                print(f"    Inference: {elapsed_time*1000:.1f} ms")
            else:
                block["entities"] = []
                block["predicted_role"] = "unknown"
                block["role_scores"] = {"user_score": 0.0, "system_score": 0.0}
                
            total_blocks_processed += 1
        print()
            
    avg_inference_time = (sum(inference_times) / len(inference_times)) if inference_times else 0.0
    total_inference_time = sum(inference_times)
    
    print("=" * 60)
    print(f"Processed {total_blocks_processed} text blocks across {len(images)} images.")
    print(f"Total Inference Time: {total_inference_time:.4f} seconds")
    print(f"Average Inference Time per text block: {avg_inference_time * 1000:.2f} ms ({avg_inference_time:.6f} s)")
    print("=" * 60)
    
    print(f"\nSaving updated dataset with GLiNER Bi-Encoder entities to {output_json_path}...")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
        
    print("GLiNER Bi-Encoder testing completed successfully!")

if __name__ == "__main__":
    test_gliner_bi_model()

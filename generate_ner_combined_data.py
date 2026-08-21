"""
Generate BIO-tagged NER training data combining BOTH Apps and Songs
for fine-tuning Davlan/xlm-roberta-base-ner-hrl.

Label schema:
  0: O
  1: B-PER,  2: I-PER
  3: B-ORG,  4: I-ORG
  5: B-LOC,  6: I-LOC
  7: B-DATE, 8: I-DATE
  9: B-SONG, 10: I-SONG
  11: B-APP, 12: I-APP

Output: train.json and val.json in HuggingFace-compatible format:
  [{"id": "0", "tokens": [...], "ner_tags": [...]}, ...]
"""

import json
import random
import os
import unicodedata

random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────
SONG_DATA_PATH = r"c:\Users\mouss\Desktop\semantic_classif\song_artist_data.json"
APP_DATA_PATH = r"c:\Users\mouss\Desktop\semantic_classif\android_apps_names.json"
OUTPUT_DIR = r"c:\Users\mouss\Desktop\semantic_classif\ner_combined_data"

# ── Label Map ──────────────────────────────────────────────────────────
LABEL2ID = {
    "O": 0,
    "B-PER": 1, "I-PER": 2,
    "B-ORG": 3, "I-ORG": 4,
    "B-LOC": 5, "I-LOC": 6,
    "B-DATE": 7, "I-DATE": 8,
    "B-SONG": 9, "I-SONG": 10,
    "B-APP": 11, "I-APP": 12,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# ── Templates ──────────────────────────────────────────────────────────

# Combined App + Song + Artist
TEMPLATES_APP_SONG_ARTIST = [
    "Now playing {song} by {artist} on {app}",
    "Listening to {song} by {artist} using {app}",
    "{app} notification: {artist} just dropped {song}",
    "Open {app} and play {song} by {artist}",
    "Stream {song} by {artist} on {app}",
    "{song} by {artist} is currently playing on {app}",
    "Played {song} from {artist} via {app}",
    "Queue {song} by {artist} in {app}",
]

# App + Song
TEMPLATES_APP_SONG = [
    "Now playing {song} on {app}",
    "Listen to {song} on {app}",
    "Open {app} to play {song}",
    "Stream {song} via {app}",
    "Search for {song} in {app}",
    "{song} playing on {app}",
    "Playing {song} using {app}",
]

# App + Artist
TEMPLATES_APP_ARTIST = [
    "Follow {artist} on {app}",
    "Listen to {artist} on {app}",
    "Open {app} to view {artist}",
    "{artist} top tracks on {app}",
    "Search for {artist} on {app}",
]

# App only (UI screens, notifications, system actions)
TEMPLATES_APP_ONLY = [
    "Open {app}",
    "Launch {app}",
    "Close {app}",
    "Switch to {app}",
    "Download {app}",
    "Install {app}",
    "Update {app}",
    "Uninstall {app}",
    "{app} is not responding",
    "{app} notification",
    "Welcome to {app}",
    "Sign in to {app}",
    "{app} settings",
    "Share via {app}",
    "Connected to {app}",
    "Allow {app} to send notifications",
    "{app}",
]

# Song + Artist
TEMPLATES_SONG_ARTIST = [
    # Now Playing / Music Player UI
    "Now playing {song} by {artist}",
    "Currently playing {song} by {artist}",
    "{song} by {artist} is now playing",
    "Playing {song} from {artist}",
    "Listening to {song} by {artist}",
    "Up next {song} by {artist}",
    "{artist} performing {song}",
    "{song} {artist}",
    "{artist} {song}",

    # Notification style
    "{artist} released {song}",
    "{artist} just dropped {song}",
    "New release {song} by {artist}",
    "Check out {song} by {artist}",
    "Trending now {song} by {artist}",
    "Top hit {song} by {artist}",

    # Search / Query style
    "Search results for {song} by {artist}",
    "Did you mean {song} by {artist}",
    "Showing results for {song} {artist}",

    # Social / Chat style
    "Have you heard {song} by {artist}",
    "I love {song} by {artist}",
    "My favorite song is {song} by {artist}",
    "Just discovered {song} by {artist}",
    "You should listen to {song} by {artist}",
    "{song} by {artist} is amazing",
    "Best track from {artist} is {song}",

    # Playlist style
    "Added {song} by {artist} to playlist",
    "Playlist includes {song} by {artist}",
    "Queue {song} by {artist}",
    "Recommended {song} by {artist}",

    # OCR / Screen capture style (short fragments)
    "{song} {artist} 3:45",
    "{artist} {song} Album",
    "{song} by {artist} Official Audio",
    "{song} {artist} Music Video",
    "{song} {artist} Lyrics",
    "{artist} {song} Live",
    "{song} feat {artist}",
    "{artist} ft {song}",
]

# Song only
TEMPLATES_SONG_ONLY = [
    "Now playing {song}",
    "Currently listening to {song}",
    "{song} is playing",
    "Up next {song}",
    "Search for {song}",
    "Download {song}",
    "Added {song} to favorites",
    "{song} Official Audio",
    "{song} Lyrics",
    "{song} Music Video",
    "Trending {song}",
    "Top hit {song}",
    "Playing {song} on repeat",
    "{song}",
    "Queue {song}",
    "Share {song}",
    "Save {song}",
]


def tokenize_simple(text: str) -> list:
    """Split text into word-level tokens, preserving punctuation as separate tokens."""
    raw_tokens = text.split()
    tokens = []
    for tok in raw_tokens:
        while tok and not tok[0].isalnum() and len(tok) > 1:
            tokens.append(tok[0])
            tok = tok[1:]
        trail = []
        while tok and not tok[-1].isalnum() and len(tok) > 1:
            trail.append(tok[-1])
            tok = tok[:-1]
        if tok:
            tokens.append(tok)
        for t in reversed(trail):
            tokens.append(t)
    return tokens


def assign_bio_tags(tokens: list, entity_tokens_list: list) -> list:
    """Assign BIO tags to tokens given a list of (entity_tokens, label_name) tuples."""
    tags = ["O"] * len(tokens)
    used = [False] * len(tokens)
    tokens_lower = [t.lower() for t in tokens]
    
    # Sort entities by length descending to match longer entities first
    entity_tokens_list = sorted(entity_tokens_list, key=lambda x: len(x[0]), reverse=True)
    
    for entity_tokens, label in entity_tokens_list:
        entity_lower = [t.lower() for t in entity_tokens]
        elen = len(entity_lower)
        if elen == 0:
            continue
        
        for i in range(len(tokens_lower) - elen + 1):
            if any(used[i:i+elen]):
                continue
            if tokens_lower[i:i+elen] == entity_lower:
                for j in range(i, i + elen):
                    used[j] = True
                tags[i] = f"B-{label}"
                for j in range(i + 1, i + elen):
                    tags[j] = f"I-{label}"
                break
    return tags


def create_example(template: str, replacements: dict, entity_mapping: dict) -> dict:
    """
    Generic example builder.
    replacements: {"song": "Hotel California", "artist": "Eagles", "app": "Spotify"}
    entity_mapping: {"song": "SONG", "artist": "PER", "app": "APP"}
    """
    sentence = template
    for key, val in replacements.items():
        sentence = sentence.replace("{" + key + "}", val)
        
    tokens = tokenize_simple(sentence)
    if not tokens:
        return None
        
    entities = []
    for key, label in entity_mapping.items():
        if "{" + key + "}" in template and key in replacements:
            val_tokens = tokenize_simple(replacements[key])
            if val_tokens:
                entities.append((val_tokens, label))
                
    bio_tags = assign_bio_tags(tokens, entities)
    ner_tags = [LABEL2ID.get(t, 0) for t in bio_tags]
    
    if all(t == 0 for t in ner_tags):
        return None
        
    return {"tokens": tokens, "ner_tags": ner_tags}


def main():
    print("Loading song and app datasets...")
    
    with open(SONG_DATA_PATH, 'r', encoding='utf-8') as f:
        songs = json.load(f)
        
    with open(APP_DATA_PATH, 'r', encoding='utf-8') as f:
        apps = json.load(f)
        
    print(f"Loaded {len(songs)} song records and {len(apps)} app records.")
    
    # Filter valid items
    valid_songs = []
    for s in songs:
        track = (s.get("track_name") or "").strip()
        artist = (s.get("artist") or "").strip()
        if track and artist and len(track) >= 2 and len(artist) >= 2:
            valid_songs.append((track, artist))
            
    valid_apps = []
    for a in apps:
        name = (a.get("name") or "").strip()
        aliases = a.get("aliases") or []
        if name and len(name) >= 2:
            valid_apps.append(name)
            # Add aliases as app variants too
            for al in aliases:
                if len(al) >= 2:
                    valid_apps.append(al)
                    
    print(f"Valid songs: {len(valid_songs)}, Valid app names & aliases: {len(valid_apps)}")
    
    # Target dataset size: ~200,000 examples
    all_examples = []
    
    # 1. Combined App + Song + Artist examples (50k)
    print("Generating App + Song + Artist examples...")
    for _ in range(50000):
        track, artist = random.choice(valid_songs)
        app_name = random.choice(valid_apps)
        tmpl = random.choice(TEMPLATES_APP_SONG_ARTIST)
        ex = create_example(tmpl, {"song": track, "artist": artist, "app": app_name},
                            {"song": "SONG", "artist": "PER", "app": "APP"})
        if ex:
            all_examples.append(ex)
            
    # 2. App + Song examples (40k)
    print("Generating App + Song examples...")
    for _ in range(40000):
        track, _ = random.choice(valid_songs)
        app_name = random.choice(valid_apps)
        tmpl = random.choice(TEMPLATES_APP_SONG)
        ex = create_example(tmpl, {"song": track, "app": app_name},
                            {"song": "SONG", "app": "APP"})
        if ex:
            all_examples.append(ex)

    # 3. App + Artist examples (20k)
    print("Generating App + Artist examples...")
    for _ in range(20000):
        _, artist = random.choice(valid_songs)
        app_name = random.choice(valid_apps)
        tmpl = random.choice(TEMPLATES_APP_ARTIST)
        ex = create_example(tmpl, {"artist": artist, "app": app_name},
                            {"artist": "PER", "app": "APP"})
        if ex:
            all_examples.append(ex)

    # 4. App only examples (40k)
    print("Generating App-only examples...")
    for _ in range(40000):
        app_name = random.choice(valid_apps)
        tmpl = random.choice(TEMPLATES_APP_ONLY)
        ex = create_example(tmpl, {"app": app_name}, {"app": "APP"})
        if ex:
            all_examples.append(ex)

    # 5. Song + Artist examples (30k)
    print("Generating Song + Artist examples...")
    for _ in range(30000):
        track, artist = random.choice(valid_songs)
        tmpl = random.choice(TEMPLATES_SONG_ARTIST)
        ex = create_example(tmpl, {"song": track, "artist": artist},
                            {"song": "SONG", "artist": "PER"})
        if ex:
            all_examples.append(ex)

    # 6. Song only examples (20k)
    print("Generating Song-only examples...")
    for _ in range(20000):
        track, _ = random.choice(valid_songs)
        tmpl = random.choice(TEMPLATES_SONG_ONLY)
        ex = create_example(tmpl, {"song": track}, {"song": "SONG"})
        if ex:
            all_examples.append(ex)

    print(f"Total generated examples: {len(all_examples)}")
    
    # Shuffle and add IDs
    random.shuffle(all_examples)
    for i, ex in enumerate(all_examples):
        ex["id"] = str(i)
        
    # Split 90% train, 10% val
    split_idx = int(len(all_examples) * 0.9)
    train_data = all_examples[:split_idx]
    val_data = all_examples[split_idx:]
    
    # Save output files
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    train_path = os.path.join(OUTPUT_DIR, "train.json")
    val_path = os.path.join(OUTPUT_DIR, "val.json")
    label_map_path = os.path.join(OUTPUT_DIR, "label_map.json")
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
        
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
        
    with open(label_map_path, 'w', encoding='utf-8') as f:
        json.dump({"label2id": LABEL2ID, "id2label": ID2LABEL}, f, indent=2)
        
    print(f"Saved combined NER dataset to {OUTPUT_DIR}:")
    print(f"  train.json: {len(train_data)} examples")
    print(f"  val.json:   {len(val_data)} examples")
    print(f"  label_map.json")

if __name__ == "__main__":
    main()

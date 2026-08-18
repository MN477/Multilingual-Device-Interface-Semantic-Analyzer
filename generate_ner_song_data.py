"""
Generate BIO-tagged NER training data from song_artist_data.json
for fine-tuning Davlan/xlm-roberta-base-ner-hrl.

Label schema (extended from the original model):
  0: O
  1: B-PER,  2: I-PER
  3: B-ORG,  4: I-ORG
  5: B-LOC,  6: I-LOC
  7: B-DATE, 8: I-DATE
  9: B-SONG, 10: I-SONG

Output: train.json and val.json in HuggingFace-compatible format:
  [{"id": "0", "tokens": [...], "ner_tags": [...]}, ...]
"""

import json
import random
import re
import os

random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────
SONG_DATA_PATH = r"c:\Users\mouss\Desktop\semantic_classif\song_artist_data.json"
OUTPUT_DIR = r"c:\Users\mouss\Desktop\semantic_classif\ner_song_data"

# ── Label Map ──────────────────────────────────────────────────────────
LABEL2ID = {
    "O": 0,
    "B-PER": 1, "I-PER": 2,
    "B-ORG": 3, "I-ORG": 4,
    "B-LOC": 5, "I-LOC": 6,
    "B-DATE": 7, "I-DATE": 8,
    "B-SONG": 9, "I-SONG": 10,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# ── Sentence Templates ────────────────────────────────────────────────
# {song} = song title placeholder, {artist} = artist placeholder
# Templates are grouped by pattern type for diversity

TEMPLATES_SONG_AND_ARTIST = [
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

TEMPLATES_ARTIST_ONLY = [
    "Songs by {artist}",
    "More from {artist}",
    "Top tracks by {artist}",
    "{artist} discography",
    "All songs by {artist}",
    "Popular tracks from {artist}",
    "Listen to {artist}",
    "{artist} playlist",
    "Follow {artist}",
    "Artist {artist}",
    "{artist}",
    "Recommended artist {artist}",
    "Similar to {artist}",
    "Fans of {artist} also like",
]


def tokenize_simple(text: str) -> list:
    """Split text into word-level tokens, preserving punctuation as separate tokens."""
    raw_tokens = text.split()
    tokens = []
    for tok in raw_tokens:
        # Separate leading punctuation
        while tok and not tok[0].isalnum() and len(tok) > 1:
            tokens.append(tok[0])
            tok = tok[1:]
        # Separate trailing punctuation
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
    """
    Assign BIO tags to a flat token list given a list of (entity_tokens, label) tuples.
    
    entity_tokens_list: [(["Hotel", "California"], "SONG"), (["Eagles"], "PER")]
    
    Uses greedy left-to-right matching to find entity spans.
    """
    tags = ["O"] * len(tokens)
    used = [False] * len(tokens)
    
    # Normalize tokens for matching (lowercase)
    tokens_lower = [t.lower() for t in tokens]
    
    for entity_tokens, label in entity_tokens_list:
        entity_lower = [t.lower() for t in entity_tokens]
        elen = len(entity_lower)
        
        # Find the entity span in the token list
        for i in range(len(tokens_lower) - elen + 1):
            if used[i]:
                continue
            if tokens_lower[i:i+elen] == entity_lower:
                # Mark as used
                for j in range(i, i + elen):
                    used[j] = True
                # Assign BIO tags
                tags[i] = f"B-{label}"
                for j in range(i + 1, i + elen):
                    tags[j] = f"I-{label}"
                break  # Only tag the first occurrence
    
    return tags


def generate_song_artist_example(song_name: str, artist_name: str, template: str) -> dict:
    """Generate a single BIO-tagged example from a template with song and artist."""
    sentence = template.replace("{song}", song_name).replace("{artist}", artist_name)
    
    tokens = tokenize_simple(sentence)
    if not tokens:
        return None
    
    song_tokens = tokenize_simple(song_name)
    artist_tokens = tokenize_simple(artist_name)
    
    entities = []
    if "{song}" in template:
        entities.append((song_tokens, "SONG"))
    if "{artist}" in template:
        entities.append((artist_tokens, "PER"))
    
    bio_tags = assign_bio_tags(tokens, entities)
    ner_tags = [LABEL2ID.get(tag, 0) for tag in bio_tags]
    
    if all(t == 0 for t in ner_tags):
        return None
    
    return {"tokens": tokens, "ner_tags": ner_tags}


def generate_song_only_example(song_name: str, template: str) -> dict:
    """Generate a BIO-tagged example with only a song entity."""
    sentence = template.replace("{song}", song_name)
    tokens = tokenize_simple(sentence)
    if not tokens:
        return None
    
    song_tokens = tokenize_simple(song_name)
    entities = [(song_tokens, "SONG")]
    bio_tags = assign_bio_tags(tokens, entities)
    ner_tags = [LABEL2ID.get(tag, 0) for tag in bio_tags]
    
    if all(t == 0 for t in ner_tags):
        return None
    
    return {"tokens": tokens, "ner_tags": ner_tags}


def generate_artist_only_example(artist_name: str, template: str) -> dict:
    """Generate a BIO-tagged example with only an artist (PER) entity."""
    sentence = template.replace("{artist}", artist_name)
    tokens = tokenize_simple(sentence)
    if not tokens:
        return None
    
    artist_tokens = tokenize_simple(artist_name)
    entities = [(artist_tokens, "PER")]
    bio_tags = assign_bio_tags(tokens, entities)
    ner_tags = [LABEL2ID.get(tag, 0) for tag in bio_tags]
    
    if all(t == 0 for t in ner_tags):
        return None
    
    return {"tokens": tokens, "ner_tags": ner_tags}


def main():
    # ── Load song data ─────────────────────────────────────────────────
    print("Loading song data...")
    with open(SONG_DATA_PATH, 'r', encoding='utf-8') as f:
        songs = json.load(f)
    
    print(f"Loaded {len(songs)} songs.")
    
    # Filter out songs with very short or empty names
    valid_songs = []
    for s in songs:
        track = (s.get("track_name") or "").strip()
        artist = (s.get("artist") or "").strip()
        if track and artist and len(track) >= 2 and len(artist) >= 2:
            valid_songs.append(s)
    
    print(f"Valid songs (after filtering): {len(valid_songs)}")
    
    # ── Sample songs ───────────────────────────────────────────────────
    # Generate ~3 examples per song entry
    # Cap at ~150k total examples for practical training
    MAX_EXAMPLES = 150000
    examples_per_song = 3
    max_songs_to_use = MAX_EXAMPLES // examples_per_song
    
    if len(valid_songs) > max_songs_to_use:
        sampled_songs = random.sample(valid_songs, max_songs_to_use)
        print(f"Sampled {max_songs_to_use} songs for dataset generation.")
    else:
        sampled_songs = valid_songs
        print(f"Using all {len(sampled_songs)} songs.")
    
    # ── Generate examples ──────────────────────────────────────────────
    print("Generating BIO-tagged examples...")
    all_examples = []
    skipped = 0
    
    for song_entry in sampled_songs:
        track = song_entry["track_name"].strip()
        artist = song_entry["artist"].strip()
        
        # 1. Song + Artist template
        template = random.choice(TEMPLATES_SONG_AND_ARTIST)
        example = generate_song_artist_example(track, artist, template)
        if example:
            all_examples.append(example)
        else:
            skipped += 1
        
        # 2. Song-only template
        template = random.choice(TEMPLATES_SONG_ONLY)
        example = generate_song_only_example(track, template)
        if example:
            all_examples.append(example)
        else:
            skipped += 1
        
        # 3. Artist-only template
        template = random.choice(TEMPLATES_ARTIST_ONLY)
        example = generate_artist_only_example(artist, template)
        if example:
            all_examples.append(example)
        else:
            skipped += 1
    
    print(f"Generated {len(all_examples)} examples. Skipped {skipped} (failed alignment).")
    
    # ── Shuffle and split ──────────────────────────────────────────────
    random.shuffle(all_examples)
    
    # Add IDs
    for i, ex in enumerate(all_examples):
        ex["id"] = str(i)
    
    # 90/10 train/val split
    split_idx = int(len(all_examples) * 0.9)
    train_data = all_examples[:split_idx]
    val_data = all_examples[split_idx:]
    
    print(f"Train: {len(train_data)} examples, Val: {len(val_data)} examples.")
    
    # ── Save ───────────────────────────────────────────────────────────
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
    
    print(f"\nSaved to {OUTPUT_DIR}:")
    print(f"  train.json  ({len(train_data)} examples)")
    print(f"  val.json    ({len(val_data)} examples)")
    print(f"  label_map.json")
    
    # ── Print samples ──────────────────────────────────────────────────
    print("\n── Sample Examples ──────────────────────────────────────────")
    for ex in train_data[:5]:
        tokens = ex["tokens"]
        tags = [ID2LABEL[t] for t in ex["ner_tags"]]
        print(f"\n  Sentence: {' '.join(tokens)}")
        print(f"  Tags:     {' '.join(tags)}")
        entities = []
        i = 0
        while i < len(tags):
            if tags[i].startswith("B-"):
                label = tags[i][2:]
                entity_tokens = [tokens[i]]
                j = i + 1
                while j < len(tags) and tags[j] == f"I-{label}":
                    entity_tokens.append(tokens[j])
                    j += 1
                entities.append(f"[{' '.join(entity_tokens)}] -> {label}")
                i = j
            else:
                i += 1
        if entities:
            print(f"  Entities: {', '.join(entities)}")
    
    # ── Validation checks ─────────────────────────────────────────────
    print("\n── Validation ──────────────────────────────────────────────")
    misaligned = 0
    for ex in all_examples:
        if len(ex["tokens"]) != len(ex["ner_tags"]):
            misaligned += 1
    print(f"  Token-tag alignment errors: {misaligned}")
    
    from collections import Counter
    tag_counter = Counter()
    for ex in all_examples:
        for t in ex["ner_tags"]:
            tag_counter[ID2LABEL[t]] += 1
    
    print(f"  Label distribution:")
    for label, count in sorted(tag_counter.items(), key=lambda x: -x[1]):
        print(f"    {label:8s}: {count:>8,}")


if __name__ == "__main__":
    main()

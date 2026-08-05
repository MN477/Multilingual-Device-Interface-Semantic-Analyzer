"""Quick validation script: loads regex_patterns.json, compiles every pattern, tests against examples."""
import json, re, sys

with open("regex_patterns.json", "r", encoding="utf-8") as f:
    data = json.load(f)

total_patterns = 0
errors = []
stats = {"system": 0, "user": 0}

for cat in data["categories"]:
    for pat in cat["patterns"]:
        total_patterns += 1
        try:
            flags = re.IGNORECASE if pat.get("case_insensitive", False) else 0
            compiled = re.compile(pat["regex"], flags)
            for ex in pat.get("examples", []):
                match = compiled.search(ex)
                if not match:
                    errors.append(f"  MISS: {pat['id']} regex failed to match example '{ex}'")
            stats[pat["signal"]] = stats.get(pat["signal"], 0) + 1
        except re.error as e:
            errors.append(f"  COMPILE ERROR: {pat['id']} -> {e}")

print(f"Categories: {len(data['categories'])}")
print(f"Total patterns: {total_patterns}")
print(f"Signal breakdown: {stats}")
print(f"Errors: {len(errors)}")
for err in errors:
    print(err)

if not errors:
    print("\nALL PATTERNS COMPILE AND MATCH THEIR EXAMPLES SUCCESSFULLY")

# Now test against the user's real data
print("\n" + "=" * 60)
print("TESTING AGAINST REAL OCR DATA")
print("=" * 60)

test_files = [
    "NER_real_dataset.json",
    "application_android_auto.json",
    "NER_dataset.json",
]

compiled_patterns = []
for cat in data["categories"]:
    for pat in cat["patterns"]:
        flags = re.IGNORECASE if pat.get("case_insensitive", False) else 0
        compiled_patterns.append({
            "compiled": re.compile(pat["regex"], flags),
            "id": pat["id"],
            "entity_type": pat["entity_type"],
            "signal": pat["signal"],
            "confidence": pat["confidence"],
            "category": cat["category"],
        })

for test_file in test_files:
    try:
        with open(test_file, "r", encoding="utf-8") as f:
            ds = json.load(f)
    except FileNotFoundError:
        continue

    print(f"\n--- {test_file} ---")
    for img in ds.get("images", []):
        for block in img.get("text_blocks", []):
            text = block.get("transcribed_text", "")
            if not text.strip():
                continue
            matches = []
            for p in compiled_patterns:
                m = p["compiled"].search(text)
                if m:
                    matches.append({
                        "pattern": p["id"],
                        "entity": p["entity_type"],
                        "signal": p["signal"],
                        "matched": m.group(),
                        "confidence": p["confidence"],
                    })
            if matches:
                best = max(matches, key=lambda x: x["confidence"])
                print(f"  [{best['signal'].upper():6s}] \"{text}\"")
                for mt in matches:
                    print(f"           -> {mt['pattern']}: '{mt['matched']}' ({mt['entity']}, signal={mt['signal']}, conf={mt['confidence']})")
            else:
                print(f"  [  --  ] \"{text}\" (no regex match)")

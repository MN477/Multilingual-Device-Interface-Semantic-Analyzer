import json, os

# Reload and check
with open('app_lookup.json', 'r', encoding='utf-8') as f:
    lookup = json.load(f)

with open('app_search_index.json', 'r', encoding='utf-8') as f:
    index = json.load(f)

print(f"Lookup entries: {len(lookup)}")
print(f"Search index terms: {len(index)}")
print(f"Entries with aliases: {sum(1 for e in lookup if e.get('aliases'))}")
print(f"Total aliases: {sum(len(e.get('aliases',[])) for e in lookup)}")

raw_size = os.path.getsize('android_apps_names.json')
lookup_size = os.path.getsize('app_lookup.json')
index_size = os.path.getsize('app_search_index.json')
print(f"\nRaw:    {raw_size/1024/1024:.1f} MB")
print(f"Lookup: {lookup_size/1024:.0f} KB")
print(f"Index:  {index_size/1024:.0f} KB")

# Show sample entries  
print("\n--- Sample entries with aliases ---")
aliased = [e for e in lookup if e.get('aliases')]
for e in aliased[:15]:
    print(f"  {e['name']:30s} aliases: {e['aliases']}")

import json
import os
from pathlib import Path

# Directory containing the i18n language files
languages_dir = Path("assets/i18n/languages")

# List of theme-related keys to remove
theme_keys = [
    "Theme",
    "Theme Mode", 
    "Theme Settings",
    "Theme settings have been saved.",
    "Save Theme Settings",
    "Select the theme mode for the application."
]

# Process each JSON file in the languages directory
for json_file in languages_dir.glob("*.json"):
    print(f"Processing {json_file.name}...")
    
    # Load the JSON file
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Remove theme-related keys
    original_length = len(data)
    for key in theme_keys:
        if key in data:
            del data[key]
            print(f"  Removed: {key}")
    
    # Write the updated JSON back to the file
    with open(json_file, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, sort_keys=True)
        f.write('\n')  # Add final newline
    
    print(f"  Updated {json_file.name}: {original_length} -> {len(data)} entries")
    
print("All theme-related entries have been removed from i18n files.")
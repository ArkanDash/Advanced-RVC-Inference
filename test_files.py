import os
import json

# Test that files can be loaded correctly with utf-8 encoding
language_files = [
    "assets/i18n/languages/vi_VN.json",
    "assets/i18n/languages/th_TH.json", 
    "assets/i18n/languages/uk_UA.json"
]

for file_path in language_files:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"SUCCESS: Successfully loaded {file_path} - {len(data)} entries")
    except Exception as e:
        print(f"ERROR: Error loading {file_path}: {e}")

# Also check existing files
existing_files = [
    "assets/i18n/languages/en_US.json",
    "assets/i18n/languages/es_ES.json"
]

for file_path in existing_files:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"SUCCESS: Successfully loaded {file_path} - {len(data)} entries")
    except Exception as e:
        print(f"ERROR: Error loading {file_path}: {e}")
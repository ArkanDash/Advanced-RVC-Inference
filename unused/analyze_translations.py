import os
import json

def analyze_translations():
    languages_dir = os.path.join("assets", "i18n", "languages")
    english_file = os.path.join(languages_dir, "en_US.json")
    
    # Load English base file
    with open(english_file, "r", encoding="utf-8") as f:
        english_data = json.load(f)
    
    total_keys = len(english_data)
    print(f"Total keys in English file: {total_keys}")
    
    # Get all language files
    language_files = [f for f in os.listdir(languages_dir) if f.endswith(".json")]
    
    # Analyze each language
    results = {}
    for lang_file in language_files:
        lang_code = lang_file.replace(".json", "")
        if lang_code == "en_US":
            continue
            
        file_path = os.path.join(languages_dir, lang_file)
        with open(file_path, "r", encoding="utf-8") as f:
            lang_data = json.load(f)
        
        # Count translated entries (where key != value)
        translated_count = 0
        untranslated_entries = []
        
        for key, value in lang_data.items():
            if key != value:
                translated_count += 1
            else:
                untranslated_entries.append(key)
        
        results[lang_code] = {
            "translated": translated_count,
            "total": total_keys,
            "percentage": (translated_count / total_keys) * 100,
            "untranslated": untranslated_entries
        }
        
        print(f"\n{lang_code}:")
        print(f"  Translated: {translated_count}/{total_keys} ({results[lang_code]['percentage']:.1f}%)")
        if untranslated_entries:
            print(f"  Untranslated entries: {len(untranslated_entries)}")
    
    return results

if __name__ == "__main__":
    analyze_translations()
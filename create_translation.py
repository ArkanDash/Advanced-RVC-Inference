import os
import json

def create_complete_translation(source_lang="en_US", target_lang="es_ES", target_lang_name="Spanish"):
    """
    Create a complete translation by copying the English file and translating all values
    """
    languages_dir = os.path.join("assets", "i18n", "languages")
    source_file = os.path.join(languages_dir, f"{source_lang}.json")
    target_file = os.path.join(languages_dir, f"{target_lang}.json")
    
    # Load source language file
    with open(source_file, "r", encoding="utf-8") as f:
        source_data = json.load(f)
    
    # Load target language file if it exists
    if os.path.exists(target_file):
        with open(target_file, "r", encoding="utf-8") as f:
            target_data = json.load(f)
    else:
        target_data = {}
    
    # Create translation mapping (this would normally come from an actual translation service)
    # For demonstration, we'll create a simple mapping
    translation_map = {
        "Download": {"es_ES": "Descargar", "fr_FR": "Télécharger", "de_DE": "Herunterladen"},
        "Model URL": {"es_ES": "URL del Modelo", "fr_FR": "URL du Modèle", "de_DE": "Modell-URL"},
        "Output Information": {"es_ES": "Información de Salida", "fr_FR": "Information de Sortie", "de_DE": "Ausgabeinformationen"},
        "The output information will be displayed here.": {
            "es_ES": "La información de salida se mostrará aquí.", 
            "fr_FR": "Les informations de sortie seront affichées ici.", 
            "de_DE": "Die Ausgabeinformationen werden hier angezeigt."
        }
    }
    
    # Create complete translation
    complete_translation = {}
    for key, value in source_data.items():
        # If there's already a translation in the target file and it's different from the key, keep it
        if key in target_data and target_data[key] != key:
            complete_translation[key] = target_data[key]
        # If we have a predefined translation, use it
        elif key in translation_map and target_lang in translation_map[key]:
            complete_translation[key] = translation_map[key][target_lang]
        # Otherwise, keep the English value (will be translated later)
        else:
            complete_translation[key] = value
    
    # Save the updated translation
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(complete_translation, f, ensure_ascii=False, indent=4)
    
    print(f"Created/updated {target_lang_name} translation file: {target_file}")
    return complete_translation

if __name__ == "__main__":
    # Example usage
    create_complete_translation("en_US", "es_ES", "Spanish")
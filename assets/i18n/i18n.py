import os, sys
import json
from pathlib import Path
from locale import getdefaultlocale

now_dir = os.getcwd()
sys.path.append(now_dir)

class I18nAuto:
    LANGUAGE_PATH = os.path.join(now_dir, "assets", "i18n", "languages")
    FALLBACK_LANGUAGE = "en_US"

    def __init__(self, language=None):
        with open(
            os.path.join(now_dir, "assets", "config.json"), "r", encoding="utf8"
        ) as file:
            config = json.load(file)
            override = config["lang"]["override"]
            lang_prefix = config["lang"]["selected_lang"]

        self.language = lang_prefix

        if override == False:
            language = language or getdefaultlocale()[0]
            lang_prefix = language[:2] if language is not None else "en"
            available_languages = self._get_available_languages()
            matching_languages = [
                lang for lang in available_languages if lang.startswith(lang_prefix)
            ]
            self.language = matching_languages[0] if matching_languages else self.FALLBACK_LANGUAGE

        self.language_map = self._load_language_list()

    def _load_language_list(self):
        try:
            file_path = Path(self.LANGUAGE_PATH) / f"{self.language}.json"
            with open(file_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            # Try to fallback to English
            try:
                file_path = Path(self.LANGUAGE_PATH) / f"{self.FALLBACK_LANGUAGE}.json"
                with open(file_path, "r", encoding="utf-8") as file:
                    return json.load(file)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Failed to load language file for {self.language} and fallback language {self.FALLBACK_LANGUAGE}."
                )

    def _get_available_languages(self):
        language_files = [path.stem for path in Path(self.LANGUAGE_PATH).glob("*.json")]
        return language_files

    def _language_exists(self, language):
        return (Path(self.LANGUAGE_PATH) / f"{language}.json").exists()

    def __call__(self, key):
        # Try to get the translation
        translation = self.language_map.get(key, key)
        
        # If the translation is the same as the key and we're not using the fallback language,
        # try to get it from the fallback language
        if translation == key and self.language != self.FALLBACK_LANGUAGE:
            try:
                fallback_file_path = Path(self.LANGUAGE_PATH) / f"{self.FALLBACK_LANGUAGE}.json"
                with open(fallback_file_path, "r", encoding="utf-8") as file:
                    fallback_language_map = json.load(file)
                    return fallback_language_map.get(key, key)
            except FileNotFoundError:
                pass
        
        return translation

    def get_current_language(self):
        """Return the current language code"""
        return self.language

    def get_display_name(self):
        """Return a human-readable name for the current language"""
        language_names = {
            "ar_SA": "Arabic (العربية)",
            "de_DE": "German (Deutsch)",
            "en_US": "English (US)",
            "es_ES": "Spanish (Español)",
            "fr_FR": "French (Français)",
            "hi_IN": "Hindi (हिन्दी)",
            "id_ID": "Indonesian (Bahasa Indonesia)",
            "it_IT": "Italian (Italiano)",
            "ja_JP": "Japanese (日本語)",
            "ko_KR": "Korean (한국어)",
            "nl_NL": "Dutch (Nederlands)",
            "pl_PL": "Polish (Polski)",
            "pt_BR": "Portuguese (Português)",
            "ru_RU": "Russian (Русский)",
            "tr_TR": "Turkish (Türkçe)",
            "zh_CN": "Chinese (中文)"
        }
        return language_names.get(self.language, self.language)

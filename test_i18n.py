import os
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)

from assets.i18n.i18n import I18nAuto

# Test the new languages
test_languages = ["vi_VN", "th_TH", "uk_UA"]

for lang in test_languages:
    print(f"\nTesting {lang}:")
    try:
        i18n = I18nAuto(language=lang)
        test_text = "Language Settings"
        result = i18n(test_text)
        print(f"  '{test_text}' -> '{result}'")
        print(f"  Current language: {i18n.get_current_language()}")
        print(f"  Display name: {i18n.get_display_name()}")
    except Exception as e:
        print(f"  Error: {e}")

# Test existing languages to ensure they still work
existing_languages = ["en_US", "es_ES", "fr_FR", "ja_JP"]

for lang in existing_languages:
    print(f"\nTesting {lang}:")
    try:
        i18n = I18nAuto(language=lang)
        test_text = "Settings"
        result = i18n(test_text)
        print(f"  '{test_text}' -> '{result}'")
        print(f"  Current language: {i18n.get_current_language()}")
        print(f"  Display name: {i18n.get_display_name()}")
    except Exception as e:
        print(f"  Error: {e}")

# Test default behavior
print(f"\nTesting default (no language specified):")
try:
    i18n = I18nAuto()
    test_text = "Theme Settings"
    result = i18n(test_text)
    print(f"  '{test_text}' -> '{result}'")
    print(f"  Current language: {i18n.get_current_language()}")
    print(f"  Display name: {i18n.get_display_name()}")
except Exception as e:
    print(f"  Error: {e}")
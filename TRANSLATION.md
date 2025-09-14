# Translation Guide

Thank you for your interest in translating Advanced RVC Inference! This guide will help you contribute translations to the project.

## How Translations Work

The application uses JSON files for translations, with each language having its own file in the `assets/i18n/languages/` directory. Each file contains key-value pairs where:
- The key is the original English string
- The value is the translated string

## Adding a New Language

1. Copy the `en_US.json` file and rename it with the appropriate language code (e.g., `fr_FR.json` for French)
2. Translate all the values in the JSON file
3. Add the language code to the `LANGUAGE_DISPLAY_NAMES` dictionary in `tabs/settings.py`
4. Submit a pull request with your changes

## Translation Tips

1. **Keep the keys unchanged** - Only translate the values
2. **Preserve formatting** - Keep any markdown or special characters
3. **Be consistent** - Use consistent terminology throughout the translation
4. **Cultural adaptation** - Adapt examples and references to be culturally appropriate

## Language Codes

We use standard language codes in the format `language_COUNTRY`:
- `language` - Two-letter lowercase language code (ISO 639-1)
- `COUNTRY` - Two-letter uppercase country code (ISO 3166-1)

Examples:
- `en_US` - English (United States)
- `fr_FR` - French (France)
- `es_ES` - Spanish (Spain)
- `zh_CN` - Chinese (China)

## Available Languages

Currently, the following languages are supported:
- 🇺🇸 English (en_US)
- 🇩🇪 German (de_DE)
- 🇪🇸 Spanish (es_ES)
- 🇫🇷 French (fr_FR)
- 🇮🇩 Indonesian (id_ID)
- 🇯🇵 Japanese (ja_JP)
- 🇧🇷 Portuguese (pt_BR)
- 🇨🇳 Chinese (zh_CN)
- 🇸🇦 Arabic (ar_SA)
- 🇮🇳 Hindi (hi_IN)
- 🇮🇹 Italian (it_IT)
- 🇰🇷 Korean (ko_KR)
- 🇳🇱 Dutch (nl_NL)
- 🇵🇱 Polish (pl_PL)
- 🇷🇺 Russian (ru_RU)
- 🇹🇷 Turkish (tr_TR)

## Testing Your Translation

1. Save your translation file in the `assets/i18n/languages/` directory
2. Select your language in the application settings
3. Restart the application
4. Verify that all strings are properly translated

## Need Help?

If you need help with translation or have questions about specific terms, please:
1. Open an issue on GitHub
2. Join our Discord community
3. Contact the development team

Thank you for helping make Advanced RVC Inference accessible to more users around the world!
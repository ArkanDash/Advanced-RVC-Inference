import json

# Load the en_US.json file
with open('assets/i18n/languages/en_US.json', encoding='utf-8') as f:
    en_us_data = json.load(f)
    
# Get all keys from en_US.json
en_us_keys = set(en_us_data.keys())

# List of strings used in settings.py i18n() calls
settings_strings = [
    "Language settings have been saved. Please restart the application to apply the changes.",
    "Theme settings have been saved.",
    "Audio settings have been saved.",
    "Performance settings have been saved.",
    "Notification settings have been saved.",
    "Discord presence setting have been saved.",
    "File management settings have been saved.",
    "Debug settings have been saved.",
    "Temporary files cleared. {deleted_count} files deleted.",
    "Backup created successfully: {backup_name}",
    "Error creating backup: {str(e)}",
    "No backup file selected.",
    "Backup restored successfully.",
    "Error restoring backup: {str(e)}",
    "Language Settings",
    "Select your preferred language for the application interface.",
    "Language",
    "Select the language you want to use for the application interface.",
    "Note: You need to restart the application for the language changes to take effect.",
    "Theme Settings",
    "Customize the appearance of the application.",
    "Theme Mode",
    "Select the theme mode for the application.",
    "Primary Color",
    "Select the primary color for the application.",
    "Font Size",
    "Select the font size for the application.",
    "Save Theme Settings",
    "Audio Settings",
    "Configure audio processing preferences.",
    "Default Audio Format",
    "Select the default format for audio output.",
    "Auto-delete Processed Files",
    "Automatically delete processed files after conversion.",
    "Max File Size (MB)",
    "Set the maximum file size for audio processing.",
    "Save Audio Settings",
    "Performance Settings",
    "Optimize application performance.",
    "Max Threads",
    "Set the maximum number of threads for processing.",
    "Memory Optimization",
    "Enable memory optimization for better performance.",
    "GPU Acceleration",
    "Enable GPU acceleration for supported operations.",
    "Save Performance Settings",
    "Notification Settings",
    "Configure application notifications.",
    "Show Completion Notifications",
    "Show notifications when tasks are completed.",
    "Show Error Notifications",
    "Show notifications when errors occur.",
    "Play Sound",
    "Play a sound when notifications are shown.",
    "Save Notification Settings",
    "File Management",
    "Manage temporary files and backups.",
    "Auto Cleanup",
    "Automatically clean temporary files at regular intervals.",
    "Enable Backups",
    "Create backups of important data.",
    "Cleanup Interval (hours)",
    "Set how often to automatically clean temporary files.",
    "Save File Management Settings",
    "Clear Temporary Files",
    "Create Backup",
    "Debug Settings",
    "Configure debugging and logging options.",
    "Verbose Logging",
    "Enable detailed logging for debugging purposes.",
    "Save Debug Logs",
    "Save debug logs to file for troubleshooting.",
    "Debug Level",
    "Select the level of detail for debug logs.",
    "Save Debug Settings",
    "Backup & Restore",
    "Create backups and restore from previous backups.",
    "Restore from Backup",
    "Select Backup",
    "Choose a backup file to restore from.",
    "Refresh Backup List",
    "Restore Selected Backup",
    "Miscellaneous Settings",
    "Other application settings.",
    "Discord Presence",
    "Show Discord rich presence when the application is running.",
    "Save Discord Settings",
    "Restart Application",
    "Restart App"
]

# Find missing translations
missing_translations = [s for s in settings_strings if s not in en_us_keys]

# Print results
print("Missing translations:")
for i, translation in enumerate(missing_translations, 1):
    print(f"{i}. {translation}")

print(f"\nTotal missing translations: {len(missing_translations)}")
print(f"Total strings in settings.py: {len(settings_strings)}")
print(f"Translations found: {len(settings_strings) - len(missing_translations)}")
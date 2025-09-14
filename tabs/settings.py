import gradio as gr
import os, sys
import json
import shutil
import zipfile
from datetime import datetime
from assets.i18n.i18n import I18nAuto

now_dir = os.getcwd()
sys.path.append(now_dir)

i18n = I18nAuto()

config_file = os.path.join(now_dir, "assets", "config.json")

# Language display names with flags or country codes
LANGUAGE_DISPLAY_NAMES = {
    "ar_SA": "🇸🇦 العربية (Arabic)",
    "de_DE": "🇩🇪 Deutsch (German)",
    "en_US": "🇺🇸 English (US)",
    "es_ES": "🇪🇸 Español (Spanish)",
    "fr_FR": "🇫🇷 Français (French)",
    "hi_IN": "🇮🇳 हिन्दी (Hindi)",
    "id_ID": "🇮🇩 Bahasa Indonesia (Indonesian)",
    "it_IT": "🇮🇹 Italiano (Italian)",
    "ja_JP": "🇯🇵 日本語 (Japanese)",
    "ko_KR": "🇰🇷 한국어 (Korean)",
    "nl_NL": "🇳🇱 Nederlands (Dutch)",
    "pl_PL": "🇵🇱 Polski (Polish)",
    "pt_BR": "🇧🇷 Português (Brazilian)",
    "ru_RU": "🇷🇺 Русский (Russian)",
    "tr_TR": "🇹🇷 Türkçe (Turkish)",
    "zh_CN": "🇨🇳 中文 (Chinese)"
}

def load_config():
    """Load configuration from file"""
    try:
        with open(config_file, "r", encoding="utf8") as file:
            return json.load(file)
    except FileNotFoundError:
        # Return default config if file doesn't exist
        return {
            "theme": {
                "file": "Grheme.py",
                "class": "Grheme",
                "mode": "light",
                "primary_hue": "blue",
                "font_size": "medium"
            },
            "discord_presence": True,
            "lang": {
                "override": False,
                "selected_lang": "en_US"
            },
            "audio": {
                "default_format": "wav",
                "auto_delete_processed": False,
                "max_file_size": 100
            },
            "performance": {
                "max_threads": 4,
                "memory_optimization": True,
                "gpu_acceleration": True
            },
            "notifications": {
                "show_completion": True,
                "show_errors": True,
                "play_sound": True
            },
            "file_management": {
                "auto_cleanup": False,
                "cleanup_interval": 24,
                "backup_enabled": False
            },
            "debug": {
                "verbose_logging": False,
                "save_debug_logs": True,
                "debug_level": "INFO"
            }
        }

def save_config(config):
    """Save configuration to file"""
    with open(config_file, "w", encoding="utf8") as file:
        json.dump(config, file, indent=2)

def get_language_settings():
    config = load_config()
    if config["lang"]["override"] == False:
        return "Language automatically detected in the system"
    else:
        selected_lang = config["lang"]["selected_lang"]
        return LANGUAGE_DISPLAY_NAMES.get(selected_lang, selected_lang)

def get_language_choices():
    """Get language choices with proper display names"""
    choices = ["Language automatically detected in the system"]
    available_languages = [path.stem for path in os.scandir(os.path.join(now_dir, "assets", "i18n", "languages")) if path.name.endswith('.json')]
    
    for lang in available_languages:
        display_name = LANGUAGE_DISPLAY_NAMES.get(lang, lang)
        choices.append(display_name)
    
    return choices

def get_language_code_from_display(display_name):
    """Convert display name back to language code"""
    if display_name == "Language automatically detected in the system":
        return display_name
    
    for code, name in LANGUAGE_DISPLAY_NAMES.items():
        if name == display_name:
            return code
    
    # If not found, return as is
    return display_name

def save_lang_settings(selected_language_display):
    config = load_config()
    
    if selected_language_display == "Language automatically detected in the system":
        config["lang"]["override"] = False
        config["lang"]["selected_lang"] = "en_US"  # Default fallback
    else:
        config["lang"]["override"] = True
        config["lang"]["selected_lang"] = get_language_code_from_display(selected_language_display)

    save_config(config)
    gr.Info(i18n("Language settings have been saved. Please restart the application to apply the changes."))

def restart_applio():
    if os.name != "nt":
        os.system("clear")
    else:
        os.system("cls")
    python = sys.executable
    os.execl(python, python, *sys.argv)

def get_theme_settings():
    """Get current theme settings"""
    config = load_config()
    return (
        config["theme"]["mode"],
        config["theme"]["primary_hue"],
        config["theme"]["font_size"]
    )

def save_theme_settings(theme_mode, primary_hue, font_size):
    """Save theme settings"""
    config = load_config()
    config["theme"]["mode"] = theme_mode
    config["theme"]["primary_hue"] = primary_hue
    config["theme"]["font_size"] = font_size
    save_config(config)
    gr.Info(i18n("Theme settings have been saved."))

def get_audio_settings():
    """Get current audio settings"""
    config = load_config()
    return (
        config["audio"]["default_format"],
        config["audio"]["auto_delete_processed"],
        config["audio"]["max_file_size"]
    )

def save_audio_settings(default_format, auto_delete_processed, max_file_size):
    """Save audio settings"""
    config = load_config()
    config["audio"]["default_format"] = default_format
    config["audio"]["auto_delete_processed"] = auto_delete_processed
    config["audio"]["max_file_size"] = max_file_size
    save_config(config)
    gr.Info(i18n("Audio settings have been saved."))

def get_performance_settings():
    """Get current performance settings"""
    config = load_config()
    return (
        config["performance"]["max_threads"],
        config["performance"]["memory_optimization"],
        config["performance"]["gpu_acceleration"]
    )

def save_performance_settings(max_threads, memory_optimization, gpu_acceleration):
    """Save performance settings"""
    config = load_config()
    config["performance"]["max_threads"] = max_threads
    config["performance"]["memory_optimization"] = memory_optimization
    config["performance"]["gpu_acceleration"] = gpu_acceleration
    save_config(config)
    gr.Info(i18n("Performance settings have been saved."))

def get_notification_settings():
    """Get current notification settings"""
    config = load_config()
    return (
        config["notifications"]["show_completion"],
        config["notifications"]["show_errors"],
        config["notifications"]["play_sound"]
    )

def save_notification_settings(show_completion, show_errors, play_sound):
    """Save notification settings"""
    config = load_config()
    config["notifications"]["show_completion"] = show_completion
    config["notifications"]["show_errors"] = show_errors
    config["notifications"]["play_sound"] = play_sound
    save_config(config)
    gr.Info(i18n("Notification settings have been saved."))

def get_discord_presence_setting():
    """Get current Discord presence setting"""
    config = load_config()
    return config["discord_presence"]

def save_discord_presence_setting(discord_presence):
    """Save Discord presence setting"""
    config = load_config()
    config["discord_presence"] = discord_presence
    save_config(config)
    gr.Info(i18n("Discord presence setting has been saved."))

def get_file_management_settings():
    """Get current file management settings"""
    config = load_config()
    file_mgmt = config.get("file_management", {})
    return (
        file_mgmt.get("auto_cleanup", False),
        file_mgmt.get("cleanup_interval", 24),
        file_mgmt.get("backup_enabled", False)
    )

def save_file_management_settings(auto_cleanup, cleanup_interval, backup_enabled):
    """Save file management settings"""
    config = load_config()
    config["file_management"] = {
        "auto_cleanup": auto_cleanup,
        "cleanup_interval": cleanup_interval,
        "backup_enabled": backup_enabled
    }
    save_config(config)
    gr.Info(i18n("File management settings have been saved."))

def get_debug_settings():
    """Get current debug settings"""
    config = load_config()
    debug = config.get("debug", {})
    return (
        debug.get("verbose_logging", False),
        debug.get("save_debug_logs", True),
        debug.get("debug_level", "INFO")
    )

def save_debug_settings(verbose_logging, save_debug_logs, debug_level):
    """Save debug settings"""
    config = load_config()
    config["debug"] = {
        "verbose_logging": verbose_logging,
        "save_debug_logs": save_debug_logs,
        "debug_level": debug_level
    }
    save_config(config)
    gr.Info(i18n("Debug settings have been saved."))

def clear_temp_files():
    """Clear temporary files"""
    temp_dirs = [
        os.path.join(now_dir, "audio_files"),
        os.path.join(now_dir, "logs")
    ]
    
    deleted_count = 0
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            try:
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith(("_temp", ".tmp", ".temp")):
                            os.remove(os.path.join(root, file))
                            deleted_count += 1
            except Exception as e:
                gr.Error(f"Error clearing temp files: {str(e)}")
                return
    
    gr.Info(i18n(f"Temporary files cleared. {deleted_count} files deleted."))

def create_backup():
    """Create a backup of important files"""
    backup_dir = os.path.join(now_dir, "backups")
    os.makedirs(backup_dir, exist_ok=True)
    
    # Files/directories to backup
    items_to_backup = [
        "assets/config.json",
        "logs"
    ]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"backup_{timestamp}.zip"
    backup_path = os.path.join(backup_dir, backup_name)
    
    try:
        with zipfile.ZipFile(backup_path, 'w') as backup_zip:
            for item in items_to_backup:
                item_path = os.path.join(now_dir, item)
                if os.path.exists(item_path):
                    if os.path.isdir(item_path):
                        for root, dirs, files in os.walk(item_path):
                            for file in files:
                                file_path = os.path.join(root, file)
                                archive_path = os.path.relpath(file_path, now_dir)
                                backup_zip.write(file_path, archive_path)
                    else:
                        backup_zip.write(item_path, item)
        
        gr.Info(i18n(f"Backup created successfully: {backup_name}"))
        return backup_path
    except Exception as e:
        gr.Error(i18n(f"Error creating backup: {str(e)}"))
        return None

def restore_backup(backup_file):
    """Restore from a backup file"""
    if not backup_file:
        gr.Error(i18n("No backup file selected."))
        return
    
    try:
        with zipfile.ZipFile(backup_file, 'r') as backup_zip:
            backup_zip.extractall(now_dir)
        
        gr.Info(i18n("Backup restored successfully."))
    except Exception as e:
        gr.Error(i18n(f"Error restoring backup: {str(e)}"))

def get_available_backups():
    """Get list of available backup files"""
    backup_dir = os.path.join(now_dir, "backups")
    if not os.path.exists(backup_dir):
        return []
    
    backups = []
    for file in os.listdir(backup_dir):
        if file.endswith(".zip"):
            backups.append(os.path.join(backup_dir, file))
    
    return sorted(backups, reverse=True)

def lang_tab():
    with gr.Column():
        gr.Markdown("### " + i18n("Language Settings"))
        gr.Markdown(i18n("Select your preferred language for the application interface."))
        
        current_lang_display = get_language_settings()
        if current_lang_display != "Language automatically detected in the system" and current_lang_display in LANGUAGE_DISPLAY_NAMES.values():
            current_value = current_lang_display
        elif current_lang_display != "Language automatically detected in the system":
            # Try to find the display name for the current language code
            current_value = LANGUAGE_DISPLAY_NAMES.get(current_lang_display, current_lang_display)
        else:
            current_value = "Language automatically detected in the system"
        
        selected_language = gr.Dropdown(
            label=i18n("Language"),
            info=i18n("Select the language you want to use for the application interface."),
            value=current_value,
            choices=get_language_choices(),
            interactive=True,
        )

        selected_language.change(
            fn=save_lang_settings,
            inputs=[selected_language],
            outputs=[],
        )
        
        gr.Markdown("*" + i18n("Note: You need to restart the application for the language changes to take effect.") + "*")

def theme_tab():
    with gr.Column():
        gr.Markdown("### " + i18n("Theme Settings"))
        gr.Markdown(i18n("Customize the appearance of the application."))
        
        theme_mode, primary_hue, font_size = get_theme_settings()
        
        with gr.Row():
            theme_mode_dropdown = gr.Dropdown(
                label=i18n("Theme Mode"),
                info=i18n("Select the theme mode for the application."),
                choices=["light", "dark"],
                value=theme_mode,
                interactive=True,
            )
            
            primary_hue_dropdown = gr.Dropdown(
                label=i18n("Primary Color"),
                info=i18n("Select the primary color for the application."),
                choices=["red", "orange", "yellow", "green", "blue", "purple", "pink", "slate", "gray", "zinc"],
                value=primary_hue,
                interactive=True,
            )
            
            font_size_dropdown = gr.Dropdown(
                label=i18n("Font Size"),
                info=i18n("Select the font size for the application."),
                choices=["small", "medium", "large"],
                value=font_size,
                interactive=True,
            )
        
        theme_save_button = gr.Button(i18n("Save Theme Settings"))
        theme_save_button.click(
            fn=save_theme_settings,
            inputs=[theme_mode_dropdown, primary_hue_dropdown, font_size_dropdown],
            outputs=[],
        )

def audio_tab():
    with gr.Column():
        gr.Markdown("### " + i18n("Audio Settings"))
        gr.Markdown(i18n("Configure audio processing preferences."))
        
        default_format, auto_delete_processed, max_file_size = get_audio_settings()
        
        with gr.Row():
            default_format_dropdown = gr.Dropdown(
                label=i18n("Default Audio Format"),
                info=i18n("Select the default format for audio output."),
                choices=["wav", "mp3", "flac", "ogg"],
                value=default_format,
                interactive=True,
            )
            
            auto_delete_checkbox = gr.Checkbox(
                label=i18n("Auto-delete Processed Files"),
                info=i18n("Automatically delete processed files after conversion."),
                value=auto_delete_processed,
                interactive=True,
            )
        
        max_file_size_slider = gr.Slider(
            label=i18n("Max File Size (MB)"),
            info=i18n("Set the maximum file size for audio processing."),
            minimum=10,
            maximum=1000,
            step=10,
            value=max_file_size,
            interactive=True,
        )
        
        audio_save_button = gr.Button(i18n("Save Audio Settings"))
        audio_save_button.click(
            fn=save_audio_settings,
            inputs=[default_format_dropdown, auto_delete_checkbox, max_file_size_slider],
            outputs=[],
        )

def performance_tab():
    with gr.Column():
        gr.Markdown("### " + i18n("Performance Settings"))
        gr.Markdown(i18n("Optimize application performance."))
        
        max_threads, memory_optimization, gpu_acceleration = get_performance_settings()
        
        with gr.Row():
            max_threads_slider = gr.Slider(
                label=i18n("Max Threads"),
                info=i18n("Set the maximum number of threads for processing."),
                minimum=1,
                maximum=16,
                step=1,
                value=max_threads,
                interactive=True,
            )
            
            memory_optimization_checkbox = gr.Checkbox(
                label=i18n("Memory Optimization"),
                info=i18n("Enable memory optimization for better performance."),
                value=memory_optimization,
                interactive=True,
            )
            
            gpu_acceleration_checkbox = gr.Checkbox(
                label=i18n("GPU Acceleration"),
                info=i18n("Enable GPU acceleration for supported operations."),
                value=gpu_acceleration,
                interactive=True,
            )
        
        performance_save_button = gr.Button(i18n("Save Performance Settings"))
        performance_save_button.click(
            fn=save_performance_settings,
            inputs=[max_threads_slider, memory_optimization_checkbox, gpu_acceleration_checkbox],
            outputs=[],
        )

def notifications_tab():
    with gr.Column():
        gr.Markdown("### " + i18n("Notification Settings"))
        gr.Markdown(i18n("Configure application notifications."))
        
        show_completion, show_errors, play_sound = get_notification_settings()
        
        with gr.Row():
            show_completion_checkbox = gr.Checkbox(
                label=i18n("Show Completion Notifications"),
                info=i18n("Show notifications when tasks are completed."),
                value=show_completion,
                interactive=True,
            )
            
            show_errors_checkbox = gr.Checkbox(
                label=i18n("Show Error Notifications"),
                info=i18n("Show notifications when errors occur."),
                value=show_errors,
                interactive=True,
            )
            
            play_sound_checkbox = gr.Checkbox(
                label=i18n("Play Sound"),
                info=i18n("Play a sound when notifications are shown."),
                value=play_sound,
                interactive=True,
            )
        
        notifications_save_button = gr.Button(i18n("Save Notification Settings"))
        notifications_save_button.click(
            fn=save_notification_settings,
            inputs=[show_completion_checkbox, show_errors_checkbox, play_sound_checkbox],
            outputs=[],
        )

def file_management_tab():
    with gr.Column():
        gr.Markdown("### " + i18n("File Management"))
        gr.Markdown(i18n("Manage temporary files and backups."))
        
        auto_cleanup, cleanup_interval, backup_enabled = get_file_management_settings()
        
        with gr.Row():
            auto_cleanup_checkbox = gr.Checkbox(
                label=i18n("Auto Cleanup"),
                info=i18n("Automatically clean temporary files at regular intervals."),
                value=auto_cleanup,
                interactive=True,
            )
            
            backup_enabled_checkbox = gr.Checkbox(
                label=i18n("Enable Backups"),
                info=i18n("Create backups of important data."),
                value=backup_enabled,
                interactive=True,
            )
        
        cleanup_interval_slider = gr.Slider(
            label=i18n("Cleanup Interval (hours)"),
            info=i18n("Set how often to automatically clean temporary files."),
            minimum=1,
            maximum=168,
            step=1,
            value=cleanup_interval,
            interactive=True,
        )
        
        with gr.Row():
            file_save_button = gr.Button(i18n("Save File Management Settings"))
            clear_temp_button = gr.Button(i18n("Clear Temporary Files"))
            backup_button = gr.Button(i18n("Create Backup"))
        
        file_save_button.click(
            fn=save_file_management_settings,
            inputs=[auto_cleanup_checkbox, cleanup_interval_slider, backup_enabled_checkbox],
            outputs=[],
        )
        
        clear_temp_button.click(
            fn=clear_temp_files,
            inputs=[],
            outputs=[],
        )
        
        backup_button.click(
            fn=create_backup,
            inputs=[],
            outputs=[],
        )

def debug_tab():
    with gr.Column():
        gr.Markdown("### " + i18n("Debug Settings"))
        gr.Markdown(i18n("Configure debugging and logging options."))
        
        verbose_logging, save_debug_logs, debug_level = get_debug_settings()
        
        with gr.Row():
            verbose_logging_checkbox = gr.Checkbox(
                label=i18n("Verbose Logging"),
                info=i18n("Enable detailed logging for debugging purposes."),
                value=verbose_logging,
                interactive=True,
            )
            
            save_debug_logs_checkbox = gr.Checkbox(
                label=i18n("Save Debug Logs"),
                info=i18n("Save debug logs to file for troubleshooting."),
                value=save_debug_logs,
                interactive=True,
            )
        
        debug_level_dropdown = gr.Dropdown(
            label=i18n("Debug Level"),
            info=i18n("Select the level of detail for debug logs."),
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            value=debug_level,
            interactive=True,
        )
        
        debug_save_button = gr.Button(i18n("Save Debug Settings"))
        debug_save_button.click(
            fn=save_debug_settings,
            inputs=[verbose_logging_checkbox, save_debug_logs_checkbox, debug_level_dropdown],
            outputs=[],
        )

def backup_restore_tab():
    with gr.Column():
        gr.Markdown("### " + i18n("Backup & Restore"))
        gr.Markdown(i18n("Create backups and restore from previous backups."))
        
        with gr.Row():
            backup_button = gr.Button(i18n("Create Backup"))
            backup_button.click(
                fn=create_backup,
                inputs=[],
                outputs=[],
            )
            
            clear_temp_button = gr.Button(i18n("Clear Temporary Files"))
            clear_temp_button.click(
                fn=clear_temp_files,
                inputs=[],
                outputs=[],
            )
        
        gr.Markdown("### " + i18n("Restore from Backup"))
        available_backups = get_available_backups()
        
        backup_dropdown = gr.Dropdown(
            label=i18n("Select Backup"),
            info=i18n("Choose a backup file to restore from."),
            choices=available_backups,
            interactive=True,
        )
        
        def update_backup_list():
            return gr.update(choices=get_available_backups())
        
        refresh_button = gr.Button(i18n("Refresh Backup List"))
        refresh_button.click(
            fn=update_backup_list,
            inputs=[],
            outputs=[backup_dropdown],
        )
        
        restore_button = gr.Button(i18n("Restore Selected Backup"))
        restore_button.click(
            fn=restore_backup,
            inputs=[backup_dropdown],
            outputs=[],
        )

def misc_tab():
    with gr.Column():
        gr.Markdown("### " + i18n("Miscellaneous Settings"))
        gr.Markdown(i18n("Other application settings."))
        
        discord_presence = get_discord_presence_setting()
        
        discord_presence_checkbox = gr.Checkbox(
            label=i18n("Discord Presence"),
            info=i18n("Show Discord rich presence when the application is running."),
            value=discord_presence,
            interactive=True,
        )
        
        discord_save_button = gr.Button(i18n("Save Discord Settings"))
        discord_save_button.click(
            fn=save_discord_presence_setting,
            inputs=[discord_presence_checkbox],
            outputs=[],
        )

def restart_tab():
    with gr.Row():
        with gr.Column():
            gr.Markdown("### " + i18n("Restart Application"))
            restart_button = gr.Button(i18n("Restart App"))
            restart_button.click(
                fn=restart_applio,
                inputs=[],
                outputs=[],
            )



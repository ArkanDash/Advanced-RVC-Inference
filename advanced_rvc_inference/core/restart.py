import os
import sys
import json
import platform
import subprocess

sys.path.append(os.getcwd())

from advanced_rvc_inference.core.ui import gr_info, gr_warning
from advanced_rvc_inference.utils.variables import python, translations, configs_json

def restart_app(app):
    gr_info(translations["30s"])
    try:
        os.system("cls" if platform.system() == "Windows" else "clear")
    except Exception:
        pass
    
    app.close()
    subprocess.run([python, os.path.join("advanced_rvc_inference", "app", "gui.py")] + [arg for arg in sys.argv[1:] if arg != "--open"])

def change_language(lang, app):
    try:
        with open(configs_json, "r", encoding="utf-8") as f:
            configs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        gr_warning("Could not load configuration file")
        return

    if lang != configs.get("language"):
        configs["language"] = lang
        try:
            with open(configs_json, "w", encoding="utf-8") as f:
                json.dump(configs, f, indent=4)
        except OSError as e:
            gr_warning(f"Could not save configuration: {e}")
            return

        restart_app(app)

def change_theme(theme, app):
    try:
        with open(configs_json, "r", encoding="utf-8") as f:
            configs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        gr_warning("Could not load configuration file")
        return
    
    if theme != configs.get("theme"):
        configs["theme"] = theme
        try:
            with open(configs_json, "w", encoding="utf-8") as f:
                json.dump(configs, f, indent=4)
        except OSError as e:
            gr_warning(f"Could not save configuration: {e}")
            return

        restart_app(app)


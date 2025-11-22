import os
import sys
import json
import platform
import subprocess

sys.path.append(os.getcwd())

from main.app.core.ui import gr_info
from main.app.variables import python, translations, configs_json

def restart_app(app):
    gr_info(translations["30s"])
    os.system("cls" if platform.system() == "Windows" else "clear")
    
    app.close()
    subprocess.run([python, os.path.join("main", "app", "app.py")] + [arg for arg in sys.argv[1:] if arg != "--open"])

def change_language(lang, app):
    configs = json.load(open(configs_json, "r"))

    if lang != configs["language"]:
        configs["language"] = lang

        with open(configs_json, "w") as f:
            json.dump(configs, f, indent=4)

        restart_app(app)

def change_theme(theme, app):
    configs = json.load(open(configs_json, "r"))
    
    if theme != configs["theme"]:
        configs["theme"] = theme
        with open(configs_json, "w") as f:
            json.dump(configs, f, indent=4)

        restart_app(app)

def change_font(font, app):
    configs = json.load(open(configs_json, "r"))

    if font != configs["font"]:
        configs["font"] = font
        with open(configs_json, "w") as f:
            json.dump(configs, f, indent=4)

        restart_app(app)
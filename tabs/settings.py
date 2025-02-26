import gradio as gr
import os, sys
import json
import gradio as gr
from assets.i18n.i18n import I18nAuto
import assets.themes.loadThemes as loadThemes

now_dir = os.getcwd()
sys.path.append(now_dir)

i18n = I18nAuto()

config_file = os.path.join(now_dir, "assets", "config.json")


def get_language_settings():
    with open(config_file, "r", encoding="utf8") as file:
        config = json.load(file)

    if config["lang"]["override"] == False:
        return "Language automatically detected in the system"
    else:
        return config["lang"]["selected_lang"]


def save_lang_settings(selected_language):
    with open(config_file, "r", encoding="utf8") as file:
        config = json.load(file)

    if selected_language == "Language automatically detected in the system":
        config["lang"]["override"] = False
    else:
        config["lang"]["override"] = True
        config["lang"]["selected_lang"] = selected_language

    gr.Info("Language have been saved. Restart Applio to apply the changes.")

    with open(config_file, "w", encoding="utf8") as file:
        json.dump(config, file, indent=2)


def restart_applio():
    if os.name != "nt":
        os.system("clear")
    else:
        os.system("cls")
    python = sys.executable
    os.execl(python, python, *sys.argv)


def lang_tab():
    with gr.Column():
        selected_language = gr.Dropdown(
            label=i18n("Language"),
            info=i18n("Select the language you want to use. (Requires restarting App)"),
            value=get_language_settings(),
            choices=["Language automatically detected in the system"]
            + i18n._get_available_languages(),
            interactive=True,
        )

        selected_language.change(
            fn=save_lang_settings,
            inputs=[selected_language],
            outputs=[],
        )


def restart_tab():
    with gr.Row():
        with gr.Column():
            restart_button = gr.Button("Restart App")
            restart_button.click(
                fn=restart_applio,
                inputs=[],
                outputs=[],
            )


def theme_tab():
    with gr.Row():
        with gr.Column():
            themes_select = gr.Dropdown(
                loadThemes.get_theme_list(),
                value=loadThemes.load_theme(),
                label=i18n("Theme"),
                info=i18n(
                    "Select the theme you want to use. (Requires restarting App)"
                ),
                visible=True,
            )
            themes_select.change(
                fn=loadThemes.select_theme,
                inputs=themes_select,
                outputs=[],
            )

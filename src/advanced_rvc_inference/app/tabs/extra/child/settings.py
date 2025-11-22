import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from main.app.core.ui import change_fp
from main.app.core.utils import stop_pid
from main.app.core.restart import change_font, change_language, change_theme
from main.app.variables import translations, theme, font, configs, language, config

def settings_tab(app):
    with gr.Row():
        gr.Markdown(translations["settings_markdown_2"])
    with gr.Row():
        toggle_button = gr.Button(translations["change_light_dark"], variant="secondary", scale=2)
    with gr.Row():
        with gr.Column():
            language_dropdown = gr.Dropdown(label=translations["lang"], interactive=True, info=translations["lang_restart"], choices=configs.get("support_language", "vi-VN"), value=language)
            change_lang = gr.Button(translations["change_lang"], variant="primary", scale=2)
        with gr.Column():
            theme_dropdown = gr.Dropdown(label=translations["theme"], interactive=True, info=translations["theme_restart"], choices=configs.get("themes", theme), value=theme, allow_custom_value=True)
            changetheme = gr.Button(translations["theme_button"], variant="primary", scale=2)
    with gr.Row():
        with gr.Column():
            fp_choice = gr.Radio(choices=["fp16","fp32"], value="fp16" if configs.get("fp16", False) else "fp32", label=translations["precision"], info=translations["precision_info"], interactive=config.device not in ["cpu", "mps", "ocl:0"])
            fp_button = gr.Button(translations["update_precision"], variant="secondary", scale=2)
        with gr.Column():
            font_choice = gr.Textbox(label=translations["font"], info=translations["font_info"], value=font, interactive=True)
            font_button = gr.Button(translations["change_font"])
    with gr.Row():
        with gr.Column():
            with gr.Accordion(translations["stop"], open=False, visible=True):
                separate_stop = gr.Button(translations["stop_separate"])
                convert_stop = gr.Button(translations["stop_convert"])
                create_dataset_stop = gr.Button(translations["stop_create_dataset"])
                with gr.Accordion(translations["stop_training"], open=False):
                    model_name_stop = gr.Textbox(label=translations["modelname"], info=translations["training_model_name"], value="", placeholder=translations["modelname"], interactive=True)
                    preprocess_stop = gr.Button(translations["stop_preprocess"])
                    extract_stop = gr.Button(translations["stop_extract"])
                    train_stop = gr.Button(translations["stop_training"])
    with gr.Row():
        toggle_button.click(fn=None, js="() => {document.body.classList.toggle('dark')}")
        fp_button.click(fn=change_fp, inputs=[fp_choice], outputs=[fp_choice])
    with gr.Row():
        change_lang.click(fn=lambda a: change_language(a, app), inputs=[language_dropdown], outputs=[])
        changetheme.click(fn=lambda a: change_theme(a, app) , inputs=[theme_dropdown], outputs=[])
        font_button.click(fn=lambda a: change_font(a, app), inputs=[font_choice], outputs=[])
    with gr.Row():
        change_lang.click(fn=None, js="setTimeout(function() {location.reload()}, 30000)", inputs=[], outputs=[])
        changetheme.click(fn=None, js="setTimeout(function() {location.reload()}, 30000)", inputs=[], outputs=[])
        font_button.click(fn=None, js="setTimeout(function() {location.reload()}, 30000)", inputs=[], outputs=[])
    with gr.Row():
        separate_stop.click(fn=lambda: stop_pid("separate_pid", None, False), inputs=[], outputs=[])
        convert_stop.click(fn=lambda: stop_pid("convert_pid", None, False), inputs=[], outputs=[])
        create_dataset_stop.click(fn=lambda: stop_pid("create_dataset_pid", None, False), inputs=[], outputs=[])
    with gr.Row():
        preprocess_stop.click(fn=lambda model_name_stop: stop_pid("preprocess_pid", model_name_stop, False), inputs=[model_name_stop], outputs=[])
        extract_stop.click(fn=lambda model_name_stop: stop_pid("extract_pid", model_name_stop, False), inputs=[model_name_stop], outputs=[])
        train_stop.click(fn=lambda model_name_stop: stop_pid("train_pid", model_name_stop, True), inputs=[model_name_stop], outputs=[])
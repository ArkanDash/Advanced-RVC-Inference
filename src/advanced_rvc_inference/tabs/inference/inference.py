import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from main.app.variables import translations, configs
from main.app.tabs.inference.child.convert import convert_tab
from main.app.tabs.inference.child.separate import separate_tab
from main.app.tabs.inference.child.convert_tts import convert_tts_tab
from main.app.tabs.inference.child.convert_with_whisper import convert_with_whisper_tab

def inference_tab():
    with gr.TabItem(translations["inference"], visible=configs.get("inference_tab", True)):
        with gr.TabItem(translations["separator_tab"], visible=configs.get("separator_tab", True)):
            gr.Markdown(f"## {translations['separator_tab']}")
            separate_tab()

        with gr.TabItem(translations["convert_audio"], visible=configs.get("convert_tab", True)):
            gr.Markdown(f"## {translations['convert_audio']}")
            convert_tab()

        with gr.TabItem(translations["convert_with_whisper"], visible=configs.get("convert_with_whisper", True)):
            gr.Markdown(f"## {translations['convert_with_whisper']}")
            convert_with_whisper_tab()

        with gr.TabItem(translations["convert_text"], visible=configs.get("tts_tab", True)):
            gr.Markdown(translations["convert_text_markdown"])
            convert_tts_tab()

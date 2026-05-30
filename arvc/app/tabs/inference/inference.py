import os
import sys

import gradio as gr


from arvc.utils.variables import translations, configs
from arvc.app.tabs.inference.child.convert import convert_tab
from arvc.app.tabs.inference.child.separate import separate_tab
from arvc.app.tabs.inference.child.convert_tts import convert_tts_tab
from arvc.app.tabs.inference.child.convert_with_whisper import convert_with_whisper_tab

def inference_tab():
    with gr.TabItem(translations["inference"], visible=configs.get("inference_tab", True)):
        with gr.TabItem(translations["convert_audio"], visible=configs.get("convert_tab", True)):
            convert_tab()
        with gr.TabItem(translations["separator_tab"], visible=configs.get("separator_tab", True)):
            separate_tab()    

        with gr.TabItem(translations["convert_with_whisper"], visible=configs.get("convert_with_whisper", True)):
            convert_with_whisper_tab()

        with gr.TabItem(translations["convert_text"], visible=configs.get("tts_tab", True)):
            gr.Markdown(translations["convert_text_markdown"])
            convert_tts_tab()

import os
import sys

import gradio as gr


from arvc.ui.feedback import shutil_move
from arvc.services.model_utils import model_info
from arvc.utils.variables import translations, configs

def read_model_tab():
    with gr.Row():
        gr.Markdown(translations["read_model_markdown_2"])
    with gr.Row():
        model = gr.File(label=translations["drop_model"], file_types=[".pth", ".onnx"]) 
    with gr.Row():
        read_button = gr.Button(translations["readmodel"], variant="primary", scale=2)
    with gr.Column():
        model_path = gr.Textbox(label=translations["model_path"], value="", placeholder="assets/weights/Model.pth", info=translations["model_path_info"], interactive=True)
        output_info = gr.Textbox(label=translations["modelinfo"], value="", interactive=False, scale=6)
    with gr.Row():
        model.upload(fn=lambda model: shutil_move(model.name, configs["weights_path"]), inputs=[model], outputs=[model_path])
        read_button.click(
            fn=model_info,
            inputs=[model_path],
            outputs=[output_info],
            api_name="read_model"
        )
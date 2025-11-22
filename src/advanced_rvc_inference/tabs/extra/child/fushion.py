import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from main.app.core.ui import visible, shutil_move
from main.app.core.model_utils import fushion_model
from main.app.variables import translations, configs

def fushion_tab():
    with gr.Row():
        gr.Markdown(translations["fushion_markdown_2"])
    with gr.Row():
        name_to_save = gr.Textbox(label=translations["modelname"], placeholder="Model.pth", value="", max_lines=1, interactive=True)
    with gr.Row():
        fushion_button = gr.Button(translations["fushion"], variant="primary", scale=4)
    with gr.Column():
        with gr.Row():
            model_a = gr.File(label=f"{translations['model_name']} 1", file_types=[".pth", ".onnx"]) 
            model_b = gr.File(label=f"{translations['model_name']} 2", file_types=[".pth", ".onnx"])
        with gr.Row():
            model_path_a = gr.Textbox(label=f"{translations['model_path']} 1", value="", placeholder="assets/weights/Model_1.pth")
            model_path_b = gr.Textbox(label=f"{translations['model_path']} 2", value="", placeholder="assets/weights/Model_2.pth")
    with gr.Row():
        ratio = gr.Slider(minimum=0, maximum=1, label=translations["model_ratio"], info=translations["model_ratio_info"], value=0.5, interactive=True)
    with gr.Row():
        output_model = gr.File(label=translations["output_model_path"], file_types=[".pth", ".onnx"], interactive=False, visible=False)
    with gr.Row():
        model_a.upload(fn=lambda model: shutil_move(model.name, configs["weights_path"]), inputs=[model_a], outputs=[model_path_a])
        model_b.upload(fn=lambda model: shutil_move(model.name, configs["weights_path"]), inputs=[model_b], outputs=[model_path_b])
    with gr.Row():
        fushion_button.click(
            fn=fushion_model,
            inputs=[
                name_to_save, 
                model_path_a, 
                model_path_b, 
                ratio
            ],
            outputs=[name_to_save, output_model],
            api_name="fushion_model"
        )
        fushion_button.click(fn=lambda: visible(True), inputs=[], outputs=[output_model])  
import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from main.app.core.ui import visible, shutil_move
from main.app.core.model_utils import onnx_export
from main.app.variables import translations, configs

def convert_model_tab():
    with gr.Row():
        gr.Markdown(translations["pytorch2onnx_markdown"])
    with gr.Row():
        model_pth_upload = gr.File(label=translations["drop_model"], file_types=[".pth"]) 
    with gr.Row():
        convert_onnx = gr.Button(translations["convert_model"], variant="primary", scale=2)
    with gr.Row():
        model_pth_path = gr.Textbox(label=translations["model_path"], value="", placeholder="assets/weights/Model.pth", info=translations["model_path_info"], interactive=True)
    with gr.Row():
        output_model2 = gr.File(label=translations["output_model_path"], file_types=[".pth", ".onnx"], interactive=False, visible=False)
    with gr.Row():
        model_pth_upload.upload(fn=lambda model_pth_upload: shutil_move(model_pth_upload.name, configs["weights_path"]), inputs=[model_pth_upload], outputs=[model_pth_path])
        convert_onnx.click(
            fn=onnx_export,
            inputs=[model_pth_path],
            outputs=[output_model2],
            api_name="model_onnx_export"
        )
        convert_onnx.click(fn=lambda: visible(True), inputs=[], outputs=[output_model2])  
import os
import sys
import gradio as gr

now_dir = os.getcwd()
sys.path.append(now_dir)

from core import run_model_information_script

def processing_tab():
    model_view_model_path = gr.Textbox(
        label="Path to Model",
        info="Introduce the model pth path",
        value="",
        interactive=True,
        placeholder="Enter path to model",
    )

    model_view_output_info = gr.Textbox(
        label="Output Information",
        info="The output information will be displayed here.",
        value="",
        max_lines=11,
    )
    model_view_button = gr.Button("View")
    model_view_button.click(
        fn=run_model_information_script,
        inputs=[model_view_model_path],
        outputs=[model_view_output_info],
    )

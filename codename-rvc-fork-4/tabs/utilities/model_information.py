import gradio as gr
from core import run_model_information_script

def model_information_tab():
    with gr.Column():
        model_name = gr.Textbox(
            label="Path to Model",
            info="Introduce the model pth path",
            placeholder="Introduce the model pth path",
            interactive=True,
        )
        model_information_output_info = gr.Textbox(
            label="Output Information",
            info="The output information will be displayed here.",
            value="",
            max_lines=12,
            interactive=False,
        )
        model_information_button = gr.Button("See Model Information")
        model_information_button.click(
            fn=run_model_information_script,
            inputs=[model_name],
            outputs=[model_information_output_info],
        )

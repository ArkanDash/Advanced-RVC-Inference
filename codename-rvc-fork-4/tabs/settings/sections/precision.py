import gradio as gr

from rvc.configs.config import Config, microarchitecture_capability_checker

config = Config()

if microarchitecture_capability_checker():
    # Ampere-Microarchitecture and higher viable:
    available_precision_choices = ["fp16", "bf16", "fp32"]
else:
    # Below Ampere-Microarchitecture viable:
    available_precision_choices = ["fp16", "fp32"]

def precision_tab():
    with gr.Row():
        with gr.Column():

            precision = gr.Radio(
                label="Precision",
                info="Select the precision you want to use for training and inference.",
                choices=available_precision_choices,
                value=config.get_precision(),
                interactive=True,
            )
            precision_output = gr.Textbox(
                label="Output Information",
                info="The output information will be displayed here.",
                value="",
                max_lines=8,
                interactive=False,
            )

            check_button = gr.Button("Check precision")
            check_button.click(
                fn=config.check_precision,
                outputs=[precision_output],
            )

            update_button = gr.Button("Update precision")
            update_button.click(
                fn=config.set_precision,
                inputs=[precision],
                outputs=[precision_output],
            )

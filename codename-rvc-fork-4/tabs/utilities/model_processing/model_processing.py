import os
import torch
import hashlib
import datetime
from collections import OrderedDict
import gradio as gr
import traceback


def extract_small_model(
    path: str,
    name: str,
    output_dir: str,
    sr: int,
    pitch_guidance: bool,
    version: str,
):
    if not path:
        return "Error: Please upload a Generator checkpoint ( Big G network .pth file)."

    try:
        if not output_dir:
            output_dir = "logs/EXTRACTED_SMALL_MODELS" 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        pth_file = f"{name}.pth"
        final_pth_path = os.path.join(output_dir, pth_file)

        ckpt = torch.load(path, map_location="cpu")

        if "model" in ckpt:
            ckpt = ckpt["model"]

        opt = OrderedDict(
            weight={
                key: value.half() for key, value in ckpt.items() if "enc_q" not in key
            }
        )

        config_map = {
            "40000": [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5]] * 3, [10, 10, 2, 2], 512, [16, 16, 4, 4], 109, 256, 40000],
            "48000": [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5]] * 3, [12, 10, 2, 2] if version != "v1" else [10, 6, 2, 2, 2], 512, [24, 20, 4, 4] if version != "v1" else [16, 16, 4, 4, 4], 109, 256, 48000],
            "32000": [513, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5]] * 3, [10, 8, 2, 2] if version != "v1" else [10, 4, 2, 2, 2], 512, [20, 16, 4, 4] if version != "v1" else [16, 16, 4, 4, 4], 109, 256, 32000],
        }
        opt["config"] = config_map.get(str(sr), [])

        opt.update(
            {
                "sr": sr,
                "f0": int(pitch_guidance),
                "version": version,
                "creation_date": datetime.datetime.now().isoformat(),
            }
        )

        torch.save(opt, final_pth_path)

        return f" Successfully extracted and saved model to {final_pth_path} ..."

    except Exception as error:
        print(f"An error occurred extracting the model: {error}")
        return f" Failed to extract model: {error}\n{traceback.format_exc()}"

def extract_small_model_tab():
    with gr.Column():
        gr.Markdown(
            """
            # Checkpoint Extractor ⚙️
            """
        )

        with gr.Row():
            model_path_input = gr.File(
                label="1. Generator network checkpoint (.pth)",
                file_types=[".pth"],
                file_count="single",
                interactive=True,
                scale=2
            )
            model_name_input = gr.Textbox(
                label="Output Model Name",
                info="The output file will be saved as `<name>.pth`.",
                value="My_extracted_model_123",
                interactive=True,
                scale=1
            )
            output_dir_input = gr.Textbox(
                label="Output Directory",
                info="The directory where the final .pth file will be saved.",
                value="logs/EXTRACTED_SMALL_MODELS",
                interactive=True,
                scale=1
            )

        with gr.Row():
            sr_input = gr.Dropdown(
                label="Sample Rate of the model (sr)",
                choices=[32000, 40000, 48000],
                value=48000, 
                type="value",
                interactive=True,
                scale=1
            )
            pitch_guidance_input = gr.Checkbox(
                label="F0-guided model", 
                value=True,
                info="Check if the model was trained with pitch (F0) guidance.",
                interactive=True,
                scale=1
            )
            version_input = gr.Dropdown(
                label="Version",
                info="Select one that corresponds to your training.",
                choices=['v1', 'v2'],
                value='v2',
                interactive=True,
                scale=1
            )

        extract_button = gr.Button("Extract Small Model", variant="primary")

        output_info = gr.Textbox(
            label="Output Information",
            info="Status messages and final file path will be displayed here.",
            value="",
            max_lines=8,
            interactive=False 
        )

        extract_button.click(
            fn=extract_small_model,
            inputs=[
                model_path_input,
                model_name_input,
                output_dir_input,
                sr_input,
                pitch_guidance_input,
                version_input,
            ],
            outputs=[output_info],
        )

if __name__ == "__main__":
    with gr.Blocks() as demo:
        extract_small_model_tab()
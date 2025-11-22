import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from main.app.core.training import create_dataset
from main.app.core.ui import visible, valueFalse_interactive, create_dataset_change
from main.app.variables import translations, sample_rate_choice, uvr_model, reverb_models, denoise_models, vr_models, mdx_models

def create_dataset_tab():
    with gr.Row():
        gr.Markdown(translations["create_dataset_markdown_2"])
    with gr.Group():
        with gr.Row():
            separate = gr.Checkbox(label=translations["separator_tab"], value=False, interactive=True)
            clean_dataset = gr.Checkbox(label=translations["clear_audio"], value=False, interactive=True)
            skip_seconds = gr.Checkbox(label=translations["skip"], value=False, interactive=True)
            separate_reverb = gr.Checkbox(label=translations["dereveb_audio"], value=False, interactive=False)
        with gr.Row(visible=False) as row:
            enable_tta = gr.Checkbox(label=translations["enable_tta"], value=False, interactive=False)
            high_end_process = gr.Checkbox(label=translations["high_end_process"], value=False, interactive=False)
            enable_post_process = gr.Checkbox(label=translations["enable_post_process"], value=False, interactive=False)
            enable_denoise = gr.Checkbox(label=translations["denoise_mdx"], value=False, interactive=False)
    with gr.Row():
        dataset_url = gr.Textbox(label=translations["url_audio"], info=translations["create_dataset_url"], value="", placeholder="https://www.youtube.com/...", interactive=True, scale=5)
        output_dataset = gr.Textbox(label=translations["output_data"], info=translations["output_data_info"], value="dataset", placeholder="dataset", interactive=True)
    with gr.Row():
        create_dataset_button = gr.Button(translations["createdataset"], variant="primary", scale=2, min_width=4000)
    with gr.Row(visible=False) as row_2:
        model_name = gr.Dropdown(label=translations["separator_model"], value=uvr_model[0], choices=uvr_model, interactive=True)
        reverb_model = gr.Dropdown(label=translations["dereveb_model"], value=list(reverb_models.keys())[0], choices=list(reverb_models.keys()), interactive=True)
        denoise_model = gr.Dropdown(label=translations["denoise_model"], value=list(denoise_models.keys())[0], choices=list(denoise_models.keys()), interactive=True, visible=False)
    with gr.Row():
        with gr.Column(visible=False) as row_3:
            with gr.Group():
                with gr.Row():
                    overlap = gr.Radio(label=translations["overlap"], info=translations["overlap_info"], choices=["0.25", "0.5", "0.75", "0.99"], value="0.25", interactive=True)
                with gr.Row():
                    window_size = gr.Slider(label=translations["window_size"], info=translations["window_size_info"], minimum=320, maximum=1024, value=512, step=32, interactive=True, visible=False)
                with gr.Row():
                    shifts = gr.Slider(label=translations["shift"], info=translations["shift_info"], minimum=1, maximum=20, value=2, step=1, interactive=True)
                    segments_size = gr.Slider(label=translations["segments_size"], info=translations["segments_size_info"], minimum=32, maximum=3072, value=256, step=32, interactive=True)
                with gr.Row():
                    batch_size = gr.Slider(label=translations["batch_size"], info=translations["mdx_batch_size_info"], minimum=1, maximum=64, value=1, step=1, interactive=True, visible=False)
                    hop_length = gr.Slider(label=translations['hop_length'], info=translations["hop_length_info"], minimum=64, maximum=8192, value=1024, step=1, interactive=True, visible=False)
                with gr.Row():
                    post_process_threshold = gr.Slider(label=translations['post_process_threshold'], info=translations["post_process_threshold_info"], minimum=0.1, maximum=0.3, value=0.2, step=0.1, interactive=True, visible=False)
                    aggression = gr.Slider(label=translations['aggression'], info=translations["aggression_info"], minimum=1, maximum=50, value=5, step=1, interactive=True, visible=False)
        with gr.Column():
            sample_rate = gr.Radio(choices=sample_rate_choice, value=48000, label=translations["sr"], info=translations["sr_info"], interactive=True)
            clean_strength = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.5, label=translations["clean_strength"], info=translations["clean_strength_info"], interactive=True, visible=False)
            with gr.Row():
                skip_start = gr.Textbox(label=translations["skip_start"], info=translations["skip_start_info"], value="", placeholder="0,...", interactive=True, visible=skip_seconds.value)
                skip_end = gr.Textbox(label=translations["skip_end"], info=translations["skip_end_info"], value="", placeholder="0,...", interactive=True, visible=skip_seconds.value)
            create_dataset_info = gr.Textbox(label=translations["create_dataset_info"], value="", interactive=False, lines=2)
    with gr.Row():
        separate.change(
            fn=lambda a: [visible(a) for _ in range(3)],
            inputs=[separate],
            outputs=[
                row,
                row_2,
                row_3
            ]
        )
        separate.change(
            fn=valueFalse_interactive,
            inputs=[separate],
            outputs=[separate_reverb]
        )
        separate.change(
            fn=create_dataset_change,
            inputs=[
                model_name, 
                reverb_model, 
                enable_post_process, 
                separate_reverb, 
                enable_denoise
            ],
            outputs=[
                reverb_model,
                overlap, 
                segments_size, 
                hop_length, 
                batch_size,
                shifts, 
                window_size, 
                aggression, 
                post_process_threshold,
                denoise_model,
                enable_tta, 
                high_end_process, 
                enable_post_process,
            ]
        )
    with gr.Row():
        model_name.change(
            fn=create_dataset_change,
            inputs=[
                model_name, 
                reverb_model, 
                enable_post_process, 
                separate_reverb, 
                enable_denoise
            ],
            outputs=[
                reverb_model,
                overlap, 
                segments_size, 
                hop_length, 
                batch_size,
                shifts, 
                window_size, 
                aggression, 
                post_process_threshold,
                denoise_model,
                enable_tta, 
                high_end_process, 
                enable_post_process,
            ]
        )
        reverb_model.change(
            fn=create_dataset_change,
            inputs=[
                model_name, 
                reverb_model, 
                enable_post_process, 
                separate_reverb, 
                enable_denoise
            ],
            outputs=[
                reverb_model,
                overlap, 
                segments_size, 
                hop_length, 
                batch_size,
                shifts, 
                window_size, 
                aggression, 
                post_process_threshold,
                denoise_model,
                enable_tta, 
                high_end_process, 
                enable_post_process,
            ]
        )
        denoise_model.change(
            fn=create_dataset_change,
            inputs=[
                model_name, 
                reverb_model, 
                enable_post_process, 
                separate_reverb, 
                enable_denoise
            ],
            outputs=[
                reverb_model,
                overlap, 
                segments_size, 
                hop_length, 
                batch_size,
                shifts, 
                window_size, 
                aggression, 
                post_process_threshold,
                denoise_model,
                enable_tta, 
                high_end_process, 
                enable_post_process,
            ]
        )
    with gr.Row():
        separate_reverb.change(
            fn=create_dataset_change,
            inputs=[
                model_name, 
                reverb_model, 
                enable_post_process, 
                separate_reverb, 
                enable_denoise
            ],
            outputs=[
                reverb_model,
                overlap, 
                segments_size, 
                hop_length, 
                batch_size,
                shifts, 
                window_size, 
                aggression, 
                post_process_threshold,
                denoise_model,
                enable_tta, 
                high_end_process, 
                enable_post_process,
            ]
        )
        enable_denoise.change(
            fn=create_dataset_change,
            inputs=[
                model_name, 
                reverb_model, 
                enable_post_process, 
                separate_reverb, 
                enable_denoise
            ],
            outputs=[
                reverb_model,
                overlap, 
                segments_size, 
                hop_length, 
                batch_size,
                shifts, 
                window_size, 
                aggression, 
                post_process_threshold,
                denoise_model,
                enable_tta, 
                high_end_process, 
                enable_post_process,
            ]
        )
    with gr.Row():
        skip_seconds.change(
            fn=lambda a: [visible(a) for _ in range(2)],
            inputs=[skip_seconds],
            outputs=[
                skip_start,
                skip_end
            ]
        )
        clean_dataset.change(
            fn=visible,
            inputs=[clean_dataset],
            outputs=[clean_strength]
        )
    with gr.Row():
        model_name.change(
            fn=lambda a: valueFalse_interactive(a in list(mdx_models.keys()) + list(vr_models.keys())), 
            inputs=[model_name], 
            outputs=[enable_denoise]
        )
        separate_reverb.change(
            fn=valueFalse_interactive, 
            inputs=[separate_reverb], 
            outputs=[enable_denoise]
        )
    with gr.Row():
        create_dataset_button.click(
            fn=create_dataset,
            inputs=[
                dataset_url,
                output_dataset,
                skip_seconds,
                skip_start,
                skip_end,
                separate,
                model_name,
                reverb_model,
                denoise_model,
                sample_rate,
                shifts,
                batch_size,
                overlap,
                aggression,
                hop_length,
                window_size,
                segments_size,
                post_process_threshold,
                enable_tta,
                enable_denoise,
                high_end_process,
                enable_post_process,
                separate_reverb,
                clean_dataset,
                clean_strength
            ],
            outputs=[create_dataset_info],
            api_name="create_dataset"
        )
import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from main.app.core.training import create_reference
from main.app.core.ui import visible, change_audios_choices, unlock_f0, shutil_move, change_embedders_mode
from main.app.variables import translations, paths_for_files, method_f0, hybrid_f0_method, file_types, configs, embedders_model, embedders_mode

def create_reference_tab():
    with gr.Row():
        gr.Markdown(translations["create_reference_markdown_2"])
    with gr.Row():
        pitch_guidance = gr.Checkbox(label=translations["training_pitch"], value=True, interactive=True)
        use_energy = gr.Checkbox(label=translations["train&energy"], value=False, interactive=True)
        f0_autotune = gr.Checkbox(label=translations["autotune"], value=False, interactive=True)
        proposal_pitch = gr.Checkbox(label=translations["proposal_pitch"], value=False, interactive=True)
    with gr.Row():
        create_reference_button = gr.Button(translations["create_reference"], variant="primary")
    with gr.Row():
        f0_up_key = gr.Slider(minimum=-20, maximum=20, step=1, info=translations["pitch_info"], label=translations["pitch"], value=0, interactive=True)
        proposal_pitch_threshold = gr.Slider(minimum=50.0, maximum=1200.0, label=translations["proposal_pitch_threshold"], info=translations["proposal_pitch_threshold_info"], value=255.0, step=0.1, interactive=True, visible=proposal_pitch.value)
    with gr.Row():
        filter_radius = gr.Slider(minimum=0, maximum=7, label=translations["filter_radius"], info=translations["filter_radius_info"], value=3, step=1, interactive=True)
        f0_autotune_strength = gr.Slider(minimum=0, maximum=1, label=translations["autotune_rate"], info=translations["autotune_rate_info"], value=1, step=0.1, interactive=True, visible=f0_autotune.value)
    with gr.Row():
        with gr.Column():
            with gr.Accordion(translations["input_output"], open=False):
                with gr.Column():
                    input_audio = gr.Dropdown(label=translations["audio_path"], value="", choices=paths_for_files, info=translations["provide_audio"], allow_custom_value=True, interactive=True)
                    reference_name = gr.Textbox(label=translations["reference_name"], value="reference", placeholder="reference", info=translations["reference_name_info"], interactive=True)
                with gr.Column():
                    refresh_audio = gr.Button(translations["refresh"])
                with gr.Column():
                    upload_audio = gr.Files(label=translations["drop_audio"], file_types=file_types)
                with gr.Column():
                    play_audio = gr.Audio(show_download_button=True, interactive=False, label=translations["input_audio"])
        with gr.Column() as f0_method_column:
            with gr.Accordion(label=translations["f0_method"], open=False):
                with gr.Group():
                    with gr.Row():
                        onnx_f0 = gr.Checkbox(label=translations["f0_onnx_mode"], value=False, interactive=True)
                        unlock_full_method = gr.Checkbox(label=translations["f0_unlock"], value=False, interactive=True)
                    f0_method = gr.Radio(label=translations["f0_method"], info=translations["f0_method_info"], choices=method_f0, value="rmvpe", interactive=True)
                    f0_hybrid_method = gr.Dropdown(label=translations["f0_method_hybrid"], info=translations["f0_method_hybrid_info"], choices=hybrid_f0_method, value=hybrid_f0_method[0], interactive=True, allow_custom_value=True, visible=f0_method.value == "hybrid")
                    with gr.Row():
                        alpha = gr.Slider(label=translations["alpha_label"], info=translations["alpha_info"], minimum=0.1, maximum=1, value=0.5, step=0.1, interactive=True, visible=False)
        with gr.Column():
            with gr.Accordion(label=translations["hubert_model"], open=False):
                with gr.Row():
                    version = gr.Radio(label=translations["training_version"], info=translations["training_version_info"], choices=["v1", "v2"], value="v2", interactive=True) 
                with gr.Group():
                    embedder_mode = gr.Radio(label=translations["embed_mode"], info=translations["embed_mode_info"], value="fairseq", choices=embedders_mode, interactive=True, visible=True)
                    embedders = gr.Radio(label=translations["hubert_model"], info=translations["hubert_info"], choices=embedders_model, value="hubert_base", interactive=True)
                with gr.Row():
                    embedders_custom = gr.Textbox(label=translations["modelname"], info=translations["modelname_info"], value="", placeholder="hubert_base", interactive=True, visible=embedders.value == "custom")
    with gr.Row():
        create_reference_info = gr.Textbox(label=translations["reference_info"], value="", interactive=False, lines=2)
    with gr.Row():
        f0_autotune.change(fn=visible, inputs=[f0_autotune], outputs=[f0_autotune_strength])
        proposal_pitch.change(fn=visible, inputs=[proposal_pitch], outputs=[proposal_pitch_threshold])
        unlock_full_method.change(fn=unlock_f0, inputs=[unlock_full_method], outputs=[f0_method])
    with gr.Row():
        input_audio.change(fn=lambda audio: audio, inputs=[input_audio], outputs=[play_audio])
        refresh_audio.click(fn=change_audios_choices, inputs=[input_audio], outputs=[input_audio])
        f0_method.change(fn=lambda method: [visible(method == "hybrid") for _ in range(2)], inputs=[f0_method], outputs=[f0_hybrid_method, alpha])
    with gr.Row():
        upload_audio.upload(fn=lambda audio_in: [shutil_move(audio.name, configs["audios_path"]) for audio in audio_in][0], inputs=[upload_audio], outputs=[input_audio])
        embedder_mode.change(fn=change_embedders_mode, inputs=[embedder_mode], outputs=[embedders])
        embedders.change(fn=lambda embedders: visible(embedders == "custom"), inputs=[embedders], outputs=[embedders_custom])
    with gr.Row():
        pitch_guidance.change(fn=visible, inputs=[pitch_guidance], outputs=[f0_method_column])
        create_reference_button.click(
            fn=create_reference,
            inputs=[
                input_audio, 
                reference_name, 
                pitch_guidance, 
                use_energy, 
                version, 
                embedders, 
                embedder_mode, 
                f0_method, 
                onnx_f0, 
                f0_up_key, 
                filter_radius, 
                f0_autotune, 
                f0_autotune_strength, 
                proposal_pitch, 
                proposal_pitch_threshold,
                alpha
            ],
            outputs=[create_reference_info],
            api_name="create_reference"
        )
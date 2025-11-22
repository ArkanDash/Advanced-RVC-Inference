import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from main.app.core.editing import apply_voice_quirk
from main.app.core.ui import change_audios_choices, shutil_move
from main.app.variables import translations, paths_for_files, configs, file_types, export_format_choices

def quirk_tab():
    with gr.Row():
        gr.Markdown(translations["quirk_markdown"])
    with gr.Row():
        input_audio_play = gr.Audio(show_download_button=True, interactive=False, label=translations["input_audio"])
    with gr.Row():
        quirk_choice = gr.Radio(label=translations["quirk_label"], info=translations["quirk_label_info"], choices=list(translations["quirk_choice"].keys()), interactive=True, value=list(translations["quirk_choice"].keys())[0])
    with gr.Row():
        apply_quirk_button = gr.Button(translations["apply"], variant="primary")
    with gr.Row():
        with gr.Accordion(translations["input_output"], open=False):
            with gr.Row():
                quirk_upload_audio = gr.Files(label=translations["drop_audio"], file_types=file_types)
            with gr.Column():
                quirk_export_format = gr.Radio(label=translations["export_format"], info=translations["export_info"], choices=export_format_choices, value="wav", interactive=True)
                quirk_input_path = gr.Dropdown(label=translations["audio_path"], value="", choices=paths_for_files, info=translations["provide_audio"], allow_custom_value=True, interactive=True)
                quirk_output_path = gr.Textbox(label=translations["output_path"], value="audios/output.wav", placeholder="audios/output.wav", info=translations["output_path_info"], interactive=True)
            with gr.Column():
                quirk_refresh = gr.Button(translations["refresh"])
    with gr.Row():
        output_audio_play = gr.Audio(show_download_button=True, interactive=False, label=translations["output_audio"])
    with gr.Row():
        quirk_upload_audio.upload(fn=lambda audio_in: [shutil_move(audio.name, configs["audios_path"]) for audio in audio_in][0], inputs=[quirk_upload_audio], outputs=[quirk_input_path])
        quirk_input_path.change(fn=lambda audio: audio if audio else None, inputs=[quirk_input_path], outputs=[input_audio_play])
        quirk_refresh.click(fn=change_audios_choices, inputs=[quirk_input_path], outputs=[quirk_input_path])
    with gr.Row():
        apply_quirk_button.click(
            fn=apply_voice_quirk,
            inputs=[
                quirk_input_path,
                quirk_choice,
                quirk_output_path,
                quirk_export_format
            ],
            outputs=[output_audio_play],
            api_name="quirk"
        )
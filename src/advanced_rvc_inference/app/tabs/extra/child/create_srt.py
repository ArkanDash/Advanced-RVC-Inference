import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from main.app.core.csrt import create_srt
from main.app.core.ui import shutil_move, change_audios_choices
from main.app.variables import translations, file_types, configs, paths_for_files

def create_srt_tab():
    with gr.Row():
        gr.Markdown(translations["create_srt_markdown_2"])
    with gr.Row():
        with gr.Column():
            srt_content = gr.Textbox(label=translations["srt_content"], value="", lines=9, max_lines=9, interactive=False)
        with gr.Column():
            word_timestamps = gr.Checkbox(label=translations["word_timestamps"], info=translations["word_timestamps_info"], value=False, interactive=True)
            model_size = gr.Radio(label=translations["model_size"], info=translations["model_size_info"], choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large-v1", "large-v2", "large-v3", "large-v3-turbo"], value="medium", interactive=True)
    with gr.Row():
        convert_button = gr.Button(translations["convert_audio"], variant="primary")
    with gr.Row():
        with gr.Accordion(translations["input_output"], open=False):
            with gr.Column():
                input_audio = gr.Dropdown(label=translations["audio_path"], value="", choices=paths_for_files, info=translations["provide_audio"], allow_custom_value=True, interactive=True)
                output_file = gr.Textbox(label=translations["srt_output_file"], value="srt/output.srt", placeholder="srt/output.srt", interactive=True)
            with gr.Column():
                refresh = gr.Button(translations["refresh"])
            with gr.Row():
                input_file = gr.Files(label=translations["drop_audio"], file_types=file_types)
    with gr.Row():
        play_audio = gr.Audio(show_download_button=True, interactive=False, label=translations["input_audio"])
    with gr.Row():
        output_srt = gr.File(label=translations["srt_output_file"], file_types=[".srt"], interactive=False, visible=False)
    with gr.Row():
        input_file.upload(fn=lambda audio_in: [shutil_move(audio.name, configs["audios_path"]) for audio in audio_in][0], inputs=[input_file], outputs=[input_audio])
        input_audio.change(fn=lambda audio: audio if os.path.isfile(audio) else None, inputs=[input_audio], outputs=[play_audio])
        refresh.click(fn=change_audios_choices, inputs=[input_audio], outputs=[input_audio])
    with gr.Row():
        convert_button.click(
            fn=create_srt,
            inputs=[
                model_size, 
                input_audio, 
                output_file, 
                word_timestamps
            ],
            outputs=[
                output_srt,
                srt_content
            ],
            api_name="create_srt"
        )



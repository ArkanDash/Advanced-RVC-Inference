import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from main.app.core.downloads import download_url
from main.app.core.separate import separate_music
from main.app.core.ui import visible, valueFalse_interactive, change_audios_choices, shutil_move, separate_change
from main.app.variables import translations, uvr_model, karaoke_models, reverb_models, vr_models, denoise_models, mdx_models, paths_for_files, sample_rate_choice, configs, file_types, export_format_choices

def separate_tab():
    with gr.Row(): 
        gr.Markdown(translations["4_part"])
    with gr.Row():
        with gr.Column():
            with gr.Group():
                with gr.Row():
                    enable_denoise = gr.Checkbox(label=translations["denoise_mdx"], value=False, interactive=False)
                    separate_backing = gr.Checkbox(label=translations["separator_backing"], value=False, interactive=True)
                    separate_reverb = gr.Checkbox(label=translations["dereveb_audio"], value=False, interactive=True)
                    enable_tta = gr.Checkbox(label=translations["enable_tta"], value=False, interactive=False)
                    high_end_process = gr.Checkbox(label=translations["high_end_process"], value=False, interactive=False)
                    enable_post_process = gr.Checkbox(label=translations["enable_post_process"], value=False, interactive=False)
                with gr.Row():
                    model_name = gr.Dropdown(label=translations["separator_model"], value=uvr_model[0], choices=uvr_model, interactive=True)
                    karaoke_model = gr.Dropdown(label=translations["separator_backing_model"], value=list(karaoke_models.keys())[0], choices=list(karaoke_models.keys()), interactive=True, visible=separate_backing.value)
                    reverb_model = gr.Dropdown(label=translations["dereveb_model"], value=list(reverb_models.keys())[0], choices=list(reverb_models.keys()), interactive=True, visible=separate_reverb.value)
                    denoise_model = gr.Dropdown(label=translations["denoise_model"], value=list(denoise_models.keys())[0], choices=list(denoise_models.keys()), interactive=True, visible=enable_denoise.value and model_name.value in list(vr_models.keys()))
    with gr.Row():
        with gr.Column():
            separate_button = gr.Button(translations["separator_tab"], variant="primary")
    with gr.Row():
        with gr.Column():
            with gr.Group():
                with gr.Row():
                    shifts = gr.Slider(label=translations["shift"], info=translations["shift_info"], minimum=1, maximum=20, value=2, step=1, interactive=True)
                    batch_size = gr.Slider(label=translations["batch_size"], info=translations["mdx_batch_size_info"], minimum=1, maximum=64, value=1, step=1, interactive=True, visible=False)
                with gr.Row():
                    segments_size = gr.Slider(label=translations["segments_size"], info=translations["segments_size_info"], minimum=32, maximum=3072, value=256, step=32, interactive=True)
                    aggression = gr.Slider(label=translations['aggression'], info=translations["aggression_info"], minimum=1, maximum=50, value=5, step=1, interactive=True, visible=False)
            drop_audio = gr.Files(label=translations["drop_audio"], file_types=file_types)    
            with gr.Accordion(translations["use_url"], open=False):
                url = gr.Textbox(label=translations["url_audio"], value="", placeholder="https://www.youtube.com/...", scale=6)
                download_button = gr.Button(translations["downloads"])
        with gr.Column():
            with gr.Group():
                with gr.Row():
                    overlap = gr.Radio(label=translations["overlap"], info=translations["overlap_info"], choices=["0.25", "0.5", "0.75", "0.99"], value="0.25", interactive=True)
                with gr.Row():
                    window_size = gr.Slider(label=translations["window_size"], info=translations["window_size_info"], minimum=320, maximum=1024, value=512, step=32, interactive=True, visible=False)
                    hop_length = gr.Slider(label=translations['hop_length'], info=translations["hop_length_info"], minimum=64, maximum=8192, value=1024, step=1, interactive=True, visible=False)
                    post_process_threshold = gr.Slider(label=translations['post_process_threshold'], info=translations["post_process_threshold_info"], minimum=0.1, maximum=0.3, value=0.2, step=0.1, interactive=True, visible=False)
            sample_rate = gr.Radio(choices=sample_rate_choice, value=44100, label=translations["sr"], info=translations["sr_info"], interactive=True)
            with gr.Accordion(translations["input_output"], open=False):
                export_format = gr.Radio(label=translations["export_format"], info=translations["export_info"], choices=export_format_choices, value="wav", interactive=True)
                input_audio = gr.Dropdown(label=translations["audio_path"], value="", choices=paths_for_files, allow_custom_value=True, interactive=True)
                refresh_audio = gr.Button(translations["refresh"])
                output_dirs = gr.Textbox(label=translations["output_folder"], value="audios", placeholder="audios", info=translations["output_folder_info"], interactive=True)
            audio_input = gr.Audio(show_download_button=True, interactive=False, label=translations["input_audio"])
    with gr.Row():
        gr.Markdown(translations["output_separator"])
    with gr.Row():
        instruments_audio = gr.Audio(show_download_button=True, interactive=False, label=translations["instruments"])
        original_vocals = gr.Audio(show_download_button=True, interactive=False, label=translations["original_vocal"])
        main_vocals = gr.Audio(show_download_button=True, interactive=False, label=translations["main_vocal"], visible=separate_backing.value)
        backing_vocals = gr.Audio(show_download_button=True, interactive=False, label=translations["backing_vocal"], visible=separate_backing.value)
    with gr.Row():
        model_name.change(fn=lambda a: valueFalse_interactive(a in list(mdx_models.keys()) + list(vr_models.keys())), inputs=[model_name], outputs=[enable_denoise])
        separate_backing.change(fn=lambda a, b: valueFalse_interactive(a or b), inputs=[separate_backing, separate_reverb], outputs=[enable_denoise])
        separate_reverb.change(fn=lambda a, b: valueFalse_interactive(a or b), inputs=[separate_backing, separate_reverb], outputs=[enable_denoise])
    with gr.Row():
        input_audio.change(fn=lambda audio: audio if os.path.isfile(audio) else None, inputs=[input_audio], outputs=[audio_input])
        drop_audio.upload(fn=lambda audio_in: [shutil_move(audio.name, configs["audios_path"]) for audio in audio_in][0], inputs=[drop_audio], outputs=[input_audio])
        refresh_audio.click(fn=change_audios_choices, inputs=[input_audio], outputs=[input_audio])
    with gr.Row():
        separate_backing.change(fn=lambda a: [visible(a) for _ in range(2)], inputs=[separate_backing], outputs=[main_vocals, backing_vocals])
        download_button.click(
            fn=download_url, 
            inputs=[url], 
            outputs=[input_audio, audio_input, url],
            api_name='download_url'
        )
    with gr.Row():
        model_name.change(
            fn=separate_change,
            inputs=[model_name, karaoke_model, reverb_model, enable_post_process, separate_backing, separate_reverb, enable_denoise],
            outputs=[
                karaoke_model,
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
        karaoke_model.change(
            fn=separate_change, 
            inputs=[model_name, karaoke_model, reverb_model, enable_post_process, separate_backing, separate_reverb, enable_denoise], 
            outputs=[
                karaoke_model,
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
        separate_backing.change(
            fn=separate_change, 
            inputs=[model_name, karaoke_model, reverb_model, enable_post_process, separate_backing, separate_reverb, enable_denoise], 
            outputs=[
                karaoke_model,
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
        reverb_model.change(
            fn=separate_change, 
            inputs=[model_name, karaoke_model, reverb_model, enable_post_process, separate_backing, separate_reverb, enable_denoise], 
            outputs=[
                karaoke_model,
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
        separate_reverb.change(
            fn=separate_change, 
            inputs=[model_name, karaoke_model, reverb_model, enable_post_process, separate_backing, separate_reverb, enable_denoise], 
            outputs=[
                karaoke_model,
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
        enable_denoise.change(
            fn=separate_change, 
            inputs=[model_name, karaoke_model, reverb_model, enable_post_process, separate_backing, separate_reverb, enable_denoise], 
            outputs=[
                karaoke_model,
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
        enable_post_process.change(
            fn=separate_change, 
            inputs=[model_name, karaoke_model, reverb_model, enable_post_process, separate_backing, separate_reverb, enable_denoise], 
            outputs=[
                karaoke_model,
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
        separate_button.click(
            fn=separate_music,
            inputs=[
                input_audio,
                output_dirs,
                export_format, 
                model_name, 
                karaoke_model,
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
                separate_backing,
                separate_reverb
            ],
            outputs=[
                original_vocals, 
                instruments_audio, 
                main_vocals, 
                backing_vocals
            ],
            api_name="separate_music"
        )
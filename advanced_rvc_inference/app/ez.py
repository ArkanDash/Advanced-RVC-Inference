import os
import sys
import shutil
import gradio as gr

sys.path.append(os.getcwd())

# Import from advanced_rvc_inference
from advanced_rvc_inference.core.presets import load_presets, save_presets
from advanced_rvc_inference.rvc.infer.inference import convert_audio, convert_selection
from advanced_rvc_inference.core.process import zip_file, save_drop_model
from advanced_rvc_inference.core.training import preprocess, extract, create_index, training
from advanced_rvc_inference.core.downloads import search_models, download_model, download_pretrained_model, download_url
from advanced_rvc_inference.core.separate import separate_music
from advanced_rvc_inference.utils.variables import (
    translations, paths_for_files, sample_rate_choice, model_name, index_path, 
    method_f0, f0_file, embedders_mode, embedders_model, presets_file, configs, 
    file_types, export_format_choices, hybrid_f0_method, models, model_options,
    pretrainedD, pretrainedG, config, reference_list, uvr_model, karaoke_models,
    reverb_models, vr_models, denoise_models, mdx_models
)
from advanced_rvc_inference.core.ui import (
    visible, valueFalse_interactive, change_audios_choices, change_f0_choices, 
    unlock_f0, change_preset_choices, change_backing_choices, hoplength_show, 
    change_models_choices, get_index, index_strength_show, change_embedders_mode, 
    shutil_move, gr_warning, get_gpu_info, pitch_guidance_lock, vocoders_lock, 
    unlock_ver, unlock_vocoder, change_pretrained_choices, gpu_number_str, 
    change_reference_choices, change_download_choices, change_download_pretrained_choices,
    separate_change
)
from advanced_rvc_inference.core.process import fetch_pretrained_data, update_sample_rate_dropdown

import glob

os.makedirs("dataset", exist_ok=True)

# Remove TTS functionality since whisperspeak is not available
tts_available = False

with gr.Blocks(title="🔊 Advanced RVC Inference", theme=gr.themes.Base(primary_hue="rose", neutral_hue="zinc")) as app:
    with gr.Row():
        gr.HTML("<h1>Advanced RVC Inference</h1>")
    
    with gr.Tabs():
        # ===== INFERENCE/CONVERT TAB =====
        with gr.TabItem("Inference"):
            gr.Markdown(translations["convert_info"])
            
            with gr.Row():
                model_pth = gr.Dropdown(
                    label=translations["model_name"], 
                    choices=model_name, 
                    value=model_name[0] if len(model_name) >= 1 else "", 
                    interactive=True, 
                    allow_custom_value=True
                )
                model_index = gr.Dropdown(
                    label=translations["index_path"], 
                    choices=index_path, 
                    value=index_path[0] if len(index_path) >= 1 else "", 
                    interactive=True, 
                    allow_custom_value=True
                )
                refresh = gr.Button(translations["refresh"])
            
            with gr.Row():
                pitch = gr.Slider(
                    minimum=-20, 
                    maximum=20, 
                    step=1, 
                    info=translations["pitch_info"], 
                    label=translations["pitch"], 
                    value=0, 
                    interactive=True
                )
            
            clean_strength0 = gr.Slider(
                label=translations["clean_strength"], 
                info=translations["clean_strength_info"], 
                minimum=0, 
                maximum=1, 
                value=0.5, 
                step=0.1, 
                interactive=True, 
                visible=False
            )
            
            with gr.Row():
                with gr.Column():
                    audio_select = gr.Dropdown(
                        label=translations["select_separate"], 
                        choices=[], 
                        value="", 
                        interactive=True, 
                        allow_custom_value=True, 
                        visible=False
                    )
                    convert_button_2 = gr.Button(translations["convert_audio"], visible=False)
            
            with gr.Row():
                with gr.Column():
                    convert_button = gr.Button(translations["convert_audio"], variant="primary")
            
            with gr.Row():
                with gr.Column():
                    with gr.Tabs():
                        with gr.TabItem("Upload"):
                            input0 = gr.Files(label=translations["drop_audio"], file_types=file_types)
                        with gr.TabItem("Record"):
                            record_button = gr.Audio(sources="microphone", label="Record Audio", type="filepath")
                    
                    play_audio = gr.Audio(interactive=True, label=translations["input_audio"])
                
                with gr.Column():
                    index_strength = gr.Slider(
                        label=translations["index_strength"], 
                        info=translations["index_strength_info"], 
                        minimum=0, 
                        maximum=1, 
                        value=0.5, 
                        step=0.01, 
                        interactive=True, 
                        visible=False
                    )
                    
                    with gr.Accordion(translations["input_output"], open=False):
                        with gr.Column():
                            export_format = gr.Radio(
                                label=translations["export_format"], 
                                info=translations["export_info"], 
                                choices=export_format_choices, 
                                value="wav", 
                                interactive=True
                            )
                            input_audio0 = gr.Dropdown(
                                label=translations["audio_path"], 
                                value="", 
                                choices=paths_for_files, 
                                info=translations["provide_audio"], 
                                allow_custom_value=True, 
                                interactive=True
                            )
                            output_audio = gr.Textbox(
                                label=translations["output_path"], 
                                value="advanced_rvc_inference/assets/audios/rvc/output.wav", 
                                placeholder="audios/output.wav", 
                                info=translations["output_path_info"], 
                                interactive=True
                            )
                        
                        refresh0 = gr.Button(translations["refresh"])
                    
                    with gr.Accordion(translations["setting"], open=False):
                        with gr.Column():
                            with gr.Row():
                                cleaner0 = gr.Checkbox(label=translations["clear_audio"], value=False, interactive=True)
                                autotune = gr.Checkbox(label=translations["autotune"], value=False, interactive=True)
                                use_audio = gr.Checkbox(label=translations["use_audio"], value=False, interactive=True)
                            
                            with gr.Row():
                                split_audio = gr.Checkbox(label=translations["split_audio"], value=False, interactive=True)
                                formant_shifting = gr.Checkbox(label=translations["formantshift"], value=False, interactive=True)
                            
                            with gr.Row():
                                proposal_pitch = gr.Checkbox(label=translations["proposal_pitch"], value=False, interactive=True)
                                audio_processing = gr.Checkbox(label=translations["audio_processing"], value=False, interactive=True)
                            
                            resample_sr = gr.Radio(
                                choices=[0] + sample_rate_choice, 
                                label=translations["resample"], 
                                info=translations["resample_info"], 
                                value=0, 
                                interactive=True
                            )
                            autotune_strength = gr.Slider(
                                minimum=0,
                                maximum=1,
                                label=translations["autotune_rate"],
                                info=translations["autotune_rate_info"],
                                value=1,
                                step=0.1, 
                                interactive=True, 
                                visible=False
                            )
                            filter_radius = gr.Slider(
                                minimum=0, 
                                maximum=7, 
                                label=translations["filter_radius"], 
                                info=translations["filter_radius_info"], 
                                value=3, 
                                step=1, 
                                interactive=True
                            )
                            rms_mix_rate = gr.Slider(
                                minimum=0, 
                                maximum=1, 
                                label=translations["rms_mix_rate"], 
                                info=translations["rms_mix_rate_info"], 
                                value=1, 
                                step=0.1, 
                                interactive=True
                            )
                            protect = gr.Slider(
                                minimum=0, 
                                maximum=1, 
                                label=translations["protect"], 
                                info=translations["protect_info"], 
                                value=0.5, 
                                step=0.01, 
                                interactive=True
                            )
            
            with gr.Row():
                gr.Markdown(translations["output_convert"])
            
            with gr.Row():
                main_convert = gr.Audio(interactive=False, label=translations["main_convert"])
            
            # ===== Event Handlers =====
            refresh.click(fn=change_models_choices, inputs=[], outputs=[model_pth, model_index])
            model_pth.change(fn=get_index, inputs=[model_pth], outputs=[model_index])
            model_index.change(fn=index_strength_show, inputs=[model_index], outputs=[index_strength])
            
            record_button.stop_recording(
                fn=lambda audio: audio.name,
                inputs=[record_button],
                outputs=[input_audio0]
            )
            
            input0.upload(
                fn=lambda audio_in: [shutil_move(audio.name, configs["audios_path"]) for audio in audio_in][0],
                inputs=[input0],
                outputs=[input_audio0]
            )
            
            input_audio0.change(
                fn=lambda audio: audio if os.path.isfile(audio) else None,
                inputs=[input_audio0],
                outputs=[play_audio]
            )
            
            refresh0.click(fn=change_audios_choices, inputs=[input_audio0], outputs=[input_audio0])
            
            autotune.change(fn=visible, inputs=[autotune], outputs=[autotune_strength])
            cleaner0.change(fn=visible, inputs=[cleaner0], outputs=[clean_strength0])
            
            convert_button.click(
                fn=convert_selection,
                inputs=[
                    cleaner0, autotune, use_audio, pitch, clean_strength0,
                    model_pth, model_index, index_strength, input_audio0,
                    output_audio, export_format, resample_sr, filter_radius,
                    rms_mix_rate, protect, split_audio, formant_shifting
                ],
                outputs=[main_convert, convert_button],
                api_name="convert_selection"
            )
        
        # ===== UVR/Separation TAB =====
        with gr.TabItem("UVR - Vocal Separation"):
            with gr.Row(): 
                gr.Markdown(translations.get("4_part", "Vocal Separation"))
            
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        with gr.Row():
                            model_name = gr.Dropdown(
                                label=translations.get("separator_model", "Separation Model"), 
                                value=uvr_model[0] if uvr_model else "", 
                                choices=uvr_model, 
                                interactive=True
                            )
                            separate_backing = gr.Checkbox(
                                label=translations.get("separator_backing", "Separate Backing Vocals"), 
                                value=False, 
                                interactive=True
                            )
                            separate_reverb = gr.Checkbox(
                                label=translations.get("dereveb_audio", "Remove Reverb"), 
                                value=False, 
                                interactive=True
                            )
                    
                    with gr.Group():
                        with gr.Row():
                            shifts = gr.Slider(
                                label=translations.get("shift", "Shifts"), 
                                info=translations.get("shift_info", "Number of shifts for processing"), 
                                minimum=1, 
                                maximum=20, 
                                value=2, 
                                step=1, 
                                interactive=True
                            )
                            segments_size = gr.Slider(
                                label=translations.get("segments_size", "Segment Size"), 
                                info=translations.get("segments_size_info", "Size of segments for processing"), 
                                minimum=32, 
                                maximum=3072, 
                                value=256, 
                                step=32, 
                                interactive=True
                            )
                    
                    with gr.Tabs():
                        with gr.TabItem("Upload Audio"):
                            drop_audio = gr.Files(label=translations.get("drop_audio", "Drop Audio Files"), file_types=file_types)
                        with gr.TabItem("From URL"):
                            url = gr.Textbox(
                                label=translations.get("url_audio", "Audio URL"), 
                                value="", 
                                placeholder="https://www.youtube.com/...", 
                                scale=6
                            )
                            download_button = gr.Button(translations.get("downloads", "Download"))
                    
                    audio_input = gr.Audio(show_download_button=True, interactive=False, label=translations.get("input_audio", "Input Audio"))
                
                with gr.Column():
                    sample_rate = gr.Radio(
                        choices=sample_rate_choice, 
                        value=44100, 
                        label=translations.get("sr", "Sample Rate"), 
                        info=translations.get("sr_info", "Output sample rate"), 
                        interactive=True
                    )
                    
                    with gr.Accordion(translations.get("input_output", "Input/Output"), open=False):
                        input_audio = gr.Dropdown(
                            label=translations.get("audio_path", "Audio Path"), 
                            value="", 
                            choices=paths_for_files, 
                            allow_custom_value=True, 
                            interactive=True
                        )
                        refresh_audio = gr.Button(translations.get("refresh", "Refresh"))
                        output_dirs = gr.Textbox(
                            label=translations.get("output_folder", "Output Folder"), 
                            value=configs.get("uvr_path", "advanced_rvc_inference/assets/audios/uvr"), 
                            placeholder="audios", 
                            info=translations.get("output_folder_info", "Output directory for separated files"), 
                            interactive=True
                        )
                    
                    separate_button = gr.Button(
                        translations.get("separator_tab", "Separate Audio"), 
                        variant="primary"
                    )
            
            with gr.Row():
                gr.Markdown(translations.get("output_separator", "Separation Results"))
            
            with gr.Row():
                instruments_audio = gr.Audio(
                    show_download_button=True, 
                    interactive=False, 
                    label=translations.get("instruments", "Instruments")
                )
                original_vocals = gr.Audio(
                    show_download_button=True, 
                    interactive=False, 
                    label=translations.get("original_vocal", "Original Vocals")
                )
                main_vocals = gr.Audio(
                    show_download_button=True, 
                    interactive=False, 
                    label=translations.get("main_vocal", "Main Vocals"), 
                    visible=False
                )
                backing_vocals = gr.Audio(
                    show_download_button=True, 
                    interactive=False, 
                    label=translations.get("backing_vocal", "Backing Vocals"), 
                    visible=False
                )
            
            # ===== Event Handlers for UVR =====
            separate_backing.change(
                fn=lambda a: [visible(a) for _ in range(2)], 
                inputs=[separate_backing], 
                outputs=[main_vocals, backing_vocals]
            )
            
            input_audio.change(
                fn=lambda audio: audio if os.path.isfile(audio) else None, 
                inputs=[input_audio], 
                outputs=[audio_input]
            )
            
            drop_audio.upload(
                fn=lambda audio_in: [shutil_move(audio.name, configs["audios_path"]) for audio in audio_in][0], 
                inputs=[drop_audio], 
                outputs=[input_audio]
            )
            
            refresh_audio.click(
                fn=change_audios_choices, 
                inputs=[input_audio], 
                outputs=[input_audio]
            )
            
            download_button.click(
                fn=download_url, 
                inputs=[url], 
                outputs=[input_audio, audio_input, url],
                api_name='download_url'
            )
            
            separate_button.click(
                fn=separate_music,
                inputs=[
                    input_audio,
                    output_dirs,
                    model_name,
                    sample_rate,
                    shifts,
                    segments_size,
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
        
        # ===== TRAIN TAB =====
        with gr.TabItem("Train"):
            with gr.Row():
                gr.Markdown(translations["training_markdown"])
            
            with gr.Row():
                with gr.Column():
                    training_name = gr.Textbox(
                        label=translations["modelname"], 
                        info=translations["training_model_name"], 
                        value="", 
                        placeholder=translations["modelname"], 
                        interactive=True
                    )
                    
                    training_sr = gr.Radio(
                        label=translations["sample_rate"], 
                        info=translations["sample_rate_info"], 
                        choices=["32k", "40k", "48k"], 
                        value="48k", 
                        interactive=True
                    )
                    
                    training_ver = gr.Radio(
                        label=translations["training_version"], 
                        info=translations["training_version_info"], 
                        choices=["v1", "v2"], 
                        value="v2", 
                        interactive=True
                    )
                    
                    with gr.Row():
                        clean_dataset = gr.Checkbox(label=translations["clear_dataset"], value=False, interactive=True)
                        process_effects = gr.Checkbox(label=translations["preprocess_effect"], value=False, interactive=True)
                        training_f0 = gr.Checkbox(label=translations["training_pitch"], value=True, interactive=True)
                    
                    with gr.Row():
                        total_epochs = gr.Slider(
                            label=translations["total_epoch"], 
                            info=translations["total_epoch_info"], 
                            minimum=1, 
                            maximum=10000, 
                            value=300, 
                            step=1, 
                            interactive=True
                        )
                        save_epochs = gr.Slider(
                            label=translations["save_epoch"], 
                            info=translations["save_epoch_info"], 
                            minimum=1, 
                            maximum=10000, 
                            value=50, 
                            step=1, 
                            interactive=True
                        )
                    
                    with gr.Column():
                        dataset_path = gr.Textbox(
                            label=translations["dataset_folder"], 
                            value="advanced_rvc_inference/assets/dataset", 
                            interactive=True
                        )
                        upload_dataset = gr.Files(label=translations["drop_audio"], file_types=file_types)
                    
                    preprocess_button = gr.Button(translations["preprocess_button"], variant="primary")
                    preprocess_info = gr.Textbox(label=translations["preprocess_info"], value="", interactive=False, lines=2)
                
                with gr.Column():
                    with gr.Accordion(label=translations["f0_method"], open=False):
                        extract_method = gr.Radio(
                            label=translations["f0_method"], 
                            info=translations["f0_method_info"], 
                            choices=method_f0, 
                            value="rmvpe", 
                            interactive=True
                        )
                    
                    with gr.Accordion(label=translations["hubert_model"], open=False):
                        embed_mode2 = gr.Radio(
                            label=translations["embed_mode"], 
                            info=translations["embed_mode_info"], 
                            value="fairseq", 
                            choices=embedders_mode, 
                            interactive=True
                        )
                        extract_embedders = gr.Radio(
                            label=translations["hubert_model"], 
                            info=translations["hubert_info"], 
                            choices=embedders_model, 
                            value="hubert_base", 
                            interactive=True
                        )
                    
                    extract_button = gr.Button(translations["extract_button"], variant="primary")
                    extract_info = gr.Textbox(label=translations["extract_info"], value="", interactive=False, lines=2)
                
                with gr.Column():
                    index_button = gr.Button(f"3. {translations['create_index']}", variant="primary")
                    training_button = gr.Button(f"4. {translations['training_model']}", variant="primary")
                    
                    with gr.Accordion(label=translations["setting"], open=False):
                        with gr.Row():
                            cache_in_gpu = gr.Checkbox(
                                label=translations["cache_in_gpu"], 
                                info=translations["cache_in_gpu_info"], 
                                value=True, 
                                interactive=True
                            )
                            save_only_latest = gr.Checkbox(
                                label=translations["save_only_latest"], 
                                info=translations["save_only_latest_info"], 
                                value=True, 
                                interactive=True
                            )
                        
                        with gr.Accordion(translations["setting_cpu_gpu"], open=False):
                            gpu_info = gr.Textbox(
                                label=translations["gpu_info"], 
                                value=get_gpu_info(), 
                                info=translations["gpu_info_2"], 
                                interactive=False
                            )
                            cpu_core = gr.Slider(
                                label=translations["cpu_core"], 
                                info=translations["cpu_core_info"], 
                                minimum=1, 
                                maximum=os.cpu_count(), 
                                value=os.cpu_count(), 
                                step=1, 
                                interactive=True
                            )
                            train_batch_size = gr.Slider(
                                label=translations["batch_size"], 
                                info=translations["batch_size_info"], 
                                minimum=1, 
                                maximum=64, 
                                value=8, 
                                step=1, 
                                interactive=True
                            )
                    
                    training_info = gr.Textbox(label=translations["train_info"], value="", interactive=False, lines=3)
            
            # ===== Event Handlers =====
            upload_dataset.upload(
                fn=lambda files, folder: [shutil_move(f.name, os.path.join(folder, os.path.split(f.name)[1])) for f in files] if folder != "" else gr_warning(translations["dataset_folder1"]),
                inputs=[upload_dataset, dataset_path],
                outputs=[],
                api_name="upload_dataset"
            )
            
            embed_mode2.change(fn=change_embedders_mode, inputs=[embed_mode2], outputs=[extract_embedders])
            
            preprocess_button.click(
                fn=preprocess,
                inputs=[
                    training_name, training_sr, cpu_core,
                    process_effects, dataset_path, clean_dataset
                ],
                outputs=[preprocess_info],
                api_name="preprocess"
            )
            
            extract_button.click(
                fn=extract,
                inputs=[
                    training_name, training_ver, extract_method,
                    training_f0, cpu_core, training_sr,
                    extract_embedders, embed_mode2
                ],
                outputs=[extract_info],
                api_name="extract"
            )
            
            index_button.click(
                fn=create_index,
                inputs=[training_name, training_ver],
                outputs=[training_info],
                api_name="create_index"
            )
            
            training_button.click(
                fn=training,
                inputs=[
                    training_name, training_ver, save_epochs,
                    save_only_latest, total_epochs, training_sr,
                    train_batch_size, training_f0, cache_in_gpu
                ],
                outputs=[training_info],
                api_name="training_model"
            )
        
        # ===== DOWNLOAD TAB =====
        with gr.TabItem("Download Models"):
            with gr.Row():
                gr.Markdown(translations["download_markdown"])
            
            with gr.Row():
                with gr.Accordion(translations["model_download"], open=True):
                    with gr.Row():
                        download_method = gr.Radio(
                            label=translations["model_download_select"],
                            choices=[
                                translations["download_url"],
                                translations["download_from_csv"],
                                translations["search_models"],
                                translations["upload"]
                            ],
                            interactive=True,
                            value=translations["download_url"]
                        )
                    
                    with gr.Column():
                        with gr.Row():
                            download_model_url = gr.Textbox(
                                label=translations["model_url"],
                                value="",
                                placeholder="https://...",
                                scale=6
                            )
                            download_model_name = gr.Textbox(
                                label=translations["modelname"],
                                value="",
                                placeholder=translations["modelname"],
                                scale=2
                            )
                        download_url_button = gr.Button(
                            value=translations["downloads"],
                            scale=2,
                            variant="primary"
                        )
                    
                    with gr.Column():
                        download_csv_model = gr.Dropdown(
                            choices=models.keys(),
                            label=translations["model_warehouse"],
                            scale=8,
                            allow_custom_value=True,
                            visible=False
                        )
                        download_csv_button = gr.Button(
                            value=translations["get_model"],
                            scale=2,
                            variant="primary",
                            visible=False
                        )
            
            # ===== Event Handlers =====
            download_method.change(
                fn=change_download_choices,
                inputs=[download_method],
                outputs=[
                    download_model_url, download_model_name, download_url_button,
                    download_csv_model, download_csv_button
                ]
            )
            
            download_url_button.click(
                fn=download_model,
                inputs=[download_model_url, download_model_name],
                outputs=[download_model_url],
                api_name="download_model"
            )
            
            download_csv_button.click(
                fn=lambda model: download_model(models[model], model),
                inputs=[download_csv_model],
                outputs=[download_csv_model],
                api_name="download_browser"
            )
    
    # ===== LAUNCH =====
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch Advanced RVC Inference GUI")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--share", action="store_true", help="Create public URL (may fail in some environments)")
    parser.add_argument("--no-share", action="store_true", help="Disable public URL, use local access only")
    parser.add_argument("--localtunnel", action="store_true", help="Use localtunnel as fallback if share fails")
    parser.add_argument("--open", action="store_true", help="Open in browser")
    parser.add_argument("--keep-alive", action="store_true", default=True, help="Keep tunnel alive (default: True)")

    args = parser.parse_args()

    sys.exit(
        launch(
            share=args.share and not args.no_share,
            server_name=args.host,
            server_port=args.port,
            inbrowser=args.open,
            enable_localtunnel=args.localtunnel,
            keep_alive=args.keep_alive,
        )
    )

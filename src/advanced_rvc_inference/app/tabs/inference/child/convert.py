import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from main.app.core.presets import load_presets, save_presets
from main.app.core.inference import convert_audio, convert_selection
from main.app.variables import translations, paths_for_files, sample_rate_choice, model_name, index_path, method_f0, f0_file, embedders_mode, embedders_model, presets_file, configs, file_types, export_format_choices, hybrid_f0_method
from main.app.core.ui import visible, valueFalse_interactive, change_audios_choices, change_f0_choices, unlock_f0, change_preset_choices, change_backing_choices, hoplength_show, change_models_choices, get_index, index_strength_show, change_embedders_mode, shutil_move

def convert_tab():
    with gr.Row():
        gr.Markdown(translations["convert_info"])
    with gr.Row():
        with gr.Column():
            with gr.Group():
                with gr.Row():
                    cleaner0 = gr.Checkbox(label=translations["clear_audio"], value=False, interactive=True)
                    autotune = gr.Checkbox(label=translations["autotune"], value=False, interactive=True)
                    use_audio = gr.Checkbox(label=translations["use_audio"], value=False, interactive=True)
                    checkpointing = gr.Checkbox(label=translations["memory_efficient_training"], value=False, interactive=True)
                with gr.Row():
                    use_original = gr.Checkbox(label=translations["convert_original"], value=False, interactive=True, visible=use_audio.value) 
                    convert_backing = gr.Checkbox(label=translations["convert_backing"], value=False, interactive=True, visible=use_audio.value)   
                    not_merge_backing = gr.Checkbox(label=translations["not_merge_backing"], value=False, interactive=True, visible=use_audio.value)
                    merge_instrument = gr.Checkbox(label=translations["merge_instruments"], value=False, interactive=True, visible=use_audio.value) 
            with gr.Row():
                pitch = gr.Slider(minimum=-20, maximum=20, step=1, info=translations["pitch_info"], label=translations["pitch"], value=0, interactive=True)
                clean_strength0 = gr.Slider(label=translations["clean_strength"], info=translations["clean_strength_info"], minimum=0, maximum=1, value=0.5, step=0.1, interactive=True, visible=cleaner0.value)
            with gr.Row(): 
                with gr.Column():
                    audio_select = gr.Dropdown(label=translations["select_separate"], choices=[], value="", interactive=True, allow_custom_value=True, visible=False)
                    convert_button_2 = gr.Button(translations["convert_audio"], visible=False)
    with gr.Row():
        with gr.Column():
            convert_button = gr.Button(translations["convert_audio"], variant="primary")
    with gr.Row():
        with gr.Column():
            input0 = gr.Files(label=translations["drop_audio"], file_types=file_types)  
            play_audio = gr.Audio(show_download_button=True, interactive=False, label=translations["input_audio"])
        with gr.Column():
            with gr.Accordion(translations["model_accordion"], open=True):
                with gr.Row():
                    model_pth = gr.Dropdown(label=translations["model_name"], choices=model_name, value=model_name[0] if len(model_name) >= 1 else "", interactive=True, allow_custom_value=True)
                    model_index = gr.Dropdown(label=translations["index_path"], choices=index_path, value=index_path[0] if len(index_path) >= 1 else "", interactive=True, allow_custom_value=True)
                with gr.Row():
                    refresh = gr.Button(translations["refresh"])
                with gr.Row():
                    index_strength = gr.Slider(label=translations["index_strength"], info=translations["index_strength_info"], minimum=0, maximum=1, value=0.5, step=0.01, interactive=True, visible=model_index.value != "")
            with gr.Accordion(translations["input_output"], open=False):
                with gr.Column():
                    export_format = gr.Radio(label=translations["export_format"], info=translations["export_info"], choices=export_format_choices, value="wav", interactive=True)
                    input_audio0 = gr.Dropdown(label=translations["audio_path"], value="", choices=paths_for_files, info=translations["provide_audio"], allow_custom_value=True, interactive=True)
                    output_audio = gr.Textbox(label=translations["output_path"], value="audios/output.wav", placeholder="audios/output.wav", info=translations["output_path_info"], interactive=True)
                with gr.Column():
                    refresh0 = gr.Button(translations["refresh"])
            with gr.Accordion(translations["setting"], open=False):
                with gr.Accordion(translations["f0_method"], open=False):
                    with gr.Group():
                        with gr.Row():
                            onnx_f0_mode = gr.Checkbox(label=translations["f0_onnx_mode"], info=translations["f0_onnx_mode_info"], value=False, interactive=True)
                            unlock_full_method = gr.Checkbox(label=translations["f0_unlock"], info=translations["f0_unlock_info"], value=False, interactive=True)
                        method = gr.Radio(label=translations["f0_method"], info=translations["f0_method_info"], choices=method_f0, value="rmvpe", interactive=True)
                        hybrid_method = gr.Dropdown(label=translations["f0_method_hybrid"], info=translations["f0_method_hybrid_info"], choices=hybrid_f0_method, value=hybrid_f0_method[0], interactive=True, allow_custom_value=True, visible=method.value == "hybrid")
                    hop_length = gr.Slider(label=translations['hop_length'], info=translations["hop_length_info"], minimum=64, maximum=512, value=160, step=1, interactive=True, visible=False)
                    alpha = gr.Slider(label=translations["alpha_label"], info=translations["alpha_info"], minimum=0.1, maximum=1, value=0.5, step=0.1, interactive=True, visible=False)
                with gr.Accordion(translations["f0_file"], open=False):
                    upload_f0_file = gr.File(label=translations["upload_f0"], file_types=[".txt"])  
                    f0_file_dropdown = gr.Dropdown(label=translations["f0_file_2"], value="", choices=f0_file, allow_custom_value=True, interactive=True)
                    refresh_f0_file = gr.Button(translations["refresh"])
                with gr.Accordion(translations["hubert_model"], open=False):
                    embed_mode = gr.Radio(label=translations["embed_mode"], info=translations["embed_mode_info"], value="fairseq", choices=embedders_mode, interactive=True, visible=True)
                    embedders = gr.Radio(label=translations["hubert_model"], info=translations["hubert_info"], choices=embedders_model, value="hubert_base", interactive=True)
                    custom_embedders = gr.Textbox(label=translations["modelname"], info=translations["modelname_info"], value="", placeholder="hubert_base", interactive=True, visible=embedders.value == "custom")    
                with gr.Accordion(translations["use_presets"], open=False):
                    with gr.Row():
                        presets_name = gr.Dropdown(label=translations["file_preset"], choices=presets_file, value=presets_file[0] if len(presets_file) > 0 else '', interactive=True, allow_custom_value=True)
                    with gr.Row():
                        load_click = gr.Button(translations["load_file"], variant="primary")
                        refresh_click = gr.Button(translations["refresh"])
                    with gr.Accordion(translations["export_file"], open=False):
                        with gr.Row():
                            with gr.Column():
                                with gr.Group():
                                    with gr.Row():
                                        cleaner_chbox = gr.Checkbox(label=translations["save_clean"], value=True, interactive=True)
                                        autotune_chbox = gr.Checkbox(label=translations["save_autotune"], value=True, interactive=True)
                                        pitch_chbox = gr.Checkbox(label=translations["save_pitch"], value=True, interactive=True)
                                        index_strength_chbox = gr.Checkbox(label=translations["save_index_2"], value=True, interactive=True)
                                        resample_sr_chbox = gr.Checkbox(label=translations["save_resample"], value=True, interactive=True)
                                        filter_radius_chbox = gr.Checkbox(label=translations["save_filter"], value=True, interactive=True)
                                        rms_mix_rate_chbox = gr.Checkbox(label=translations["save_envelope"], value=True, interactive=True)
                                        protect_chbox = gr.Checkbox(label=translations["save_protect"], value=True, interactive=True)
                                        split_audio_chbox = gr.Checkbox(label=translations["save_split"], value=True, interactive=True)
                                        formant_shifting_chbox = gr.Checkbox(label=translations["formantshift"], value=True, interactive=True)
                        with gr.Row():
                            with gr.Column():
                                name_to_save_file = gr.Textbox(label=translations["filename_to_save"])
                                save_file_button = gr.Button(translations["export_file"])
                    with gr.Row():
                        upload_presets = gr.Files(label=translations["upload_presets"], file_types=[".conversion.json"])  
                with gr.Column():
                    with gr.Group():
                        with gr.Row():
                            split_audio = gr.Checkbox(label=translations["split_audio"], value=False, interactive=True)
                            formant_shifting = gr.Checkbox(label=translations["formantshift"], value=False, interactive=True)
                        with gr.Row():
                            proposal_pitch = gr.Checkbox(label=translations["proposal_pitch"], value=False, interactive=True)
                            audio_processing = gr.Checkbox(label=translations["audio_processing"], value=False, interactive=True)
                    resample_sr = gr.Radio(choices=[0]+sample_rate_choice, label=translations["resample"], info=translations["resample_info"], value=0, interactive=True)
                    proposal_pitch_threshold = gr.Slider(minimum=50.0, maximum=1200.0, label=translations["proposal_pitch_threshold"], info=translations["proposal_pitch_threshold_info"], value=255.0, step=0.1, interactive=True, visible=proposal_pitch.value)
                    f0_autotune_strength = gr.Slider(minimum=0, maximum=1, label=translations["autotune_rate"], info=translations["autotune_rate_info"], value=1, step=0.1, interactive=True, visible=autotune.value)
                    filter_radius = gr.Slider(minimum=0, maximum=7, label=translations["filter_radius"], info=translations["filter_radius_info"], value=3, step=1, interactive=True)
                    rms_mix_rate = gr.Slider(minimum=0, maximum=1, label=translations["rms_mix_rate"], info=translations["rms_mix_rate_info"], value=1, step=0.1, interactive=True)
                    protect = gr.Slider(minimum=0, maximum=1, label=translations["protect"], info=translations["protect_info"], value=0.5, step=0.01, interactive=True)
                with gr.Row():
                    formant_qfrency = gr.Slider(value=1.0, label=translations["formant_qfrency"], info=translations["formant_qfrency"], minimum=0.0, maximum=16.0, step=0.1, interactive=True, visible=False)
                    formant_timbre = gr.Slider(value=1.0, label=translations["formant_timbre"], info=translations["formant_timbre"], minimum=0.0, maximum=16.0, step=0.1, interactive=True, visible=False)
    with gr.Row():
        gr.Markdown(translations["output_convert"])
    with gr.Row():
        main_convert = gr.Audio(show_download_button=True, interactive=False, label=translations["main_convert"])
        backing_convert = gr.Audio(show_download_button=True, interactive=False, label=translations["convert_backing"], visible=convert_backing.value)
        main_backing = gr.Audio(show_download_button=True, interactive=False, label=translations["main_or_backing"], visible=convert_backing.value)  
    with gr.Row():
        original_convert = gr.Audio(show_download_button=True, interactive=False, label=translations["convert_original"], visible=use_original.value)
        vocal_instrument = gr.Audio(show_download_button=True, interactive=False, label=translations["voice_or_instruments"], visible=merge_instrument.value)  
    with gr.Row():
        upload_f0_file.upload(fn=lambda inp: shutil_move(inp.name, configs["f0_path"]), inputs=[upload_f0_file], outputs=[f0_file_dropdown])
        refresh_f0_file.click(fn=change_f0_choices, inputs=[], outputs=[f0_file_dropdown])
        unlock_full_method.change(fn=unlock_f0, inputs=[unlock_full_method], outputs=[method])
    with gr.Row():
        load_click.click(
            fn=load_presets, 
            inputs=[
                presets_name, 
                cleaner0, 
                autotune, 
                pitch, 
                clean_strength0, 
                index_strength, 
                resample_sr, 
                filter_radius, 
                rms_mix_rate, 
                protect, 
                split_audio, 
                f0_autotune_strength,
                formant_shifting, 
                formant_qfrency, 
                formant_timbre,
                proposal_pitch,
                proposal_pitch_threshold
            ], 
            outputs=[
                cleaner0, 
                autotune, 
                pitch, 
                clean_strength0, 
                index_strength, 
                resample_sr, 
                filter_radius, 
                rms_mix_rate, 
                protect, 
                split_audio, 
                f0_autotune_strength, 
                formant_shifting, 
                formant_qfrency, 
                formant_timbre,
                proposal_pitch,
                proposal_pitch_threshold
            ]
        )
        refresh_click.click(fn=change_preset_choices, inputs=[], outputs=[presets_name])
        save_file_button.click(
            fn=save_presets, 
            inputs=[
                name_to_save_file, 
                cleaner0, 
                autotune, 
                pitch, 
                clean_strength0, 
                index_strength, 
                resample_sr, 
                filter_radius, 
                rms_mix_rate, 
                protect, 
                split_audio, 
                f0_autotune_strength, 
                cleaner_chbox, 
                autotune_chbox, 
                pitch_chbox, 
                index_strength_chbox, 
                resample_sr_chbox, 
                filter_radius_chbox, 
                rms_mix_rate_chbox, 
                protect_chbox, 
                split_audio_chbox, 
                formant_shifting_chbox, 
                formant_shifting, 
                formant_qfrency, 
                formant_timbre,
                proposal_pitch,
                proposal_pitch_threshold
            ], 
            outputs=[presets_name]
        )
    with gr.Row():
        upload_presets.upload(fn=lambda presets_in: [shutil_move(preset.name, configs["presets_path"]) for preset in presets_in][0], inputs=[upload_presets], outputs=[presets_name])
        autotune.change(fn=visible, inputs=[autotune], outputs=[f0_autotune_strength])
        use_audio.change(fn=lambda a: [visible(a), visible(a), visible(a), visible(a), visible(a), valueFalse_interactive(a), valueFalse_interactive(a), valueFalse_interactive(a), valueFalse_interactive(a), visible(not a), visible(not a), visible(not a), visible(not a)], inputs=[use_audio], outputs=[main_backing, use_original, convert_backing, not_merge_backing, merge_instrument, use_original, convert_backing, not_merge_backing, merge_instrument, input_audio0, output_audio, input0, play_audio])
    with gr.Row():
        convert_backing.change(fn=lambda a,b: [change_backing_choices(a, b), visible(a)], inputs=[convert_backing, not_merge_backing], outputs=[use_original, backing_convert])
        use_original.change(fn=lambda audio, original: [visible(original), visible(not original), visible(audio and not original), valueFalse_interactive(not original), valueFalse_interactive(not original)], inputs=[use_audio, use_original], outputs=[original_convert, main_convert, main_backing, convert_backing, not_merge_backing])
        cleaner0.change(fn=visible, inputs=[cleaner0], outputs=[clean_strength0])
    with gr.Row():
        merge_instrument.change(fn=visible, inputs=[merge_instrument], outputs=[vocal_instrument])
        not_merge_backing.change(fn=lambda audio, merge, cvb: [visible(audio and not merge), change_backing_choices(cvb, merge)], inputs=[use_audio, not_merge_backing, convert_backing], outputs=[main_backing, use_original])
        method.change(fn=lambda method, hybrid: [visible(method == "hybrid"), visible(method == "hybrid"), hoplength_show(method, hybrid)], inputs=[method, hybrid_method], outputs=[hybrid_method, alpha, hop_length])
    with gr.Row():
        hybrid_method.change(fn=hoplength_show, inputs=[method, hybrid_method], outputs=[hop_length])
        refresh.click(fn=change_models_choices, inputs=[], outputs=[model_pth, model_index])
        model_pth.change(fn=get_index, inputs=[model_pth], outputs=[model_index])
    with gr.Row():
        input0.upload(fn=lambda audio_in: [shutil_move(audio.name, configs["audios_path"]) for audio in audio_in][0], inputs=[input0], outputs=[input_audio0])
        input_audio0.change(fn=lambda audio: audio if os.path.isfile(audio) else None, inputs=[input_audio0], outputs=[play_audio])
        formant_shifting.change(fn=lambda a: [visible(a) for _ in range(2)], inputs=[formant_shifting], outputs=[formant_qfrency, formant_timbre])
    with gr.Row():
        embedders.change(fn=lambda embedders: visible(embedders == "custom"), inputs=[embedders], outputs=[custom_embedders])
        refresh0.click(fn=change_audios_choices, inputs=[input_audio0], outputs=[input_audio0])
        model_index.change(fn=index_strength_show, inputs=[model_index], outputs=[index_strength])
    with gr.Row():
        convert_button.click(fn=lambda: visible(False), inputs=[], outputs=[convert_button])
        convert_button_2.click(fn=lambda: [visible(False), visible(False)], inputs=[], outputs=[audio_select, convert_button_2])
    with gr.Row():
        proposal_pitch.change(fn=visible, inputs=[proposal_pitch], outputs=[proposal_pitch_threshold])
        embed_mode.change(fn=change_embedders_mode, inputs=[embed_mode], outputs=[embedders])
    with gr.Row():
        convert_button.click(
            fn=convert_selection,
            inputs=[
                cleaner0,
                autotune,
                use_audio,
                use_original,
                convert_backing,
                not_merge_backing,
                merge_instrument,
                pitch,
                clean_strength0,
                model_pth,
                model_index,
                index_strength,
                input_audio0,
                output_audio,
                export_format,
                method,
                hybrid_method,
                hop_length,
                embedders,
                custom_embedders,
                resample_sr,
                filter_radius,
                rms_mix_rate,
                protect,
                split_audio,
                f0_autotune_strength,
                checkpointing,
                onnx_f0_mode,
                formant_shifting, 
                formant_qfrency, 
                formant_timbre,
                f0_file_dropdown,
                embed_mode,
                proposal_pitch,
                proposal_pitch_threshold,
                audio_processing,
                alpha
            ],
            outputs=[audio_select, main_convert, backing_convert, main_backing, original_convert, vocal_instrument, convert_button, convert_button_2],
            api_name="convert_selection"
        )
        convert_button_2.click(
            fn=convert_audio,
            inputs=[
                cleaner0,
                autotune,
                use_audio,
                use_original,
                convert_backing,
                not_merge_backing,
                merge_instrument,
                pitch,
                clean_strength0,
                model_pth,
                model_index,
                index_strength,
                input_audio0,
                output_audio,
                export_format,
                method,
                hybrid_method,
                hop_length,
                embedders,
                custom_embedders,
                resample_sr,
                filter_radius,
                rms_mix_rate,
                protect,
                split_audio,
                f0_autotune_strength,
                audio_select,
                checkpointing,
                onnx_f0_mode,
                formant_shifting, 
                formant_qfrency, 
                formant_timbre,
                f0_file_dropdown,
                embed_mode,
                proposal_pitch,
                proposal_pitch_threshold,
                audio_processing,
                alpha
            ],
            outputs=[main_convert, backing_convert, main_backing, original_convert, vocal_instrument, convert_button],
            api_name="convert_audio"
        )
import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from advanced_rvc_inference.utils.variables import translations, configs, model_name, index_path, method_f0, embedders_mode, embedders_model
from advanced_rvc_inference.core.ui import change_models_choices, get_index, index_strength_show, unlock_f0, hoplength_show, change_embedders_mode, visible, update_dropdowns_from_json, update_button_from_json

def realtime_client_tab():
    with gr.TabItem(translations["realtime_client"], visible=configs.get("realtime_client_tab", True)):
        gr.Markdown(translations["realtime_markdown"])
        with gr.Row():
            gr.Markdown(translations["realtime_markdown_2"])
        with gr.Row():
            gr.Label(label=translations["realtime_latency"], value=translations["realtime_not_startup"], elem_id="realtime-status-info")
        with gr.Row():
            monitor = gr.Checkbox(label=translations["monitor"], value=False, interactive=True)
            exclusive_mode = gr.Checkbox(label=translations["exclusive_mode"], value=False, interactive=True)
            vad_enabled = gr.Checkbox(label=translations["vad_enabled"], value=False, interactive=True)
            clean_audio = gr.Checkbox(label=translations["clear_audio"], value=False, interactive=True)
        with gr.Row():
            with gr.Accordion(translations["audio_device"], open=True):
                with gr.Row():
                    input_audio_device = gr.Dropdown(label=translations["input_audio_device_label"], info=translations["input_audio_device_info"], choices=[], value=None, interactive=True)
                    output_audio_device = gr.Dropdown(label=translations["output_audio_device_label"], info=translations["output_audio_device_info"], choices=[], value=None, interactive=True)
                    monitor_output_device = gr.Dropdown(label=translations["monitor_output_device_label"], info=translations["monitor_output_device_info"], choices=[], value=None, interactive=True, visible=False)
                with gr.Row():
                    input_audio_gain = gr.Slider(minimum=0, maximum=2500, label=translations["input_audio_gain_label"], info=translations["input_audio_gain_info"], value=100, step=1, interactive=True)
                    output_audio_gain = gr.Slider(minimum=0, maximum=4000, label=translations["output_audio_gain_label"], info=translations["output_audio_gain_info"], value=100, step=1, interactive=True)
                    monitor_audio_gain = gr.Slider(minimum=0, maximum=4000, label=translations["monitor_audio_gain_label"], info=translations["monitor_audio_gain_info"], value=100, step=1, interactive=True, visible=False)
                with gr.Row():
                    refresh_audio_device = gr.Button(value=translations["refresh_audio_device"], variant="secondary")
        with gr.Row():
            start_realtime = gr.Button(value=translations["start_realtime_button"], variant="primary", interactive=True)
            stop_realtime = gr.Button(value=translations["stop_realtime_button"], variant="stop", interactive=False)
        with gr.Row():
            chunk_size = gr.Slider(minimum=2.7, maximum=2730.7, step=0.1, label=translations["chunk_size"], info=translations["chunk_size_info"], value=1024, interactive=True)
            pitch = gr.Slider(minimum=-20, maximum=20, step=1, info=translations["pitch_info"], label=translations["pitch"], value=0, interactive=True)
        with gr.Row():
            with gr.Column():
                with gr.Accordion(translations["model_accordion"], open=True):
                    with gr.Row():
                        model_pth = gr.Dropdown(label=translations["model_name"], choices=model_name, value=model_name[0] if len(model_name) >= 1 else "", interactive=True, allow_custom_value=True)
                        model_index = gr.Dropdown(label=translations["index_path"], choices=index_path, value=index_path[0] if len(index_path) >= 1 else "", interactive=True, allow_custom_value=True)
                    with gr.Row():
                        model_refresh = gr.Button(translations["refresh"])
                    with gr.Row():
                        index_strength = gr.Slider(label=translations["index_strength"], info=translations["index_strength_info"], minimum=0, maximum=1, value=0.5, step=0.01, interactive=True, visible=model_index.value != "")
            with gr.Column():
                with gr.Accordion(translations["f0_method"], open=True):
                    with gr.Group():
                        with gr.Row():
                            onnx_f0_mode = gr.Checkbox(label=translations["f0_onnx_mode"], info=translations["f0_onnx_mode_info"], value=False, interactive=True)
                            unlock_full_method = gr.Checkbox(label=translations["f0_unlock"], info=translations["f0_unlock_info"], value=False, interactive=True)
                        f0_method = gr.Radio(label=translations["f0_method"], info=translations["f0_method_info"], choices=[m for m in method_f0 if m != "hybrid"], value="rmvpe", interactive=True)
                    hop_length = gr.Slider(label=translations['hop_length'], info=translations["hop_length_info"], minimum=64, maximum=512, value=160, step=1, interactive=True, visible=False)
            with gr.Column():
                with gr.Accordion(translations["hubert_model"], open=True):
                    embed_mode = gr.Radio(label=translations["embed_mode"], info=translations["embed_mode_info"], value="fairseq", choices=embedders_mode, interactive=True, visible=True)
                    embedders = gr.Radio(label=translations["hubert_model"], info=translations["hubert_info"], choices=embedders_model, value="hubert_base", interactive=True)
                    custom_embedders = gr.Textbox(label=translations["modelname"], info=translations["modelname_info"], value="", placeholder="hubert_base", interactive=True, visible=embedders.value == "custom")
        with gr.Row():
                with gr.Accordion(translations["setting"], open=True):
                    with gr.Row():
                        f0_autotune = gr.Checkbox(label=translations["autotune"], value=False, interactive=True)
                        proposal_pitch = gr.Checkbox(label=translations["proposal_pitch"], value=False, interactive=True)
                    with gr.Group():
                        with gr.Row():
                            f0_autotune_strength = gr.Slider(minimum=0, maximum=1, label=translations["autotune_rate"], info=translations["autotune_rate_info"], value=1, step=0.1, interactive=True, visible=f0_autotune.value)
                            proposal_pitch_threshold = gr.Slider(minimum=50.0, maximum=1200.0, label=translations["proposal_pitch_threshold"], info=translations["proposal_pitch_threshold_info"], value=255.0, step=0.1, interactive=True, visible=proposal_pitch.value)
                        with gr.Row():
                            rms_mix_rate = gr.Slider(minimum=0, maximum=1, label=translations["rms_mix_rate"], info=translations["rms_mix_rate_info"], value=1, step=0.1, interactive=True)
                            protect = gr.Slider(minimum=0, maximum=1, label=translations["protect"], info=translations["protect_info"], value=0.5, step=0.01, interactive=True)
                        with gr.Row():
                            clean_strength = gr.Slider(label=translations["clean_strength"], info=translations["clean_strength_info"], minimum=0, maximum=1, value=0.5, step=0.1, interactive=True, visible=False)
                            filter_radius = gr.Slider(minimum=0, maximum=7, label=translations["filter_radius"], info=translations["filter_radius_info"], value=3, step=1, interactive=True)
                    with gr.Column():
                        silent_threshold = gr.Slider(minimum=-90, maximum=-60, label=translations["silent_threshold_label"], info=translations["silent_threshold_info"], value=-90, step=1, interactive=True)
                        extra_convert_size = gr.Slider(minimum=0.1, maximum=5, label=translations["extra_convert_size_label"], info=translations["extra_convert_size_info"], value=0.5, step=0.1, interactive=True)
                        cross_fade_overlap_size = gr.Slider(minimum=0.05, maximum=0.2, label=translations["cross_fade_overlap_size_label"], info=translations["cross_fade_overlap_size_info"], value=0.1, step=0.01, interactive=True)
                    with gr.Row():
                        vad_sensitivity = gr.Slider(minimum=0, maximum=3, label=translations["vad_sensitivity_label"], info=translations["vad_sensitivity_info"], value=3, step=1, interactive=True, visible=vad_enabled.value)
                        vad_frame_ms = gr.Slider(minimum=10, maximum=30, label=translations["vad_frame_ms_label"], info=translations["vad_frame_ms_info"], value=30, step=10, interactive=True, visible=vad_enabled.value)
        with gr.Row():
            json_audio_hidden = gr.JSON(visible=False)
            json_button_hidden = gr.JSON(visible=False)
        with gr.Row():
            model_pth.change(
                fn=get_index, 
                inputs=[model_pth], 
                outputs=[model_index]
            )
            model_index.change(
                fn=index_strength_show, 
                inputs=[model_index], 
                outputs=[index_strength]
            )
            model_refresh.click(
                fn=change_models_choices, 
                inputs=[], 
                outputs=[model_pth, model_index]
            )
        with gr.Row():
            unlock_full_method.change(
                fn=lambda f0_method: {"choices": [m for m in unlock_f0(f0_method)["choices"] if m != "hybrid"], "value": "rmvpe", "__type__": "update"}, 
                inputs=[unlock_full_method], 
                outputs=[f0_method]
            )
            f0_method.change(
                fn=lambda f0_method: hoplength_show(f0_method, None), 
                inputs=[f0_method], 
                outputs=[hop_length]
            )
            embed_mode.change(
                fn=change_embedders_mode, 
                inputs=[embed_mode], 
                outputs=[embedders]
            )
        with gr.Row():
            embedders.change(
                fn=lambda embedders: visible(embedders == "custom"), 
                inputs=[embedders], 
                outputs=[custom_embedders]
            )
            f0_autotune.change(
                fn=visible, 
                inputs=[f0_autotune], 
                outputs=[f0_autotune_strength]
            )
            clean_audio.change(
                fn=visible, 
                inputs=[clean_audio], 
                outputs=[clean_strength]
            )
        with gr.Row():
            proposal_pitch.change(
                fn=visible, 
                inputs=[proposal_pitch], 
                outputs=[proposal_pitch_threshold]
            )
            vad_enabled.change(
                fn=lambda a: [visible(a) for _ in range(2)],
                inputs=[vad_enabled],
                outputs=[vad_sensitivity, vad_frame_ms]
            )
            refresh_audio_device.click(
                fn=None,
                js="getAudioDevices",
                inputs=[],
                outputs=json_audio_hidden
            )
        with gr.Row():
            json_audio_hidden.change(
                fn=update_dropdowns_from_json,
                inputs=[json_audio_hidden],
                outputs=[input_audio_device, output_audio_device, monitor_output_device]
            )
            json_button_hidden.change(
                fn=update_button_from_json,
                inputs=[json_button_hidden],
                outputs=[start_realtime, stop_realtime]
            )
        with gr.Row():
            start_realtime.click(
                fn=None,
                js="StreamAudioRealtime",
                inputs=[
                    monitor,
                    vad_enabled,
                    input_audio_device,
                    output_audio_device,
                    monitor_output_device,
                    input_audio_gain,
                    output_audio_gain,
                    monitor_audio_gain,
                    chunk_size,
                    pitch,
                    model_pth,
                    model_index,
                    index_strength,
                    onnx_f0_mode,
                    f0_method,
                    hop_length,
                    embed_mode,
                    embedders,
                    custom_embedders,
                    f0_autotune,
                    proposal_pitch,
                    f0_autotune_strength,
                    proposal_pitch_threshold,
                    rms_mix_rate,
                    protect,
                    filter_radius,
                    silent_threshold,
                    extra_convert_size,
                    cross_fade_overlap_size,
                    vad_sensitivity,
                    vad_frame_ms,
                    clean_audio,
                    clean_strength,
                    exclusive_mode
                ],
                outputs=[json_button_hidden]
            )
        stop_realtime.click(
            fn=None,
            js="StopAudioStream",
            inputs=[],
            outputs=[json_button_hidden]
        )
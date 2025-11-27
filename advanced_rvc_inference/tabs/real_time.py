"""
Real-time Inference Tab - Live Voice Conversion
Advanced RVC Inference - Ultimate Performance Edition
Version 4.0.0

Authors: ArkanDash & BF667
Last Updated: November 26, 2025
"""

import gradio as gr
import os, sys
from .core import real_time_voice_conversion
from ...lib.i18n import I18nAuto
from ...lib.path_manager import path

i18n = I18nAuto()

now_dir = os.getcwd()
sys.path.append(now_dir)

# Get available models
model_root = str(path('logs_dir'))
names = [
    os.path.join(root, file)
    for root, _, files in os.walk(model_root, topdown=False)
    for file in files
    if (
        file.endswith((".pth", ".onnx"))
        and not (file.startswith("G_") or file.startswith("D_"))
    )
]

indexes_list = [
    os.path.join(root, name)
    for root, _, files in os.walk(model_root, topdown=False)
    for name in files
    if name.endswith(".index") and "trained" not in name
]

def real_time_inference_tab():
    with gr.Column():
        gr.Markdown("## 🎙️ Real-Time Voice Conversion")
        
        with gr.Row():
            with gr.Column():
                model_file = gr.Dropdown(
                    label=i18n("Voice Model"),
                    info=i18n("Select the voice model for real-time conversion."),
                    choices=sorted(names),
                    value=names[0] if names else "",
                    interactive=True,
                    allow_custom_value=True,
                )
                
                index_file = gr.Dropdown(
                    label=i18n("Index File"),
                    info=i18n("Select the index file for real-time conversion."),
                    choices=sorted(indexes_list),
                    value=indexes_list[0] if indexes_list else "",
                    interactive=True,
                    allow_custom_value=True,
                )
                
                with gr.Row():
                    pitch = gr.Slider(
                        label=i18n("Pitch"),
                        info=i18n("Adjust the pitch of the audio."),
                        minimum=-12,
                        maximum=12,
                        step=1,
                        value=0,
                        interactive=True,
                    )
                    
                    index_rate = gr.Slider(
                        minimum=0,
                        maximum=1,
                        label=i18n("Search Feature Ratio"),
                        info=i18n(
                            "Influence exerted by the index file; a higher value corresponds to greater influence."
                        ),
                        value=0.75,
                        interactive=True,
                    )
            
            with gr.Column():
                embedder_model = gr.Dropdown(
                    label=i18n("Embedder Model"),
                    info=i18n("Model used for learning speaker embedding."),
                    choices=[
                        "contentvec",
                        "chinese-hubert-base",
                        "japanese-hubert-base",
                        "korean-hubert-base",
                        "vietnamese-hubert-base",
                        "english-hubert-base",
                        "whisper-large-v2",
                        "whisper-medium",
                        "whisper-small"
                    ],
                    value="contentvec",
                    interactive=True,
                )
                
                pitch_extract = gr.Dropdown(
                    label=i18n("Pitch Extractor"),
                    info=i18n("Advanced pitch extraction algorithm."),
                    choices=[
                        "rmvpe", 
                        "mangio-crepe", 
                        "mangio-crepe-tiny", 
                        "crepe", 
                        "crepe-tiny", 
                        "mangio-dbs", 
                        "fcpe", 
                        "mangio-dt",
                        "pm", 
                        "harvest", 
                        "dio", 
                        "pyin"
                    ],
                    value="rmvpe",
                    interactive=True,
                )
        
        with gr.Row():
            start_button = gr.Button(i18n("Start Real-Time Conversion"), variant="primary")
            stop_button = gr.Button(i18n("Stop Conversion"))
        
        with gr.Row():
            with gr.Column():
                input_audio = gr.Microphone(
                    label=i18n("Input Microphone"),
                    type="filepath",
                    interactive=True
                )
            
            with gr.Column():
                output_audio = gr.Audio(
                    label=i18n("Converted Audio"),
                    interactive=False,
                    autoplay=True
                )
        
        with gr.Accordion("Advanced Settings", open=False):
            with gr.Row():
                filter_radius = gr.Slider(
                    minimum=0,
                    maximum=7,
                    label=i18n("Filter Radius"),
                    info=i18n(
                        "If the number is greater than or equal to three, employing median filtering on the collected tone results has the potential to decrease respiration."
                    ),
                    value=3,
                    step=1,
                    interactive=True,
                )
                
                rms_mix_rate = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Volume Envelope"),
                    info=i18n(
                        "Substitute or blend with the volume envelope of the input."
                    ),
                    value=0.25,
                    interactive=True,
                )
                
                protect = gr.Slider(
                    minimum=0,
                    maximum=0.5,
                    label=i18n("Protect Voiceless Consonants"),
                    info=i18n(
                        "Safeguard distinct consonants and breathing sounds."
                    ),
                    value=0.33,
                    interactive=True,
                )
        
        def start_real_time_conversion(model_path, index_path, embedder, p, filter_rad, idx_rate, rms_rate, prot, pitch_extractor):
            try:
                result = real_time_voice_conversion(
                    model_path, index_path, embedder, p, filter_rad, idx_rate, rms_rate, prot, pitch_extractor
                )
                return result
            except Exception as e:
                return f"Error during real-time conversion: {str(e)}"

        start_button.click(
            start_real_time_conversion,
            inputs=[
                model_file, index_file, embedder_model,
                pitch, filter_radius, index_rate,
                rms_mix_rate, protect, pitch_extract
            ],
            outputs=[]
        )

        def stop_real_time_conversion():
            # Placeholder for stopping the real-time conversion
            return "Real-time conversion stopped"

        stop_button.click(
            stop_real_time_conversion,
            inputs=[],
            outputs=[]
        )

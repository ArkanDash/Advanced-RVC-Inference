import os
import sys
from pathlib import Path
import gradio as gr
import torch
import shutil
import unicodedata
import regex as re

# Add project root to path
project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))

from advanced_rvc_inference.assets.i18n.i18n import I18nAuto
from advanced_rvc_inference.rvc.lib.utils import format_title

i18n = I18nAuto()

now_dir = str(project_root)

model_root = os.path.join("advanced_rvc_inference", "logs")
audio_root = os.path.join("advanced_rvc_inference", "assets", "audios")

model_root_relative = os.path.relpath(model_root, now_dir)
audio_root_relative = os.path.relpath(audio_root, now_dir)

sup_audioext = {
    "wav",
    "mp3",
    "flac",
    "ogg",
    "opus",
    "m4a",
    "mp4",
    "aac",
    "alac",
    "wma",
    "aiff",
    "webm",
    "ac3",
}

vocals_model_names = [
    "Mel-Roformer by KimberleyJSN",
    "BS-Roformer by ViperX",
    "MDX23C",
]

karaoke_models_names = [
    "Mel-Roformer Karaoke by aufr33 and viperx",
    "UVR-BVE",
]

denoise_models_names = [
    "Mel-Roformer Denoise Normal by aufr33",
    "Mel-Roformer Denoise Aggressive by aufr33",
    "UVR Denoise",
]

dereverb_models_names = [
    "MDX23C DeReverb by aufr33 and jarredou",
    "UVR-Deecho-Dereverb",
    "MDX Reverb HQ by FoxJoy",
    "BS-Roformer Dereverb by anvuew",
]

deecho_models_names = ["UVR-Deecho-Normal", "UVR-Deecho-Aggressive"]


def get_number_of_gpus():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        return "-".join(map(str, range(num_gpus)))
    else:
        return "-"


def format_title(title):
    formatted_title = (
        unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("utf-8")
    )
    formatted_title = re.sub(r"[\u2500-\u257F]+", "", formatted_title)
    formatted_title = re.sub(r"[^\w\s.-]", "", formatted_title)
    formatted_title = re.sub(r"\s+", "_", formatted_title)
    return formatted_title


def save_to_wav(upload_audio):
    file_path = upload_audio
    formated_name = format_title(os.path.basename(file_path))
    target_path = os.path.join(audio_root_relative, formated_name)

    if os.path.exists(target_path):
        os.remove(target_path)

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    shutil.copy(file_path, target_path)
    return target_path


def delete_outputs():
    gr.Info(f"Outputs cleared!")
    for root, _, files in os.walk(audio_root_relative, topdown=False):
        for name in files:
            if name.endswith(tuple(sup_audioext)) and name.__contains__("_output"):
                os.remove(os.path.join(root, name))


def get_models_list():
    models_list = [
        os.path.join(root, file)
        for root, _, files in os.walk(model_root_relative, topdown=False)
        for file in files
        if (
            file.endswith((".pth", ".onnx"))
            and not (file.startswith("G_") or file.startswith("D_"))
        )
    ]
    return sorted(models_list, key=lambda path: os.path.getsize(path))


def get_indexes_list():
    indexes_list = [
        os.path.join(root, name)
        for root, _, files in os.walk(model_root_relative, topdown=False)
        for name in files
        if name.endswith(".index") and "trained" not in name
    ]
    return indexes_list if indexes_list else []


def full_inference_tab():
    with gr.Row():
        gr.Markdown(i18n("## RVC x UVR Full Inference"))
    
    with gr.Row():
        gr.Markdown(i18n("This is a work in progress feature. More options will be added soon."))
    
    with gr.Row():
        with gr.Column():
            # Voice Model and Index
            model_file = gr.Dropdown(
                label=i18n("Voice Model"),
                info=i18n("Select the voice model to use for the conversion."),
                choices=get_models_list(),
                interactive=True,
                allow_custom_value=True,
            )
            index_file = gr.Dropdown(
                label=i18n("Index File"),
                info=i18n("Select the index file to use for the conversion."),
                choices=get_indexes_list(),
                interactive=True,
                allow_custom_value=True,
            )
        
        with gr.Column():
            # Audio input
            upload_audio = gr.Audio(
                label=i18n("Upload Audio"),
                type="filepath",
                editable=False,
                sources=["upload"]
            )
    
    with gr.Accordion(i18n("Audio Separation Settings"), open=True):
        with gr.Row():
            vocal_model = gr.Dropdown(
                label=i18n("Vocals Model"),
                info=i18n("Select the vocals model to use for the separation."),
                choices=sorted(vocals_model_names),
                interactive=True,
                value="Mel-Roformer by KimberleyJSN",
                allow_custom_value=False,
            )
            karaoke_model = gr.Dropdown(
                label=i18n("Karaoke Model"),
                info=i18n("Select the karaoke model to use for the separation."),
                choices=sorted(karaoke_models_names),
                interactive=True,
                value="Mel-Roformer Karaoke by aufr33 and viperx",
                allow_custom_value=False,
            )
        with gr.Row():
            dereverb_model = gr.Dropdown(
                label=i18n("Dereverb Model"),
                info=i18n("Select the dereverb model to use for the separation."),
                choices=sorted(dereverb_models_names),
                interactive=True,
                value="UVR-Deecho-Dereverb",
                allow_custom_value=False,
            )
            deecho = gr.Checkbox(
                label=i18n("Apply De-echo"),
                info=i18n("Apply de-echo to the audio."),
                visible=True,
                value=True,
                interactive=True,
            )
        with gr.Row(visible=True) as deecho_row:
            deeecho_model = gr.Dropdown(
                label=i18n("De-echo Model"),
                info=i18n("Select the de-echo model to use for the separation."),
                choices=sorted(deecho_models_names),
                interactive=True,
                value="UVR-Deecho-Normal",
                allow_custom_value=False,
            )
        with gr.Row():
            denoise = gr.Checkbox(
                label=i18n("Apply Denoise"),
                info=i18n("Apply denoise to the audio."),
                visible=True,
                value=False,
                interactive=True,
            )
        with gr.Row(visible=False) as denoise_row:
            denoise_model = gr.Dropdown(
                label=i18n("Denoise Model"),
                info=i18n("Select the denoise model to use for the separation."),
                choices=sorted(denoise_models_names),
                interactive=True,
                value="Mel-Roformer Denoise Normal by aufr33",
                allow_custom_value=False,
                visible=False,
            )
    
    with gr.Accordion(i18n("RVC Conversion Settings"), open=True):
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
        with gr.Row():
            index_rate = gr.Slider(
                minimum=0,
                maximum=1,
                label=i18n("Search Feature Ratio"),
                info=i18n(
                    "Influence exerted by the index file; a higher value corresponds to greater influence. However, opting for lower values can help mitigate artifacts present in the audio."
                ),
                value=0.75,
                interactive=True,
            )
            rms_mix_rate = gr.Slider(
                minimum=0,
                maximum=1,
                label=i18n("Volume Envelope"),
                info=i18n(
                    "Substitute or blend with the volume envelope of the output. The closer the ratio is to 1, the more the output envelope is employed."
                ),
                value=0.25,
                interactive=True,
            )
        with gr.Row():
            protect = gr.Slider(
                minimum=0,
                maximum=0.5,
                label=i18n("Protect Voiceless Consonants"),
                info=i18n(
                    "Safeguard distinct consonants and breathing sounds to prevent electro-acoustic tearing and other artifacts. Pulling the parameter to its maximum value of 0.5 offers comprehensive protection. However, reducing this value might decrease the extent of protection while potentially mitigating the indexing effect."
                ),
                value=0.33,
                interactive=True,
            )
            split_audio = gr.Checkbox(
                label=i18n("Split Audio"),
                info=i18n(
                    "Split the audio into chunks for inference to obtain better results in some cases."
                ),
                visible=True,
                value=False,
                interactive=True,
            )
        with gr.Row():
            pitch_extract = gr.Radio(
                label=i18n("Pitch Extractor"),
                info=i18n("Pitch extract Algorithm."),
                choices=[
                    "pm-ac", 
                    "pm-cc", 
                    "pm-shs", 
                    "dio", 
                    "mangio-crepe-tiny", 
                    "mangio-crepe-small", 
                    "mangio-crepe-medium", 
                    "mangio-crepe-large", 
                    "mangio-crepe-full", 
                    "crepe-tiny", 
                    "crepe-small", 
                    "crepe-medium", 
                    "crepe-large", 
                    "crepe-full", 
                    "fcpe", 
                    "fcpe-legacy", 
                    "fcpe-previous", 
                    "rmvpe", 
                    "rmvpe-clipping", 
                    "rmvpe-medfilt", 
                    "rmvpe-clipping-medfilt", 
                    "harvest", 
                    "yin", 
                    "pyin", 
                    "swipe", 
                    "piptrack", 
                    "penn", 
                    "mangio-penn", 
                    "djcm", 
                    "djcm-clipping", 
                    "djcm-medfilt", 
                    "djcm-clipping-medfilt", 
                    "swift", 
                    "pesto", 
                    "hybrid"
                ],
                value="rmvpe",
                interactive=True,
            )
            embedder_model = gr.Radio(
                label=i18n("Embedder Model"),
                info=i18n("Model used for learning speaker embedding."),
                choices=[
                    "contentvec",
                    "japanese_hubert_base", 
                    "korean_hubert_base", 
                    "chinese_hubert_base", 
                    "portuguese_hubert_base", 
                    "custom",
                ],
                value="contentvec",
                interactive=True,
            )
        with gr.Row():
            autotune = gr.Checkbox(
                label=i18n("Autotune"),
                info=i18n(
                    "Apply a soft autotune to your inferences, recommended for singing conversions."
                ),
                visible=True,
                value=False,
                interactive=True,
            )
            use_tta = gr.Checkbox(
                label=i18n("Use TTA"),
                info=i18n("Use Test Time Augmentation."),
                visible=True,
                value=False,
                interactive=True,
            )
    
    with gr.Row():
        batch_size = gr.Slider(
            minimum=1,
            maximum=24,
            step=1,
            label=i18n("Batch Size"),
            info=i18n("Set the batch size for the separation."),
            value=1,
            interactive=True,
        )
        devices = gr.Textbox(
            label=i18n("Device"),
            info=i18n(
                "Select the device to use for the conversion. 0 to ∞ separated by - and for CPU leave only an -"
            ),
            value=get_number_of_gpus(),
            interactive=True,
        )
    
    with gr.Row():
        export_format_final = gr.Radio(
            label=i18n("Export Format"),
            info=i18n("Select the format to export the audio."),
            choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
            value="FLAC",
            interactive=True,
        )
        export_format_rvc = gr.Radio(
            label=i18n("RVC Export Format"),
            info=i18n("Select the format for RVC output."),
            choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
            value="FLAC",
            interactive=True,
        )
    
    # Add more advanced settings for RVC x UVR
    with gr.Accordion(i18n("Advanced Settings"), open=False):
        with gr.Row():
            vocals_volume = gr.Slider(
                label=i18n("Vocals Volume"),
                info=i18n("Adjust the volume of the vocals."),
                minimum=-10,
                maximum=0,
                step=1,
                value=-3,
                interactive=True,
            )
            instrumentals_volume = gr.Slider(
                label=i18n("Instrumentals Volume"),
                info=i18n("Adjust the volume of the Instrumentals."),
                minimum=-10,
                maximum=0,
                step=1,
                value=-3,
                interactive=True,
            )
        with gr.Row():
            change_inst_pitch = gr.Slider(
                label=i18n("Change Instrumental Pitch"),
                info=i18n("Change the pitch of the instrumental."),
                minimum=-12,
                maximum=12,
                step=1,
                value=0,
                interactive=True,
            )
            delete_audios = gr.Checkbox(
                label=i18n("Delete Audios"),
                info=i18n("Delete all audio files after processing except the final result."),
                visible=True,
                value=True,
                interactive=True,
            )
    
    with gr.Row():
        convert_button = gr.Button(i18n("Convert"))
    
    with gr.Row():
        output_info = gr.Textbox(
            label=i18n("Output Information"),
            info=i18n("The output information will be displayed here."),
        )
        output_audio = gr.Audio(
            label=i18n("Final Output Audio")
        )

    def update_dropdown_visibility(checkbox):
        return gr.update(visible=checkbox)

    def update_visibility_denoise(denoise_checked):
        return gr.update(visible=denoise_checked)

    def run_full_inference(
        model_file,
        index_file,
        upload_audio,
        export_format_rvc,
        split_audio,
        autotune,
        vocal_model,
        karaoke_model,
        dereverb_model,
        deecho,
        deeecho_model,
        denoise,
        denoise_model,
        reverb,
        vocals_volume,
        instrumentals_volume,
        export_format_final,
        devices,
        pitch,
        filter_radius,
        index_rate,
        rms_mix_rate,
        protect,
        pitch_extract,
        embedder_model,
        delete_audios,
        use_tta,
        batch_size,
        change_inst_pitch,
    ):
        from core import (
            run_infer_script,
        )

        # Validate inputs
        if not upload_audio or not model_file:
            return "Error: Please provide both audio file and model.", None

        # For now, run a basic RVC inference with the provided parameters
        # In a full implementation, this would run the complete COVERMAKER pipeline
        try:
            # Create output path
            import os
            audio_basename = os.path.splitext(os.path.basename(upload_audio))[0]
            output_path = os.path.join(audio_root, f"{audio_basename}_output.wav")

            # Run basic RVC inference with the provided parameters
            message, output_file = run_infer_script(
                pitch=pitch,
                index_rate=index_rate,
                volume_envelope=rms_mix_rate,
                protect=protect,
                f0_method=pitch_extract,
                input_path=upload_audio,
                output_path=output_path,
                pth_path=model_file,
                index_path=index_file,
                split_audio=split_audio,
                f0_autotune=autotune,
                f0_autotune_strength=1.0,  # Default strength
                proposed_pitch=False,
                proposed_pitch_threshold=155.0,
                clean_audio=False,
                clean_strength=0.5,
                export_format=export_format_rvc,
                embedder_model=embedder_model,
                filter_radius=filter_radius,
            )

            return f"RVC x UVR processing completed:\n{message}\n\nParameters used:\n- Model: {model_file}\n- Audio: {upload_audio}\n- Pitch: {pitch}\n- F0 Method: {pitch_extract}\n- Embedder: {embedder_model}", output_file

        except Exception as e:
            return f"Error during processing: {str(e)}", None
    
    # Set up event handlers
    deecho.change(
        fn=update_dropdown_visibility,
        inputs=[deecho],
        outputs=[deeecho_model],
    )
    
    denoise.change(
        fn=update_visibility_denoise,
        inputs=[denoise],
        outputs=[denoise_model],
    )
    
    convert_button.click(
        fn=run_full_inference,
        inputs=[
            model_file,
            index_file,
            upload_audio,
            export_format_rvc,
            split_audio,
            autotune,
            vocal_model,
            karaoke_model,
            dereverb_model,
            deecho,
            deeecho_model,
            denoise,
            denoise_model,
            gr.Checkbox(value=False, visible=False), # reverb - not needed for now
            vocals_volume,
            instrumentals_volume,
            export_format_final,
            devices,
            pitch,
            filter_radius,
            index_rate,
            rms_mix_rate,
            protect,
            pitch_extract,
            embedder_model,
            delete_audios,
            use_tta,
            batch_size,
            change_inst_pitch,
        ],
        outputs=[output_info, output_audio],
    )
    
    # Upload handler
    upload_audio.upload(
        fn=save_to_wav,
        inputs=[upload_audio],
        outputs=[upload_audio],
    )
    
    return convert_button, output_info, output_audio
import os
import sys
import gradio as gr
import subprocess
import shutil

from assets.i18n.i18n import I18nAuto
from uvr.constants import (
    models_vocals,
    karaoke_models,
    denoise_models,
    dereverb_models,
    deecho_models,
    get_model_info_by_name,
    download_file
)

i18n = I18nAuto()

now_dir = os.getcwd()
sys.path.append(now_dir)

# Define available models for separation from constants
vocals_model_names = [model["name"] for model in models_vocals]
karaoke_models_names = [model["name"] for model in karaoke_models]
denoise_models_names = [model["name"] for model in denoise_models]
dereverb_models_names = [model["name"] for model in dereverb_models]
deecho_models_names = [model["name"] for model in deecho_models]


def get_cuda_devices():
    """Get available CUDA devices"""
    try:
        import torch
        if torch.cuda.is_available():
            devices = []
            for i in range(torch.cuda.device_count()):
                devices.append(f"{i}: {torch.cuda.get_device_name(i)}")
            return devices
        else:
            return ["cpu"]
    except:
        return ["cpu"]


def audio_separation(
    input_audio,
    output_dir,
    model_type,
    vocals_model,
    karaoke_model,
    dereverb_model,
    deecho,
    deecho_model,
    denoise,
    denoise_model,
    use_tta,
    batch_size,
    device,
    export_format
):
    """Function to handle audio separation using UVR features"""
    if not input_audio:
        return "Please provide an input audio file", None

    if not output_dir:
        output_dir = "assets/audios/separated"
        os.makedirs(output_dir, exist_ok=True)

    # Define the separation script from COVERMAKER
    separation_script = os.path.join(now_dir, "uvr", "music_separation", "inference.py")

    if not os.path.exists(separation_script):
        return f"Separation script not found: {separation_script}", None

    try:
        # Determine device for separation
        if device == "cpu" or device.startswith("cpu"):
            device_flag = ["--force_cpu"]
        else:
            # Extract GPU ID from device string like "0: GeForce RTX..."
            gpu_id = device.split(":")[0] if ":" in device else "0"
            device_flag = ["--device_ids", gpu_id]

        # Map model names to appropriate model configurations
        model_mapping = {
            "Mel-Roformer by KimberleyJSN": {
                "type": "mel_band_roformer",
                "config_path": "assets/models/mel-vocals/config.yaml",
                "model_path": "assets/models/mel-vocals/model.ckpt",
                "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml",
                "model_url": "https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt",
            },
            "BS-Roformer by ViperX": {
                "type": "bs_roformer",
                "config_path": "assets/models/bs-vocals/config.yaml",
                "model_path": "assets/models/bs-vocals/model.ckpt",
                "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml",
                "model_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            },
            "MDX23C": {
                "type": "mdx23c",
                "config_path": "assets/models/mdx23c-vocals/config.yaml",
                "model_path": "assets/models/mdx23c-vocals/model.ckpt",
                "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_vocals_mdx23c.yaml",
                "model_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_mdx23c_sdr_10.17.ckpt",
            },
            "Mel-Roformer Karaoke by aufr33 and viperx": {
                "type": "mel_band_roformer",
                "config_path": "assets/models/mel-kara/config.yaml",
                "model_path": "assets/models/mel-kara/model.ckpt",
                "config_url": "https://huggingface.co/shiromiya/audio-separation-models/resolve/main/mel_band_roformer_karaoke_aufr33_viperx/config_mel_band_roformer_karaoke.yaml",
                "model_url": "https://huggingface.co/shiromiya/audio-separation-models/resolve/main/mel_band_roformer_karaoke_aufr33_viperx/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
            },
            "UVR-BVE": {
                "type": "vr",  # vr arch
                "model_path": "assets/models/uve5/UVR-DeNoise.pth",  # Placeholder - would need to download
            },
            "Mel-Roformer Denoise Normal by aufr33": {
                "type": "mel_band_roformer",
                "config_path": "assets/models/mel-denoise/config.yaml",
                "model_path": "assets/models/mel-denoise/model.ckpt",
                "config_url": "https://huggingface.co/shiromiya/audio-separation-models/resolve/main/mel-denoise/model_mel_band_roformer_denoise.yaml",
                "model_url": "https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
            },
            "Mel-Roformer Denoise Aggressive by aufr33": {
                "type": "mel_band_roformer",
                "config_path": "assets/models/mel-denoise-aggr/config.yaml",
                "model_path": "assets/models/mel-denoise-aggr/model.ckpt",
                "config_url": "https://huggingface.co/shiromiya/audio-separation-models/resolve/main/mel-denoise/model_mel_band_roformer_denoise.yaml",
                "model_url": "https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt",
            },
            "UVR Denoise": {
                "type": "vr",
                "model_path": "assets/models/uve5/UVR-DeNoise.pth",  # Placeholder
            },
            "MDX23C DeReverb by aufr33 and jarredou": {
                "type": "mdx23c",
                "config_path": "assets/models/mdx23c-dereveb/config.yaml",
                "model_path": "assets/models/mdx23c-dereveb/model.ckpt",
                "config_url": "https://huggingface.co/jarredou/aufr33_jarredou_MDXv3_DeReverb/resolve/main/config_dereverb_mdx23c.yaml",
                "model_url": "https://huggingface.co/jarredou/aufr33_jarredou_MDXv3_DeReverb/resolve/main/dereverb_mdx23c_sdr_6.9096.ckpt",
            },
            "BS-Roformer Dereverb by anvuew": {
                "type": "bs_roformer",
                "config_path": "assets/models/bs-dereveb/config.yaml",
                "model_path": "assets/models/bs-dereveb/model.ckpt",
                "config_url": "https://huggingface.co/anvuew/deverb_bs_roformer/resolve/main/deverb_bs_roformer_8_384dim_10depth.yaml",
                "model_url": "https://huggingface.co/anvuew/deverb_bs_roformer/resolve/main/deverb_bs_roformer_8_384dim_10depth.ckpt",
            },
            "UVR-Deecho-Dereverb": {
                "type": "vr",
                "model_path": "assets/models/uve5/UVR-DeEcho-DeReverb.pth",  # Placeholder
            },
            "MDX Reverb HQ by FoxJoy": {
                "type": "mdx",  # ONNX architecture
                "model_path": "assets/models/uve5/Reverb_HQ_By_FoxJoy.onnx",  # Placeholder
            },
        }

        # Get model configuration
        model_info = model_mapping.get(vocals_model)
        if not model_info:
            return f"Model {vocals_model} configuration not found", None

        model_type = model_info["type"]
        model_path = model_info.get("model_path", "")
        config_path = model_info.get("config_path", "")

        # Check if model files exist, if not download them
        if model_path and not os.path.exists(model_path):
            model_url = model_info.get("model_url", "")
            if model_url:
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                download_file(model_url, os.path.dirname(model_path), os.path.basename(model_path))

        if config_path and model_type != "vr" and not os.path.exists(config_path):  # VR models don't always need a config
            config_url = model_info.get("config_url", "")
            if config_url:
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                download_file(config_url, os.path.dirname(config_path), os.path.basename(config_path))

        # Prepare arguments for separation
        cmd = [
            sys.executable,
            separation_script,
            "--model_type", model_info.get("type", model_type),
            "--config_path", config_path if config_path and os.path.exists(config_path) else "",
            "--start_check_point", model_path if model_path and os.path.exists(model_path) else "",
            "--input_file", input_audio,
            "--store_dir", output_dir,
        ]

        # Add device flags
        cmd.extend(device_flag)

        # Add additional flags based on settings
        if use_tta:
            cmd.append("--use_tta")

        # Add batch size
        cmd.extend(["--batch_size", str(batch_size)])

        # Add format based on export_format
        if export_format.lower() == "flac":
            cmd.append("--flac_file")
        elif export_format.lower() == "wav":
            cmd.append("--wav_file")
        else:
            cmd.append("--flac_file")  # Default to high quality

        # Run the separation
        try:
            print(f"Running separation with command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 minute timeout

            if result.returncode == 0:
                output_msg = f"Separation completed successfully!\n- Input: {os.path.basename(input_audio)}\n- Output: {output_dir}\n- Model: {vocals_model}\n- Device: {device}"

                # Find the output file in the output directory
                output_files = []
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        if file.endswith(('.wav', '.flac', '.mp3')) and os.path.basename(input_audio).split('.')[0] in file:
                            output_files.append(os.path.join(root, file))

                if output_files:
                    # Return the first output file found
                    return output_msg, output_files[0]
                else:
                    return output_msg, None
            else:
                error_msg = f"Separation failed with error:\n{result.stderr}\n\nCommand: {' '.join(cmd)}"
                return error_msg, None
        except subprocess.TimeoutExpired:
            return "Separation process timed out (10 minutes)", None
        except Exception as e:
            return f"Error during separation: {str(e)}", None

    except Exception as e:
        return f"Error during separation setup: {str(e)}", None


# The get_model_info_by_name and download_file functions are now imported from uvr.constants
# So we don't need to redefine them here


def separation_tab():
    """Create the audio separation tab interface"""
    with gr.Row():
        gr.Markdown(i18n("## Audio Separation (UVR Features)"))
    with gr.Row():
        with gr.Column():
            input_audio = gr.Audio(
                label=i18n("Input Audio"),
                type="filepath",
                interactive=True
            )
            with gr.Row():
                model_type = gr.Dropdown(
                    label=i18n("Separation Model Type"),
                    choices=[
                        "mdx23c", "bs_roformer", "mel_band_roformer", 
                        "vr", "htdemucs", "segm_models", "torchseg", "swin_upernet", "scnet", "scnet_unofficial"
                    ],
                    value="mdx23c",
                    interactive=True
                )
                device = gr.Dropdown(
                    label=i18n("Device"), 
                    choices=get_cuda_devices(),
                    value=get_cuda_devices()[0] if get_cuda_devices() else "cpu",
                    interactive=True
                )
            with gr.Row():
                vocals_model = gr.Dropdown(
                    label=i18n("Vocals Model"),
                    choices=vocals_model_names,
                    value=vocals_model_names[0],
                    interactive=True
                )
                karaoke_model = gr.Dropdown(
                    label=i18n("Karaoke Model"),
                    choices=karaoke_models_names,
                    value=karaoke_models_names[0],
                    interactive=True
                )
            with gr.Row():
                dereverb_model = gr.Dropdown(
                    label=i18n("Dereverb Model"),
                    choices=dereverb_models_names,
                    value=dereverb_models_names[0],
                    interactive=True
                )
            with gr.Row():
                deecho = gr.Checkbox(
                    label=i18n("Apply De-echo"),
                    value=True,
                    interactive=True
                )
                deecho_model = gr.Dropdown(
                    label=i18n("De-echo Model"),
                    choices=deecho_models_names,
                    value=deecho_models_names[0],
                    interactive=True,
                    visible=True
                )
            with gr.Row():
                denoise = gr.Checkbox(
                    label=i18n("Apply Denoise"),
                    value=False,
                    interactive=True
                )
                denoise_model = gr.Dropdown(
                    label=i18n("Denoise Model"),
                    choices=denoise_models_names,
                    value=denoise_models_names[0],
                    interactive=True,
                    visible=False
                )
        with gr.Column():
            output_dir = gr.Textbox(
                label=i18n("Output Directory"),
                placeholder=i18n("Enter output directory path"),
                value="assets/audios/separated",
                interactive=True
            )
            export_format = gr.Radio(
                label=i18n("Export Format"),
                choices=["WAV", "FLAC", "MP3"],
                value="WAV",
                interactive=True
            )
            use_tta = gr.Checkbox(
                label=i18n("Use TTA (Test Time Augmentation)"),
                value=False,
                interactive=True
            )
            batch_size = gr.Slider(
                minimum=1,
                maximum=24,
                step=1,
                label=i18n("Batch Size"),
                info=i18n("Set the batch size for the separation."),
                value=1,
                interactive=True,
            )
            with gr.Accordion(i18n("Advanced Options"), open=False):
                # Additional advanced options can be added here
                gr.Markdown(i18n("Additional advanced options will be available here"))

    with gr.Row():
        separate_button = gr.Button(i18n("Separate Audio"), variant="primary")
    
    with gr.Row():
        with gr.Column():
            output_info = gr.Textbox(
                label=i18n("Output Information"),
                info=i18n("The separation process information will be displayed here."),
                interactive=False
            )
        with gr.Column():
            separated_audio = gr.Audio(
                label=i18n("Separated Audio Preview"),
                interactive=False
            )
    
    # Update visibility based on checkbox states
    def update_deecho_visibility(checked):
        return gr.update(visible=checked)
    
    def update_denoise_visibility(checked):
        return gr.update(visible=checked)
    
    deecho.change(
        fn=update_deecho_visibility,
        inputs=[deecho],
        outputs=[deecho_model]
    )
    
    denoise.change(
        fn=update_denoise_visibility,
        inputs=[denoise],
        outputs=[denoise_model]
    )
    
    # Handle the separation process
    separate_button.click(
        fn=audio_separation,
        inputs=[
            input_audio,
            output_dir,
            model_type,
            vocals_model,
            karaoke_model,
            dereverb_model,
            deecho,
            deecho_model,
            denoise,
            denoise_model,
            use_tta,
            batch_size,
            device,
            export_format
        ],
        outputs=[output_info, separated_audio]
    )
    
    return separate_button, output_info, separated_audio
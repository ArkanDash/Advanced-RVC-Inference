import os
import sys
import yt_dlp
import subprocess
import logging
import json
from logging.handlers import RotatingFileHandler
from contextlib import suppress

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment

# =============================================================================
# Import the UVR separator – ensure that the module is installed or available.
# =============================================================================
try:
    from audio_separator.separator import Separator
except ImportError:
    raise ImportError("Make sure the 'audio_separator' module is installed or in your working directory.")

# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(log_level=logging.DEBUG, log_file="kuro_rvc.log"):
    """
    Set up logging with both console and rotating file handlers.
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s", 
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=2)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.debug("...logging has been configured.")

setup_logging()

# =============================================================================
# Global Directories and Paths
# =============================================================================

# current working directory
current_dir = os.getcwd()

# where model folders (named after the model, e.g. "Sonic") are stored
rvc_models_dir = os.path.join(current_dir, 'logs')

# output folder for RVC (mixed song)
rvc_output_dir = os.path.join(current_dir, 'song_output')

# where downloaded YouTube files will be placed
download_dir = os.path.join(current_dir, "downloads")

# temporary output folder for UVR separation
uvr_output_dir = os.path.join(current_dir, "output_uvr")

# RVC inference CLI script – must be present in the current directory
rvc_cli_filename = "scrpt.py"
rvc_cli_file = os.path.join(current_dir, rvc_cli_filename)
if not os.path.exists(rvc_cli_file):
    logging.error("scrpt.py not found in the current directory: %s", current_dir)
    raise FileNotFoundError("scrpt.py not found in the current directory.")

# =============================================================================
# Function Definitions
# =============================================================================

def download_youtube_audio(url, download_dir):
    """
    Download audio from a YouTube URL and return the path (or list) of the downloaded WAV file(s).
    """
    logging.debug("Starting YouTube audio download. URL: %s", url)
    os.makedirs(download_dir, exist_ok=True)
    outtmpl = os.path.join(download_dir, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192"
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
    if "entries" in info_dict:  # playlist support
        downloaded_files = [os.path.join(download_dir, f"{entry['title']}.wav")
                            for entry in info_dict["entries"] if entry]
    else:
        downloaded_files = os.path.join(download_dir, f"{info_dict['title']}.wav")
    logging.debug("Downloaded audio file(s): %s", downloaded_files)
    return downloaded_files

def separator_uvr(input_audio, output_dir):
    """
    Run the two‐step UVR separation:
      1. Separate instrumental and vocals.
      2. Split vocals into lead and backing vocals.
    The separated files are renamed and saved in output_dir.
    Returns the full paths for lead vocals, backing vocals, and instrumental.
    """
    logging.debug("Starting UVR separation for file: %s", input_audio)
    os.makedirs(output_dir, exist_ok=True)
    
    # First separation: instrumentals + vocals
    uvr_separator = Separator(output_dir=output_dir)
    logging.debug("Loading first UVR model for instrumental/vocals separation.")
    uvr_separator.load_model('model_bs_roformer_ep_317_sdr_12.9755.ckpt')
    separated_files = uvr_separator.separate(input_audio)
    if len(separated_files) < 2:
        error_msg = "UVR separation did not produce expected files for instrumental/vocals."
        logging.error(error_msg)
        raise RuntimeError(error_msg)
    # Rename the files to fixed names
    instrumental_path = os.path.join(output_dir, 'Instrumental.wav')
    vocals_path = os.path.join(output_dir, 'Vocals.wav')
    os.rename(os.path.join(output_dir, separated_files[0]), instrumental_path)
    os.rename(os.path.join(output_dir, separated_files[1]), vocals_path)
    logging.debug("Separated instrumental saved to: %s", instrumental_path)
    logging.debug("Separated vocals saved to: %s", vocals_path)
    
    # Second separation: split vocals
    logging.debug("Loading second UVR model for vocal splitting.")
    uvr_separator.load_model('mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt')
    separated_vocals = uvr_separator.separate(vocals_path)
    if len(separated_vocals) < 2:
        error_msg = "UVR separation did not produce expected files for vocal split."
        logging.error(error_msg)
        raise RuntimeError(error_msg)
    backing_vocals_path = os.path.join(output_dir, 'Backing_Vocals.wav')
    lead_vocals_path = os.path.join(output_dir, 'Lead_Vocals.wav')
    os.rename(os.path.join(output_dir, separated_vocals[0]), backing_vocals_path)
    os.rename(os.path.join(output_dir, separated_vocals[1]), lead_vocals_path)
    logging.debug("Separated backing vocals saved to: %s", backing_vocals_path)
    logging.debug("Separated lead vocals saved to: %s", lead_vocals_path)
    
    return lead_vocals_path, backing_vocals_path, instrumental_path

def run_rvc(f0_up_key, filter_radius, rms_mix_rate, index_rate, hop_length, protect,
            f0_method, input_path, output_path, pth_file, index_file, split_audio,
            clean_audio, clean_strength, export_format, f0_autotune,
            embedder_model, embedder_model_custom, rvc_cli_file):
    """
    Run the RVC inference pipeline via the CLI script.
    """
    logging.debug("Preparing RVC inference command for input file: %s", input_path)
    command = [
        sys.executable, rvc_cli_file, "infer",
        "--pitch", str(f0_up_key),
        "--filter_radius", str(filter_radius),
        "--volume_envelope", str(rms_mix_rate),
        "--index_rate", str(index_rate),
        "--hop_length", str(hop_length),
        "--protect", str(protect),
        "--f0_method", f0_method,
        "--f0_autotune", str(f0_autotune),
        "--input_path", input_path,
        "--output_path", output_path,
        "--pth_path", pth_file,
        "--index_path", index_file,
        "--split_audio", str(split_audio),
        "--clean_audio", str(clean_audio),
        "--clean_strength", str(clean_strength),
        "--export_format", export_format,
        "--embedder_model", embedder_model,
        "--embedder_model_custom", embedder_model_custom
    ]
    logging.info("Running RVC inference. Command: %s", " ".join(command))
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    logging.debug("RVC inference stdout: %s", result.stdout)
    if result.stderr:
        logging.debug("RVC inference stderr: %s", result.stderr)
    logging.info("RVC inference completed for input: %s", input_path)

def load_audio(file_path):
    """
    Load an audio file using pydub.
    """
    if file_path and os.path.exists(file_path):
        logging.debug("Loading audio file: %s", file_path)
        return AudioSegment.from_file(file_path)
    else:
        logging.warning("Audio file not found: %s", file_path)
        return None

def run_advanced_rvc(model_name, youtube_url, export_format, f0_method, f0_up_key, filter_radius,
                     rms_mix_rate, protect, index_rate, hop_length, clean_strength, split_audio,
                     clean_audio, f0_autotune, backing_vocal_infer, embedder_model, embedder_model_custom):
    """
    This function wraps the entire Advanced-RVC pipeline:
      • Checks model files in logs/{model_name}
      • Downloads the audio from YouTube
      • Runs UVR separation (first to separate instrumental/vocals then to split vocals)
      • Runs RVC inference on lead (and optionally backing) vocals
      • Mixes the processed vocals with the instrumental and exports the final file.
      
    Returns a status message and (if successful) the path to the mixed audio file.
    """
    try:
        logging.info("Starting Advanced-RVC pipeline.")
        # Recompute directories in case the working directory has changed
        current_dir = os.getcwd()
        rvc_models_dir = os.path.join(current_dir, 'logs')
        rvc_output_dir = os.path.join(current_dir, 'song_output')
        download_dir = os.path.join(current_dir, "downloads")
        uvr_output_dir = os.path.join(current_dir, "output_uvr")
        
        # Ensure the RVC CLI script exists
        rvc_cli_file = os.path.join(current_dir, "scrpt.py")
        if not os.path.exists(rvc_cli_file):
            error_msg = f"scrpt.py not found in {current_dir}"
            logging.error(error_msg)
            return error_msg, None
        
        # Check model folder and required model files (.pth and .index)
        model_folder = os.path.join(rvc_models_dir, model_name)
        if not os.path.exists(model_folder):
            error_msg = f"Model directory not found: {model_folder}"
            logging.error(error_msg)
            return error_msg, None
        files_in_folder = os.listdir(model_folder)
        pth_filename = next((f for f in files_in_folder if f.endswith(".pth")), None)
        index_filename = next((f for f in files_in_folder if f.endswith(".index")), None)
        if not pth_filename or not index_filename:
            error_msg = "Required model files (.pth or .index) were not found in the model folder."
            logging.error(error_msg)
            return error_msg, None
        pth_file = os.path.join(model_folder, pth_filename)
        index_file = os.path.join(model_folder, index_filename)
        logging.debug("Model files located. PTH: %s, Index: %s", pth_file, index_file)
        
        # Download audio from YouTube
        logging.info("Downloading audio from YouTube...")
        downloaded_audio = download_youtube_audio(youtube_url, download_dir)
        input_audio = downloaded_audio[0] if isinstance(downloaded_audio, list) else downloaded_audio
        if not os.path.exists(input_audio):
            error_msg = f"Downloaded audio file not found: {input_audio}"
            logging.error(error_msg)
            return error_msg, None
        logging.info("Audio downloaded successfully: %s", input_audio)
        
        # Run UVR separation
        logging.info("Running UVR separation...")
        lead_vocals_path, backing_vocals_path, instrumental_path = separator_uvr(input_audio, uvr_output_dir)
        logging.info("UVR separation completed.\n  Lead vocals: %s\n  Backing vocals: %s\n  Instrumental: %s",
                     lead_vocals_path, backing_vocals_path, instrumental_path)
        
        os.makedirs(rvc_output_dir, exist_ok=True)
        rvc_lead_output = os.path.join(rvc_output_dir, "rvc_result_lead.wav")
        rvc_backing_output = os.path.join(rvc_output_dir, "rvc_result_backing.wav")
        
        # Run RVC inference for lead vocals
        logging.info("Running RVC inference for lead vocals...")
        run_rvc(f0_up_key, filter_radius, rms_mix_rate, index_rate, hop_length, protect,
                f0_method, lead_vocals_path, rvc_lead_output, pth_file, index_file,
                split_audio, clean_audio, clean_strength, export_format, f0_autotune,
                embedder_model, embedder_model_custom, rvc_cli_file)
        
        # Optionally run RVC inference for backing vocals
        if backing_vocal_infer:
            logging.info("Running RVC inference for backing vocals...")
            run_rvc(f0_up_key, filter_radius, rms_mix_rate, index_rate, hop_length, protect,
                    f0_method, backing_vocals_path, rvc_backing_output, pth_file, index_file,
                    split_audio, clean_audio, clean_strength, export_format, f0_autotune,
                    embedder_model, embedder_model_custom, rvc_cli_file)
        
        # Load the processed tracks for mixing
        logging.info("Loading audio tracks for final mix.")
        lead_vocals_audio = load_audio(rvc_lead_output)
        instrumental_audio = load_audio(instrumental_path)
        # Use the RVC-processed backing if available; otherwise, the separated backing track
        backing_vocals_audio = load_audio(rvc_backing_output) if backing_vocal_infer else load_audio(backing_vocals_path)
        
        if not instrumental_audio:
            error_msg = "Instrumental track is required for mixing!"
            logging.error(error_msg)
            return error_msg, None
        
        # Mix the tracks (overlay vocals onto the instrumental)
        final_mix = instrumental_audio
        if lead_vocals_audio:
            logging.debug("Overlaying lead vocals onto instrumental.")
            final_mix = final_mix.overlay(lead_vocals_audio)
        if backing_vocals_audio:
            logging.debug("Overlaying backing vocals onto instrumental.")
            final_mix = final_mix.overlay(backing_vocals_audio)
        
        # Save the final mix
        output_filename = f"aicover_{model_name}"
        output_file = f"{output_filename}.{export_format.lower()}"
        final_mix.export(output_file, format=export_format.lower())
        logging.info("Mixed file saved as: %s", output_file)
        return f"Mixed file saved as: {output_file}", output_file, lead_vocals_audio, backing_vocals_audio
    except Exception as e:
        logging.exception("An error occurred during execution: %s", e)
        return f"An error occurred: {e}", None

# =============================================================================
# Gradio Blocks Web UI
# =============================================================================

def inference_tab():
    gr.Markdown("# Advanced RVC Pipeline WebUI")
    gr.Markdown("Set parameters below and click the **Run Advanced RVC Pipeline** button to start the process.")
    with gr.Row():
        model_name_input = gr.Textbox(label="Model Name", value="Sonic")
        youtube_url_input = gr.Textbox(label="YouTube URL", value="https://youtu.be/eCkWlRL3_N0?si=y6xHAs1m8fYVLTUV")
        export_format_input = gr.Dropdown(label="Export Format", choices=["WAV", "MP3", "FLAC", "OGG", "M4A"], value="WAV")
        
        f0_method_input = gr.Dropdown(label="F0 Method", choices=["crepe", "crepe-tiny", "rmvpe", "fcpe", "hybrid[rmvpe+fcpe]"],
                                      value="hybrid[rmvpe+fcpe]")
    with gr.Row():
        f0_up_key_input = gr.Slider(label="F0 Up Key", minimum=-24, maximum=24, step=1, value=0)
        filter_radius_input = gr.Slider(label="Filter Radius", minimum=0, maximum=10, step=1, value=3)
        rms_mix_rate_input = gr.Slider(label="RMS Mix Rate", minimum=0.0, maximum=1.0, step=0.1, value=0.8)
        protect_input = gr.Slider(label="Protect", minimum=0.0, maximum=0.5, step=0.1, value=0.5)
    with gr.Row():
        index_rate_input = gr.Slider(label="Index Rate", minimum=0.0, maximum=1.0, step=0.1, value=0.6)
        hop_length_input = gr.Slider(label="Hop Length", minimum=1, maximum=512, step=1, value=128)
        clean_strength_input = gr.Slider(label="Clean Strength", minimum=0.0, maximum=1.0, step=0.1, value=0.7)    
        split_audio_input = gr.Checkbox(label="Split Audio", value=False)
    with gr.Row():
        clean_audio_input = gr.Checkbox(label="Clean Audio", value=False)
        f0_autotune_input = gr.Checkbox(label="F0 Autotune", value=False)
        backing_vocal_infer_input = gr.Checkbox(label="Infer Backing Vocals", value=False)
    with gr.Row():
        embedder_model_input = gr.Dropdown(
            label="Embedder Model", 
            choices=["contentvec", "chinese-hubert-base", "japanese-hubert-base", "korean-hubert-base", "custom"],
            value="contentvec"
        )
        embedder_model_custom_input = gr.Textbox(label="Custom Embedder Model", value="")
    run_button = gr.Button("Convert")
    with gr.Row():
        output_message = gr.Textbox(label="Status")
        output_audio = gr.Audio(label="Final Mixed Audio", type="filepath")
        with gr.Row():
            output_lead = gr.Audio(label="Output Lead Ai Cover:", type="filepath")
            output_backing = gr.Audio(label="Output Backing Ai Cover:", type="filepath")
    
    run_button.click(
        run_advanced_rvc,
        inputs=[
            model_name_input, youtube_url_input, export_format_input, f0_method_input,
            f0_up_key_input, filter_radius_input, rms_mix_rate_input, protect_input,
            index_rate_input, hop_length_input, clean_strength_input, split_audio_input,
            clean_audio_input, f0_autotune_input, backing_vocal_infer_input,
            embedder_model_input, embedder_model_custom_input
        ],
        outputs=[output_message, output_audio, output_lead, output_backing]
    )

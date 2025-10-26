"""
Advanced RVC Inference Core Module
Optimized for performance, readability and maintainability

This module provides comprehensive audio processing capabilities for voice conversion,
including vocals separation, noise reduction, reverb removal, echo cancellation,
and high-quality voice conversion using RVC models.

The main processing pipeline includes:
1. Audio input and preprocessing
2. Multi-stage audio separation (vocals, instrumentals, backing vocals)
3. Audio enhancement (denoise, dereverb, deecho)
4. Voice conversion using RVC models
5. Post-processing and audio merging
"""

import sys
import os
import subprocess
import torch
import json
from functools import lru_cache
import shutil
from pedalboard import Pedalboard, Reverb
from pedalboard.io import AudioFile
from pydub import AudioSegment
from audio_separator.separator import Separator
import logging
import yaml
from typing import Optional, Dict, Any, Tuple, Union

from programs.applio_code.rvc.infer.infer import VoiceConverter
from programs.applio_code.rvc.lib.tools.model_download import model_download_pipeline
from programs.music_separation_code.inference import proc_file
from assets.presence.discord_presence import RPCManager, track_presence


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Get current directory
now_dir = os.getcwd()
os.path.dirname(os.path.abspath(__file__))

# Define model configurations
MODELS_CONFIG = {
    "vocals": [
        {
            "name": "Mel-Roformer by KimberleyJSN",
            "path": os.path.join(now_dir, "models", "mel-vocals"),
            "model": os.path.join(now_dir, "models", "mel-vocals", "model.ckpt"),
            "config": os.path.join(now_dir, "models", "mel-vocals", "config.yaml"),
            "type": "mel_band_roformer",
            "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml",
            "model_url": "https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt",
        },
        {
            "name": "BS-Roformer by ViperX",
            "path": os.path.join(now_dir, "models", "bs-vocals"),
            "model": os.path.join(now_dir, "models", "bs-vocals", "model.ckpt"),
            "config": os.path.join(now_dir, "models", "bs-vocals", "config.yaml"),
            "type": "bs_roformer",
            "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml",
            "model_url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        },
        {
            "name": "MDX23C",
            "path": os.path.join(now_dir, "models", "mdx23c-vocals"),
            "model": os.path.join(now_dir, "models", "mdx23c-vocals", "model.ckpt"),
            "config": os.path.join(now_dir, "models", "mdx23c-vocals", "config.yaml"),
            "type": "mdx23c",
            "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_vocals_mdx23c.yaml",
            "model_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_mdx23c_sdr_10.17.ckpt",
        },
    ],
    "karaoke": [
        {
            "name": "Mel-Roformer Karaoke by aufr33 and viperx",
            "path": os.path.join(now_dir, "models", "mel-kara"),
            "model": os.path.join(now_dir, "models", "mel-kara", "model.ckpt"),
            "config": os.path.join(now_dir, "models", "mel-kara", "config.yaml"),
            "type": "mel_band_roformer",
            "config_url": "https://huggingface.co/shiromiya/audio-separation-models/resolve/main/mel_band_roformer_karaoke_aufr33_viperx/config_mel_band_roformer_karaoke.yaml",
            "model_url": "https://huggingface.co/shiromiya/audio-separation-models/resolve/main/mel_band_roformer_karaoke_aufr33_viperx/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
        },
        {
            "name": "UVR-BVE",
            "full_name": "UVR-BVE-4B_SN-44100-1.pth",
            "arch": "vr",
        },
    ],
    "denoise": [
        {
            "name": "Mel-Roformer Denoise Normal by aufr33",
            "path": os.path.join(now_dir, "models", "mel-denoise"),
            "model": os.path.join(now_dir, "models", "mel-denoise", "model.ckpt"),
            "config": os.path.join(now_dir, "models", "mel-denoise", "config.yaml"),
            "type": "mel_band_roformer",
            "config_url": "https://huggingface.co/shiromiya/audio-separation-models/resolve/main/mel-denoise/model_mel_band_roformer_denoise.yaml",
            "model_url": "https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
        },
        {
            "name": "Mel-Roformer Denoise Aggressive by aufr33",
            "path": os.path.join(now_dir, "models", "mel-denoise-aggr"),
            "model": os.path.join(now_dir, "models", "mel-denoise-aggr", "model.ckpt"),
            "config": os.path.join(now_dir, "models", "mel-denoise-aggr", "config.yaml"),
            "type": "mel_band_roformer",
            "config_url": "https://huggingface.co/shiromiya/audio-separation-models/resolve/main/mel-denoise/model_mel_band_roformer_denoise.yaml",
            "model_url": "https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt",
        },
        {
            "name": "UVR Denoise",
            "full_name": "UVR-DeNoise.pth",
            "arch": "vr",
        },
    ],
    "dereverb": [
        {
            "name": "MDX23C DeReverb by aufr33 and jarredou",
            "path": os.path.join(now_dir, "models", "mdx23c-dereveb"),
            "model": os.path.join(now_dir, "models", "mdx23c-dereveb", "model.ckpt"),
            "config": os.path.join(now_dir, "models", "mdx23c-dereveb", "config.yaml"),
            "type": "mdx23c",
            "config_url": "https://huggingface.co/jarredou/aufr33_jarredou_MDXv3_DeReverb/resolve/main/config_dereverb_mdx23c.yaml",
            "model_url": "https://huggingface.co/jarredou/aufr33_jarredou_MDXv3_DeReverb/resolve/main/dereverb_mdx23c_sdr_6.9096.ckpt",
        },
        {
            "name": "BS-Roformer Dereverb by anvuew",
            "path": os.path.join(now_dir, "models", "mdx23c-dereveb"),
            "model": os.path.join(now_dir, "models", "mdx23c-dereveb", "model.ckpt"),
            "config": os.path.join(now_dir, "models", "mdx23c-dereveb", "config.yaml"),
            "type": "bs_roformer",
            "config_url": "https://huggingface.co/anvuew/deverb_bs_roformer/resolve/main/deverb_bs_roformer_8_384dim_10depth.yaml",
            "model_url": "https://huggingface.co/anvuew/deverb_bs_roformer/resolve/main/deverb_bs_roformer_8_384dim_10depth.ckpt",
        },
        {
            "name": "UVR-Deecho-Dereverb",
            "full_name": "UVR-DeEcho-DeReverb.pth",
            "arch": "vr",
        },
        {
            "name": "MDX Reverb HQ by FoxJoy",
            "full_name": "Reverb_HQ_By_FoxJoy.onnx",
            "arch": "mdx",
        },
    ],
    "deecho": [
        {
            "name": "UVR-Deecho-Normal",
            "full_name": "UVR-De-Echo-Normal.pth",
            "arch": "vr",
        },
        {
            "name": "UVR-Deecho-Agggressive",
            "full_name": "UVR-De-Echo-Aggressive.pth",
            "arch": "vr",
        },
    ],
}

config_file = os.path.join(now_dir, "assets", "config.json")


@lru_cache(maxsize=None)
def import_voice_converter():
    """Cached import of VoiceConverter to avoid repeated imports."""
    from programs.applio_code.rvc.infer.infer import VoiceConverter
    return VoiceConverter()


def load_config_presence():
    """Load Discord presence configuration."""
    with open(config_file, "r", encoding="utf8") as file:
        config = json.load(file)
        return config["discord_presence"]


def initialize_presence():
    """Initialize Discord presence if enabled."""
    if load_config_presence():
        RPCManager.start_presence()


initialize_presence()


@lru_cache(maxsize=1)
def get_config():
    """Cached configuration getter."""
    from programs.applio_code.rvc.configs.config import Config
    return Config()


def download_file(url: str, path: str, filename: str) -> bool:
    """
    Download file with progress and error handling.
    
    Args:
        url (str): URL to download the file from
        path (str): Local directory to save the file
        filename (str): Name of the file to download
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, filename)

        if os.path.exists(file_path):
            logging.info(f"File '{filename}' already exists at '{path}'.")
            return True

        torch.hub.download_url_to_file(url, file_path)
        logging.info(f"File '{filename}' downloaded successfully")
        return True
    except Exception as e:
        logging.error(f"Error downloading file '{filename}' from '{url}': {e}")
        return False


def get_model_info_by_name(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve model information by name from all configured model types.
    
    Args:
        model_name (str): The name of the model to look up
        
    Returns:
        Optional[Dict[str, Any]]: Model configuration dictionary if found, None otherwise
    """
    all_models = []
    for model_list in MODELS_CONFIG.values():
        all_models.extend(model_list)
    
    for model in all_models:
        if model["name"] == model_name:
            return model
    return None


def get_last_modified_file(folder_path: str) -> Optional[str]:
    """
    Get the most recently modified file in a folder.
    
    Args:
        folder_path (str): Path to the directory to search
        
    Returns:
        Optional[str]: Name of the most recently modified file, or None if no files exist
        
    Raises:
        NotADirectoryError: If the provided path is not a directory
    """
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"{folder_path} is not a valid directory.")
    
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    if not files:
        return None
    
    return max(files, key=lambda x: os.path.getmtime(os.path.join(folder_path, x)))


def search_with_word(folder: str, word: str) -> Optional[str]:
    """
    Search for a file containing a specific word in a folder.
    
    Args:
        folder (str): Path to the directory to search
        word (str): Word to search for in file names
        
    Returns:
        Optional[str]: Name of the most recently modified file containing the word, 
                      or None if no files match
                      
    Raises:
        NotADirectoryError: If the provided path is not a directory
    """
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"{folder} is not a valid directory.")
    
    files = [file for file in os.listdir(folder) if word in file]
    if not files:
        return None
    
    return max(files, key=lambda file: os.path.getmtime(os.path.join(folder, file)))


def search_with_two_words(folder: str, word1: str, word2: str) -> Optional[str]:
    """
    Search for a file containing two specific words in a folder.
    
    Args:
        folder (str): Path to the directory to search
        word1 (str): First word to search for in file names
        word2 (str): Second word to search for in file names
        
    Returns:
        Optional[str]: Name of the most recently modified file containing both words, 
                      or None if no files match
                      
    Raises:
        NotADirectoryError: If the provided path is not a directory
    """
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"{folder} is not a valid directory.")
    
    files = [file for file in os.listdir(folder) if word1 in file and word2 in file]
    if not files:
        return None
    
    return max(files, key=lambda file: os.path.getmtime(os.path.join(folder, file)))


def get_last_modified_folder(path: str) -> Optional[str]:
    """
    Get the most recently modified folder in a path.
    
    Args:
        path (str): Path to the directory to search
        
    Returns:
        Optional[str]: Name of the most recently modified subdirectory, 
                      or None if no subdirectories exist
    """
    directories = [
        os.path.join(path, d)
        for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d))
    ]
    if not directories:
        return None
    
    return os.path.basename(max(directories, key=os.path.getmtime))


def add_audio_effects(
    audio_path: str,
    reverb_size: float,
    reverb_wet: float,
    reverb_dry: float,
    reverb_damping: float,
    reverb_width: float,
    output_path: str,
) -> str:
    """
    Add reverb effects to an audio file.
    
    Args:
        audio_path (str): Path to the input audio file
        reverb_size (float): Size of the reverb room (0.0-1.0)
        reverb_wet (float): Wet level of the reverb (0.0-1.0)
        reverb_dry (float): Dry level of the reverb (0.0-1.0)
        reverb_damping (float): Damping factor (0.0-1.0)
        reverb_width (float): Width of the reverb (0.0-1.0)
        output_path (str): Path to save the output audio file
        
    Returns:
        str: Path to the output audio file with reverb applied
    """
    board = Pedalboard([
        Reverb(
            room_size=reverb_size,
            dry_level=reverb_dry,
            wet_level=reverb_wet,
            damping=reverb_damping,
            width=reverb_width,
        )
    ])
    
    with AudioFile(audio_path) as f:
        with AudioFile(output_path, "w", f.samplerate, f.num_channels) as o:
            while f.tell() < f.frames:
                chunk = f.read(int(f.samplerate))
                effected = board(chunk, f.samplerate, reset=False)
                o.write(effected)
    
    return output_path


def merge_audios(
    vocals_path: str,
    inst_path: str,
    backing_path: str,
    output_path: str,
    main_gain: float,
    inst_gain: float,
    backing_Vol: float,
    output_format: str,
) -> str:
    """
    Merge multiple audio tracks with volume adjustments.
    
    Args:
        vocals_path (str): Path to the main vocals track
        inst_path (str): Path to the instrumental track
        backing_path (str): Path to the backing vocals track
        output_path (str): Path to save the merged output
        main_gain (float): Volume adjustment for main vocals (dB)
        inst_gain (float): Volume adjustment for instrumentals (dB)
        backing_Vol (float): Volume adjustment for backing vocals (dB)
        output_format (str): Output audio format (e.g. 'wav', 'mp3', 'flac')
        
    Returns:
        str: Path to the merged output audio file
    """
    main_vocal_audio = AudioSegment.from_file(vocals_path, format="flac") + main_gain
    instrumental_audio = AudioSegment.from_file(inst_path, format="flac") + inst_gain
    backing_vocal_audio = AudioSegment.from_file(backing_path, format="flac") + backing_Vol
    
    combined_audio = main_vocal_audio.overlay(
        instrumental_audio.overlay(backing_vocal_audio)
    )
    combined_audio.export(output_path, format=output_format)
    return output_path


def check_fp16_support(device: str) -> bool:
    """
    Check if the GPU supports FP16 operations.
    
    Args:
        device (str): Device identifier (e.g. "cuda:0")
        
    Returns:
        bool: True if FP16 is supported, False otherwise
    """
    try:
        i_device = int(str(device).split(":")[-1])
        gpu_name = torch.cuda.get_device_name(i_device)
        low_end_gpus = ["16", "P40", "P10", "1060", "1070", "1080"]
        
        if any(gpu in gpu_name for gpu in low_end_gpus) and "V100" not in gpu_name.upper():
            print(f"Your GPU {gpu_name} not support FP16 inference. Using FP32 instead.")
            return False
        return True
    except Exception as e:
        print(f"Error checking FP16 support: {e}")
        return False


@track_presence("Infer the Audio")
def full_inference_program(
    model_path: str,
    index_path: str,
    input_audio_path: str,
    output_path: str,
    export_format_rvc: str,
    split_audio: bool,
    autotune: bool,
    vocal_model: str,
    karaoke_model: str,
    dereverb_model: str,
    deecho: bool,
    deecho_model: str,
    denoise: bool,
    denoise_model: str,
    reverb: bool,
    vocals_volume: float,
    instrumentals_volume: float,
    backing_vocals_volume: float,
    export_format_final: str,
    devices: str,
    pitch: int,
    filter_radius: int,
    index_rate: float,
    rms_mix_rate: float,
    protect: float,
    pitch_extract: str,
    hop_lenght: int,
    reverb_room_size: float,
    reverb_damping: float,
    reverb_wet_gain: float,
    reverb_dry_gain: float,
    reverb_width: float,
    embedder_model: str,
    delete_audios: bool,
    use_tta: bool,
    batch_size: int,
    infer_backing_vocals: bool,
    infer_backing_vocals_model: str,
    infer_backing_vocals_index: str,
    change_inst_pitch: int,
    pitch_back: int,
    filter_radius_back: int,
    index_rate_back: float,
    rms_mix_rate_back: float,
    protect_back: float,
    pitch_extract_back: str,
    hop_length_back: int,
    export_format_rvc_back: str,
    split_audio_back: bool,
    autotune_back: bool,
    embedder_model_back: str,
) -> Tuple[str, str]:
    """
    Main inference function that orchestrates the entire audio processing pipeline.
    
    This function performs a complete audio transformation process including:
    1. Vocal and instrumental separation
    2. Karaoke/Backing vocal extraction
    3. Audio enhancement (noise reduction, reverb removal, echo cancellation)
    4. Voice conversion using RVC models
    5. Backing vocal processing (optional)
    6. Post processing (reverb, pitch adjustment)
    7. Audio mixing and export
    
    Args:
        model_path (str): Path to the RVC model file
        index_path (str): Path to the index file for the model
        input_audio_path (str): Path to the input audio file
        output_path (str): Path to save the final output (not used in this implementation)
        export_format_rvc (str): Audio format for RVC output (e.g. 'wav', 'flac')
        split_audio (bool): Whether to split audio for processing
        autotune (bool): Whether to apply autotune
        vocal_model (str): Name of the vocal separation model
        karaoke_model (str): Name of the karaoke separation model
        dereverb_model (str): Name of the dereverb model
        deecho (bool): Whether to apply de-echo processing
        deecho_model (str): Name of the de-echo model
        denoise (bool): Whether to apply noise reduction
        denoise_model (str): Name of the denoise model
        reverb (bool): Whether to apply reverb in post-processing
        vocals_volume (float): Volume adjustment for main vocals (dB)
        instrumentals_volume (float): Volume adjustment for instrumentals (dB)
        backing_vocals_volume (float): Volume adjustment for backing vocals (dB)
        export_format_final (str): Final export format for output
        devices (str): Device specification (e.g. '0' for GPU 0, 'cpu')
        pitch (int): Pitch shift amount in semitones
        filter_radius (int): Filter radius for conversion
        index_rate (float): Index rate for the model (0.0-1.0)
        rms_mix_rate (float): RMS mix rate (0.0-1.0)
        protect (float): Protect value (0.0-1.0)
        pitch_extract (str): Method for pitch extraction ('harvest', 'crepe', etc.)
        hop_lenght (int): Hop length for audio processing
        reverb_room_size (float): Room size for reverb (0.0-1.0)
        reverb_damping (float): Damping factor for reverb (0.0-1.0)
        reverb_wet_gain (float): Wet gain for reverb (0.0-1.0)
        reverb_dry_gain (float): Dry gain for reverb (0.0-1.0)
        reverb_width (float): Width for reverb (0.0-1.0)
        embedder_model (str): Name of the embedder model ('contentvec', etc.)
        delete_audios (bool): Whether to delete temporary audio files after processing
        use_tta (bool): Whether to use test-time augmentation
        batch_size (int): Batch size for processing
        infer_backing_vocals (bool): Whether to infer backing vocals
        infer_backing_vocals_model (str): Model path for backing vocal inference
        infer_backing_vocals_index (str): Index path for backing vocal inference
        change_inst_pitch (int): Pitch change for instrumentals
        pitch_back (int): Pitch for backing vocals
        filter_radius_back (int): Filter radius for backing vocals
        index_rate_back (float): Index rate for backing vocals
        rms_mix_rate_back (float): RMS mix rate for backing vocals
        protect_back (float): Protect value for backing vocals
        pitch_extract_back (str): Pitch extraction method for backing vocals
        hop_length_back (int): Hop length for backing vocal processing
        export_format_rvc_back (str): Export format for backing vocal output
        split_audio_back (bool): Whether to split audio for backing vocal processing
        autotune_back (bool): Whether to apply autotune to backing vocals
        embedder_model_back (str): Embedder model for backing vocals
        
    Returns:
        Tuple[str, str]: A tuple containing:
            - Status message indicating success
            - Path to the final output audio file
    """
    # Device setup
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        devices = devices.replace("-", " ")
        print(f"Number of GPUs available: {n_gpu}")
        first_device = devices.split()[0] if devices else "cpu"
        fp16 = check_fp16_support(first_device)
    else:
        devices = "cpu"
        print("Using CPU")
        fp16 = False

    # Initialize variables
    music_folder = os.path.splitext(os.path.basename(input_audio_path))[0]
    input_audio_basename = os.path.splitext(os.path.basename(input_audio_path))[0]

    # Step 1: Separate vocals
    vocals_dir = os.path.join(now_dir, "audio_files", music_folder, "vocals")
    inst_dir = os.path.join(now_dir, "audio_files", music_folder, "instrumentals")
    os.makedirs(vocals_dir, exist_ok=True)
    os.makedirs(inst_dir, exist_ok=True)
    
    model_info = get_model_info_by_name(vocal_model)
    
    # Download model if necessary
    model_ckpt_path = os.path.join(model_info["path"], "model.ckpt")
    if not os.path.exists(model_ckpt_path):
        download_file(model_info["model_url"], model_info["path"], "model.ckpt")
    
    config_json_path = os.path.join(model_info["path"], "config.yaml")
    if not os.path.exists(config_json_path):
        download_file(model_info["config_url"], model_info["path"], "config.yaml")
    
    if not fp16:
        with open(model_info["config"], "r") as file:
            config = yaml.safe_load(file)
        config["training"]["use_amp"] = False
        with open(model_info["config"], "w") as file:
            yaml.safe_dump(config, file)
    
    search_result = search_with_word(vocals_dir, "vocals")
    if search_result:
        print("Vocals already separated...")
    else:
        print("Separating vocals...")
        command = [
            "python",
            os.path.join(now_dir, "programs", "music_separation_code", "inference.py"),
            "--model_type", model_info["type"],
            "--config_path", model_info["config"],
            "--start_check_point", model_info["model"],
            "--input_file", input_audio_path,
            "--store_dir", vocals_dir,
            "--flac_file", "--pcm_type", "PCM_16",
            "--extract_instrumental",
        ]

        if devices == "cpu":
            command.append("--force_cpu")
        else:
            device_ids = [str(int(device)) for device in devices.split()]
            command.extend(["--device_ids"] + device_ids)

        try:
            subprocess.run(command, check=True, timeout=300)  # 5 minute timeout
        except subprocess.TimeoutExpired:
            logging.error(f"Vocal separation command timed out: {' '.join(command)}")
            raise
        except subprocess.CalledProcessError as e:
            logging.error(f"Vocal separation failed with return code {e.returncode}: {' '.join(command)}")
            raise
        
        # Rename instrumental file
        instrumental_file = search_with_two_words(
            vocals_dir,
            os.path.basename(input_audio_path).split(".")[0],
            "instrumental",
        )
        if instrumental_file:
            try:
                os.rename(
                    os.path.join(vocals_dir, instrumental_file),
                    os.path.join(
                        inst_dir,
                        f"{os.path.basename(input_audio_path).split('.')[0]}_instrumentals.flac",
                    ),
                )
            except OSError as e:
                logging.error(f"Failed to rename instrumental file: {e}")
                raise
    
    inst_file = os.path.join(
        inst_dir,
        search_with_two_words(
            inst_dir, os.path.basename(input_audio_path).split(".")[0], "instrumentals"
        ),
    )

    # Step 2: Karaoke separation
    karaoke_dir = os.path.join(now_dir, "audio_files", music_folder, "karaoke")
    os.makedirs(karaoke_dir, exist_ok=True)
    
    vocals_file = search_with_word(vocals_dir, "vocals")
    input_file = os.path.join(vocals_dir, vocals_file) if vocals_file else None
    
    model_info = get_model_info_by_name(karaoke_model)
    if model_info["name"] == "Mel-Roformer Karaoke by aufr33 and viperx":
        # Handle Mel-Roformer model
        model_ckpt_path = os.path.join(model_info["path"], "model.ckpt")
        if not os.path.exists(model_ckpt_path):
            download_file(model_info["model_url"], model_info["path"], "model.ckpt")
        
        config_json_path = os.path.join(model_info["path"], "config.yaml")
        if not os.path.exists(config_json_path):
            download_file(model_info["config_url"], model_info["path"], "config.yaml")
        
        if not fp16:
            with open(model_info["config"], "r") as file:
                config = yaml.safe_load(file)
            config["training"]["use_amp"] = False
            with open(model_info["config"], "w") as file:
                yaml.safe_dump(config, file)
        
        if input_file:
            command = [
                "python",
                os.path.join(now_dir, "programs", "music_separation_code", "inference.py"),
                "--model_type", model_info["type"],
                "--config_path", model_info["config"],
                "--start_check_point", model_info["model"],
                "--input_file", input_file,
                "--store_dir", karaoke_dir,
                "--flac_file", "--pcm_type", "PCM_16",
                "--extract_instrumental",
            ]
            
            if devices == "cpu":
                command.append("--force_cpu")
            else:
                device_ids = [str(int(device)) for device in devices.split()]
                command.extend(["--device_ids"] + device_ids)
            
            subprocess.run(command, check=True)
    else:
        # Handle VR model (UVR-BVE)
        separator = Separator(
            model_file_dir=os.path.join(now_dir, "models", "karaoke"),
            log_level=logging.WARNING,
            normalization_threshold=1.0,
            output_format="flac",
            output_dir=karaoke_dir,
            vr_params={
                "batch_size": batch_size,
                "enable_tta": use_tta,
            },
        )
        separator.load_model(model_filename=model_info["full_name"])
        if input_file:
            separator.separate(input_file)
        
        vocals_result = search_with_two_words(
            karaoke_dir,
            os.path.basename(input_audio_path).split(".")[0],
            "Vocals",
        )
        instrumental_result = search_with_two_words(
            karaoke_dir,
            os.path.basename(input_audio_path).split(".")[0],
            "Instrumental",
        )
        
        if vocals_result and "UVR-BVE-4B_SN-44100-1" in os.path.basename(vocals_result):
            os.rename(
                os.path.join(karaoke_dir, vocals_result),
                os.path.join(
                    karaoke_dir,
                    f"{os.path.basename(input_audio_path).split('.')[0]}_karaoke.flac",
                ),
            )
        if instrumental_result and "UVR-BVE-4B_SN-44100-1" in os.path.basename(instrumental_result):
            os.rename(
                os.path.join(karaoke_dir, instrumental_result),
                os.path.join(
                    karaoke_dir,
                    f"{os.path.basename(input_audio_path).split('.')[0]}_instrumental.flac",
                ),
            )

    # Step 3: Dereverb
    dereverb_dir = os.path.join(now_dir, "audio_files", music_folder, "dereverb")
    os.makedirs(dereverb_dir, exist_ok=True)
    
    karaoke_file = search_with_word(karaoke_dir, "karaoke")
    input_file = os.path.join(karaoke_dir, karaoke_file) if karaoke_file else None
    
    model_info = get_model_info_by_name(dereverb_model)
    if model_info["name"] in ["BS-Roformer Dereverb by anvuew", "MDX23C DeReverb by aufr33 and jarredou"]:
        # Handle model-specific dereverb
        model_ckpt_path = os.path.join(model_info["path"], "model.ckpt")
        if not os.path.exists(model_ckpt_path):
            download_file(model_info["model_url"], model_info["path"], "model.ckpt")
        
        config_json_path = os.path.join(model_info["path"], "config.yaml")
        if not os.path.exists(config_json_path):
            download_file(model_info["config_url"], model_info["path"], "config.yaml")
        
        if not fp16:
            with open(model_info["config"], "r") as file:
                config = yaml.safe_load(file)
            config["training"]["use_amp"] = False
            with open(model_info["config"], "w") as file:
                yaml.safe_dump(config, file)
        
        if input_file:
            command = [
                "python",
                os.path.join(now_dir, "programs", "music_separation_code", "inference.py"),
                "--model_type", model_info["type"],
                "--config_path", model_info["config"],
                "--start_check_point", model_info["model"],
                "--input_file", input_file,
                "--store_dir", dereverb_dir,
                "--flac_file", "--pcm_type", "PCM_16",
            ]
            
            if devices == "cpu":
                command.append("--force_cpu")
            else:
                device_ids = [str(int(device)) for device in devices.split()]
                command.extend(["--device_ids"] + device_ids)
            
            subprocess.run(command, check=True)
    else:
        # Handle VR models
        separator = Separator(
            model_file_dir=os.path.join(now_dir, "models", "dereverb"),
            log_level=logging.WARNING,
            normalization_threshold=1.0,
            output_format="flac",
            output_dir=dereverb_dir,
            output_single_stem="No Reverb",
            vr_params={"batch_size": batch_size, "enable_tta": use_tta} if model_info["arch"] == "vr" else None,
        )
        separator.load_model(model_filename=model_info["full_name"])
        if input_file:
            separator.separate(input_file)
        
        search_result = search_with_two_words(
            dereverb_dir,
            os.path.basename(input_audio_path).split(".")[0],
            "No Reverb",
        )
        
        if search_result and ("UVR-DeEcho-DeReverb" in os.path.basename(search_result) or 
                             "MDX Reverb HQ by FoxJoy" in os.path.basename(search_result)):
            os.rename(
                os.path.join(dereverb_dir, search_result),
                os.path.join(
                    dereverb_dir,
                    f"{os.path.basename(input_audio_path).split('.')[0]}_noreverb.flac",
                ),
            )

    # Step 4: Deecho (if requested)
    deecho_dir = os.path.join(now_dir, "audio_files", music_folder, "deecho")
    os.makedirs(deecho_dir, exist_ok=True)
    if deecho:
        print("Removing echo...")
        model_info = get_model_info_by_name(deecho_model)

        noreverb_file = search_with_word(dereverb_dir, "noreverb")
        input_file = os.path.join(dereverb_dir, noreverb_file) if noreverb_file else None

        if input_file:
            separator = Separator(
                model_file_dir=os.path.join(now_dir, "models", "deecho"),
                log_level=logging.WARNING,
                normalization_threshold=1.0,
                output_format="flac",
                output_dir=deecho_dir,
                output_single_stem="No Echo",
                vr_params={
                    "batch_size": batch_size,
                    "enable_tta": use_tta,
                },
            )
            separator.load_model(model_filename=model_info["full_name"])
            separator.separate(input_file)
            
            search_result = search_with_two_words(
                deecho_dir,
                os.path.basename(input_audio_path).split(".")[0],
                "No Echo",
            )
            
            if search_result and ("UVR-De-Echo-Normal" in os.path.basename(search_result) or 
                                 "UVR-Deecho-Agggressive" in os.path.basename(search_result)):
                os.rename(
                    os.path.join(deecho_dir, search_result),
                    os.path.join(
                        deecho_dir,
                        f"{os.path.basename(input_audio_path).split('.')[0]}_noecho.flac",
                    ),
                )

    # Step 5: Denoise (if requested)
    denoise_dir = os.path.join(now_dir, "audio_files", music_folder, "denoise")
    os.makedirs(denoise_dir, exist_ok=True)
    if denoise:
        model_info = get_model_info_by_name(denoise_model)
        print("Removing noise")
        
        # Determine input file based on processing chain
        if deecho:
            # Use deecho output if deecho was applied
            deecho_file = search_with_word(deecho_dir, "noecho")
            input_file = os.path.join(deecho_dir, deecho_file) if deecho_file else None
        else:
            # Use dereverb output otherwise
            noreverb_file = search_with_word(dereverb_dir, "noreverb")
            input_file = os.path.join(dereverb_dir, noreverb_file) if noreverb_file else None

        if input_file and (model_info["name"] == "Mel-Roformer Denoise Normal by aufr33" or 
                          model_info["name"] == "Mel-Roformer Denoise Aggressive by aufr33"):
            model_ckpt_path = os.path.join(model_info["path"], "model.ckpt")
            if not os.path.exists(model_ckpt_path):
                download_file(model_info["model_url"], model_info["path"], "model.ckpt")
            
            config_json_path = os.path.join(model_info["path"], "config.yaml")
            if not os.path.exists(config_json_path):
                download_file(model_info["config_url"], model_info["path"], "config.yaml")
            
            if not fp16:
                with open(model_info["config"], "r") as file:
                    config = yaml.safe_load(file)
                config["training"]["use_amp"] = False
                with open(model_info["config"], "w") as file:
                    yaml.safe_dump(config, file)
            
            command = [
                "python",
                os.path.join(now_dir, "programs", "music_separation_code", "inference.py"),
                "--model_type", model_info["type"],
                "--config_path", model_info["config"],
                "--start_check_point", model_info["model"],
                "--input_file", input_file,
                "--store_dir", denoise_dir,
                "--flac_file", "--pcm_type", "PCM_16",
            ]
            
            if devices == "cpu":
                command.append("--force_cpu")
            else:
                device_ids = [str(int(device)) for device in devices.split()]
                command.extend(["--device_ids"] + device_ids)
            
            try:
                subprocess.run(command, check=True, timeout=300)  # 5 minute timeout
            except subprocess.TimeoutExpired:
                logging.error(f"Denoise separation command timed out: {' '.join(command)}")
                raise
            except subprocess.CalledProcessError as e:
                logging.error(f"Denoise separation failed with return code {e.returncode}: {' '.join(command)}")
                raise
        elif input_file:
            # Handle VR models
            separator = Separator(
                model_file_dir=os.path.join(now_dir, "models", "denoise"),
                log_level=logging.WARNING,
                normalization_threshold=1.0,
                output_format="flac",
                output_dir=denoise_dir,
                output_single_stem="No Noise",
                vr_params={
                    "batch_size": batch_size,
                    "enable_tta": use_tta,
                },
            )
            separator.load_model(model_filename=model_info["full_name"])
            separator.separate(input_file)
            
            search_result = search_with_two_words(
                denoise_dir,
                os.path.basename(input_audio_path).split(".")[0],
                "No Noise",
            )
            if search_result and "UVR Denoise" in os.path.basename(search_result):
                os.rename(
                    os.path.join(denoise_dir, search_result),
                    os.path.join(
                        denoise_dir,
                        f"{os.path.basename(input_audio_path).split('.')[0]}_dry.flac",
                    ),
                )

    # Step 6: Apply RVC voice conversion
    # Determine final input path based on processing chain
    if denoise:
        dry_file = search_with_two_words(denoise_dir, os.path.basename(input_audio_path).split(".")[0], "dry")
        final_path = os.path.join(denoise_dir, dry_file) if dry_file else None
    elif deecho:
        noecho_file = search_with_two_words(deecho_dir, os.path.basename(input_audio_path).split(".")[0], "noecho")
        final_path = os.path.join(deecho_dir, noecho_file) if noecho_file else None
    elif search_with_word(dereverb_dir, "noreverb"):
        noreverb_file = search_with_word(dereverb_dir, "noreverb")
        final_path = os.path.join(dereverb_dir, noreverb_file) if noreverb_file else None
    else:
        final_path = None

    rvc_dir = os.path.join(now_dir, "audio_files", music_folder, "rvc")
    os.makedirs(rvc_dir, exist_ok=True)
    print("Making RVC inference...")
    output_rvc = os.path.join(
        rvc_dir,
        f"{os.path.basename(input_audio_path).split('.')[0]}_rvc.wav",
    )
    
    try:
        inference_vc = import_voice_converter()
        inference_vc.convert_audio(
            audio_input_path=final_path,
            audio_output_path=output_rvc,
            model_path=model_path,
            index_path=index_path,
            embedder_model=embedder_model,
            pitch=pitch,
            f0_file=None,
            f0_method=pitch_extract,
            filter_radius=filter_radius,
            index_rate=index_rate,
            volume_envelope=rms_mix_rate,
            protect=protect,
            split_audio=split_audio,
            f0_autotune=autotune,
            hop_length=hop_lenght,
            export_format=export_format_rvc,
            embedder_model_custom=None,
        )
        logging.info(f"RVC conversion completed successfully for {input_audio_basename}")
    except Exception as e:
        logging.error(f"RVC conversion failed for {input_audio_basename}: {e}")
        raise

    # Step 7: Process backing vocals (if requested)
    backing_vocals = os.path.join(karaoke_dir, search_with_word(karaoke_dir, "instrumental"))
    if infer_backing_vocals:
        print("Inferring backing vocals...")
        instrumental_file = search_with_word(karaoke_dir, "instrumental")
        if instrumental_file:
            input_file = os.path.join(karaoke_dir, instrumental_file)
            output_backing_vocals = os.path.join(
                karaoke_dir, f"{input_audio_basename}_instrumental_output.wav"
            )
            inference_vc.convert_audio(
                audio_input_path=input_file,
                audio_output_path=output_backing_vocals,
                model_path=infer_backing_vocals_model,
                index_path=infer_backing_vocals_index,
                embedder_model=embedder_model_back,
                pitch=pitch_back,
                f0_file=None,
                f0_method=pitch_extract_back,
                filter_radius=filter_radius_back,
                index_rate=index_rate_back,
                volume_envelope=rms_mix_rate_back,
                protect=protect_back,
                split_audio=split_audio_back,
                f0_autotune=autotune_back,
                hop_length=hop_length_back,
                export_format=export_format_rvc_back,
                embedder_model_custom=None,
            )
            backing_vocals = output_backing_vocals

    # Step 8: Apply post-processing (reverb, pitch shift)
    if reverb:
        print("Applying reverb...")
        last_rvc_file = get_last_modified_file(rvc_dir)
        if last_rvc_file:
            input_file = os.path.join(rvc_dir, last_rvc_file)
            output_file = os.path.join(
                rvc_dir,
                os.path.basename(input_audio_path),
            )
            add_audio_effects(
                input_file,
                reverb_room_size,
                reverb_wet_gain,
                reverb_dry_gain,
                reverb_damping,
                reverb_width,
                output_file,
            )
    
    if change_inst_pitch != 0:
        print("Changing instrumental pitch...")
        inst_path = os.path.join(
            now_dir,
            "audio_files",
            music_folder,
            "instrumentals",
            search_with_word(
                os.path.join(now_dir, "audio_files", music_folder, "instrumentals"),
                "instrumentals",
            ),
        )
        if os.path.exists(inst_path):
            audio = AudioSegment.from_file(inst_path)
            factor = 2 ** (change_inst_pitch / 12)
            new_frame_rate = int(audio.frame_rate * factor)
            audio = audio._spawn(audio.raw_data, overrides={"frame_rate": new_frame_rate})
            audio = audio.set_frame_rate(audio.frame_rate)
            
            output_dir_pitch = os.path.join(
                now_dir, "audio_files", music_folder, "instrumentals"
            )
            output_path_pitch = os.path.join(
                output_dir_pitch, "inst_with_changed_pitch.flac"
            )
            audio.export(output_path_pitch, format="flac")

    # Step 9: Merge all audio tracks
    final_dir = os.path.join(now_dir, "audio_files", music_folder, "final")
    os.makedirs(final_dir, exist_ok=True)

    # Determine vocals file to use
    last_rvc_file = get_last_modified_file(rvc_dir)
    vocals_file = os.path.join(rvc_dir, last_rvc_file) if last_rvc_file else output_rvc

    # Determine karaoke file to use
    karaoke_file = search_with_word(karaoke_dir, "Instrumental") or search_with_word(karaoke_dir, "instrumental")
    karaoke_file = os.path.join(karaoke_dir, karaoke_file) if karaoke_file else None
    
    final_output_path = os.path.join(
        final_dir,
        f"{os.path.basename(input_audio_path).split('.')[0]}_final.{export_format_final.lower()}",
    )
    print("Merging audios...")
    try:
        result = merge_audios(
            vocals_file,
            inst_file,
            backing_vocals,
            final_output_path,
            vocals_volume,
            instrumentals_volume,
            backing_vocals_volume,
            export_format_final,
        )
        print("Audios merged successfully")
    except Exception as e:
        logging.error(f"Audio merging failed: {e}")
        raise
    
    # Step 10: Cleanup (if requested)
    if delete_audios:
        print("Cleaning up temporary files...")
        main_directory = os.path.join(now_dir, "audio_files", music_folder)
        folder_to_keep = "final"
        for folder_name in os.listdir(main_directory):
            folder_path = os.path.join(main_directory, folder_name)
            if os.path.isdir(folder_path) and folder_name != folder_to_keep:
                shutil.rmtree(folder_path)
    
    return (
        f"Audio file {os.path.basename(input_audio_path).split('.')[0]} converted with success",
        result,
    )


def download_model(link: str) -> str:
    """
    Download a model using the model download pipeline.
    
    Args:
        link (str): URL or identifier for the model to download
        
    Returns:
        str: Status message indicating success or error
    """
    try:
        model_download_pipeline(link)
        return "Model downloaded successfully"
    except Exception as e:
        logging.error(f"Error downloading model: {e}")
        return f"Error: {str(e)}"


def download_music(link: str) -> str:
    """
    Download music from a URL using yt-dlp.
    
    Args:
        link (str): URL of the music to download
        
    Returns:
        str: Status message indicating success or error
    """
    if not link or not isinstance(link, str):
        logging.error("Invalid link provided.")
        return "Error: Invalid link"

    try:
        current_dir = os.getcwd()
    except OSError as e:
        logging.error(f"Error accessing current working directory: {e}")
        return "Error: Unable to access current working directory"

    output_dir = os.path.join(current_dir, "audio_files", "original_files")
    os.makedirs(output_dir, exist_ok=True)

    output_template = os.path.join(output_dir, "%(title)s.%(ext)s")

    command = [
        "yt-dlp",
        "-x",
        "--audio-format",
        "wav",
        "--output",
        output_template,
        "--cookies",
        f"{current_dir}/assets/ytdlstuff.txt",
        link,
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logging.info(f"Download successful: {result.stdout}")
        return "Music downloaded successfully"
    except FileNotFoundError:
        logging.error("yt-dlp is not installed. Please install it first.")
        return "Error: yt-dlp not found"
    except subprocess.CalledProcessError as e:
        logging.error(f"Download failed: {e.stderr}")
        return f"Error: {e.stderr}"



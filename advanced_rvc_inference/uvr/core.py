"""
UVR (Ultimate Vocal Remover) Core Module
This module contains the core functionality for audio separation features.
"""

import argparse
import glob
import os
import sys
import time
import warnings
try:
    import librosa
except ImportError:
    print("Warning: librosa not available. UVR functionality may be limited.")
    librosa = None
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from tqdm import tqdm
import logging

# Add the current directory to sys.path to allow importing from submodules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

from uvr.music_separation.inference import proc_file

# Import the music separation functions from the moved location
def run_separation(
    input_file,
    store_dir,
    model_type="mdx23c",
    config_path=None,
    checkpoint_path="",
    extract_instrumental=True,
    flac_file=False,
    pcm_type="PCM_24",
    use_tta=False,
    device_ids=None,
    force_cpu=False,
    disable_detailed_pbar=False,
    batch_size=1
):
    """
    Run audio separation using the UVR models from COVERMAKER.

    Args:
        input_file (str): Path to the input audio file.
        store_dir (str): Directory to store the output files.
        model_type (str): Type of model to use ('mdx23c', 'bs_roformer', etc.)
        config_path (str): Path to the model config file.
        checkpoint_path (str): Path to the model checkpoint file.
        extract_instrumental (bool): Whether to extract instrumental track.
        flac_file (bool): Whether to output in FLAC format.
        pcm_type (str): PCM type for FLAC ('PCM_16' or 'PCM_24').
        use_tta (bool): Whether to use Test-Time-Augmentation.
        device_ids (list): List of GPU IDs to use.
        force_cpu (bool): Force CPU usage even if GPU is available.
        disable_detailed_pbar (bool): Disable detailed progress bar.
        batch_size (int): Batch size for the separation process.
    """

    # Prepare command line arguments following COVERMAKER format
    cmd_args = [
        "--model_type", model_type,
        "--input_file", input_file,
        "--store_dir", store_dir,
    ]

    if config_path and os.path.exists(config_path):
        cmd_args.extend(["--config_path", config_path])

    if checkpoint_path and os.path.exists(checkpoint_path):
        cmd_args.extend(["--start_check_point", checkpoint_path])

    # Add device settings
    if force_cpu:
        cmd_args.append("--force_cpu")
    elif device_ids:
        if isinstance(device_ids, list):
            cmd_args.extend(["--device_ids"] + [str(d) for d in device_ids])
        else:
            cmd_args.extend(["--device_ids", str(device_ids)])

    # Add enhancement options
    if extract_instrumental:
        cmd_args.append("--extract_instrumental")

    if use_tta:
        cmd_args.append("--use_tta")

    if disable_detailed_pbar:
        cmd_args.append("--disable_detailed_pbar")

    # Set format based on flac_file flag
    if flac_file:
        cmd_args.append("--flac_file")
    else:
        cmd_args.append("--wav_file")

    # Add batch size
    cmd_args.extend(["--batch_size", str(batch_size)])

    # Add PCM type if using flac
    cmd_args.extend(["--pcm_type", pcm_type])

    # Call the music separation inference directly
    try:
        from uvr.music_separation.inference import proc_file
        proc_file(cmd_args)
        return True
    except ImportError:
        print("Could not import proc_file from uvr.music_separation.inference")
        print("Make sure the COVERMAKER music separation module is properly implemented")
        return False
    except Exception as e:
        print(f"Error during separation: {str(e)}")
        return False


def list_available_models():
    """
    List available UVR models that can be used for separation.
    """
    models = [
        "mel_band_roformer",
        "bs_roformer",
        "mdx23c",
        "htdemucs",
        "scnet",
        "scnet_unofficial",
        "segm_models",
        "swin_upernet",
        "torchseg",
        "bandit",
        "bandit_v2"
    ]
    return models


def get_model_info(model_type):
    """
    Get information about a specific model type.
    """
    model_info = {
        "mdx23c": {
            "description": "MDX23C model for music separation",
            "recommended_for": ["vocals", "instrumental", "drums", "bass"]
        },
        "bs_roformer": {
            "description": "BS-Roformer for high-quality music separation",
            "recommended_for": ["vocals", "instrumental", "drums", "bass"]
        },
        "mel_band_roformer": {
            "description": "MelBandRoformer using mel-scale bands",
            "recommended_for": ["vocals", "instrumental"]
        },
        "htdemucs": {
            "description": "HTDemucs high-time resolution model",
            "recommended_for": ["vocals", "drum", "bass", "other"]
        }
    }

    return model_info.get(model_type, {"description": "Unknown model type", "recommended_for": []})


def validate_audio_file(file_path):
    """
    Validates an audio file by trying to load it.
    """
    if librosa is None:
        return False, {"error": "librosa not available for audio validation"}

    try:
        # Try to load the audio file to validate it
        y, sr = librosa.load(file_path)
        return True, {"duration": librosa.get_duration(y=y, sr=sr), "sample_rate": sr}
    except Exception as e:
        return False, str(e)


def batch_separation(
    input_files,
    output_dir,
    model_type="mdx23c",
    **kwargs
):
    """
    Perform batch separation on multiple audio files.

    Args:
        input_files (list): List of paths to input audio files.
        output_dir (str): Base directory for output files.
        model_type (str): Type of model to use for separation.
        **kwargs: Additional arguments passed to run_separation.
    """
    results = []
    for i, input_file in enumerate(input_files):
        file_name = os.path.splitext(os.path.basename(input_file))[0]
        store_dir = os.path.join(output_dir, file_name)

        os.makedirs(store_dir, exist_ok=True)

        success = run_separation(
            input_file=input_file,
            store_dir=store_dir,
            model_type=model_type,
            **kwargs
        )

        results.append({"file": input_file, "success": success, "output_dir": store_dir})

    return results


# The original proc_file function
def proc_file_with_args(args=None):
    """Wrapper for the original proc_file function"""
    proc_file(args)


if __name__ == "__main__":
    # If called directly, run the original proc_file
    proc_file(None)

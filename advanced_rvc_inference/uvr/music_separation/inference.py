"""
UVR (Ultimate Vocal Remover) Inference Module
This module enables advanced audio separation using COVERMAKER features.
"""

import os
import sys
import argparse
import glob
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

# Add the current directory to sys.path to allow importing from submodules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

# Import core functionality from the enhanced separation models in the project
from uvr.music_separation.models import *

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_from_config(model_type, config_path):
    """
    Get model and configuration from model type and config path.
    """
    if model_type == "mel_band_roformer":
        from uvr.music_separation.models.mel_band_roformer import MelBandRoformer
        # Load config and model
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        model = MelBandRoformer(
            **config["model"],
            sample_rate=config["audio"]["sample_rate"]
        )
        return model, config
    elif model_type == "bs_roformer":
        from uvr.music_separation.models.bs_roformer import BSRoformer
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        model = BSRoformer(
            **config["model"],
            sample_rate=config["audio"]["sample_rate"]
        )
        return model, config
    elif model_type == "mdx23c":
        from uvr.music_separation.models.mdx23c import MDXNet
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        model = MDXNet(
            **config["model"],
            sample_rate=config["audio"]["sample_rate"]
        )
        return model, config
    else:
        # VR models like UVR, Demucs, etc.
        raise NotImplementedError(f"Model type {model_type} not implemented yet.")


def demix(config, model, mix, device, pbar=False, model_type="mel_band_roformer"):
    """
    Demix audio using the given model and configuration.
    """
    # Prepare the mixture for processing
    if len(mix.shape) == 1:
        mix = np.vstack([mix, mix])  # Convert mono to stereo

    # Normalize if needed
    if "normalize" in config.inference:
        if config.inference["normalize"] is True:
            mono = mix.mean(0)
            mean = mono.mean()
            std = mono.std()
            mix = (mix - mean) / std

    # Convert to tensor and move to device
    mix_tensor = torch.tensor(mix, dtype=torch.float32, device=device)
    
    # Process based on model type
    if model_type == "mel_band_roformer":
        return demix_mel_band_roformer(model, mix_tensor)
    elif model_type == "bs_roformer":
        return demix_bs_roformer(model, mix_tensor)
    elif model_type == "mdx23c":
        return demix_mdx23c(model, mix_tensor)
    else:
        # Default processing
        return demix_generic(model, mix_tensor)


def demix_mel_band_roformer(model, mix):
    """
    Demix using Mel-Band Roformer model.
    """
    model.eval()
    with torch.no_grad():
        # Process the mixture
        result = model.forward(mix.unsqueeze(0))
        
        # Extract stems
        vocals = result[0].cpu().numpy() # Assuming first output is vocals
        instrumentals = (mix - result[0]).cpu().numpy() # Instrumental is the difference
        
        return {"vocals": vocals, "instrumental": instrumentals}


def demix_bs_roformer(model, mix):
    """
    Demix using BS-Roformer model.
    """
    model.eval()
    with torch.no_grad():
        # Process the mixture
        result = model.forward(mix.unsqueeze(0))
        
        # Extract stems
        vocals = result[0].cpu().numpy() # Assuming first output is vocals
        instrumentals = (mix - result[0]).cpu().numpy() # Instrumental is the difference
        
        return {"vocals": vocals, "instrumental": instrumentals}


def demix_mdx23c(model, mix):
    """
    Demix using MDX23C model.
    """
    model.eval()
    with torch.no_grad():
        # Process the mixture
        result = model.forward(mix.unsqueeze(0))
        
        # Extract stems
        vocals = result[0].cpu().numpy() # Assuming first output is vocals
        instrumentals = (mix - result[0]).cpu().numpy() # Instrumental is the difference
        
        return {"vocals": vocals, "instrumental": instrumentals}


def demix_generic(model, mix):
    """
    Generic demix function for other model types.
    """
    model.eval()
    with torch.no_grad():
        # Generic processing approach
        result = model.forward(mix.unsqueeze(0))
        
        # Extract stems (this might vary depending on model)
        vocals = result[0].cpu().numpy() if len(result) > 0 else (mix / 2).cpu().numpy()
        instrumentals = (mix - vocals).cpu().numpy() if len(result) > 0 else (mix / 2).cpu().numpy()
        
        return {"vocals": vocals, "instrumental": instrumentals}


def proc_file(args=None):
    """
    Process a single audio file using music separation models.
    This function handles the entire pipeline from audio loading to separation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="mdx23c", 
                        help="Model type to use: mel_band_roformer, bs_roformer, mdx23c, etc.")
    parser.add_argument("--config_path", type=str, help="Path to the config file")
    parser.add_argument("--start_check_point", type=str, default="", 
                        help="Path to start checkpoint")
    parser.add_argument("--input_file", type=str, help="Path to input file to process")
    parser.add_argument("--store_dir", type=str, help="Path to store results")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--use_tta", action="store_true", 
                        help="Use Test Time Augmentation - improves quality but takes longer")
    parser.add_argument("--flac_file", action="store_true", help="Store as FLAC")
    parser.add_argument("--wav_file", action="store_true", help="Store as WAV")
    parser.add_argument("--device_ids", nargs="+", type=int, default=0, 
                        help="List of device IDs to use")
    parser.add_argument("--extract_instrumental", action="store_true", 
                        help="Extract instrumental instead of vocals")
    parser.add_argument("--disable_detailed_pbar", action="store_true", 
                        help="Disable detailed progress bar")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="Batch Size for separation")
    parser.add_argument("--pcm_type", type=str, choices=["PCM_16", "PCM_24", "FLOAT"], 
                        default="FLOAT", help="PCM type for audio files")

    if args is None:
        args = parser.parse_args()
    else:
        # If args is provided as a list, parse it as arguments
        if isinstance(args, list):
            args = parser.parse_args(args)
        # Otherwise use it as is

    # Determine device
    device = "cpu"
    if not args.force_cpu:
        if torch.cuda.is_available():
            if isinstance(args.device_ids, list):
                device = f"cuda:{args.device_ids[0]}"
            else:
                device = f"cuda:{args.device_ids}"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
    print(f"Using device: {device}")

    # Load model
    start_time = time.time()
    model, config = get_model_from_config(args.model_type, args.config_path)

    # Load checkpoint if provided
    if args.start_check_point != "":
        print(f"Loading checkpoint from: {args.start_check_point}")
        checkpoint = torch.load(args.start_check_point, map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

    # Move model to device
    model = model.to(device)

    if isinstance(args.device_ids, list) and len(args.device_ids) > 1:
        model = nn.DataParallel(model, device_ids=args.device_ids)

    print(f"Model loaded in {time.time() - start_time:.2f} seconds")

    # Process the input file
    if not os.path.exists(args.input_file):
        print(f"Input file does not exist: {args.input_file}")
        return

    print(f"Processing file: {args.input_file}")
    processing_start = time.time()

    # Load audio file
    if librosa is None:
        print("Error: librosa is not available. Cannot load audio file.")
        return
    mix, sr = librosa.load(args.input_file, sr=44100, mono=False)

    # Create output directory
    os.makedirs(args.store_dir, exist_ok=True)

    # Process the audio
    try:
        result = demix(config, model, mix, device, model_type=args.model_type)

        # Save the separated tracks
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]

        for stem_name, stem_audio in result.items():
            # Transpose to correct shape if needed (from channels-first to channels-last)
            if stem_audio.shape[0] == 2 and stem_audio.shape[1] < stem_audio.shape[0]*1000:  # Likely in [channels, samples] format
                stem_audio = stem_audio.T

            # Determine output format
            if args.flac_file:
                output_ext = ".flac"
                subtype = "PCM_24" if args.pcm_type == "PCM_24" else "PCM_16"
            elif args.wav_file:
                output_ext = ".wav"
                subtype = "FLOAT"
            else:
                output_ext = ".wav"  # Default
                subtype = "FLOAT"

            output_path = os.path.join(args.store_dir, f"{base_name}_{stem_name}{output_ext}")
            sf.write(output_path, stem_audio, sr, subtype=subtype)
            print(f"Saved {stem_name} to {output_path}")

        processing_time = time.time() - processing_start
        print(f"Processing completed in {processing_time:.2f} seconds")

        # If extract instrumental flag is set, create instrumental track as inverse of vocals
        if args.extract_instrumental and "vocals" in result:
            instrumental_audio = mix.T - result["vocals"]  # Transpose mix to match shape
            output_ext = ".flac" if args.flac_file else ".wav"
            subtype = "PCM_24" if (args.flac_file and args.pcm_type == "PCM_24") else "FLOAT"
            instrumental_path = os.path.join(args.store_dir, f"{base_name}_instrumental{output_ext}")
            sf.write(instrumental_path, instrumental_audio, sr, subtype=subtype)
            print(f"Saved instrumental to {instrumental_path}")

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    proc_file()
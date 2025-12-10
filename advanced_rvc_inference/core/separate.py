import os
import sys
import subprocess

sys.path.append(os.getcwd())

from advanced_rvc_inference.core.ui import gr_info, gr_warning
from advanced_rvc_inference.variables import python, translations, configs
from advanced_rvc_inference.infer.create_reference import proc_file
import yaml
from audio_separator.separator import Separator

# Get the base models directory from translations
models_base_dir = translations["uvr5_path"]
# Get the audios output directory from translations
audios_output_dir = translations["audios_path"]

models_vocals = [
    {
        "name": "Mel-Roformer by KimberleyJSN",
        "path": os.path.join(models_base_dir, "mel-vocals"),
        "model": os.path.join(models_base_dir, "mel-vocals", "model.ckpt"),
        "config": os.path.join(models_base_dir, "mel-vocals", "config.yaml"),
        "type": "mel_band_roformer",
        "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml",
        "model_url": "https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt",
    },
    {
        "name": "BS-Roformer by ViperX",
        "path": os.path.join(models_base_dir, "bs-vocals"),
        "model": os.path.join(models_base_dir, "bs-vocals", "model.ckpt"),
        "config": os.path.join(models_base_dir, "bs-vocals", "config.yaml"),
        "type": "bs_roformer",
        "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml",
        "model_url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    },
    {
        "name": "MDX23C",
        "path": os.path.join(models_base_dir, "mdx23c-vocals"),
        "model": os.path.join(models_base_dir, "mdx23c-vocals", "model.ckpt"),
        "config": os.path.join(models_base_dir, "mdx23c-vocals", "config.yaml"),
        "type": "mdx23c",
        "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_vocals_mdx23c.yaml",
        "model_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_mdx23c_sdr_10.17.ckpt",
    },
]

karaoke_models = [
    {
        "name": "Mel-Roformer Karaoke by aufr33 and viperx",
        "path": os.path.join(models_base_dir, "mel-kara"),
        "model": os.path.join(models_base_dir, "mel-kara", "model.ckpt"),
        "config": os.path.join(models_base_dir, "mel-kara", "config.yaml"),
        "type": "mel_band_roformer",
        "config_url": "https://huggingface.co/shiromiya/audio-separation-models/resolve/main/mel_band_roformer_karaoke_aufr33_viperx/config_mel_band_roformer_karaoke.yaml",
        "model_url": "https://huggingface.co/shiromiya/audio-separation-models/resolve/main/mel_band_roformer_karaoke_aufr33_viperx/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
    },
    {
        "name": "UVR-BVE",
        "full_name": "UVR-BVE-4B_SN-44100-1.pth",
        "path": models_base_dir,
        "model": os.path.join(models_base_dir, "UVR-BVE-4B_SN-44100-1.pth"),
        "arch": "vr",
    },
]

denoise_models = [
    {
        "name": "Mel-Roformer Denoise Normal by aufr33",
        "path": os.path.join(models_base_dir, "mel-denoise"),
        "model": os.path.join(models_base_dir, "mel-denoise", "model.ckpt"),
        "config": os.path.join(models_base_dir, "mel-denoise", "config.yaml"),
        "type": "mel_band_roformer",
        "config_url": "https://huggingface.co/shiromiya/audio-separation-models/resolve/main/mel-denoise/model_mel_band_roformer_denoise.yaml",
        "model_url": "https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
    },
    {
        "name": "Mel-Roformer Denoise Aggressive by aufr33",
        "path": os.path.join(models_base_dir, "mel-denoise-aggr"),
        "model": os.path.join(models_base_dir, "mel-denoise-aggr", "model.ckpt"),
        "config": os.path.join(models_base_dir, "mel-denoise-aggr", "config.yaml"),
        "type": "mel_band_roformer",
        "config_url": "https://huggingface.co/shiromiya/audio-separation-models/resolve/main/mel-denoise/model_mel_band_roformer_denoise.yaml",
        "model_url": "https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt",
    },
    {
        "name": "UVR Denoise",
        "full_name": "UVR-DeNoise.pth",
        "path": models_base_dir,
        "model": os.path.join(models_base_dir, "UVR-DeNoise.pth"),
        "arch": "vr",
    },
]

dereverb_models = [
    {
        "name": "Mel-Roformer Dereverb by aufr33",
        "path": os.path.join(models_base_dir, "mel-dereverb"),
        "model": os.path.join(models_base_dir, "mel-dereverb", "model.ckpt"),
        "config": os.path.join(models_base_dir, "mel-dereverb", "config.yaml"),
        "type": "mel_band_roformer",
        "config_url": "https://huggingface.co/shiromiya/audio-separation-models/resolve/main/mel-dereverb/config_mel_band_roformer_dereverb.yaml",
        "model_url": "https://huggingface.co/shiromiya/audio-separation-models/resolve/main/mel-dereverb/dereverb_mel_band_roformer_aufr33_sdr_9.4778.ckpt",
    },
    {
        "name": "UVR DeReverb",
        "full_name": "UVR-DeEcho-DeReverb.pth",
        "path": models_base_dir,
        "model": os.path.join(models_base_dir, "UVR-DeEcho-DeReverb.pth"),
        "arch": "vr",
    },
]

# Define model categories for better organization
model_categories = {
    "vocals": models_vocals,
    "karaoke": karaoke_models,
    "denoise": denoise_models,
    "dereverb": dereverb_models
}

def initialize_model_directories():
    """Create model directories if they don't exist"""
    # Create base models directory first
    os.makedirs(models_base_dir, exist_ok=True)
    
    for category, models in model_categories.items():
        for model in models:
            if "path" in model:
                os.makedirs(model["path"], exist_ok=True)
    
    # Create audios output directory
    os.makedirs(audios_output_dir, exist_ok=True)
    
    gr_info(f"Model directories initialized at: {models_base_dir}")
    gr_info(f"Audio output directory initialized at: {audios_output_dir}")

def download_model_if_missing(model_info):
    """Download model if it doesn't exist locally"""
    # Check if model file exists
    model_file = model_info.get("model") or model_info.get("full_name")
    if not model_file:
        gr_warning(f"No model file specified for {model_info['name']}")
        return False
    
    # For UVR models (with full_name), check if file exists
    if "full_name" in model_info and not os.path.exists(model_info["model"]):
        gr_info(f"UVR model {model_info['name']} not found at {model_info['model']}")
        gr_info(f"Please download {model_info['full_name']} manually to {models_base_dir}")
        return False
    
    # For other models with model_url
    elif "model" in model_info and not os.path.exists(model_info["model"]):
        gr_info(f"Downloading {model_info['name']}...")
        try:
            # Download config if missing
            if "config" in model_info and not os.path.exists(model_info["config"]):
                subprocess.run([python, "-m", "wget", "-O", model_info["config"], model_info["config_url"]], 
                              check=True, capture_output=True, text=True)
            
            # Download model if missing
            if "model_url" in model_info:
                subprocess.run([python, "-m", "wget", "-O", model_info["model"], model_info["model_url"]], 
                              check=True, capture_output=True, text=True)
                
            gr_info(f"Successfully downloaded {model_info['name']}")
            return True
            
        except subprocess.CalledProcessError as e:
            gr_warning(f"Failed to download {model_info['name']}: {e.stderr}")
            return False
        except Exception as e:
            gr_warning(f"Failed to download {model_info['name']}: {str(e)}")
            return False
    
    return True

def get_model_choices(category="vocals"):
    """Get list of model names for a given category"""
    if category in model_categories:
        return [model["name"] for model in model_categories[category]]
    return []

def get_model_info(model_name, category="vocals"):
    """Get model information by name and category"""
    if category in model_categories:
        for model in model_categories[category]:
            if model["name"] == model_name:
                return model
    return None

def create_separator_for_model(model_name, category="vocals", output_dir=None):
    """Create a Separator instance for the specified model"""
    model_info = get_model_info(model_name, category)
    
    if not model_info:
        gr_warning(f"Model {model_name} not found in category {category}")
        return None
    
    # Ensure model is available
    if not download_model_if_missing(model_info):
        gr_warning(f"Model {model_name} is not available")
        return None
    
    # Use provided output_dir or default to audios_output_dir
    output_directory = output_dir or audios_output_dir
    
    try:
        # Create separator based on model type
        if "type" in model_info and model_info["type"] in ["mel_band_roformer", "bs_roformer", "mdx23c"]:
            return Separator(
                model_file_path=model_info["model"],
                model_config_path=model_info["config"],
                output_dir=output_directory,
                use_cuda=True,
                output_format="WAV"
            )
        elif "arch" in model_info and model_info["arch"] == "vr":
            # For UVR models
            return Separator(
                model_file_path=model_info["model"],
                output_dir=output_directory,
                use_cuda=True,
                output_format="WAV"
            )
    except Exception as e:
        gr_warning(f"Failed to create separator for {model_name}: {str(e)}")
    
    return None

def separate_audio(audio_path, model_name, category="vocals", output_format="WAV", custom_output_dir=None):
    """Separate audio using specified model"""
    # Create output directory
    output_directory = custom_output_dir or audios_output_dir
    os.makedirs(output_directory, exist_ok=True)
    
    separator = create_separator_for_model(model_name, category, output_directory)
    if not separator:
        return None
    
    try:
        gr_info(f"Starting separation with {model_name}...")
        gr_info(f"Input: {audio_path}")
        gr_info(f"Output directory: {output_directory}")
        
        # Perform separation
        output_files = separator.separate(audio_path)
        
        gr_info(f"Separation completed successfully")
        gr_info(f"Output files: {output_files}")
        return output_files
        
    except Exception as e:
        gr_warning(f"Separation failed: {str(e)}")
        import traceback
        gr_warning(f"Traceback: {traceback.format_exc()}")
        return None

def get_available_models():
    """Get all available models grouped by category"""
    available_models = {}
    for category, models in model_categories.items():
        available_models[category] = []
        for model in models:
            # Check if model file exists
            if "model" in model and os.path.exists(model.get("model", "")):
                available_models[category].append({
                    "name": model["name"],
                    "available": True,
                    "path": model["model"]
                })
            elif "full_name" in model:
                # For UVR models, check if they exist
                model_file = model.get("model") or os.path.join(models_base_dir, model["full_name"])
                available = os.path.exists(model_file)
                available_models[category].append({
                    "name": model["name"],
                    "available": available,
                    "path": model_file if available else model.get("full_name")
                })
    
    return available_models

def get_separated_audios():
    """Get list of separated audio files in the output directory"""
    if not os.path.exists(audios_output_dir):
        return []
    
    audio_files = []
    for file in os.listdir(audios_output_dir):
        if file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
            audio_files.append({
                "name": file,
                "path": os.path.join(audios_output_dir, file),
                "size": os.path.getsize(os.path.join(audios_output_dir, file))
            })
    
    # Sort by modification time (newest first)
    audio_files.sort(key=lambda x: os.path.getmtime(x["path"]), reverse=True)
    
    return audio_files

def clear_separated_audios():
    """Clear all separated audio files from output directory"""
    if not os.path.exists(audios_output_dir):
        return 0
    
    count = 0
    for file in os.listdir(audios_output_dir):
        if file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
            try:
                os.remove(os.path.join(audios_output_dir, file))
                count += 1
            except Exception as e:
                gr_warning(f"Failed to delete {file}: {str(e)}")
    
    gr_info(f"Cleared {count} audio files from {audios_output_dir}")
    return count

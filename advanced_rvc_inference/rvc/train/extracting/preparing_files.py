import os
import sys
import shutil
from random import shuffle

sys.path.append(os.getcwd())

from advanced_rvc_inference.core.ui import configs, config
from advanced_rvc_inference.rvc.train.extracting.embedding import create_mute_file

def mute_file(embedders_mode, embedder_model, mute_base_path, rvc_version):
    """Get the path to the mute feature file based on embedder configuration."""
    if embedders_mode.startswith(("spin", "whisper")):
        mute_file_name = f"mute_{embedder_model}.npy"
    else:
        mute_file_name = {
            "contentvec_base": "mute.npy",
            "hubert_base": "mute.npy",
            "vietnamese_hubert_base": "mute_vietnamese.npy",
            "japanese_hubert_base": "mute_japanese.npy",
            "korean_hubert_base": "mute_korean.npy",
            "chinese_hubert_base": "mute_chinese.npy",
            "portuguese_hubert_base": "mute_portuguese.npy"
        }.get(embedder_model, None)

    if mute_file_name is None:
        # Create the mute file if it doesn't exist
        create_mute_file(rvc_version, embedder_model, embedders_mode, config.is_half)
        mute_file_name = f"mute_{embedder_model}.npy"

    return os.path.join(mute_base_path, f"{rvc_version}_extracted", mute_file_name)

def generate_config(rvc_version, sample_rate, model_path):
    """Generate config file for the model if it doesn't exist."""
    config_save_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_save_path):
        src_config_path = os.path.join(
            os.getcwd(), 
            "advanced_rvc_inference", 
            "configs", 
            rvc_version, 
            f"{sample_rate}.json"
        )
        if os.path.exists(src_config_path):
            shutil.copy(src_config_path, config_save_path)
        else:
            print(f"Warning: Config file not found at {src_config_path}")

def generate_filelist(pitch_guidance, model_path, rvc_version, sample_rate, 
                     embedders_mode="fairseq", embedder_model="hubert_base", rms_extract=False):
    """Generate filelist for training."""
    gt_wavs_dir = os.path.join(model_path, "sliced_audios")
    feature_dir = os.path.join(model_path, f"{rvc_version}_extracted")
    f0_dir, f0nsf_dir, energy_dir = None, None, None

    # Check for necessary directories
    if not os.path.exists(gt_wavs_dir):
        raise FileNotFoundError(f"Directory not found: {gt_wavs_dir}")
    if not os.path.exists(feature_dir):
        raise FileNotFoundError(f"Directory not found: {feature_dir}")

    if pitch_guidance:
        f0_dir = os.path.join(model_path, "f0")
        f0nsf_dir = os.path.join(model_path, "f0_voiced")
        if not os.path.exists(f0_dir):
            raise FileNotFoundError(f"Directory not found: {f0_dir}")
        if not os.path.exists(f0nsf_dir):
            raise FileNotFoundError(f"Directory not found: {f0nsf_dir}")

    if rms_extract:
        energy_dir = os.path.join(model_path, "energy")
        if not os.path.exists(energy_dir):
            raise FileNotFoundError(f"Directory not found: {energy_dir}")

    # Get all file names without extensions
    gt_wavs_files = set(
        os.path.splitext(name)[0] for name in os.listdir(gt_wavs_dir) 
        if name.endswith(".wav")
    )
    feature_files = set(
        os.path.splitext(name)[0] for name in os.listdir(feature_dir) 
        if name.endswith(".npy")
    )
    
    # Start with intersection of wav and feature files
    names = gt_wavs_files & feature_files

    # Further intersect with other required files
    if pitch_guidance:
        f0_files = set(
            os.path.splitext(name)[0].replace(".wav", "") for name in os.listdir(f0_dir) 
            if name.endswith(".npy")
        )
        f0nsf_files = set(
            os.path.splitext(name)[0].replace(".wav", "") for name in os.listdir(f0nsf_dir) 
            if name.endswith(".npy")
        )
        names = names & f0_files & f0nsf_files
    
    if rms_extract:
        energy_files = set(
            os.path.splitext(name)[0].replace(".wav", "") for name in os.listdir(energy_dir) 
            if name.endswith(".npy")
        )
        names = names & energy_files
    
    if not names:
        raise ValueError("No matching files found across all required directories")
    
    options = []
    mute_base_path = os.path.join(configs["logs_path"], "mute")

    # Create mute directories if they don't exist
    os.makedirs(os.path.join(mute_base_path, f"{rvc_version}_extracted"), exist_ok=True)
    os.makedirs(os.path.join(mute_base_path, "sliced_audios"), exist_ok=True)
    os.makedirs(os.path.join(mute_base_path, "f0"), exist_ok=True)
    os.makedirs(os.path.join(mute_base_path, "f0_voiced"), exist_ok=True)
    if rms_extract:
        os.makedirs(os.path.join(mute_base_path, "energy"), exist_ok=True)

    # Add regular file entries
    for name in sorted(names):  # Sort for consistent ordering
        if pitch_guidance:
            if rms_extract:
                option = f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{f0_dir}/{name}.wav.npy|{f0nsf_dir}/{name}.wav.npy|{energy_dir}/{name}.wav.npy|0"
            else:
                option = f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{f0_dir}/{name}.wav.npy|{f0nsf_dir}/{name}.wav.npy|0"
        else:
            if rms_extract:
                option = f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{energy_dir}/{name}.wav.npy|0"
            else:
                option = f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|0"
        options.append(option)

    # Add mute entries
    mute_audio_path = os.path.join(mute_base_path, "sliced_audios", f"mute{sample_rate}.wav")
    mute_feature_path = mute_file(embedders_mode, embedder_model, mute_base_path, rvc_version)
    
    for _ in range(2):  # Add two mute entries
        if pitch_guidance:
            if rms_extract:
                option = f"{mute_audio_path}|{mute_feature_path}|{os.path.join(mute_base_path, 'f0', 'mute.wav.npy')}|{os.path.join(mute_base_path, 'f0_voiced', 'mute.wav.npy')}|{os.path.join(mute_base_path, 'energy', 'mute.wav.npy')}|0"
            else:
                option = f"{mute_audio_path}|{mute_feature_path}|{os.path.join(mute_base_path, 'f0', 'mute.wav.npy')}|{os.path.join(mute_base_path, 'f0_voiced', 'mute.wav.npy')}|0"
        else:
            if rms_extract:
                option = f"{mute_audio_path}|{mute_feature_path}|{os.path.join(mute_base_path, 'energy', 'mute.wav.npy')}|0"
            else:
                option = f"{mute_audio_path}|{mute_feature_path}|0"
        options.append(option)

    # Shuffle and write to file
    shuffle(options)
    filelist_path = os.path.join(model_path, "filelist.txt")
    with open(filelist_path, "w", encoding="utf-8") as f:
        f.write("\n".join(options))
    
    return filelist_path, len(options)

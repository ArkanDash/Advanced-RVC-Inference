import os
import sys
import json
import shutil
from random import shuffle

sys.path.append(os.getcwd())

from arvc.utils.variables import configs, config, logger
from arvc.engine.training.extract.embedding import create_mute_file

def mute_file(embedders_mode, embedder_model, mute_base_path, rvc_version):
    """Get the path to the mute feature file based on embedder configuration.
    
    VRVC addition: supports spin-v1/v2 embedder models.
    """
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
            "portuguese_hubert_base": "mute_portuguese.npy",
            # VRVC additions:
            "spin-v1": "mute_spin-v1.npy",
            "spin-v2": "mute_spin-v2.npy",
        }.get(embedder_model, None)

    if mute_file_name is None or not os.path.exists(os.path.join(mute_base_path, f"{rvc_version}_extracted", mute_file_name)):
        create_mute_file(rvc_version, embedder_model, embedders_mode, config.is_half)
        mute_file_name = f"mute_{embedder_model}.npy"

    return os.path.join(mute_base_path, f"{rvc_version}_extracted", mute_file_name)

def generate_config(rvc_version, sample_rate, model_path, architecture="RVC"):
    """Generate config file for the model if it doesn't exist.
    
    VRVC addition: architecture parameter — SVC mode sets hop_length=441.
    """
    config_save_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_save_path):
        src_config_path = os.path.join(
            os.getcwd(), 
            "arvc", 
            "configs", 
            rvc_version, 
            f"{sample_rate}.json"
        )
        if os.path.exists(src_config_path):
            shutil.copy(src_config_path, config_save_path)
        else:
            print(f"Warning: Config file not found at {src_config_path}")

    # VRVC: SVC architecture requires hop_length=441
    if os.path.exists(config_save_path) and architecture == "SVC": 
        with open(config_save_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            data["data"]["hop_length"] = 441

        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

def generate_filelist(pitch_guidance, model_path, rvc_version, sample_rate, 
                     embedders_mode="fairseq", embedder_model="hubert_base", 
                     rms_extract=False, include_mutes=2):
    """Generate filelist for training.
    
    VRVC additions:
    - include_mutes: configurable number of mute entries per speaker (default 2)
    - Multi-speaker support: mute entries are added per-speaker, not just flat
    """
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

    # If no overlap, try stripping .wav suffix from feature files
    if not names and feature_files:
        feature_stripped = set(
            name.replace(".wav", "") for name in feature_files
        )
        names = gt_wavs_files & feature_stripped

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
        # Diagnostic: log what's in each directory to help debug
        import logging as _log
        _logger = _log.getLogger(__name__)
        
        wav16k_dir = os.path.join(model_path, "sliced_audios_16k")
        wav16k_info = ""
        if os.path.exists(wav16k_dir):
            wav16k_files = [f for f in os.listdir(wav16k_dir) if f.endswith(".wav")]
            wav16k_info = f"\n  sliced_audios_16k ({len(wav16k_files)} wavs): exists - embedding extraction may have failed to produce output"
        else:
            wav16k_info = f"\n  sliced_audios_16k: MISSING - preprocessing may not have created 16kHz resamples"
        
        _logger.error(
            f"File matching failed! Directory contents:\n"
            f"  sliced_audios ({len(gt_wavs_files)} wavs): {sorted(gt_wavs_files)[:5]}...\n"
            f"  {rvc_version}_extracted ({len(feature_files)} npys): {sorted(feature_files)[:5]}...\n"
            f"  intersection before f0: {sorted(gt_wavs_files & feature_files)[:5]}...{wav16k_info}"
        )
        if pitch_guidance:
            _logger.error(
                f"  f0 ({len(f0_files)} npys): {sorted(f0_files)[:5]}...\n"
                f"  f0_voiced ({len(f0nsf_files)} npys): {sorted(f0nsf_files)[:5]}..."
            )
        raise ValueError("No matching files found across all required directories")
    
    options = []
    sids = []  # VRVC: track speaker IDs for per-speaker mute entries
    mute_base_path = os.path.join(configs["logs_path"], "mute")

    # Create mute directories if they don't exist
    os.makedirs(os.path.join(mute_base_path, f"{rvc_version}_extracted"), exist_ok=True)
    os.makedirs(os.path.join(mute_base_path, "sliced_audios"), exist_ok=True)
    os.makedirs(os.path.join(mute_base_path, "f0"), exist_ok=True)
    os.makedirs(os.path.join(mute_base_path, "f0_voiced"), exist_ok=True)
    if rms_extract:
        os.makedirs(os.path.join(mute_base_path, "energy"), exist_ok=True)

    # Add regular file entries and track speaker IDs
    for name in sorted(names):
        sid = name.split("_")[0]
        if sid not in sids: sids.append(sid)

        option = {
            True: {
                True: f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{f0_dir}/{name}.wav.npy|{f0nsf_dir}/{name}.wav.npy|{energy_dir}/{name}.wav.npy|{sid}",
                False: f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{f0_dir}/{name}.wav.npy|{f0nsf_dir}/{name}.wav.npy|{sid}"
            },
            False: {
                True: f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{energy_dir}/{name}.wav.npy|{sid}",
                False: f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{sid}"
            }
        }[pitch_guidance][rms_extract]

        options.append(option)

    # VRVC: Add mute entries — per-speaker with configurable count
    if include_mutes > 0:
        mute_audio_path = os.path.join(mute_base_path, "sliced_audios", f"mute{sample_rate}.wav")
        mute_feature_path = mute_file(embedders_mode, embedder_model, mute_base_path, rvc_version)
        mute_f0_path = os.path.join(mute_base_path, 'f0', 'mute.wav.npy')
        mute_f0nsf_path = os.path.join(mute_base_path, 'f0_voiced', 'mute.wav.npy')
        mute_energy_path = os.path.join(mute_base_path, 'energy', 'mute.wav.npy')

        for sid in sids * include_mutes:
            option = {
                True: {
                    True: f"{mute_audio_path}|{mute_feature_path}|{mute_f0_path}|{mute_f0nsf_path}|{mute_energy_path}|{sid}",
                    False: f"{mute_audio_path}|{mute_feature_path}|{mute_f0_path}|{mute_f0nsf_path}|{sid}"
                },
                False: {
                    True: f"{mute_audio_path}|{mute_feature_path}|{mute_energy_path}|{sid}",
                    False: f"{mute_audio_path}|{mute_feature_path}|{sid}"
                }
            }[pitch_guidance][rms_extract]

            options.append(option)

    # Shuffle and write to file
    shuffle(options)
    filelist_path = os.path.join(model_path, "filelist.txt")
    with open(filelist_path, "w", encoding="utf-8") as f:
        f.write("\n".join(options))

    # VRVC: Update config.json with speaker count
    configs_path = os.path.join(model_path, "config.json")
    if os.path.exists(configs_path):
        with open(configs_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            data["sid"] = len(sids)
        with open(configs_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    return filelist_path, len(options)

import os
import sys
import shutil

from random import shuffle

sys.path.append(os.getcwd())

from main.app.core.ui import configs, config
from main.inference.extracting.embedding import create_mute_file

def mute_file(embedders_mode, embedders_model, mute_base_path, rvc_version):
    if embedders_mode.startswith(("spin", "whisper")):
        mute_file = f"mute_{embedders_model}.npy"
    else:
        mute_file = {
            "contentvec_base": "mute.npy",
            "hubert_base": "mute.npy",
            "vietnamese_hubert_base": "mute_vietnamese.npy",
            "japanese_hubert_base": "mute_japanese.npy",
            "korean_hubert_base": "mute_korean.npy",
            "chinese_hubert_base": "mute_chinese.npy",
            "portuguese_hubert_base": "mute_portuguese.npy"
        }.get(embedders_model, None)

    if mute_file is None:
        create_mute_file(rvc_version, embedders_model, embedders_mode, config.is_half)
        mute_file = f"mute_{embedders_model}.npy"

    return os.path.join(mute_base_path, f"{rvc_version}_extracted", mute_file)

def generate_config(rvc_version, sample_rate, model_path):
    config_save_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_save_path): shutil.copy(os.path.join("main", "configs", rvc_version, f"{sample_rate}.json"), config_save_path)

def generate_filelist(pitch_guidance, model_path, rvc_version, sample_rate, embedders_mode = "fairseq", embedder_model = "hubert_base", rms_extract = False):
    gt_wavs_dir, feature_dir = os.path.join(model_path, "sliced_audios"), os.path.join(model_path, f"{rvc_version}_extracted")
    f0_dir, f0nsf_dir, energy_dir = None, None, None

    if pitch_guidance: f0_dir, f0nsf_dir = os.path.join(model_path, "f0"), os.path.join(model_path, "f0_voiced")
    if rms_extract: energy_dir = os.path.join(model_path, "energy")

    gt_wavs_files, feature_files = set(name.split(".")[0] for name in os.listdir(gt_wavs_dir)), set(name.split(".")[0] for name in os.listdir(feature_dir))
    names = gt_wavs_files & feature_files

    if pitch_guidance: names = names & set(name.split(".")[0] for name in os.listdir(f0_dir)) & set(name.split(".")[0] for name in os.listdir(f0nsf_dir))
    if rms_extract: names = names & set(name.split(".")[0] for name in os.listdir(energy_dir))
    
    options = []
    mute_base_path = os.path.join(configs["logs_path"], "mute")

    for name in names:
        option = {
            True: {
                True: f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{f0_dir}/{name}.wav.npy|{f0nsf_dir}/{name}.wav.npy|{energy_dir}/{name}.wav.npy|0",
                False: f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{f0_dir}/{name}.wav.npy|{f0nsf_dir}/{name}.wav.npy|0"
            },
            False: {
                True: f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{energy_dir}/{name}.wav.npy|0",
                False: f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|0"
            }
        }[pitch_guidance][rms_extract]

        options.append(option)

    mute_audio_path, mute_feature_path = os.path.join(mute_base_path, "sliced_audios", f"mute{sample_rate}.wav"), mute_file(embedders_mode, embedder_model, mute_base_path, rvc_version)
    
    for _ in range(2):
        option = {
            True: {
                True: f"{mute_audio_path}|{mute_feature_path}|{os.path.join(mute_base_path, 'f0', 'mute.wav.npy')}|{os.path.join(mute_base_path, 'f0_voiced', 'mute.wav.npy')}|{os.path.join(mute_base_path, 'energy', 'mute.wav.npy')}|0",
                False: f"{mute_audio_path}|{mute_feature_path}|{os.path.join(mute_base_path, 'f0', 'mute.wav.npy')}|{os.path.join(mute_base_path, 'f0_voiced', 'mute.wav.npy')}|0"
            },
            False: {
                True: f"{mute_audio_path}|{mute_feature_path}|{os.path.join(mute_base_path, 'energy', 'mute.wav.npy')}|0",
                False: f"{mute_audio_path}|{mute_feature_path}|0"
            }
        }[pitch_guidance][rms_extract]

        options.append(option)

    shuffle(options)
    with open(os.path.join(model_path, "filelist.txt"), "w") as f:
        f.write("\n".join(options))
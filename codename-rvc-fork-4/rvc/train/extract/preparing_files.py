import os
import shutil
from random import shuffle
from rvc.configs.config import Config
import json

config = Config()
current_directory = os.getcwd()


def generate_config(sample_rate: int, model_path: str, vocoder_arch: str):
    config_path = os.path.join("rvc", "configs", vocoder_arch, f"{sample_rate}.json")
    config_save_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_save_path):
        shutil.copyfile(config_path, config_save_path)
        print(f"Config saved at {config_save_path}")
    else:
        print(f"Config file already exists at {config_save_path}")

def generate_filelist(
    model_path: str, sample_rate: int, include_mutes: int = 2, embedder_model: str = "contentvec", vocoder_arch: str = "hifi_mrf_refine"
):
    gt_wavs_dir = os.path.join(model_path, "sliced_audios")
    feature_dir = os.path.join(model_path, f"extracted")

    f0_dir, f0nsf_dir = None, None
    f0_dir = os.path.join(model_path, "f0")
    f0nsf_dir = os.path.join(model_path, "f0_voiced")

    gt_wavs_files = set(name.split(".")[0] for name in os.listdir(gt_wavs_dir))
    feature_files = set(name.split(".")[0] for name in os.listdir(feature_dir))

    f0_files = set(name.split(".")[0] for name in os.listdir(f0_dir))
    f0nsf_files = set(name.split(".")[0] for name in os.listdir(f0nsf_dir))
    names = gt_wavs_files & feature_files & f0_files & f0nsf_files

    options = []

    if embedder_model == "contentvec":
        mute_folder = "mute"
    elif embedder_model == "spin_v1":
        mute_folder = "mute_spin_v1"
    else:
        mute_folder = "mute_spin_v2"

    mute_base_path = os.path.join(current_directory, "logs", mute_folder)

    sids = []
    
    vocoder_arch = vocoder_arch

    for name in names:
        sid = name.split("_")[0]
        if sid not in sids:
            sids.append(sid)
        options.append(
            f"{os.path.join(gt_wavs_dir, name)}.wav|{os.path.join(feature_dir, name)}.npy|{os.path.join(f0_dir, name)}.wav.npy|{os.path.join(f0nsf_dir, name)}.wav.npy|{sid}"
        )

    if include_mutes > 0:
        mute_audio_path = os.path.join(
            mute_base_path, "sliced_audios", f"mute{sample_rate}.wav"
        )
        mute_feature_path = os.path.join(
            mute_base_path, f"extracted", "mute.npy"
        )
        mute_f0_path = os.path.join(mute_base_path, "f0", "mute.wav.npy")
        mute_f0nsf_path = os.path.join(mute_base_path, "f0_voiced", "mute.wav.npy")

        # adding x files per sid
        for sid in sids * include_mutes:
            options.append(
                f"{mute_audio_path}|{mute_feature_path}|{mute_f0_path}|{mute_f0nsf_path}|{sid}"
            )

    file_path = os.path.join(model_path, "model_info.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    data["speakers_id"] = len(sids)
    data["vocoder_architecture"] = vocoder_arch

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    shuffle(options)


    with open(os.path.join(model_path, "filelist.txt"), "w") as f:
        f.write("\n".join(options))

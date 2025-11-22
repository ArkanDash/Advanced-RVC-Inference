import os
import sys
import glob
import torch

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from collections import OrderedDict

sys.path.append(os.getcwd())

from main.app.variables import config, translations

MATPLOTLIB_FLAG = False

def optimizer_device(optimizer, device="cpu"):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v): state[k] = v.to(device)

    return optimizer

def replace_keys_in_dict(d, old_key_part, new_key_part):
    updated_dict = OrderedDict() if isinstance(d, OrderedDict) else {}

    for key, value in d.items():
        updated_dict[(key.replace(old_key_part, new_key_part) if isinstance(key, str) else key)] = (replace_keys_in_dict(value, old_key_part, new_key_part) if isinstance(value, dict) else value)
    
    return updated_dict

def load_checkpoint(logger, checkpoint_path, model, optimizer=None, load_opt=1):
    assert os.path.isfile(checkpoint_path), translations["not_found_checkpoint"].format(checkpoint_path=checkpoint_path)

    checkpoint_dict = replace_keys_in_dict(replace_keys_in_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True), ".weight_v", ".parametrizations.weight.original1"), ".weight_g", ".parametrizations.weight.original0")
    new_state_dict = {k: checkpoint_dict["model"].get(k, v) for k, v in (model.module.state_dict() if hasattr(model, "module") else model.state_dict()).items()}
    model.module.load_state_dict(new_state_dict, strict=False) if hasattr(model, "module") else model.load_state_dict(new_state_dict, strict=False)

    if optimizer and load_opt == 1: optimizer.load_state_dict(checkpoint_dict.get("optimizer", {}))
    logger.debug(translations["save_checkpoint"].format(checkpoint_path=checkpoint_path, checkpoint_dict=checkpoint_dict['iteration']))

    return (model, optimizer, checkpoint_dict.get("learning_rate", 0), checkpoint_dict["iteration"], checkpoint_dict.get("scaler", {}))

def save_checkpoint(logger, model, optimizer, learning_rate, iteration, checkpoint_path, scaler):
    state_dict = (model.module.state_dict() if hasattr(model, "module") else model.state_dict())
    torch.save(
        replace_keys_in_dict(
            replace_keys_in_dict({
                "model": state_dict if not config.device.startswith("privateuseone") else {key: value.detach().cpu() for key, value in state_dict.items()}, 
                "iteration": iteration, 
                "optimizer": (optimizer if not config.device.startswith("privateuseone") else optimizer_device(optimizer)).state_dict(), 
                "learning_rate": learning_rate, 
                "scaler": scaler.state_dict()
            }, ".parametrizations.weight.original1", ".weight_v"), 
            ".parametrizations.weight.original0", ".weight_g"
        ), 
        checkpoint_path
    )

    if config.device.startswith("privateuseone"): optimizer_device(optimizer, config.device)
    logger.info(translations["save_model"].format(checkpoint_path=checkpoint_path, iteration=iteration))

def summarize(writer, global_step, scalars={}, histograms={}, images={}, audios={}, audio_sample_rate=22050):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)

    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)

    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")

    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sample_rate)

def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    checkpoints = sorted(glob.glob(os.path.join(dir_path, regex)), key=lambda f: int("".join(filter(str.isdigit, f))))
    return checkpoints[-1] if checkpoints else None

def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG

    if not MATPLOTLIB_FLAG:
        plt.switch_backend("Agg")
        MATPLOTLIB_FLAG = True

    fig, ax = plt.subplots(figsize=(10, 2))
    plt.colorbar(ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none"), ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()
    fig.canvas.draw()
    plt.close(fig)

    try:
        data = np.array(fig.canvas.renderer.buffer_rgba(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    except:
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="").reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data

def load_wav_to_torch(full_path):
    data, sample_rate = sf.read(full_path, dtype=np.float32)
    return torch.FloatTensor(data.astype(np.float32)), sample_rate

def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding="utf-8") as f:
        return [line.strip().split(split) for line in f]
    
class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = HParams(**v) if isinstance(v, dict) else v

    def keys(self):
        return self.__dict__.keys()
    
    def items(self):
        return self.__dict__.items()
    
    def values(self):
        return self.__dict__.values()
    
    def __len__(self):
        return len(self.__dict__)
    
    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __contains__(self, key):
        return key in self.__dict__
    
    def __repr__(self):
        return repr(self.__dict__)
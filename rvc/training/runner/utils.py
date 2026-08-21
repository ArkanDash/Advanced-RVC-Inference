import os
import sys
import glob
import torch
import torch.nn.functional as F

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from collections import OrderedDict


from arvc.utils.variables import config, translations
from arvc.engine.models.weight_norm import convert_old_to_new, convert_new_to_old

MATPLOTLIB_FLAG = False

def optimizer_state_dict_cpu(optimizer):
    import copy

    opt_state = copy.deepcopy(optimizer.state_dict())

    for state in opt_state["state"].values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.detach().cpu()

    return opt_state

def replace_keys_in_dict(d, old_key_part, new_key_part):
    updated_dict = OrderedDict() if isinstance(d, OrderedDict) else {}

    for key, value in d.items():
        updated_dict[(
            key.replace(old_key_part, new_key_part) if isinstance(key, str) else key
        )] = (
            replace_keys_in_dict(value, old_key_part, new_key_part) if isinstance(value, dict) else value
        )
    
    return updated_dict

def load_checkpoint(logger, checkpoint_path, model, optimizer=None, load_opt=1):
    assert os.path.isfile(checkpoint_path), translations["not_found_checkpoint"].format(checkpoint_path=checkpoint_path)

    from arvc.engine.models.safe_load import safe_torch_load
    checkpoint_dict = convert_old_to_new(
        safe_torch_load(checkpoint_path)
    )

    new_state_dict = {
        k: checkpoint_dict["model"].get(k, v) 
        for k, v in (model.module.state_dict() if hasattr(model, "module") else model.state_dict()).items()
    }

    model.module.load_state_dict(new_state_dict, strict=False) if hasattr(model, "module") else model.load_state_dict(new_state_dict, strict=False)
    if optimizer and load_opt == 1: optimizer.load_state_dict(checkpoint_dict.get("optimizer", {}))
    logger.debug(translations["save_checkpoint"].format(checkpoint_path=checkpoint_path, checkpoint_dict=checkpoint_dict['iteration']))

    return (
        model, 
        optimizer, 
        checkpoint_dict.get("learning_rate", 0), 
        checkpoint_dict["iteration"], 
        checkpoint_dict.get("scaler", {})
    )

def save_checkpoint(logger, model, optimizer, learning_rate, iteration, checkpoint_path, scaler):
    state_dict = (model.module.state_dict() if hasattr(model, "module") else model.state_dict())

    if config.device.startswith("privateuseone"):
        model_state = {k: v.detach().cpu() for k, v in state_dict.items()}
        model_optimizer = optimizer_state_dict_cpu(optimizer)
    else:
        model_state = state_dict
        model_optimizer = optimizer.state_dict()

    torch.save(
        convert_new_to_old({
            "model": model_state, 
            "iteration": iteration, 
            "optimizer": model_optimizer, 
            "learning_rate": learning_rate, 
            "scaler": scaler.state_dict()
        }), 
        checkpoint_path
    )

    logger.info(translations["save_model"].format(checkpoint_path=checkpoint_path, iteration=iteration))

def summarize(
    writer,
    global_step,
    scalars={},
    histograms={},
    images={},
    audios={},
    audio_sample_rate=22050
):
    for k, v in scalars.items():
        # BUG FIX: Detach tensors before logging to prevent GPU memory leak.
        # Without .detach(), the computation graph is retained, causing OOM over
        # many epochs as gradients accumulate.
        if torch.is_tensor(v):
            v = v.detach()
        writer.add_scalar(k, v, global_step)

    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)

    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")

    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sample_rate)

def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    checkpoints = sorted(
        glob.glob(
            os.path.join(dir_path, regex)
        ), 
        key=lambda f: int("".join(filter(str.isdigit, f)))
    )
    return checkpoints[-1] if checkpoints else None

def plot_spectrogram_to_numpy(spectrogram):
    """Render a mel spectrogram to a numpy RGB array for TensorBoard.

    BACKPORT (PolTrain): normalize to dB range [-10, 0] with a fixed colorbar
    so spectrograms are directly comparable across epochs. The original
    auto-scaling meant early-epoch (noisy) and late-epoch (clean) spectrograms
    looked similar because the color range rescaled each time.
    """
    global MATPLOTLIB_FLAG

    if not MATPLOTLIB_FLAG:
        plt.switch_backend("Agg")
        MATPLOTLIB_FLAG = True

    fig, ax = plt.subplots(figsize=(10, 2))
    # Convert to dB if the spectrogram looks linear (values >> 1.0)
    spec_db = spectrogram
    try:
        if np.max(spectrogram) > 1.0:
            spec_db = 10 * np.log10(np.maximum(spectrogram, 1e-10))
    except Exception:
        pass

    # Use fixed dB range so images are comparable across epochs
    im = ax.imshow(
        spec_db,
        aspect="auto",
        origin="lower",
        interpolation="none",
        cmap="viridis",
        norm=Normalize(vmin=-10, vmax=0),
    )
    plt.colorbar(im, ax=ax, format="%+2.0f dB")
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()
    fig.canvas.draw()
    plt.close(fig)

    try:
        data = np.array(
            fig.canvas.renderer.buffer_rgba(),
            dtype=np.uint8
        ).reshape(
            fig.canvas.get_width_height()[::-1] + (4,)
        )[:, :, :3]
    except Exception:
        # BUG FIX: np.fromstring is deprecated since NumPy 1.14 and removed in
        # NumPy 2.0+. Use np.frombuffer which is the modern, supported API.
        try:
            data = np.frombuffer(
                fig.canvas.tostring_rgb(),
                dtype=np.uint8
            ).reshape(
                fig.canvas.get_width_height()[::-1] + (3,)
            )
        except Exception:
            # Final fallback: use buffer_rgba with RGB conversion
            rgba = np.array(fig.canvas.renderer.buffer_rgba(), dtype=np.uint8)
            data = rgba[:, :, :3].reshape(
                fig.canvas.get_width_height()[::-1] + (3,)
            )

    return data


def mel_spectrogram_similarity(y_hat_mel, y_mel):
    """Compute a 0-100% similarity score between generated and real mel spectrograms.

    BACKPORT (PolTrain): converts the L1 mel loss into a human-interpretable
    similarity percentage. 100% = identical, 0% = completely different.
    Much easier to monitor in TensorBoard than the raw L1 loss value.

    Args:
        y_hat_mel: Generated mel spectrogram (predicted).
        y_mel: Real mel spectrogram (ground truth).

    Returns:
        torch.Tensor: Scalar similarity score in [0, 100].
    """
    device = y_hat_mel.device
    y_mel = y_mel.to(device)

    # Trim to matching length if shapes differ (e.g. due to rounding)
    if y_hat_mel.shape != y_mel.shape:
        trimmed_shape = tuple(min(a, b) for a, b in zip(y_hat_mel.shape, y_mel.shape))
        y_hat_mel = y_hat_mel[..., :trimmed_shape[-1]]
        y_mel = y_mel[..., :trimmed_shape[-1]]

    loss_mel = F.l1_loss(y_hat_mel, y_mel)
    similarity = 100.0 - (loss_mel * 100.0)
    return similarity.clamp(0.0, 100.0)

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

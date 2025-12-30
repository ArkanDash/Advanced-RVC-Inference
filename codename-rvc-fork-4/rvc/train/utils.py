import os
import glob
import json
import signal
import sys

import torch
from torch.nn import functional as F
import numpy as np
import soundfile as sf
from collections import OrderedDict
import matplotlib.pyplot as plt

MATPLOTLIB_FLAG = False

debug_save_load = False

from itertools import chain
from utils_cdnm import check_optimizer_coverage, verify_optimizer_has_all_params
from mel_processing import mel_spectrogram_torch

def replace_keys_in_dict(d, old_key_part, new_key_part):
    """
    Recursively replace parts of the keys in a dictionary.

    Args:
        d (dict or OrderedDict): The dictionary to update.
        old_key_part (str): The part of the key to replace.
        new_key_part (str): The new part of the key.
    """
    updated_dict = OrderedDict() if isinstance(d, OrderedDict) else {}
    for key, value in d.items():
        new_key = (
            key.replace(old_key_part, new_key_part) if isinstance(key, str) else key
        )
        updated_dict[new_key] = (
            replace_keys_in_dict(value, old_key_part, new_key_part)
            if isinstance(value, dict)
            else value
        )
    return updated_dict


def load_checkpoint(architecture, checkpoint_path, model, optimizer=None, load_opt=1):
    """
    Load a checkpoint into a model and optionally the optimizer.

    Args:
        architecture (str): Chosen architecture for training.
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load the checkpoint into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load the state from. Defaults to None.
        load_opt (int, optional): Whether to load the optimizer state. Defaults to 1.
    """
    assert os.path.isfile(checkpoint_path), f"Checkpoint file not found: {checkpoint_path}"

    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Backwards compatibility for mainline for "RVC" architecture
    if architecture == "RVC":
        checkpoint_dict = replace_keys_in_dict(
            checkpoint_dict, 
            ".weight_v", 
            ".parametrizations.weight.original1"
        )
        checkpoint_dict = replace_keys_in_dict(
            checkpoint_dict, 
            ".weight_g", 
            ".parametrizations.weight.original0"
        )

    # Safety check / fix for Future RingFormer and RefineGAN models that were saved before the better key handling was added:
    if architecture != "RVC":
        if any(key.endswith(".weight_v") for key in checkpoint_dict["model"].keys()) and any(key.endswith(".weight_g") for key in checkpoint_dict["model"].keys()):
            print(f"[WARNING] Detected {architecture} architecture, fixing keys by converting .weight_v and .weight_g back to .parametrizations.weight.original1 and .parametrizations.weight.original0.")
            checkpoint_dict = replace_keys_in_dict(
                checkpoint_dict, 
                ".weight_v", 
                ".parametrizations.weight.original1"
            )
            checkpoint_dict = replace_keys_in_dict(
                checkpoint_dict, 
                ".weight_g", 
                ".parametrizations.weight.original0"
            )

    model_state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    new_state_dict = {k: checkpoint_dict["model"].get(k, v) for k, v in model_state_dict.items()}

    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)

    if optimizer and load_opt == 1:
        opt_state = checkpoint_dict.get("optimizer", None)

        if opt_state is None:
            raise ValueError(f"[ERROR] Missing optimizer state in checkpoint: {checkpoint_path}")

        try:
            optimizer.load_state_dict(opt_state)
        except Exception as e:
            raise ValueError(f"[ERROR] Failed to load optimizer state from checkpoint: {checkpoint_path}. Reason: {e}")

        if not hasattr(optimizer, 'state_dict'):
            raise ValueError(f"ERROR: Optimizer does not have a valid state_dict method after loading. Check the optimizer setup.")

        print(f"Loaded optimizer state from checkpoint successfully.")

    print(f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint_dict['iteration']})")

    return (
        model,
        optimizer,
        checkpoint_dict.get("learning_rate", 0),
        checkpoint_dict["iteration"],
        checkpoint_dict.get("gradscaler", {})
    )


def save_checkpoint(architecture, model, optimizer, learning_rate, iteration, checkpoint_path, gradscaler=None):
    """
    Save the model and optimizer state to a checkpoint file.

    Args:
        architecture (str): Chosen architecture for training.
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save the state of.
        learning_rate (float): The current learning rate.
        iteration (int): The current iteration.
        checkpoint_path (str): The path to save the checkpoint to.
    """
    state_dict = (
        model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    )
    checkpoint_data = {
        "model": state_dict,
        "iteration": iteration,
        "optimizer": optimizer.state_dict(),
        "learning_rate": learning_rate,
    }

    # for fp16 trainings
    if gradscaler is not None:
            checkpoint_data["gradscaler"] = gradscaler.state_dict()

    # Backwards compatibility for mainline, for "RVC" architecture
    if architecture == "RVC":
        checkpoint_data = replace_keys_in_dict(
            checkpoint_data, ".parametrizations.weight.original1", ".weight_v"
        )
        checkpoint_data = replace_keys_in_dict(
            checkpoint_data, ".parametrizations.weight.original0", ".weight_g"
        )

    # Save the checkpoint data
    torch.save(checkpoint_data, checkpoint_path)
    print(f"Saved model '{checkpoint_path}' (epoch {iteration})")


def load_checkpoints(checkpoint_path, *models, optimizer=None, load_opt=1):
    """
    Load a checkpoint into multiple models and optionally a shared optimizer.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        *models (torch.nn.Module): List of models to load in the same order they were saved.
        optimizer (torch.optim.Optimizer, optional): Shared optimizer. Defaults to None.
        load_opt (int, optional): Whether to load the optimizer state. Defaults to 1.

    Returns:
        tuple: (models..., optimizer, learning_rate, iteration)
    """
    assert os.path.isfile(checkpoint_path), f"Checkpoint file not found: {checkpoint_path}"

    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Reverse key name remapping
    checkpoint_dict = replace_keys_in_dict(
        replace_keys_in_dict(checkpoint_dict, ".weight_v", ".parametrizations.weight.original1"),
        ".weight_g", ".parametrizations.weight.original0"
    )

    model_names = ["mpd", "cqt"]
    assert len(models) == len(model_names), f"Expected {len(model_names)} models, got {len(models)}"

    for model, name in zip(models, model_names):
        assert name in checkpoint_dict, f"Missing key '{name}' in checkpoint."
        model_state_dict = (
            model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        )
        new_state_dict = {
            k: checkpoint_dict[name].get(k, v) for k, v in model_state_dict.items()
        }
        if hasattr(model, "module"):
            res = model.module.load_state_dict(new_state_dict, strict=False)
        else:
            res = model.load_state_dict(new_state_dict, strict=False)

        if debug_save_load:
            print(f"[DEBUG][{name}] missing_keys: {res.missing_keys}")
            print(f"[DEBUG][{name}] unexpected_keys: {res.unexpected_keys}")


    if optimizer and load_opt == 1:
        opt_state = checkpoint_dict.get("optimizer", {})
        optimizer.load_state_dict(opt_state)

        if debug_save_load:
            check_optimizer_coverage([models[0], models[1]], optimizer)

            print("═══════════════ OPTIMIZER VERIFYING ═══════════════")
            print(f"[VERIF] Optimizer keys: {list(opt_state.keys())}")
            print(f"[VERIF] Param groups count: {len(opt_state.get('param_groups', []))}")
            print(f"[VERIF] State entries count: {len(opt_state.get('state', {}))}")
            print("═══════════════════════════════════════════════════")

            total_mpd = sum(p.numel() for p in models[0].parameters())
            total_cqt = sum(p.numel() for p in models[1].parameters())
            print(f"[VERIF] Param MPD: {total_mpd}, CQT: {total_cqt}")
            print(f"[VERIF] optim_d.state after  load: {len(optimizer.state)}")


    print(f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint_dict['iteration']})")

    return (*models, optimizer, checkpoint_dict.get("learning_rate", 0), checkpoint_dict["iteration"])


def save_checkpoints(models, optimizer, learning_rate, iteration, checkpoint_path):
    """
    Save multiple models and a shared optimizer state to a checkpoint file,
    after verifying optimizer coverage.
    """
    model_names = ["mpd", "cqt"]
    assert len(models) == len(model_names), "Expected exactly two models: mpd and cqt"

    if debug_save_load:
        verify_optimizer_has_all_params(models, optimizer, model_names)

    checkpoint_data = {
        "iteration":      iteration,
        "optimizer":      optimizer.state_dict(),
        "learning_rate":  learning_rate,
    }
    for model, name in zip(models, model_names):
        sd = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        checkpoint_data[name] = sd

    checkpoint_data = replace_keys_in_dict(
        replace_keys_in_dict(checkpoint_data,
                             ".parametrizations.weight.original1", ".weight_v"),
        ".parametrizations.weight.original0", ".weight_g",
    )

    torch.save(checkpoint_data, checkpoint_path)
    print(f"✅ Saved checkpoint '{checkpoint_path}' (epoch {iteration})")



def summarize(
    writer,
    global_step,
    scalars={},
    histograms={},
    images={},
    audios={},
    audio_sample_rate=22050,
):
    """
    Log various summaries to a TensorBoard writer.

    Args:
        writer (SummaryWriter): The TensorBoard writer.
        global_step (int): The current global step.
        scalars (dict, optional): Dictionary of scalar values to log.
        histograms (dict, optional): Dictionary of histogram values to log.
        images (dict, optional): Dictionary of image values to log.
        audios (dict, optional): Dictionary of audio values to log.
        audio_sample_rate (int, optional): Sampling rate of the audio data.
    """
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sample_rate)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    """
    Get the latest checkpoint file in a directory.

    Args:
        dir_path (str): The directory to search for checkpoints.
        regex (str, optional): The regular expression to match checkpoint files.
    """
    checkpoints = sorted(
        glob.glob(os.path.join(dir_path, regex)),
        key=lambda f: int("".join(filter(str.isdigit, f))),
    )
    return checkpoints[-1] if checkpoints else None


def plot_spectrogram_to_numpy(spectrogram):
    """
    Convert a spectrogram to a NumPy array for visualization.

    Args:
        spectrogram (numpy.ndarray): The spectrogram to plot.
    """
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        plt.switch_backend("Agg")
        MATPLOTLIB_FLAG = True

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data


def load_wav_to_torch(full_path):
    """
    Load a WAV file into a PyTorch tensor.

    Args:
        full_path (str): The path to the WAV file.
    """
    data, sample_rate = sf.read(full_path, dtype="float32")
    return torch.FloatTensor(data), sample_rate


def load_filepaths_and_text(filename, split="|"):
    """
    Load filepaths and associated text from a file.

    Args:
        filename (str): The path to the file.
        split (str, optional): The delimiter used to split the lines.
    """
    with open(filename, encoding="utf-8") as f:
        return [line.strip().split(split) for line in f]


class HParams:
    """
    A class for storing and accessing hyperparameters.
    """

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


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_config_from_json(config_save_path):
    try:
        with open(config_save_path, "r") as f:
            config = json.load(f)
        config = HParams(**config)
        return config
    except FileNotFoundError:
        print(
            f"Model config file not found at {config_save_path}. Did you run preprocessing and feature extraction steps?"
        )
        sys.exit(1)


def mel_spec_similarity(y_hat_mel, y_mel):
    device = y_hat_mel.device
    y_mel = y_mel.to(device)

    if y_hat_mel.shape != y_mel.shape:
        trimmed_shape = tuple(min(dim_a, dim_b) for dim_a, dim_b in zip(y_hat_mel.shape, y_mel.shape))
        y_hat_mel = y_hat_mel[..., :trimmed_shape[-1]]
        y_mel = y_mel[..., :trimmed_shape[-1]]
    
    loss_mel = F.l1_loss(y_hat_mel, y_mel)

    mel_spec_similarity = 100.0 - (loss_mel * 100.0)
    mel_spec_similarity = mel_spec_similarity.clamp(0.0, 100.0)

    return mel_spec_similarity


def flush_writer(writer, rank):
    if rank == 0 and writer is not None:
        writer.flush()


# Currently has no use, kept for future ig.
def flush_writer_grad(writer, rank, global_step):
    if rank == 0 and writer is not None and global_step % 10 == 0:
        writer.flush()


def block_tensorboard_flush_on_exit(writer):
    def handler(signum, frame):
        print("[Warning] Training interrupted. Skipping flush to avoid partial logs.")
        try:
            writer.close()
        except:
            pass
        os._exit(1)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def si_sdr(preds, target, eps=1e-8):
    """Scale-Invariant SDR"""
    preds = preds - preds.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    target_energy = (target ** 2).sum(dim=-1, keepdim=True)
    scaling_factor = (preds * target).sum(dim=-1, keepdim=True) / (target_energy + eps)
    projection = scaling_factor * target

    noise = preds - projection

    si_sdr_value = 10 * torch.log10((projection ** 2).sum(dim=-1) / (noise ** 2).sum(dim=-1) + eps)

    return si_sdr_value.mean()


def wave_to_mel(config, waveform, half):
    mel_spec = mel_spectrogram_torch(
        waveform.float().squeeze(1),
        config.data.filter_length,
        config.data.n_mel_channels,
        config.data.sample_rate,
        config.data.hop_length,
        config.data.win_length,
        config.data.mel_fmin,
        config.data.mel_fmax,
    )
    if half == torch.float16:
        mel_spec = mel_spec.half()
    elif half == torch.float32 or half == torch.bfloat16:
        pass

    return mel_spec


def small_model_naming(model_name, epoch, global_step):
    return f"{model_name}_{epoch}e_{global_step}s.pth"


def old_session_cleanup(now_dir, model_name):
    for root, dirs, files in os.walk(os.path.join(now_dir, "logs", model_name), topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            file_name, file_extension = os.path.splitext(name)
            if (
                file_extension == ".0"
                or (file_name.startswith("D_") and file_extension == ".pth")
                or (file_name.startswith("G_") and file_extension == ".pth")
                or (file_name.startswith("added") and file_extension == ".index")
            ):
                os.remove(file_path)
        for name in dirs:
            if name == "eval":
                folder_path = os.path.join(root, name)
                for item in os.listdir(folder_path):
                    item_path = os.path.join(folder_path, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                os.rmdir(folder_path)

    print("    ██████  Cleanup done!")


def verify_remap_checkpoint(checkpoint_path, model, architecture):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_state_dict = checkpoint["model"]
    print(f"[ ] Verifying checkpoint for selected architecture: {architecture}")

    strict_mode = True

    try:
        if architecture == "RVC":
            if hasattr(model, "module"):
                model.module.load_state_dict(checkpoint_state_dict, strict=strict_mode)
            else:
                model.load_state_dict(checkpoint_state_dict, strict=strict_mode)
        elif architecture != "RVC":
            print("[ ] Non-RVC architecture pretrains detected. Checking for old keys...")
            
            # Check for old keys and remap them if found
            if any(key.endswith(".weight_v") for key in checkpoint_state_dict.keys()) and \
               any(key.endswith(".weight_g") for key in checkpoint_state_dict.keys()):
                print("[ ] Old keys detected. Converting .weight_v and .weight_g to new format...")
                checkpoint_state_dict = replace_keys_in_dict(
                    checkpoint_state_dict,
                    ".weight_v",
                    ".parametrizations.weight.original1"
                )
                print("[ ] Remapping `.weight_v` keys completed.")
                checkpoint_state_dict = replace_keys_in_dict(
                    checkpoint_state_dict,
                    ".weight_g",
                    ".parametrizations.weight.original0"
                )
                print("[ ] Remapping `.weight_g` keys completed.")
            else:
                print("[ ] No old keys detected. Proceeding without remapping.")
            
            # Load the state dictionary after remapping
            if hasattr(model, "module"):
                model.module.load_state_dict(checkpoint_state_dict, strict=strict_mode)
            else:
                model.load_state_dict(checkpoint_state_dict, strict=strict_mode)

    except RuntimeError as e:
        error_message = str(e)
        if "size mismatch for" in error_message:
            print("\nError: A size mismatch was detected between the checkpoint and the model.")
            print("This usually means the model's architecture or sample rate is different from the checkpoint's.")
            print("Please check your model settings.")
            print("Detailed mismatch report:")
            for mismatch in error_message.split("\n"):
                if "size mismatch for" in mismatch:
                    print(f"  - {mismatch.strip()}")
        elif "Missing key(s) in state_dict:" in error_message or "Unexpected key(s) in state_dict:" in error_message:
            print("\nError: Key mismatch detected. The checkpoint's parameters do not match the model's.")
            print("This may be due to a different architecture or a corrupted checkpoint.")
            print("Missing or unexpected keys details:")
            print(error_message)
        else:
            print(f"An unknown runtime error occurred when loading the checkpoint.")
            print(f"Please check your model settings for compatibility with the checkpoint.")
            print(f"Original PyTorch error: {e}")
        
        sys.exit(1)
    else:
        del checkpoint
        del checkpoint_state_dict
        print("[ ] Checkpoint successfully verified and loaded.")


def print_init_setup(
    warmup_duration,
    rank,
    use_warmup,
    config,
    optimizer_choice,
    use_validation,
    lr_scheduler,
    exp_decay_gamma,
    use_kl_annealing,
    kl_annealing_cycle_duration,
):
    # Warmup init msg:
    if rank == 0:

        # Precision init msg:
        if not (config.train.bf16_run or config.train.fp16_run):
            if torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32:
                print("    ██████  PRECISION: TF32")
            else:
                print("    ██████  PRECISION: FP32")
        elif config.train.bf16_run:
            if torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32:
                print("    ██████  PRECISION: TF32 / BF16 - AMP")
            else:
                print("    ██████  PRECISION: FP32 / BF16 - AMP")
        elif config.train.fp16_run:
            if torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32:
                print("    ██████  PRECISION: TF32 / FP16 - AMP")
            else:
                print("    ██████  PRECISION: FP32 / FP16 - AMP")
        # cudnn backend checks:
        if torch.backends.cudnn.benchmark:
            print("    ██████  cudnn.benchmark: True")
        else:
            print("    ██████  cudnn.benchmark: False")

        if torch.backends.cudnn.deterministic:
            print("    ██████  cudnn.deterministic: True")
        else:
            print("    ██████  cudnn.deterministic: False")

        # Optimizer check:
        print(f"    ██████  Optimizer used: {optimizer_choice}")

        # Validation check:
        print(f"    ██████  Using Validation: {use_validation}")

        # LR scheduler check:
        if lr_scheduler == "exp decay":
            print(f"    ██████  lr scheduler: exponential lr decay with gamma of: {exp_decay_gamma}")
        elif lr_scheduler == "cosine annealing":
            print(f"    ██████  lr scheduler: cosine annealing")

        if use_warmup:
            print(f"    ██████  Warmup Enabled for: {warmup_duration} epochs.")

        if use_kl_annealing:
            print(f"    ██████  KL loss annealing enabled with cycle duration of: {kl_annealing_cycle_duration} epochs.")


def train_loader_safety(benchmark_mode, train_loader):
    if not benchmark_mode:
        if len(train_loader) < 3:
            print("Not enough data present in the training set. Perhaps you didn't slice the audio files? ( Preprocessing step )")
            os._exit(2333333)


def verify_spk_dim(
    config,
    model_info_path,
    experiment_dir,
    latest_checkpoint_path,
    rank,
    pretrainG
):
    embedder_name = "contentvec" # Default embedder
    spk_dim = config.model.spk_embed_dim  # 109 default speakers

    if rank == 0:
        try:
            with open(model_info_path, "r") as f:
                model_info = json.load(f)
                embedder_name = model_info["embedder_model"]
                spk_dim = model_info["speakers_id"]
        except Exception as e:
            print(f"Could not load model info file: {e}. Using defaults.")

        # Try to load speaker dim from latest checkpoint or pretrainG
        try:
            last_g = latest_checkpoint_path(experiment_dir, "G_*.pth")
            chk_path = (last_g if last_g else (pretrainG if pretrainG not in ("", "None") else None))
            if chk_path:
                ckpt = torch.load(chk_path, map_location="cpu", weights_only=True)
                spk_dim = ckpt["model"]["emb_g.weight"].shape[0]
                del ckpt

        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Using default number of speakers.")

        # update config before the model init
        print(f"    ██████  Initializing the generator with: {spk_dim} speakers.")

    return spk_dim

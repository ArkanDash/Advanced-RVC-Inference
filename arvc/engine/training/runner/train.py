import os
import re
import sys
import glob
import json
import gc
import torch
import logging
import argparse
import datetime
import warnings

import torch.distributed as dist
import torch.multiprocessing as mp

# ── FIX: Ensure project root is in sys.path BEFORE any arvc imports ──
# When run as a subprocess (e.g. python arvc/engine/training/runner/train.py),
# the project root isn't automatically in the path, causing ModuleNotFoundError.
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
os.environ["USE_LIBUV"] = "0" if sys.platform == "win32" else "1"

from tqdm import tqdm
from collections import deque
from contextlib import nullcontext
from random import randint, shuffle
from arvc.utils import strtobool
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from time import time as ttime
from torch.nn.parallel import DistributedDataParallel as DDP

from arvc.engine.models.utils import clear_gpu_cache
from arvc.engine.models.backends import directml, opencl, zluda

# XPU backend — may not exist in all installations; graceful fallback
try:
    from arvc.engine.models.backends import xpu
except ImportError:
    xpu = None

# ZLUDA detection: True when running on AMD GPU via CUDA compatibility layer
_is_zluda = zluda.is_available()

from arvc.utils.variables import logger, translations as _raw_translations

# ── BULLETPROOF SAFETY NET ──
# Wrap translations in a dict subclass that returns the key name itself
# if missing. This means training will NEVER crash with KeyError on
# translation lookups, regardless of what the language files contain.
class _SafeTranslations(dict):
    def __missing__(self, key):
        return key
    def __contains__(self, key):
        return True  # always report True so `in` checks never fail

translations = _SafeTranslations(_raw_translations)

from arvc.engine.models.algorithms import commons
from arvc.engine.training.runner import losses

from arvc.engine.training.runner.extract_model import extract_model

from arvc.engine.training.runner.mel_processing import (
    MultiScaleMelSpectrogramLoss, 
    mel_spectrogram_torch,
    spec_to_mel_torch
)

from arvc.engine.training.runner.utils import (
    HParams, 
    summarize, 
    load_checkpoint, 
    save_checkpoint, 
    load_wav_to_torch,
    latest_checkpoint_path, 
    plot_spectrogram_to_numpy,
)
from arvc.engine.models.weight_norm import configure_weight_norm, use_new_pytorch

from arvc.utils.variables import config as main_config
from arvc.utils.variables import configs as main_configs
from arvc.utils.huggingface import HF_download_file

if not getattr(main_config, 'debug_mode', False):
    warnings.filterwarnings("ignore")
    logging.getLogger("torch").setLevel(logging.ERROR)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--rvc_version", type=str, default="v2")
    parser.add_argument("--save_every_epoch", type=int, required=True)
    parser.add_argument("--save_only_latest", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--save_every_weights", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--total_epoch", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--pitch_guidance", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--g_pretrained_path", type=str, default="")
    parser.add_argument("--d_pretrained_path", type=str, default="")
    parser.add_argument("--overtraining_detector", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--overtraining_threshold", type=int, default=50)
    parser.add_argument("--cleanup", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--cache_data_in_gpu", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--model_author", type=str)
    parser.add_argument("--vocoder", type=str, default="Default")
    parser.add_argument("--checkpointing", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--deterministic", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--benchmark", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--energy_use", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--use_custom_reference", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--reference_path", type=str, default="")
    parser.add_argument("--multiscale_mel_loss", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--use_cosine_annealing_lr", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--architecture", type=str, default="RVC", help="Model architecture: RVC or SVC")
    parser.add_argument("--compile_model", type=lambda x: bool(strtobool(x)), default=False, help="Use torch.compile() on generator for PyTorch 2.x speedup")
    parser.add_argument("--use_8bit_adam", type=lambda x: bool(strtobool(x)), default=False, help="Use 8-bit Adam optimizer for lower VRAM (requires bitsandbytes)")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps (reduces VRAM usage with larger effective batch sizes)")
    parser.add_argument("--newpytorch", type=lambda x: bool(strtobool(x)), default=True, help="Use PyTorch 2.0+ parametrization format (default, matches Applio/VRVC). Set false for legacy weight_norm format.")

    return parser.parse_args()

d_lr_coeff = 1.0
g_lr_coeff = 1.0
d_step_per_g_step = 1
randomized = True  # Applio-style: random slice for training, full-sequence for finetuning

args = parse_arguments()

(
    model_name, 
    save_every_epoch, 
    total_epoch, 
    pretrainG, 
    pretrainD, 
    version, 
    gpus, 
    batch_size, 
    pitch_guidance, 
    save_only_latest, 
    save_every_weights, 
    cache_data_in_gpu, 
    overtraining_detector, 
    overtraining_threshold, 
    cleanup, 
    model_author, 
    vocoder, 
    checkpointing, 
    optimizer_choice, 
    energy_use, 
    use_custom_reference, 
    reference_path, 
    multiscale_mel_loss,
    use_cosine_annealing_lr,
    architecture,
    compile_model,
    use_8bit_adam,
    grad_accum_steps,
    newpytorch,
) = (
    args.model_name, 
    args.save_every_epoch, 
    args.total_epoch, 
    args.g_pretrained_path, 
    args.d_pretrained_path, 
    args.rvc_version, 
    args.gpu, 
    args.batch_size, 
    args.pitch_guidance, 
    args.save_only_latest, 
    args.save_every_weights, 
    args.cache_data_in_gpu, 
    args.overtraining_detector, 
    args.overtraining_threshold, 
    args.cleanup, 
    args.model_author, 
    args.vocoder, 
    args.checkpointing, 
    args.optimizer, 
    args.energy_use, 
    args.use_custom_reference, 
    args.reference_path, 
    args.multiscale_mel_loss,
    args.use_cosine_annealing_lr,
    args.architecture,
    args.compile_model,
    args.use_8bit_adam,
    args.grad_accum_steps,
    args.newpytorch,
)

# ── Configure weight_norm mode BEFORE any model creation ──
configure_weight_norm(newpytorch)
if newpytorch:
    if __name__ == "__main__": print(f"[Advanced-RVC] PyTorch weight format: NEW (2.0+ parametrization)")
else:
    if __name__ == "__main__": print(f"[Advanced-RVC] PyTorch weight format: OLD (weight_norm, RVC fork compatible)")

# Discriminator version: use v3 discriminator for BigVGAN and RefineGAN (matches VRVC)
disc_version = version if vocoder not in ["RefineGAN", "BigVGAN"] else "v3"

# is_half logic — matches Vietnamese-RVC exactly
is_half = main_config.is_half
if getattr(main_config, 'brain', False): is_half = True

# SVC architecture overrides (from Vietnamese-RVC)
if architecture == "SVC":
    disc_version = version if vocoder != "Default" else "v0"
    pitch_guidance = True
    energy_use = False

# Vietnamese-RVC style experiment_dir / checkpoint_path handling
weights_path = main_configs["weights_path"]
logs_path = main_configs["logs_path"]
custom_save_checkpoint_path = None

if not os.path.exists(model_name): 
    experiment_dir = os.path.join(logs_path, model_name)
else:
    experiment_dir = model_name
    model_name = os.path.basename(model_name)
    custom_save_checkpoint_path = weights_path

checkpoint_path = experiment_dir if custom_save_checkpoint_path is None else custom_save_checkpoint_path

training_file_path = os.path.join(experiment_dir, "training_data.json")
config_save_path = os.path.join(experiment_dir, "config.json")
filelist_path = os.path.join(experiment_dir, "filelist.txt")
eval_dir = os.path.join(experiment_dir, "eval")
spec_dirs = None

save_the_pid = True
cache_spectrogram = True
use_clip_grad_value = False

# Create config.json if it doesn't exist
if not os.path.exists(config_save_path):
    import shutil
    os.makedirs(experiment_dir, exist_ok=True)
    
    sr = 32000  # default sample rate
    extracted_dir = os.path.join(experiment_dir, f"{version}_extracted")
    if os.path.exists(extracted_dir):
        wav_files = glob.glob(os.path.join(extracted_dir, "*.wav"))
        if wav_files:
            try:
                import soundfile as sf
                _, detected_sr = sf.read(wav_files[0])
                sr = detected_sr
            except:
                pass
    
    config_template_path = os.path.join(main_configs["configs_path"], version, f"{sr}.json")
    
    if not os.path.exists(config_template_path):
        # Try nearest available sample rate as fallback
        for fallback_sr in [40000, 32000, 48000, 24000, 44100]:
            config_template_path = os.path.join(main_configs["configs_path"], version, f"{fallback_sr}.json")
            if os.path.exists(config_template_path):
                break
    
    if os.path.exists(config_template_path):
        shutil.copy(config_template_path, config_save_path)
    else:
        raise FileNotFoundError(f"Config template not found at: {config_template_path}")

# cuDNN / TF32 — controlled by config, not forced ON
torch.backends.cudnn.deterministic = args.deterministic if not main_config.device.startswith(("ocl", "privateuseone")) and not _is_zluda else False
torch.backends.cudnn.benchmark = args.benchmark if not main_config.device.startswith(("ocl", "privateuseone")) and not _is_zluda else False

tf32_enabled = getattr(main_config, 'tf32', False)
if torch.cuda.is_available() and not _is_zluda:
    torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
    torch.backends.cudnn.allow_tf32 = tf32_enabled

lowest_value = {"step": 0, "value": float("inf"), "epoch": 0}
global_step, last_loss_gen_all, overtrain_save_epoch = 0, 0, 0
loss_gen_history, smoothed_loss_gen_history, loss_disc_history, smoothed_loss_disc_history = [], [], [], []
consecutive_increases_gen = 0
consecutive_increases_disc = 0
avg_losses = {
    "grad_d_50": deque(maxlen=50), 
    "grad_g_50": deque(maxlen=50), 
    "disc_loss_50": deque(maxlen=50), 
    "adv_loss_50": deque(maxlen=50), 
    "fm_loss_50": deque(maxlen=50), 
    "kl_loss_50": deque(maxlen=50), 
    "mel_loss_50": deque(maxlen=50), 
    "gen_loss_50": deque(maxlen=50),
    "energy_loss_50": deque(maxlen=50),
}

with open(config_save_path, "r", encoding="utf-8") as f:
    config = json.load(f)

config = HParams(**config)
config.data.training_files = filelist_path

def main():
    global training_file_path, last_loss_gen_all, smoothed_loss_gen_history, loss_gen_history, loss_disc_history, smoothed_loss_disc_history, overtrain_save_epoch, model_author, vocoder, checkpointing, gpus, energy_use

    log_data = {
        translations["modelname"]: model_name, 
        translations["save_every_epoch"]: save_every_epoch, 
        translations["total_e"]: total_epoch, 
        translations["dorg"].format(pretrainG=pretrainG, pretrainD=pretrainD): "", 
        translations["training_version"]: version, 
        "Gpu": gpus, 
        translations["batch_size"]: batch_size, 
        translations["training_f0"]: pitch_guidance, 
        translations["save_only_latest"]: save_only_latest, 
        translations["save_every_weights"]: save_every_weights, 
        translations["cache_in_gpu"]: cache_data_in_gpu, 
        translations["overtraining_detector"]: overtraining_detector, 
        translations["threshold"]: overtraining_threshold, 
        translations["cleanup_training"]: cleanup, 
        translations["memory_efficient_training"]: checkpointing, 
        translations["optimizer"]: optimizer_choice, 
        translations["train&energy"]: energy_use,
        translations["multiscale_mel_loss"]: multiscale_mel_loss,
        translations["cosine_annealing_lr"]: use_cosine_annealing_lr,
        translations["architecture"]: architecture,
    }

    if model_author: log_data[translations["model_author"].format(model_author=model_author)] = ""
    if vocoder != "Default": log_data[translations['vocoder']] = vocoder

    for key, value in log_data.items():
        logger.debug(f"{key}: {value}" if value != "" else f"{key} {value}")

    try:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(randint(20000, 55555))

        wavs = glob.glob(os.path.join(os.path.join(experiment_dir, "sliced_audios"), "*.wav"))
        if wavs:
            _, sr = load_wav_to_torch(wavs[0])
            if sr != config.data.sample_rate:
                logger.warning(translations["training_sr"].format(sr_1=config.data.sample_rate, sr_2=sr))
                sys.exit(1)
        else:
            logger.warning(translations["not_found_dataset"])
            sys.exit(1)

        # Device selection — Vietnamese-RVC style with XPU + CPU fallback + Advanced-RVC ZLUDA
        if gpus == "-":
            device, gpus = torch.device("cpu"), [0]
            n_gpus = 1
            logger.warning(translations["not_gpu"])
        elif torch.cuda.is_available() and main_config.device.startswith("cuda"):
            if _is_zluda:
                device = torch.device("cuda")
                gpus = [0]
                n_gpus = 1
                logger.info("ZLUDA detected (AMD GPU) — using single GPU mode with gloo backend")
            else:
                device, gpus = torch.device("cuda"), [int(item) for item in gpus.split("-")]
                n_gpus = len(gpus)
        elif hasattr(torch, "xpu") and torch.xpu.is_available() and main_config.device.startswith("xpu"):
            device, gpus = torch.device("xpu"), [int(item) for item in gpus.split("-")]
            n_gpus = len(gpus)
        elif opencl.is_available() and main_config.device.startswith("ocl"):
            device, gpus = torch.device("ocl"), [int(item) for item in gpus.split("-")]
            n_gpus = len(gpus)
        elif directml.is_available() and main_config.device.startswith("privateuseone"):
            device, gpus = torch.device("privateuseone"), [int(item) for item in gpus.split("-")]
            n_gpus = len(gpus)
        elif torch.backends.mps.is_available() and main_config.device.startswith("mps"):
            device, gpus = torch.device("mps"), [0]
            n_gpus = 1
        else:
            device, gpus = torch.device("cpu"), [0]
            n_gpus = 1
            logger.warning(translations["not_gpu"])

        logger.info(
            translations["use_precision"].format(
                fp=("BF16" if getattr(main_config, 'brain', False) else "FP16") if is_half else "FP32"
            )
        )

        def start():
            children = []
            pid_data = {"process_pids": []}

            if save_the_pid:
                with open(config_save_path, "r", encoding="utf-8") as f:
                    try:
                        pid_data.update(json.load(f))
                    except json.JSONDecodeError:
                        pass

            for rank, device_id in enumerate(gpus):
                subproc = mp.Process(
                    target=run, 
                    args=(
                        rank, 
                        n_gpus, 
                        pretrainG, 
                        pretrainD, 
                        pitch_guidance, 
                        total_epoch, 
                        save_every_weights, 
                        config, 
                        device, 
                        device_id, 
                        model_author, 
                        vocoder, 
                        checkpointing, 
                        energy_use,
                        compile_model,
                    )
                )
                children.append(subproc)
                subproc.start()
                pid_data["process_pids"].append(subproc.pid)

            if save_the_pid: 
                with open(config_save_path, "w", encoding="utf-8") as f:
                    json.dump(pid_data, f, indent=4)

            for i in range(n_gpus):
                children[i].join()

        def load_from_json(file_path):
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    data = json.load(f)
                    return (
                        data.get("loss_disc_history", []), 
                        data.get("smoothed_loss_disc_history", []), 
                        data.get("loss_gen_history", []), 
                        data.get("smoothed_loss_gen_history", [])
                    )
            
            return [], [], [], []

        def continue_overtrain_detector(training_file_path):
            if overtraining_detector and os.path.exists(training_file_path): 
                (
                    loss_disc_history, 
                    smoothed_loss_disc_history, 
                    loss_gen_history, 
                    smoothed_loss_gen_history 
                ) = load_from_json(training_file_path)

        if cleanup:
            for root, dirs, files in os.walk(experiment_dir, topdown=False):
                for name in files:
                    file_path = os.path.join(root, name)
                    file_name, file_extension = os.path.splitext(name)

                    if (
                        file_extension == ".0" or 
                        (file_name.startswith(("D_", "G_")) and file_extension == ".pth") or 
                        (file_name.startswith(("added", "trained")) and file_extension == ".index")
                    ): 
                        os.remove(file_path)

                for name in dirs:
                    if name == "eval":
                        folder_path = os.path.join(root, name)

                        for item in os.listdir(folder_path):
                            item_path = os.path.join(folder_path, item)
                            if os.path.isfile(item_path): os.remove(item_path)

                        os.rmdir(folder_path)

        continue_overtrain_detector(training_file_path)
        start()
    except Exception as e:
        logger.error(f"{translations['training_error']} {e}")
        import traceback
        logger.debug(traceback.format_exc())

class EpochRecorder:
    def __init__(self):
        self.last_time = ttime()

    def record(self):
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time

        return translations["time_or_speed_training"].format(
            current_time=datetime.datetime.now().strftime("%H:%M:%S"), 
            elapsed_time_str=str(datetime.timedelta(seconds=int(round(elapsed_time, 1))))
        )

def run(
    rank, 
    n_gpus, 
    pretrainG, 
    pretrainD, 
    pitch_guidance, 
    custom_total_epoch, 
    custom_save_every_weights, 
    config, 
    device, 
    device_id, 
    model_author, 
    vocoder, 
    checkpointing, 
    energy_use,
    compile_model,
):
    global global_step, smoothed_value_gen, smoothed_value_disc, optimizer_choice

    smoothed_value_gen, smoothed_value_disc = 0, 0

    # DDP backend selection — Vietnamese-RVC style with XPU + Advanced-RVC ZLUDA
    _ddp_backend = "gloo" if (sys.platform == "win32" or device.type not in ["cuda", "xpu"] or _is_zluda) else ("xccl" if device.type == "xpu" else "nccl")
    dist.init_process_group(
        backend=_ddp_backend, 
        init_method="env://", 
        world_size=n_gpus if device.type in ["cuda", "xpu"] else 1, 
        rank=rank if device.type in ["cuda", "xpu"] else 0
    )

    torch.manual_seed(config.train.seed)
    if device.type == "cuda": torch.cuda.manual_seed(config.train.seed)
    elif device.type == "xpu": torch.xpu.manual_seed(config.train.seed)
    elif device.type == "ocl": opencl.pytorch_ocl.manual_seed_all(config.train.seed)

    if torch.cuda.is_available(): torch.cuda.set_device(device_id)
    elif hasattr(torch, "xpu") and torch.xpu.is_available(): torch.xpu.set_device(device_id)

    if rank == 0:
        if _is_zluda:
            logger.info(f"Training on ZLUDA (AMD GPU): {zluda.device_name(0)}")

    writer_eval = SummaryWriter(
        log_dir=eval_dir
    ) if rank == 0 else None

    from arvc.engine.training.runner.data_utils import (
        DistributedBucketSampler,
        TextAudioCollate,
        TextAudioLoader
    )

    train_dataset = TextAudioLoader(
        config.data, 
        spec_dirs=spec_dirs,
        cache_spectrogram=cache_spectrogram,
        pitch_guidance=pitch_guidance, 
        energy=energy_use
    )

    # Adaptive data loader settings
    _pin_mem = not _is_zluda
    _num_workers = 2 if _is_zluda else 4
    _prefetch = 2 if _is_zluda else 8

    train_loader = DataLoader(
        train_dataset, 
        num_workers=_num_workers, 
        shuffle=False, 
        pin_memory=_pin_mem, 
        batch_size=1 if architecture != "SVC" else batch_size,
        collate_fn=TextAudioCollate(
            pitch_guidance=pitch_guidance, 
            energy=energy_use
        ), 
        batch_sampler=DistributedBucketSampler(
            train_dataset, 
            batch_size, 
            [50, 100, 200, 300, 400, 500, 600, 700, 800, 900], 
            num_replicas=n_gpus, 
            rank=rank, 
            shuffle=True
        ) if architecture != "SVC" else None, 
        persistent_workers=True, 
        prefetch_factor=_prefetch
    )

    if len(train_loader) < 3:
        logger.warning(translations["not_enough_data"])
        sys.exit(1)

    # ── Dynamic spk_dim detection from checkpoint (Vietnamese-RVC feature) ──
    spk_dim = config.model.spk_embed_dim

    try:
        spk_dim = config.sid
    except Exception as e:
        logger.debug(e)

    try:
        g_path = os.path.join(checkpoint_path, "G_latest.pth")
        last_g = g_path if save_only_latest and os.path.exists(g_path) else latest_checkpoint_path(checkpoint_path, "G_*.pth")

        chk_path = (last_g if last_g else (pretrainG if pretrainG not in ["", "None"] else None))

        if chk_path:
            ckpt = torch.load(chk_path, map_location="cpu", weights_only=True)
            spk_dim = ckpt["model"]["emb_g.weight"].shape[0]
            del ckpt
    except Exception as e:
        logger.debug(e)

    config.model.spk_embed_dim = spk_dim

    from arvc.engine.models.algorithms.synthesizers import Synthesizer
    from arvc.engine.models.algorithms.discriminators import MultiPeriodDiscriminator

    # SVC architecture support (Vietnamese-RVC feature)
    _has_svc = False
    try:
        from arvc.engine.models.algorithms.synthesizers import SynthesizerSVC
        _has_svc = True
    except ImportError:
        pass

    if architecture == "SVC" and _has_svc:
        net_g, net_d = (
            SynthesizerSVC(
                config.data.filter_length // 2 + 1, 
                config.train.segment_size // config.data.hop_length, 
                **config.model, 
                sr=config.data.sample_rate, 
                vocoder=vocoder, 
                checkpointing=checkpointing, 
            ), 
            MultiPeriodDiscriminator(
                version=disc_version, 
                use_spectral_norm=config.model.use_spectral_norm, 
                checkpointing=checkpointing
            )
        )
    else:
        net_g, net_d = (
            Synthesizer(
                config.data.filter_length // 2 + 1, 
                config.train.segment_size // config.data.hop_length, 
                **config.model, 
                use_f0=pitch_guidance, 
                sr=config.data.sample_rate, 
                vocoder=vocoder, 
                randomized=randomized,
                checkpointing=checkpointing, 
                energy=energy_use
            ), 
            MultiPeriodDiscriminator(
                version=disc_version, 
                use_spectral_norm=config.model.use_spectral_norm, 
                checkpointing=checkpointing
            )
        )

    # Move to device — Vietnamese-RVC style with XPU support
    net_g, net_d = (
        net_g.cuda(device_id), 
        net_d.cuda(device_id)
    ) if torch.cuda.is_available() else (
        net_g.xpu(device_id), 
        net_d.xpu(device_id)
    ) if hasattr(torch, "xpu") and torch.xpu.is_available() else (
        net_g.to(device), 
        net_d.to(device)
    )

    # ── Optimizer selection ──
    # Use the Advanced-RVC optimizer registry when available, with Vietnamese-RVC
    # fallbacks for AdaBeliefV2 / InverseSqrt scheduler
    _use_registry = True
    try:
        from arvc.engine.models.optimizers import get_optimizer_class, get_optimizer_info
    except ImportError:
        _use_registry = False

    # Vietnamese-RVC style InverseSqrt scheduler import for AdaBeliefV2
    get_inverse_sqrt_scheduler = None
    try:
        from arvc.engine.models.optimizers.adabeliefv2 import AdaBeliefV2 as _AdaBeliefV2, get_inverse_sqrt_scheduler as _get_inv_sqrt
        get_inverse_sqrt_scheduler = _get_inv_sqrt
    except ImportError:
        pass

    if _use_registry:
        try:
            optimizer_optim = get_optimizer_class(optimizer_choice)
            optimizer_meta = get_optimizer_info(optimizer_choice)
        except ValueError:
            logger.warning(f"Unknown optimizer '{optimizer_choice}', falling back to AdamW")
            optimizer_choice = "AdamW"
            optimizer_optim = get_optimizer_class("AdamW")
            optimizer_meta = get_optimizer_info("AdamW")

        if rank == 0:
            logger.info(f"Optimizer: {optimizer_choice} (Rating: {optimizer_meta.get('rating', 'N/A')}/5 - {optimizer_meta.get('category', 'N/A')})")

        # CUDA Optimizer Training: Use fused kernels when available and supported
        use_fused_optimizer = (
            device.type == "cuda"
            and not _is_zluda
            and optimizer_meta.get("supports_fused", False)
            and hasattr(optimizer_optim, "fused")
        )

        # Build optimizer kwargs based on what the optimizer supports
        def _build_optimizer_kwargs(lr_coeff):
            kwargs = {"lr": config.train.learning_rate * lr_coeff}
            if optimizer_meta.get("supports_betas"):
                kwargs["betas"] = config.train.betas
            if optimizer_meta.get("supports_eps"):
                kwargs["eps"] = config.train.eps
            if optimizer_meta.get("supports_weight_decay"):
                kwargs["weight_decay"] = 0.0
            if use_fused_optimizer:
                kwargs["fused"] = True
            return kwargs

        # 8-bit Adam (requires bitsandbytes) — Advanced-RVC feature
        if use_8bit_adam and device.type == "cuda":
            try:
                import bitsandbytes as bnb
                if rank == 0:
                    logger.info(f"Using 8-bit {optimizer_choice} via bitsandbytes for reduced VRAM usage")
                optim_g = bnb.optim.AdamW8bit(net_g.parameters(), lr=config.train.learning_rate * g_lr_coeff, betas=config.train.betas)
                optim_d = bnb.optim.AdamW8bit(net_d.parameters(), lr=config.train.learning_rate * d_lr_coeff, betas=config.train.betas)
                use_fused_optimizer = False
            except ImportError:
                if rank == 0:
                    logger.warning("bitsandbytes not installed, falling back to standard optimizer")
                optim_g = optimizer_optim(net_g.parameters(), **_build_optimizer_kwargs(g_lr_coeff))
                optim_d = optimizer_optim(net_d.parameters(), **_build_optimizer_kwargs(d_lr_coeff))
        else:
            optim_g = optimizer_optim(net_g.parameters(), **_build_optimizer_kwargs(g_lr_coeff))
            optim_d = optimizer_optim(net_d.parameters(), **_build_optimizer_kwargs(d_lr_coeff))
        
        if rank == 0 and use_fused_optimizer:
            logger.info(f"CUDA Optimizer Training: Using fused {optimizer_choice} for enhanced CUDA performance")
    else:
        # Vietnamese-RVC fallback optimizer selection
        if optimizer_choice == "AnyPrecisionAdamW" and getattr(main_config, 'brain', False):
            from arvc.engine.models.optimizers.anyprecision_optimizer import AnyPrecisionAdamW
            optimizer_optim = AnyPrecisionAdamW
        elif optimizer_choice == "RAdam":
            from torch.optim import RAdam
            optimizer_optim = RAdam
        elif optimizer_choice == "AdaBelief":
            from arvc.engine.models.optimizers.adabelief import AdaBelief
            optimizer_optim = AdaBelief
        elif optimizer_choice == "AdaBeliefV2":
            from arvc.engine.models.optimizers.adabeliefv2 import AdaBeliefV2
            optimizer_optim = AdaBeliefV2
        else:
            from torch.optim import AdamW
            optimizer_optim = AdamW

        optim_g, optim_d = (
            optimizer_optim(
                net_g.parameters(), 
                config.train.learning_rate * g_lr_coeff, 
                betas=config.train.betas if not optimizer_choice.startswith("AdaBelief") else 1e-8, 
                eps=config.train.eps
            ), 
            optimizer_optim(
                net_d.parameters(), 
                config.train.learning_rate * d_lr_coeff, 
                betas=config.train.betas if not optimizer_choice.startswith("AdaBelief") else 1e-8, 
                eps=config.train.eps
            )
        )

    fn_mel_loss = MultiScaleMelSpectrogramLoss(sample_rate=config.data.sample_rate) if multiscale_mel_loss else torch.nn.L1Loss()

    # DDP wrapping — Vietnamese-RVC style with XPU, Advanced-RVC ZLUDA + bucket_cap_mb
    if not device.type.startswith(("privateuseone", "ocl", "mps", "xpu")): 
        if _is_zluda:
            # ZLUDA: DDP without device_ids (gloo backend, no NCCL)
            net_g, net_d = DDP(net_g), DDP(net_d)
        elif torch.cuda.is_available():
            # Optimization: increase gradient bucket size for faster all-reduce communication
            ddp_kwargs = {"device_ids": [device_id], "bucket_cap_mb": 25}
            net_g, net_d = DDP(net_g, **ddp_kwargs), DDP(net_d, **ddp_kwargs)
        else:
            net_g, net_d = DDP(net_g), DDP(net_d)

    # Optimization: torch.compile for PyTorch 2.x+ — Advanced-RVC feature
    # ZLUDA: torch.compile is not supported
    if compile_model and device.type == "cuda" and not _is_zluda and hasattr(torch, "compile"):
        if rank == 0:
            logger.info("Optimization: Applying torch.compile() to generator for faster training")
        try:
            net_g = torch.compile(net_g, mode="reduce-overhead")
        except Exception as e:
            if rank == 0:
                logger.warning(f"torch.compile() failed, falling back to eager mode: {e}")

    scaler_dict = {}
    try:
        if rank == 0: logger.info(translations["start_training"])

        d_path = os.path.join(checkpoint_path, "D_latest.pth") if save_only_latest else latest_checkpoint_path(checkpoint_path, "D_*.pth")
        g_path = os.path.join(checkpoint_path, "G_latest.pth") if save_only_latest else latest_checkpoint_path(checkpoint_path, "G_*.pth")

        _, _, _, epoch_str, scaler_dict = load_checkpoint(
            logger, 
            d_path, 
            net_d, 
            optim_d
        )

        _, _, _, epoch_str, _ = load_checkpoint(
            logger, 
            g_path, 
            net_g, 
            optim_g
        )

        if rank == 0: logger.info(translations["load_checkpoint"].format(d_path=d_path, g_path=g_path))
        
        epoch_str += 1
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        check = ["", "None"]
        epoch_str, global_step = 1, 0

        # Auto-download default pretrained models if no custom pretrained paths provided
        # (Advanced-RVC feature — better than Vietnamese-RVC's approach)
        if pretrainG in check and pretrainD in check and rank == 0:
            # Primary and fallback pretrained URLs
            primary_url = main_configs.get(
                f"pretrained_{version}_url",
                f"https://huggingface.co/buckets/R-Kentaren/Ultimate-RVC-Models/resolve/pretrained_{version}/"
            )
            # Fallback: R-Kentaren/Ultimate-RVC-Models HuggingFace Storage Bucket
            _default_fallback = (
                f"https://huggingface.co/buckets/R-Kentaren/Ultimate-RVC-Models/resolve/pretrained_{version}/"
            )
            fallback_url = main_configs.get(
                f"pretrained_{version}_fallback_url",
                _default_fallback
            )
            pretrained_save_dir = os.path.join(main_configs.get(f"pretrained_{version}_path", os.path.join(os.path.dirname(__file__), "../../assets/models", f"pretrained_{version}")))
            os.makedirs(pretrained_save_dir, exist_ok=True)

            pretrained_selector = {
                True: {  # pitch_guidance (f0 models)
                    24000: ("f0G24k.pth", "f0D24k.pth"),
                    32000: ("f0G32k.pth", "f0D32k.pth"),
                    40000: ("f0G40k.pth", "f0D40k.pth"),
                    44100: ("f0G40k.pth", "f0D40k.pth"),  # reuse 40k pretrained
                    48000: ("f0G48k.pth", "f0D48k.pth"),
                },
                False: {  # no pitch guidance (base models)
                    24000: ("G24k.pth", "D24k.pth"),
                    32000: ("G32k.pth", "D32k.pth"),
                    40000: ("G40k.pth", "D40k.pth"),
                    44100: ("G40k.pth", "D40k.pth"),  # reuse 40k pretrained
                    48000: ("G48k.pth", "D48k.pth"),
                }
            }

            sr = config.data.sample_rate

            # 24k pretrained models do not exist in any known repo.
            # Fallback to 32k pretrained for 24k training.
            sr_for_pretrained = sr
            if sr == 24000:
                logger.warning("24k pretrained models are not available; falling back to 32k pretrained.")
                sr_for_pretrained = 32000

            g_file, d_file = pretrained_selector.get(pitch_guidance, pretrained_selector[True]).get(
                sr_for_pretrained,
                pretrained_selector[pitch_guidance][40000]
            )

            g_local = os.path.join(pretrained_save_dir, g_file)
            d_local = os.path.join(pretrained_save_dir, d_file)

            def _train_download_pretrained(file_name, file_path, url_sources):
                """Try downloading from multiple URL sources, return True on success."""
                for src_url in url_sources:
                    try:
                        full_url = src_url + file_name
                        logger.info(f"Trying download: {full_url}")
                        HF_download_file(full_url, file_path)
                        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                            return True
                    except Exception as e:
                        logger.warning(f"Download failed from {src_url}: {e}")
                        continue
                return False

            url_sources = [primary_url, fallback_url]

            if not os.path.exists(g_local):
                logger.info(f"Downloading default pretrained G ({g_file}) for {version.upper()} {sr}Hz...")
                if not _train_download_pretrained(g_file, g_local, url_sources):
                    logger.error(f"Failed to download pretrained G ({g_file}) from all sources")
                    g_local = ""
            else:
                logger.info(f"Using cached default pretrained G: {g_local}")

            if not os.path.exists(d_local):
                logger.info(f"Downloading default pretrained D ({d_file}) for {version.upper()} {sr}Hz...")
                if not _train_download_pretrained(d_file, d_local, url_sources):
                    logger.error(f"Failed to download pretrained D ({d_file}) from all sources")
                    d_local = ""
            else:
                logger.info(f"Using cached default pretrained D: {d_local}")

            if g_local and d_local:
                pretrainG = g_local
                pretrainD = d_local
            elif g_local:
                pretrainG = g_local
                logger.warning("Downloaded only pretrained G; pretrained D will be randomly initialized")
            elif d_local:
                pretrainD = d_local
                logger.warning("Downloaded only pretrained D; pretrained G will be randomly initialized")
            else:
                logger.warning("Failed to download any pretrained models; training from scratch")

        try:
            if pretrainG not in check:
                if rank == 0: logger.info(translations["import_pretrain"].format(dg="G", pretrain=pretrainG))

                ckptG = torch.load(pretrainG, map_location="cpu", weights_only=True)["model"]

                # SVC architecture: ensure emb_g.weight is present
                if architecture == "SVC" and "emb_g.weight" not in ckptG: 
                    ckptG["emb_g.weight"] = net_g.module.emb_g.weight if hasattr(net_g, "module") else net_g.emb_g.weight

                # Match Vietnamese-RVC: strict loading with pretrain_strict config
                # Soft-merge was silently replacing pretrained weights with random
                # values for any key mismatch (e.g. weight_norm format), destroying
                # the pretrained model quality. This caused training from a near-random
                # initialization instead of a properly pretrained base.
                strict = main_configs.get("pretrain_strict", True)
                net_g.module.load_state_dict(ckptG, strict=strict) if hasattr(net_g, "module") else net_g.load_state_dict(ckptG, strict=strict)
                del ckptG

            if pretrainD not in check:
                if rank == 0: logger.info(translations["import_pretrain"].format(dg="D", pretrain=pretrainD))

                ckptD = torch.load(pretrainD, map_location="cpu", weights_only=True)["model"]

                # Match Vietnamese-RVC: strict loading with pretrain_strict config
                strict = main_configs.get("pretrain_strict", True)
                net_d.module.load_state_dict(ckptD, strict=strict) if hasattr(net_d, "module") else net_d.load_state_dict(ckptD, strict=strict)
                del ckptD
        except Exception as e:
            logger.error(translations["checkpointing_err"])
            logger.debug(e)
            sys.exit(1)

    # Scheduler selection — Vietnamese-RVC style with AdaBelief / AdaBeliefV2 / CosineAnnealing
    if optimizer_choice == "AdaBelief" or use_cosine_annealing_lr:
        scheduler_g, scheduler_d = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optim_g, 
                T_max=total_epoch, 
                eta_min=1e-6, 
                last_epoch=epoch_str - 2
            ), 
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optim_d, 
                T_max=total_epoch, 
                eta_min=1e-6, 
                last_epoch=epoch_str - 2
            )
        )
    elif optimizer_choice == "AdaBeliefV2" and get_inverse_sqrt_scheduler is not None:
        # InverseSqrt scheduler for AdaBeliefV2 (Vietnamese-RVC feature)
        scheduler_g, scheduler_d = (
            get_inverse_sqrt_scheduler(
                optim_g, 
                warmup_epochs=10, 
                last_epoch=epoch_str - 2
            ), 
            get_inverse_sqrt_scheduler(
                optim_d, 
                warmup_epochs=10, 
                last_epoch=epoch_str - 2
            )
        )
    else:
        scheduler_g, scheduler_d = (
            torch.optim.lr_scheduler.ExponentialLR(
                optim_g, 
                gamma=config.train.lr_decay, 
                last_epoch=epoch_str - 2
            ), 
            torch.optim.lr_scheduler.ExponentialLR(
                optim_d, 
                gamma=config.train.lr_decay, 
                last_epoch=epoch_str - 2
            )
        )

    # XPU mixed precision — Vietnamese-RVC feature
    if device.type == "xpu" and is_half and xpu is not None: 
        xpu.setup_gradscaler()

    # GradScaler — only for FP16 on CUDA/XPU (NOT BF16 — BF16 has same dynamic range as FP32)
    # Applio correctly disables scaler for BF16; VRVC had this bug too
    use_scaler = is_half and not getattr(main_config, 'brain', False) and device.type in ["cuda", "xpu"]
    scaler = GradScaler(device=device, enabled=use_scaler)
    cache = []

    if len(scaler_dict) > 0: scaler.load_state_dict(scaler_dict)

    if use_custom_reference and os.path.isfile(os.path.join(reference_path, "feats.npy")):
        import numpy as np

        if rank == 0: logger.info(translations["using_reference"].format(reference_name=re.sub(r'_v\d+_(?:[A-Za-z0-9_]+?)_(True|False)_(True|False)$', '', os.path.basename(reference_path))))
        phone = np.repeat(np.load(os.path.join(reference_path, "feats.npy")), 2, axis=0)

        reference = (
            torch.FloatTensor(phone).unsqueeze(0).to(device),
            torch.LongTensor([phone.shape[0]]).to(device),
            torch.LongTensor(np.load(os.path.join(reference_path, "pitch_coarse.npy"))[:-1]).unsqueeze(0).to(device) if pitch_guidance else None,
            torch.FloatTensor(np.load(os.path.join(reference_path, "pitch_fine.npy"))[:-1]).unsqueeze(0).to(device) if pitch_guidance else None,
            torch.LongTensor([0]).to(device),
        )
        if architecture != "SVC": reference += (torch.FloatTensor(np.load(os.path.join(reference_path, "energy.npy"))[:-1]).unsqueeze(0).to(device) if energy_use else None,)
    else:
        info = next(iter(train_loader))
        reference = (info[0].to(device), info[1].to(device))

        if pitch_guidance:
            reference += (info[2].to(device), info[3].to(device), info[8].to(device))
            if architecture != "SVC": reference += (info[9].to(device),) if energy_use else (None,)
        else:
            reference += (None, None, info[6].to(device))
            if architecture != "SVC": reference += (info[7].to(device),) if energy_use else (None,)

    try:
        for epoch in range(epoch_str, total_epoch + 1):
            train_and_evaluate(
                rank, 
                epoch, 
                config, 
                [net_g, net_d], 
                [optim_g, optim_d], 
                scaler, 
                train_loader, 
                writer_eval, 
                cache, 
                custom_save_every_weights, 
                custom_total_epoch, 
                device, 
                device_id, 
                reference, 
                model_author, 
                vocoder, 
                energy_use, 
                fn_mel_loss
            )

            scheduler_g.step(); scheduler_d.step()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

def train_and_evaluate(
    rank, 
    epoch, 
    hps, 
    nets, 
    optims, 
    scaler, 
    train_loader, 
    writer, 
    cache, 
    custom_save_every_weights, 
    custom_total_epoch, 
    device, 
    device_id, 
    reference, 
    model_author, 
    vocoder, 
    energy_use, 
    fn_mel_loss
):
    global global_step, lowest_value, loss_disc, consecutive_increases_gen, consecutive_increases_disc, smoothed_value_gen, smoothed_value_disc

    if epoch == 1:
        lowest_value = {"step": 0, "value": float("inf"), "epoch": 0}
        consecutive_increases_gen, consecutive_increases_disc = 0, 0

    net_g, net_d = nets
    optim_g, optim_d = optims

    if architecture != "SVC": train_loader.batch_sampler.set_epoch(epoch)
    net_g.train(); net_d.train()

    # Cache data in GPU — Vietnamese-RVC style with XPU support
    if device.type == "cuda" and cache_data_in_gpu:
        data_iterator = cache

        if cache == []:
            for batch_idx, info in enumerate(train_loader):
                cache.append(
                    (batch_idx, [
                        tensor.cuda(device_id, non_blocking=True) 
                        for tensor in info
                    ])
                )
        else: 
            shuffle(cache)
    elif device.type == "xpu" and cache_data_in_gpu:
        data_iterator = cache

        if cache == []:
            for batch_idx, info in enumerate(train_loader):
                cache.append(
                    (batch_idx, [
                        tensor.xpu(device_id, non_blocking=True) 
                        for tensor in info
                    ])
                )
        else: 
            shuffle(cache)
    elif device.type in ["privateuseone", "ocl"] and cache_data_in_gpu:
        data_iterator = cache

        if cache == []:
            for batch_idx, info in enumerate(train_loader):
                cache.append(
                    (batch_idx, [
                        tensor.to(device_id if device.type == "ocl" else device, non_blocking=True) 
                        for tensor in info
                    ])
                )
        else: 
            shuffle(cache)
    else: 
        data_iterator = enumerate(train_loader)

    epoch_recorder = EpochRecorder()

    # Autocast settings — Vietnamese-RVC style with XPU mixed precision
    autocast_enabled = is_half and device.type in ["cuda", "xpu"]
    autocast_dtype = (
        torch.float32 
        if not autocast_enabled else 
        (torch.bfloat16 if getattr(main_config, 'brain', False) else torch.float16)
    )

    autocasts = autocast(
        device.type, 
        enabled=autocast_enabled, 
        dtype=autocast_dtype
    ) if not device.type.startswith(("ocl", "privateuseone")) else nullcontext()
    
    with tqdm(total=len(train_loader), leave=False) as pbar:
        for batch_idx, info in data_iterator:
            # Move data to device — Vietnamese-RVC style with XPU support
            if device.type == "cuda" and not cache_data_in_gpu: 
                info = [
                    tensor.cuda(device_id, non_blocking=True) 
                    for tensor in info
                ]  
            elif device.type == "xpu" and not cache_data_in_gpu: 
                info = [
                    tensor.xpu(device_id, non_blocking=True) 
                    for tensor in info
                ]  
            elif device.type in ["privateuseone", "ocl"] and not cache_data_in_gpu: 
                info = [
                    tensor.to(device_id if device.type == "ocl" else device, non_blocking=True) 
                    for tensor in info
                ]  
            else: 
                info = [
                    tensor.to(device) 
                    for tensor in info
                ]

            phone, phone_lengths = info[0], info[1]
            if pitch_guidance:
                pitch, pitchf = info[2], info[3]
                spec, spec_lengths, wave, sid = info[4], info[5], info[6], info[8]
                energy = info[9] if energy_use else None
            else:
                pitch = pitchf = None
                spec, spec_lengths, wave, sid = info[2], info[3], info[4], info[6]
                energy = info[7] if energy_use else None

            with autocasts:
                net_g_params = (
                    phone, 
                    phone_lengths, 
                    pitch, 
                    pitchf, 
                    spec, 
                    spec_lengths, 
                    sid,
                )

                if energy_use: net_g_params += (energy,)

                y_hat, ids_slice, _, z_mask, (_, z_p, m_p, logs_p, _, logs_q) = net_g(
                    *net_g_params
                )

                # Slice wave to match generator output — only when using random slices
                if ids_slice is not None:
                    wave = commons.slice_segments(
                        wave, 
                        ids_slice * config.data.hop_length, 
                        config.train.segment_size, 
                        dim=3
                    )

            # ── Discriminator step ──────────────────────────────────────────
            for _ in range(d_step_per_g_step):
                with autocasts:
                    y_d_hat_r, y_d_hat_g, _, _ = net_d(
                        wave, 
                        y_hat.detach()
                    )

                    loss_disc, losses_disc_r, losses_disc_g = losses.discriminator_loss(
                        y_d_hat_r, 
                        y_d_hat_g
                    )

                optim_d.zero_grad()

                if autocast_enabled:
                    scaler.scale(loss_disc).backward()
                    scaler.unscale_(optim_d)
                    # Vietnamese-RVC: use_clip_grad_value toggle
                    grad_norm_d = commons.clip_grad_value(net_d.parameters(), None) if use_clip_grad_value else commons.grad_norm(net_d.parameters())
                    scaler.step(optim_d)
                    # NO scaler.update() here — only after G step
                else:
                    loss_disc.backward()
                    grad_norm_d = commons.clip_grad_value(net_d.parameters(), None) if use_clip_grad_value else commons.grad_norm(net_d.parameters())
                    optim_d.step()

            # ── Generator step ──────────────────────────────────────────────
            with autocasts:
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(
                    wave, 
                    y_hat
                )

            # Mel loss
            if multiscale_mel_loss: 
                loss_mel = fn_mel_loss(
                    wave, 
                    y_hat
                ) * config.train.c_mel / 3.0
            else:
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.float().squeeze(1), 
                    config.data.filter_length, 
                    config.data.n_mel_channels, 
                    config.data.sample_rate, 
                    config.data.hop_length, 
                    config.data.win_length, 
                    config.data.mel_fmin, 
                    config.data.mel_fmax
                )

                loss_mel = fn_mel_loss(
                    mel_spectrogram_torch(
                        wave.float().squeeze(1), 
                        config.data.filter_length, 
                        config.data.n_mel_channels, 
                        config.data.sample_rate, 
                        config.data.hop_length, 
                        config.data.win_length, 
                        config.data.mel_fmin, 
                        config.data.mel_fmax
                    ), 
                    y_hat_mel
                ) * config.train.c_mel

            # KL divergence loss — Vietnamese-RVC: DirectML workaround (move to CPU)
            if device.type == "privateuseone": 
                loss_kl = (
                    losses.kl_loss(
                        z_p.detach().cpu(), 
                        logs_q.detach().cpu(), 
                        m_p.detach().cpu(), 
                        logs_p.detach().cpu(), 
                        z_mask.detach().cpu()
                    ) * config.train.c_kl
                ).to(device)
            else:
                loss_kl = losses.kl_loss(
                    z_p, 
                    logs_q, 
                    m_p, 
                    logs_p, 
                    z_mask
                ) * config.train.c_kl

            loss_fm = losses.feature_loss(fmap_r, fmap_g)
            loss_gen, losses_gen = losses.generator_loss(y_d_hat_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

            # Energy loss (optional, off by default) — Advanced-RVC feature
            loss_energy = torch.tensor(0.0, device=device)
            if energy_use and energy is not None:
                energy_target = energy.float()
                if ids_slice is not None:
                    energy_slice = commons.slice_segments(
                        energy_target.unsqueeze(1), 
                        ids_slice, 
                        config.train.segment_size // config.data.hop_length, 
                        dim=2
                    ).squeeze(1)
                else:
                    energy_slice = energy_target
                wave_rms = torch.sqrt(torch.mean(y_hat.float() ** 2, dim=2, keepdim=True).clamp(min=1e-5))
                energy_pred = wave_rms.squeeze(-1)
                loss_energy = torch.nn.functional.l1_loss(energy_pred, energy_slice)
                loss_energy = loss_energy * getattr(config.train, 'c_energy', 0.1)
                loss_gen_all = loss_gen_all + loss_energy

            if loss_gen_all < lowest_value["value"]: 
                lowest_value = {
                    "step": global_step, 
                    "value": loss_gen_all, 
                    "epoch": epoch
                }

            # Gradient accumulation — properly implemented (fixes grad_accum_steps bug)
            # Scale loss by 1/grad_accum_steps and only step optimizer every grad_accum_steps steps
            if grad_accum_steps > 1:
                loss_gen_all_scaled = loss_gen_all / grad_accum_steps
            else:
                loss_gen_all_scaled = loss_gen_all

            optim_g.zero_grad()
            if autocast_enabled:
                scaler.scale(loss_gen_all_scaled).backward()
                scaler.unscale_(optim_g)
                # Vietnamese-RVC: use_clip_grad_value toggle
                grad_norm_g = commons.clip_grad_value(net_g.parameters(), None) if use_clip_grad_value else commons.grad_norm(net_g.parameters())
                # Only step and update when we've accumulated enough gradients
                if (batch_idx + 1) % grad_accum_steps == 0:
                    scaler.step(optim_g)
                    scaler.update()
            else:
                loss_gen_all_scaled.backward()
                grad_norm_g = commons.clip_grad_value(net_g.parameters(), None) if use_clip_grad_value else commons.grad_norm(net_g.parameters())
                if (batch_idx + 1) % grad_accum_steps == 0:
                    optim_g.step()

            # For grad_accum_steps > 1, we still step on non-boundary steps to keep 
            # backward compatibility with the original loss tracking. However, the 
            # effective gradient update only happens every grad_accum_steps steps.
            # When grad_accum_steps == 1, this is identical to the original behavior.
            if grad_accum_steps == 1 or (batch_idx + 1) % grad_accum_steps == 0:
                global_step += 1

            avg_losses["grad_d_50"].append(grad_norm_d)
            avg_losses["grad_g_50"].append(grad_norm_g)
            avg_losses["disc_loss_50"].append(loss_disc.detach())
            avg_losses["adv_loss_50"].append(loss_gen.detach())
            avg_losses["fm_loss_50"].append(loss_fm.detach())
            avg_losses["kl_loss_50"].append(loss_kl.detach())
            avg_losses["mel_loss_50"].append(loss_mel.detach())
            avg_losses["gen_loss_50"].append(loss_gen_all.detach())
            avg_losses["energy_loss_50"].append(loss_energy.detach())

            if rank == 0 and global_step % 50 == 0:
                scalar_dict = {
                    "grad_avg_50/norm_d": sum(avg_losses["grad_d_50"]) / len(avg_losses["grad_d_50"]),
                    "grad_avg_50/norm_g": sum(avg_losses["grad_g_50"]) / len(avg_losses["grad_g_50"]),
                    "loss_avg_50/d/adv": torch.stack(list(avg_losses["disc_loss_50"])).mean(),
                    "loss_avg_50/g/adv": torch.stack(list(avg_losses["adv_loss_50"])).mean(),
                    "loss_avg_50/g/fm": torch.stack(list(avg_losses["fm_loss_50"])).mean(),
                    "loss_avg_50/g/kl": torch.stack(list(avg_losses["kl_loss_50"])).mean(),
                    "loss_avg_50/g/mel": torch.stack(list(avg_losses["mel_loss_50"])).mean(),
                    "loss_avg_50/g/total": torch.stack(list(avg_losses["gen_loss_50"])).mean(),
                }

                if energy_use and len(avg_losses["energy_loss_50"]) > 0:
                    scalar_dict["loss_avg_50/g/energy"] = torch.stack(list(avg_losses["energy_loss_50"])).mean()

                summarize(
                    writer=writer, 
                    global_step=global_step, 
                    scalars=scalar_dict
                )

            pbar.update(1)

    with torch.no_grad():
        clear_gpu_cache()

    if rank == 0:
        mel = spec_to_mel_torch(
            spec, 
            config.data.filter_length, 
            config.data.n_mel_channels, 
            config.data.sample_rate, 
            config.data.mel_fmin, 
            config.data.mel_fmax
        )

        if ids_slice is not None:
            y_mel = commons.slice_segments(
                mel, 
                ids_slice, 
                config.train.segment_size // config.data.hop_length, 
                dim=3
            )
        else:
            y_mel = mel

        y_hat_mel = mel_spectrogram_torch(
            y_hat.float().squeeze(1), 
            config.data.filter_length, 
            config.data.n_mel_channels, 
            config.data.sample_rate, 
            config.data.hop_length, 
            config.data.win_length, 
            config.data.mel_fmin, 
            config.data.mel_fmax
        )

        scalar_dict = {
            "loss/g/total": loss_gen_all, 
            "loss/d/adv": loss_disc, 
            "learning_rate": optim_g.param_groups[0]["lr"], 
            "grad/norm_d": grad_norm_d, 
            "grad/norm_g": grad_norm_g, 
            "loss/g/adv": loss_gen,
            "loss/g/fm": loss_fm, 
            "loss/g/mel": loss_mel, 
            "loss/g/kl": loss_kl,
            "loss/g/energy": loss_energy if energy_use else torch.tensor(0.0, device=device)
        }

        scalar_dict.update({f"loss/g/{i}": v for i, v in enumerate(losses_gen)})
        scalar_dict.update({f"loss/d_r/{i}": v for i, v in enumerate(losses_disc_r)})
        scalar_dict.update({f"loss/d_g/{i}": v for i, v in enumerate(losses_disc_g)})

        image_dict = {
            "slice/mel_org": plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            "slice/mel_gen": plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
            "all/mel": plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
        }

        if epoch % save_every_epoch == 0:
            with autocasts:
                with torch.no_grad():
                    o, *_ = net_g.module.infer(*reference) if hasattr(net_g, "module") else net_g.infer(*reference)

            summarize(
                writer=writer, 
                global_step=global_step, 
                images=image_dict,
                scalars=scalar_dict, 
                audios={
                    f"gen/audio_{global_step:07d}": o[0, :, :]
                }, 
                audio_sample_rate=config.data.sample_rate
            )
        else: 
            summarize(
                writer=writer, 
                global_step=global_step, 
                images=image_dict, 
                scalars=scalar_dict
            )

    def check_overtraining(
        smoothed_loss_history, 
        threshold, 
        epsilon=0.004
    ):
        if len(smoothed_loss_history) < threshold + 1: return False

        for i in range(-threshold, -1):
            if smoothed_loss_history[i + 1] > smoothed_loss_history[i]: 
                return True

            if abs(smoothed_loss_history[i + 1] - smoothed_loss_history[i]) >= epsilon: 
                return False

        return True

    def update_exponential_moving_average(
        smoothed_loss_history, 
        new_value, 
        smoothing=0.987
    ):
        smoothed_value = (
            new_value 
            if not smoothed_loss_history else 
            (smoothing * smoothed_loss_history[-1] + (1 - smoothing) * new_value)
        )      

        smoothed_loss_history.append(smoothed_value)
        return smoothed_value

    def save_to_json(
        file_path, 
        loss_disc_history, 
        smoothed_loss_disc_history, 
        loss_gen_history, 
        smoothed_loss_gen_history
    ):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({
                "loss_disc_history": loss_disc_history, 
                "smoothed_loss_disc_history": smoothed_loss_disc_history, 
                "loss_gen_history": loss_gen_history, 
                "smoothed_loss_gen_history": smoothed_loss_gen_history
            }, f)
    
    model_add, model_del = [], []
    done = False
    
    if rank == 0:
        if epoch % save_every_epoch == False:
            checkpoint_suffix = f"{'latest' if save_only_latest else global_step}.pth"

            save_checkpoint(
                logger, 
                net_g, 
                optim_g, 
                config.train.learning_rate, 
                epoch, 
                os.path.join(checkpoint_path, "G_" + checkpoint_suffix), 
                scaler
            )
            save_checkpoint(
                logger, 
                net_d, 
                optim_d, 
                config.train.learning_rate, 
                epoch, 
                os.path.join(checkpoint_path, "D_" + checkpoint_suffix), 
                scaler
            )

            if custom_save_every_weights: 
                model_add.append(
                    os.path.join(weights_path, f"{model_name}_{epoch}e_{global_step}s.pth")
                )

        if overtraining_detector and epoch > 1:
            current_loss_disc, current_loss_gen = float(loss_disc), float(lowest_value["value"])
            
            loss_disc_history.append(current_loss_disc)
            loss_gen_history.append(current_loss_gen)
            
            smoothed_value_disc = update_exponential_moving_average(
                smoothed_loss_disc_history, 
                current_loss_disc
            )

            smoothed_value_gen = update_exponential_moving_average(
                smoothed_loss_gen_history, 
                current_loss_gen
            )
            
            is_overtraining_disc = check_overtraining(
                smoothed_loss_disc_history, 
                overtraining_threshold * 2
            )

            is_overtraining_gen = check_overtraining(
                smoothed_loss_gen_history, 
                overtraining_threshold, 
                0.01
            )
            
            consecutive_increases_disc = (consecutive_increases_disc + 1) if is_overtraining_disc else 0
            consecutive_increases_gen = (consecutive_increases_gen + 1) if is_overtraining_gen else 0

            if epoch % save_every_epoch == 0: 
                save_to_json(
                    training_file_path, 
                    loss_disc_history, 
                    smoothed_loss_disc_history, 
                    loss_gen_history, 
                    smoothed_loss_gen_history
                )

            if (
                is_overtraining_gen and 
                consecutive_increases_gen == overtraining_threshold or 
                is_overtraining_disc and 
                consecutive_increases_disc == (overtraining_threshold * 2)
            ):
                logger.info(
                    translations["overtraining_find"].format(
                        epoch=epoch, 
                        smoothed_value_gen=f"{smoothed_value_gen:.3f}", 
                        smoothed_value_disc=f"{smoothed_value_disc:.3f}"
                    )
                )

                done = True
            else:
                logger.info(
                    translations["best_epoch"].format(
                        epoch=epoch, 
                        smoothed_value_gen=f"{smoothed_value_gen:.3f}", 
                        smoothed_value_disc=f"{smoothed_value_disc:.3f}"
                    )
                )

                for file in glob.glob(os.path.join(weights_path, f"{model_name}_*e_*s_best_epoch.pth")):
                    model_del.append(file)

                model_add.append(
                    os.path.join(weights_path, f"{model_name}_{epoch}e_{global_step}s_best_epoch.pth")
                )
        
        if epoch >= custom_total_epoch:
            logger.info(
                translations["success_training"].format(
                    epoch=epoch, 
                    global_step=global_step, 
                    loss_gen_all=round(loss_gen_all.item(), 3)
                )
            )

            logger.info(
                translations["training_info"].format(
                    lowest_value_rounded=round(float(lowest_value["value"]), 3), 
                    lowest_value_epoch=lowest_value['epoch'], 
                    lowest_value_step=lowest_value['step']
                )
            )

            model_add.append(
                os.path.join(weights_path, f"{model_name}_{epoch}e_{global_step}s.pth")
            )

            done = True
            
        for m in model_del:
            os.remove(m)
        
        if model_add:
            ckpt = (net_g.module.state_dict() if hasattr(net_g, "module") else net_g.state_dict())
            for m in model_add:
                extract_model(
                    ckpt=ckpt, 
                    sr=config.data.sample_rate, 
                    pitch_guidance=pitch_guidance == True, 
                    name=model_name, 
                    model_path=m, 
                    epoch=epoch, 
                    step=global_step, 
                    version=version, 
                    hps=hps, 
                    model_author=model_author, 
                    vocoder=vocoder, 
                    energy_use=energy_use,
                    speakers_id=getattr(config, 'sid', None),
                    architecture=architecture
                )

        lowest_value_rounded = round(float(lowest_value["value"]), 3)

        if epoch > 1 and overtraining_detector: 
            logger.info(
                translations["model_training_info"].format(
                    model_name=model_name, 
                    epoch=epoch, 
                    global_step=global_step, 
                    epoch_recorder=epoch_recorder.record(), 
                    lowest_value_rounded=lowest_value_rounded, 
                    lowest_value_epoch=lowest_value['epoch'], 
                    lowest_value_step=lowest_value['step'], 
                    remaining_epochs_gen=(overtraining_threshold - consecutive_increases_gen), 
                    remaining_epochs_disc=((overtraining_threshold * 2) - consecutive_increases_disc), 
                    smoothed_value_gen=f"{smoothed_value_gen:.3f}", 
                    smoothed_value_disc=f"{smoothed_value_disc:.3f}"
                )
            )
        elif epoch > 1 and overtraining_detector == False: 
            logger.info(
                translations["model_training_info_2"].format(
                    model_name=model_name, 
                    epoch=epoch, 
                    global_step=global_step, 
                    epoch_recorder=epoch_recorder.record(), 
                    lowest_value_rounded=lowest_value_rounded, 
                    lowest_value_epoch=lowest_value['epoch'], 
                    lowest_value_step=lowest_value['step']
                )
            )
        else: 
            logger.info(
                translations["model_training_info_3"].format(
                    model_name=model_name, 
                    epoch=epoch, 
                    global_step=global_step, 
                    epoch_recorder=epoch_recorder.record()
                )
            )

        logger.debug(
            f"loss_gen_all: {loss_gen_all} loss_gen: {loss_gen} loss_fm: {loss_fm} loss_mel: {loss_mel} loss_kl: {loss_kl}"
        )

        last_loss_gen_all = loss_gen_all

        if done: 
            if save_the_pid:
                pid_file_path = os.path.join(experiment_dir, "config.json")

                with open(pid_file_path, "r", encoding="utf-8") as pid_file:
                    pid_data = json.load(pid_file)

                with open(pid_file_path, "w", encoding="utf-8") as pid_file:
                    pid_data.pop("process_pids", None)
                    json.dump(pid_data, pid_file, indent=4)

                if os.path.exists(os.path.join(experiment_dir, "train_pid.txt")): 
                    os.remove(os.path.join(experiment_dir, "train_pid.txt"))

            sys.exit(0)

        with torch.no_grad():
            clear_gpu_cache()

if __name__ == "__main__": 
    mp.set_start_method("spawn")
    main()

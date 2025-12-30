import os
import datetime
import glob
import itertools
import json
import math
import re
import subprocess
import sys

pid_data = {"process_pids": []}
os.environ["USE_LIBUV"] = "0" if sys.platform == "win32" else "1"

from typing import Tuple, Optional
from collections import deque
from distutils.util import strtobool
from random import randint, shuffle
from time import time as ttime, sleep

import numpy as np
import psutil
from tqdm import tqdm
from pesq import pesq

import torch
import torch.nn as nn
import torchaudio
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
import torch.multiprocessing as mp
import auraloss

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

from utils import (
    HParams,
    plot_spectrogram_to_numpy,
    summarize,
    load_checkpoint,
    save_checkpoint,
    latest_checkpoint_path,
    load_wav_to_torch,
    load_config_from_json,
    mel_spec_similarity,
    flush_writer,
    block_tensorboard_flush_on_exit,
    si_sdr,
    wave_to_mel,
    small_model_naming,
    old_session_cleanup,
    verify_remap_checkpoint,
    print_init_setup,
    train_loader_safety,
    verify_spk_dim,
)
from losses import (
    discriminator_loss,
    generator_loss,
    discriminator_loss_v2,
    generator_loss_v2,
    HingeAdversarialLoss,
    feature_loss,
    kl_loss,
    kl_loss_clamped,
    phase_loss,
    envelope_loss,
)
from mel_processing import (
    spec_to_mel_torch,
    MultiScaleMelSpectrogramLoss,
)
from rvc.train.process.extract_model import extract_model
from rvc.lib.algorithm import commons
from rvc.train.utils import replace_keys_in_dict

# Parse command line arguments start region ===========================

model_name = sys.argv[1]
epoch_save_frequency = int(sys.argv[2])
total_epoch_count = int(sys.argv[3])
pretrainG = sys.argv[4]
pretrainD = sys.argv[5]
gpus = sys.argv[6]
batch_size = int(sys.argv[7])
sample_rate = int(sys.argv[8])
save_only_latest_net_models = strtobool(sys.argv[9])
save_weight_models = strtobool(sys.argv[10])
cache_data_in_gpu = strtobool(sys.argv[11])
use_warmup = strtobool(sys.argv[12])
warmup_duration = int(sys.argv[13])
cleanup = strtobool(sys.argv[14])
vocoder = sys.argv[15]
architecture = sys.argv[16]
optimizer_choice = sys.argv[17]
use_checkpointing = strtobool(sys.argv[18])
use_tf32 = bool(strtobool(sys.argv[19]))
use_benchmark = bool(strtobool(sys.argv[20]))
use_deterministic = bool(strtobool(sys.argv[21]))
spectral_loss = sys.argv[22]
lr_scheduler = sys.argv[23]
exp_decay_gamma = float(sys.argv[24])
use_validation = strtobool(sys.argv[25])
use_kl_annealing = strtobool(sys.argv[26])
kl_annealing_cycle_duration = int(sys.argv[27])
vits2_mode = strtobool(sys.argv[28])
use_custom_lr = strtobool(sys.argv[29])
custom_lr_g, custom_lr_d = (float(sys.argv[30]), float(sys.argv[31])) if use_custom_lr else (None, None)
assert not use_custom_lr or (custom_lr_g and custom_lr_d), "Invalid custom LR values."

# Parse command line arguments end region ===========================

current_dir = os.getcwd()
experiment_dir = os.path.join(current_dir, "logs", model_name)
config_save_path = os.path.join(experiment_dir, "config.json")
dataset_path = os.path.join(experiment_dir, "sliced_audios")
model_info_path = os.path.join(experiment_dir, "model_info.json")


# Load the config from json
config = load_config_from_json(config_save_path)
config.data.training_files = os.path.join(experiment_dir, "filelist.txt")


# AMP precision / dtype init
if config.train.bf16_run:
    train_dtype = torch.bfloat16
elif config.train.fp16_run: 
    train_dtype = torch.float16
else:
    train_dtype = torch.float32


# Globals ( do not touch these. )
global_step = 0
warmup_completed = False
from_scratch = False
use_lr_scheduler = lr_scheduler != "none"


# Torch backends config
torch.backends.cuda.matmul.allow_tf32 = use_tf32
torch.backends.cudnn.allow_tf32 = use_tf32
torch.backends.cudnn.benchmark = use_benchmark
torch.backends.cudnn.deterministic = use_deterministic


# Globals ( tweakable )
randomized = True
benchmark_mode = False
enable_persistent_workers = True
debug_shapes = False

# EXPERIMENTAL
c_stft = 21.0 # Seems close enough to multi-scale mel loss's magnitude, but needs more testing.
adversarial_loss = "lsgan" # Supported adv losses: "lsgan" ( default rvc / vits ), "tprls" or "hinge"

pretrain_preview = True
override_pretrain_lr = False
new_pretrain_lr = 9e-5

##################################################################

import logging
logging.getLogger("torch").setLevel(logging.ERROR)


def eval_infer(net_g, reference):
    net_g.eval()
    with torch.no_grad():
        if hasattr(net_g, "module"):
            o, *_ = net_g.module.infer(*reference)
        else:
            o, *_ = net_g.infer(*reference)
    net_g.train()
    return o

class EpochRecorder:
    """
    Records the time elapsed per epoch.
    """

    def __init__(self):
        self.last_time = ttime()

    def record(self):
        """
        Records the elapsed time and returns a formatted string.
        """
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time = round(elapsed_time, 1)
        elapsed_time_str = str(datetime.timedelta(seconds=int(elapsed_time)))
        current_time = datetime.datetime.now().strftime("%H:%M:%S")

        return f"Current time: {current_time} | Time per epoch: {elapsed_time_str}"

def setup_env_and_distr(rank, n_gpus, device, device_id, config):
    if rank == 0:
        writer_eval = SummaryWriter(
            log_dir=os.path.join(experiment_dir, "eval"),
            flush_secs=86400 # Periodic background flush's timer workarouand.
        )
        block_tensorboard_flush_on_exit(writer_eval)
    else:
        writer_eval = None

    dist.init_process_group(
        backend="gloo" if sys.platform == "win32" or device.type != "cuda" else "nccl",
        init_method="env://",
        world_size=n_gpus if device.type == "cuda" else 1,
        rank=rank if device.type == "cuda" else 0,
    )

    torch.manual_seed(config.train.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)

    return writer_eval

def prepare_dataloaders(config, n_gpus, rank, batch_size, use_validation, benchmark_mode):
    from data_utils import (
        DistributedBucketSampler,
        TextAudioCollateMultiNSFsid,
        TextAudioLoaderMultiNSFsid
    )

    if not benchmark_mode and use_validation:
        full_dataset = TextAudioLoaderMultiNSFsid(config.data)
        train_len = int(0.90 * len(full_dataset))
        val_len = len(full_dataset) - train_len
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(config.train.seed)
        )
        train_dataset.lengths = [full_dataset.lengths[i] for i in train_dataset.indices]
        val_dataset.lengths = [full_dataset.lengths[i] for i in val_dataset.indices]
    else:
        train_dataset = TextAudioLoaderMultiNSFsid(config.data)
        val_dataset = None

    train_sampler = DistributedBucketSampler(
        train_dataset,
        batch_size * n_gpus,
        [50, 100, 200, 300, 400, 500, 600, 700, 800, 900],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True
    )

    collate_fn = TextAudioCollateMultiNSFsid()
    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=enable_persistent_workers,
        prefetch_factor=8
    )
    val_loader = None
    if val_dataset:
        val_sampler = DistributedBucketSampler(
            val_dataset,
            batch_size * n_gpus,
            [50, 100, 200, 300, 400, 500, 600, 700, 800, 900],
            num_replicas=n_gpus,
            rank=rank,
            shuffle=False
        )
        val_loader = DataLoader(
            val_dataset, batch_sampler=val_sampler, shuffle=False, collate_fn=collate_fn,
            num_workers=1, pin_memory=True
        )
    
    train_loader_safety(benchmark_mode, train_loader)

    return train_loader, val_loader

def get_g_model(config, sample_rate, vocoder, use_checkpointing, randomized):
    from rvc.lib.algorithm.synthesizers import Synthesizer
    return Synthesizer(
        config.data.filter_length // 2 + 1,
        config.train.segment_size // config.data.hop_length,
        **config.model,
        use_f0 = True,
        sr = sample_rate,
        vocoder = vocoder,
        checkpointing = use_checkpointing,
        randomized = randomized,
        vits2_mode = vits2_mode,
    )

def get_d_model(config, vocoder, use_checkpointing):
    if vocoder in ["RingFormer_v1", "RingFormer_v2"]:
        from rvc.lib.algorithm.discriminators.multi import MPD_MSD_MRD_Combined
        # MPD + MSD + MRD ( unified ) - RingFormer architecture v1 and v2
        return MPD_MSD_MRD_Combined(
            config.model.use_spectral_norm,
            use_checkpointing=use_checkpointing,
            **dict(config.mrd)
        )
    elif vocoder == "PCPH-GAN": # Potentially changed in future - Trial
        from rvc.lib.algorithm.discriminators.multi import MPD_MSD_MRD_Combined
        return MPD_MSD_MRD_Combined(
            config.model.use_spectral_norm,
            use_checkpointing=use_checkpointing,
            **dict(config.mrd)
        )
    elif vocoder == "RefineGAN":
        from rvc.lib.algorithm.discriminators.multi import MPD_MSD_MRD_Combined_RefineGan
        # Trimmed MPD + MSD + MRD ( unified )
        return MPD_MSD_MRD_Combined_RefineGan(
            config.model.use_spectral_norm,
            use_checkpointing=use_checkpointing
        )
    else: # For HiFi-GAN or MRF-HiFi-GAN
        from rvc.lib.algorithm.discriminators.multi import MPD_MSD_Combined
        # MPD + MSD ( unified ) - Original RVC Setup
        return MPD_MSD_Combined(
            config.model.use_spectral_norm,
            use_checkpointing=use_checkpointing
        )

def get_optimizers(
    net_g,
    net_d,
    config,
    optimizer_choice,
    custom_lr_g,
    custom_lr_d,
    use_custom_lr,
    total_epoch_count,
    train_loader
):
    # Common args for optims
    common_args_g = dict(
        lr=custom_lr_g if use_custom_lr else config.train.learning_rate,
        betas=(0.8, 0.99),
        eps=1e-9,
        weight_decay=0.0,
    )
    common_args_d = dict(
        lr=custom_lr_d if use_custom_lr else config.train.learning_rate,
        betas=(0.8, 0.99),
        eps=1e-9,
        weight_decay=0.0,
    )
    adamwspd_args_g = dict(
        lr=custom_lr_g if use_custom_lr else config.train.learning_rate,
        betas=(0.8, 0.99),
        eps=1e-9,
        weight_decay=0.5,
    )
    adamwspd_args_d = dict(
        lr=custom_lr_d if use_custom_lr else config.train.learning_rate,
        betas=(0.8, 0.99),
        eps=1e-9,
        weight_decay=0.5,
    )

    # For exotic optimizers
    ranger_args = dict(
        num_epochs=total_epoch_count,
        num_batches_per_epoch=len(train_loader),
        use_madgrad=False,
        use_warmup=False,
        warmdown_active=False,
        use_cheb=False,
        lookahead_active=True,
        normloss_active=False,
        normloss_factor=1e-4,
        softplus=False,
        use_adaptive_gradient_clipping=True,
        agc_clipping_value=0.01,
        agc_eps=1e-3,
        using_gc=True,
        gc_conv_only=True,
        using_normgc=False,
    )

    if optimizer_choice == "AdamW":
        optim_g = torch.optim.AdamW(filter(lambda p: p.requires_grad, net_g.parameters()), **common_args_g, fused=True)
        optim_d = torch.optim.AdamW(filter(lambda p: p.requires_grad, net_d.parameters()), **common_args_d, fused=True)

    elif optimizer_choice == "AdamW BF16":
        from optimi import AdamW as AdamW_BF16
        optim_g = AdamW_BF16(filter(lambda p: p.requires_grad, net_g.parameters()), **common_args_g, kahan_sum=True, foreach=True)
        optim_d = AdamW_BF16(filter(lambda p: p.requires_grad, net_d.parameters()), **common_args_d, kahan_sum=True, foreach=True)

    elif optimizer_choice == "RAdam":
        optim_g = torch.optim.RAdam(filter(lambda p: p.requires_grad, net_g.parameters()), **common_args_g)
        optim_d = torch.optim.RAdam(filter(lambda p: p.requires_grad, net_d.parameters()), **common_args_d)

    elif optimizer_choice == "DiffGrad":
        from rvc.train.custom_optimizers.diffgrad import diffgrad
        optim_g = diffgrad(filter(lambda p: p.requires_grad, net_g.parameters()), **common_args_g)
        optim_d = diffgrad(filter(lambda p: p.requires_grad, net_d.parameters()), **common_args_d)

    elif optimizer_choice == "Prodigy":
        from rvc.train.custom_optimizers.prodigy import Prodigy
        optim_g = Prodigy(filter(lambda p: p.requires_grad, net_g.parameters()), lr=custom_lr_g if use_custom_lr else 1.0, betas=(0.8, 0.99), weight_decay=0.0, decouple=True)
        optim_d = Prodigy(filter(lambda p: p.requires_grad, net_d.parameters()), lr=custom_lr_d if use_custom_lr else 1.0, betas=(0.8, 0.99), weight_decay=0.0, decouple=True)

    elif optimizer_choice == "Ranger21":
        from rvc.train.custom_optimizers.ranger21 import Ranger21
        optim_g = Ranger21(filter(lambda p: p.requires_grad, net_g.parameters()), **common_args_g, **ranger_args)
        optim_d = Ranger21(filter(lambda p: p.requires_grad, net_d.parameters()), **common_args_d, **ranger_args)

    elif optimizer_choice == "AdamSPD":
        import copy
        from rvc.train.custom_optimizers.adamspd import AdamSPD

        # Get trainable parameters and cache the pre-trained weights
        params_to_opt_g = [p for p in net_g.parameters() if p.requires_grad]
        params_anchor_g = copy.deepcopy(params_to_opt_g) 
        # Parameter group with the anchor
        param_group_g = [{'params': params_to_opt_g, 'pre': params_anchor_g}]

        # Get trainable parameters and cache the pre-trained weights
        params_to_opt_d = [p for p in net_d.parameters() if p.requires_grad]
        params_anchor_d = copy.deepcopy(params_to_opt_d) 
        # Parameter group with the anchor
        param_group_d = [{'params': params_to_opt_d, 'pre': params_anchor_d}]

        optim_g = AdamSPD(param_group_g, **adamwspd_args_g)
        optim_d = AdamSPD(param_group_d, **adamwspd_args_d,)

        proj_strength_mult_g = adamwspd_args_g['weight_decay']
        proj_strength_mult_d = adamwspd_args_d['weight_decay']
        print(f"    ██████  Proj. Strength Mult. for AdamSPD: G; {proj_strength_mult_g}, D; {proj_strength_mult_d}")
    else:
        raise ValueError(f"Unknown optimizer choice: {optimizer_choice}")
    return optim_g, optim_d

def setup_models_for_training(net_g, net_d, device, device_id, n_gpus):
    net_g = net_g.to(device_id) if device.type == "cuda" else net_g.to(device)
    net_d = net_d.to(device_id) if device.type == "cuda" else net_d.to(device)

    if n_gpus > 1 and device.type == "cuda":
        net_g = DDP(net_g, device_ids=[device_id]) # find_unused_parameters=True)
        net_d = DDP(net_d, device_ids=[device_id]) # find_unused_parameters=True)

    return net_g, net_d

def load_models_and_optimizers(config, pretrainG, pretrainD, vocoder, use_checkpointing, randomized, sample_rate, optimizer_choice, custom_lr_g, custom_lr_d, use_custom_lr, total_epoch_count, train_loader, device, device_id, n_gpus, rank):
    # Init the models
    net_g = get_g_model(config, sample_rate, vocoder, use_checkpointing, randomized)
    net_d = get_d_model(config, vocoder, use_checkpointing)
    try:
        print("    ██████  Starting the training ...")

        # Confirm presence of checkpoints
        g_checkpoint_path = latest_checkpoint_path(experiment_dir, "G_*.pth")
        d_checkpoint_path = latest_checkpoint_path(experiment_dir, "D_*.pth")

        # If they exist, we attempt to resume the training
        if g_checkpoint_path and d_checkpoint_path:

            # Init the optimizers
            optim_g, optim_d = get_optimizers(net_g, net_d, config, optimizer_choice, custom_lr_g, custom_lr_d, use_custom_lr, total_epoch_count, train_loader)
            # Move the models to an appropriate device ( And optionally wrap with DDP for multi-gpu )
            net_g, net_d = setup_models_for_training(net_g, net_d, device, device_id, n_gpus)

            # Load the model and optim states
            _, _, _, epoch_str, gradscaler_dict = load_checkpoint(architecture, g_checkpoint_path, net_g, optim_g)
            _, _, _, epoch_str, _ = load_checkpoint(architecture, d_checkpoint_path, net_d, optim_d)

            if override_pretrain_lr:
                new_lr_for_pretrain = new_pretrain_lr
                for param_group in optim_g.param_groups:
                    param_group['lr'] = new_lr_for_pretrain
                    param_group['initial_lr'] = new_lr_for_pretrain
                for param_group in optim_d.param_groups:
                    param_group['lr'] = new_lr_for_pretrain
                    param_group['initial_lr'] = new_lr_for_pretrain
                print(f"[OVERRIDE] Pretrain LR Override: {new_lr_for_pretrain}")

            epoch_str += 1
            global_step = (epoch_str - 1) * len(train_loader)
            print(f"[RESUMING] (G) & (D) at global_step: {global_step} and epoch count: {epoch_str - 1}")
        else:
            raise FileNotFoundError("No checkpoints found.")

    except FileNotFoundError:
    # If no checkpoints are available, using the Pretrains directly
        epoch_str = 1
        global_step = 0
        gradscaler_dict = {}
        
        # Loading the pretrained Generator model
        if (pretrainG != "" and pretrainG != "None"):
            if rank == 0:
                print(f"[ ] Loading pretrained (G) '{pretrainG}'")
            verify_remap_checkpoint(pretrainG, net_g, architecture)

        # Loading the pretrained Discriminator model
        if pretrainD != "" and pretrainD != "None":
            if rank == 0:
                print(f"[ ] Loading pretrained (D) '{pretrainD}'")
            verify_remap_checkpoint(pretrainD, net_d, architecture)

        # Load the models and optionally wrap with DDP
        net_g, net_d = setup_models_for_training(net_g, net_d, device, device_id, n_gpus)

        # Init the optimizers
        optim_g, optim_d = get_optimizers(net_g, net_d, config, optimizer_choice, custom_lr_g, custom_lr_d, use_custom_lr, total_epoch_count, train_loader)

    return net_g, net_d, optim_g, optim_d, epoch_str, global_step, gradscaler_dict

def prepare_schedulers(optim_g, optim_d, use_warmup, warmup_duration, use_lr_scheduler, lr_scheduler, exp_decay_gamma, total_epoch_count, epoch_str, global_step, train_loader):
    warmup_scheduler_g, warmup_scheduler_d = None, None
    scheduler_g, scheduler_d = None, None

    num_batches_per_epoch = len(train_loader)

    if override_pretrain_lr:
        scheduler_resume_epoch = -1
        scheduler_resume_step = -1
    else:
        scheduler_resume_epoch = epoch_str - 1
        scheduler_resume_step = global_step - 1

    if use_warmup:
        warmup_scheduler_g = torch.optim.lr_scheduler.LambdaLR(
            optim_g, lr_lambda=lambda epoch: min(1.0, (epoch + 1) / warmup_duration)
        )
        warmup_scheduler_d = torch.optim.lr_scheduler.LambdaLR(
            optim_d, lr_lambda=lambda epoch: min(1.0, (epoch + 1) / warmup_duration)
        )

    if not use_warmup:
        for param_group in optim_g.param_groups: # For Generator
            if 'initial_lr' not in param_group:
                param_group['initial_lr'] = param_group['lr']
        for param_group in optim_d.param_groups: # For Discriminator
            if 'initial_lr' not in param_group:
                param_group['initial_lr'] = param_group['lr']

    if use_lr_scheduler:
        if lr_scheduler == "exp decay step":
            exp_decay_gamma_step = exp_decay_gamma ** (1.0 / num_batches_per_epoch)

        if lr_scheduler == "exp decay epoch":
            # Exponential decay lr scheduler per epoch
            scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=exp_decay_gamma, last_epoch=scheduler_resume_epoch)
            scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=exp_decay_gamma, last_epoch=scheduler_resume_epoch)

        elif lr_scheduler == "exp decay step":
            # Exponential decay lr scheduler per step
            scheduler_g = torch.optim.lr_scheduler.LambdaLR(optim_g, lr_lambda=lambda step: exp_decay_gamma_step ** step, last_epoch=scheduler_resume_step)
            scheduler_d = torch.optim.lr_scheduler.LambdaLR(optim_d, lr_lambda=lambda step: exp_decay_gamma_step ** step, last_epoch=scheduler_resume_step)

        elif lr_scheduler == "cosine annealing epoch":
            # Cosine annealing lr scheduler per epoch
            scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optim_g, T_max=total_epoch_count, eta_min=3e-5, last_epoch=scheduler_resume_epoch)
            scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optim_d, T_max=total_epoch_count, eta_min=3e-5, last_epoch=scheduler_resume_epoch)

    return warmup_scheduler_g, warmup_scheduler_d, scheduler_g, scheduler_d

def get_reference_sample(train_loader, device, config):
    reference_path = os.path.join("logs", "reference")
    use_custom_ref = all([
        os.path.isfile(os.path.join(reference_path, "ref_feats.npy")),
        os.path.isfile(os.path.join(reference_path, "ref_f0c.npy")),
        os.path.isfile(os.path.join(reference_path, "ref_f0f.npy")),
    ])

    if use_custom_ref:
        print("[REFERENCE] Using custom reference input from 'logs\\reference\\'")

        phone = torch.FloatTensor(np.repeat(np.load(os.path.join(reference_path, "ref_feats.npy")), 2, axis=0)).unsqueeze(0).to(device)
        pitch = torch.LongTensor(np.load(os.path.join(reference_path, "ref_f0c.npy"))).unsqueeze(0).to(device)
        pitchf = torch.FloatTensor(np.load(os.path.join(reference_path, "ref_f0f.npy"))).unsqueeze(0).to(device)

        min_len = min(phone.shape[1], pitch.shape[1], pitchf.shape[1])

        phone, pitch, pitchf = phone[:, :min_len, :], pitch[:, :min_len], pitchf[:, :min_len]
        phone_lengths = torch.LongTensor([phone.shape[1]]).to(device)

        sid = torch.LongTensor([0]).to(device)
    else:
        print("[REFERENCE] No custom reference found. Fetching from the first batch of the train_loader.")

        info = next(iter(train_loader))
        phone, phone_lengths, pitch, pitchf, _, _, _, _, sid = info
        phone, phone_lengths, pitch, pitchf, sid = phone.to(device), phone_lengths.to(device), pitch.to(device), pitchf.to(device), sid.to(device)

        batch_indices = []
        for batch in train_loader.batch_sampler:
            batch_indices = batch
            break

        if isinstance(train_loader.dataset, torch.utils.data.Subset):
            file_paths = train_loader.dataset.dataset.get_file_paths(batch_indices)
        else:
            file_paths = train_loader.dataset.get_file_paths(batch_indices)

        file_name = os.path.basename(file_paths[0])
        print(f"[REFERENCE] Origin of the ref: {file_name}")

    return (phone, phone_lengths, pitch, pitchf, sid, config.train.seed)

def main():
    """
    Main function to start the training process.
    """
    global gpus

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))

    wavs = glob.glob(os.path.join(os.path.join(experiment_dir, "sliced_audios"), "*.wav"))
    if wavs:
        _, sr = load_wav_to_torch(wavs[0])
        if sr != sample_rate:
            print(f"Error: Pretrained model sample rate ({sample_rate} Hz) does not match dataset audio sample rate ({sr} Hz).")
            os._exit(1)
    else:
        print("No wav file found.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpus = [int(item) for item in gpus.split("-")]
        n_gpus = len(gpus) 
    else:
        device = torch.device("cpu")
        gpus = [0]
        n_gpus = 1
        print("No GPU detected, fallback to CPU. This will take a very long time ...")

    def start():
        """
        Starts the training process with multi-GPU support or CPU.
        """
        children = []

        for rank, device_id in enumerate(gpus):
            subproc = mp.Process(
                target=run,
                args=(
                    rank,
                    n_gpus,
                    experiment_dir,
                    pretrainG,
                    pretrainD,
                    total_epoch_count,
                    epoch_save_frequency,
                    save_weight_models,
                    save_only_latest_net_models,
                    config,
                    device,
                    device_id,
                ),
            )
            children.append(subproc)
            subproc.start()
            pid_data["process_pids"].append(subproc.pid)

        for i in range(n_gpus):
            children[i].join()

    if cleanup:
        old_session_cleanup(now_dir, model_name)
    start()

def run(
    rank,
    n_gpus,
    experiment_dir,
    pretrainG,
    pretrainD,
    total_epoch_count,
    epoch_save_frequency,
    save_weight_models,
    save_only_latest_net_models,
    config,
    device,
    device_id,
):
    """
    Runs the training loop on a specific GPU or CPU.

    Args:
        rank (int): The rank of the current process within the distributed training setup.
        n_gpus (int): The total number of GPUs available for training.
        experiment_dir (str): The directory where experiment logs and checkpoints will be saved.
        pretrainG (str): Path to the pre-trained generator model.
        pretrainD (str): Path to the pre-trained discriminator model.
        total_epoch_count (int): The total number of epochs for training.
        epoch_save_frequency (int): Frequency of saving epochs.
        save_weight_models (int): Whether to save small weight models. 0 for no, 1 for yes.
        save_only_latest_net_models (int): Whether to save only latest G/D or for each epoch.
        config (object): Configuration object containing training parameters.
        device (torch.device): The device to use for training (CPU or GPU).
    """
    global global_step, warmup_completed, optimizer_choice, from_scratch

    if 'warmup_completed' not in globals():
        warmup_completed = False

    # Initial print / session info for console
    print_init_setup(
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
    )

    # Initial setup
    writer_eval = setup_env_and_distr(
        rank,
        n_gpus,
        device,
        device_id,
        config
    )

    # Dataloading and loaders preparation
    train_loader, val_loader = prepare_dataloaders(
        config,
        n_gpus,
        rank,
        batch_size,
        use_validation,
        benchmark_mode
    )

    # Spk dim verif
    spk_dim = verify_spk_dim(config, model_info_path, experiment_dir, latest_checkpoint_path, rank, pretrainG)
    config.model.spk_embed_dim = spk_dim

    # Spectral loss init
    if spectral_loss == "L1 Mel Loss":
        fn_spectral_loss = torch.nn.L1Loss()
        print("    ██████  Spectral loss: Single-Scale (L1) Mel loss function")
    elif spectral_loss == "Multi-Scale Mel Loss":
        fn_spectral_loss = MultiScaleMelSpectrogramLoss(sample_rate=sample_rate)
        print("    ██████  Spectral loss: Multi-Scale Mel loss function")
    elif spectral_loss == "Multi-Res STFT Loss":
        fn_spectral_loss = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes = [1024, 2048, 4096],
            hop_sizes = [256, 512, 1024],
            win_lengths = [1024, 2048, 4096],
            window = "hann_window",
            scale = "mel",
            n_bins = 128,
            sample_rate = sample_rate,
            perceptual_weighting = True,
            device=device,
        )
        print("    ██████  Spectral loss: Multi-Resolution STFT loss function")
    else:
        print("ERROR: Chosen spectral loss is undefined. Exiting.")
        sys.exit(1)

    # Hinge adversarial loss
    fn_hinge_loss = HingeAdversarialLoss() if adversarial_loss == "hinge" else None

    # Loading of models and optims
    net_g, net_d, optim_g, optim_d, epoch_str, global_step, gradscaler_dict = load_models_and_optimizers(
        config,
        pretrainG,
        pretrainD,
        vocoder,
        use_checkpointing,
        randomized,
        sample_rate, 
        optimizer_choice,
        custom_lr_g,
        custom_lr_d,
        use_custom_lr, 
        total_epoch_count,
        train_loader,
        device,
        device_id,
        n_gpus,
        rank
    )

    # from-scratch checker ( disables average loss )
    if pretrainG in ["", "None"] and pretrainD in ["", "None"]:
        from_scratch = True
        if rank == 0:
            print("[INIT] No pretrains used: Average loss disabled!")

    # Prepare the schedulers
    warmup_scheduler_g, warmup_scheduler_d, scheduler_g, scheduler_d = prepare_schedulers(
        optim_g,
        optim_d,
        use_warmup,
        warmup_duration,
        use_lr_scheduler, 
        lr_scheduler,
        exp_decay_gamma,
        total_epoch_count,
        epoch_str,
        global_step,
        train_loader
    )

    # Hann window for stft ( for RingFormer only. )
    hann_window = torch.hann_window(config.model.gen_istft_n_fft).to(device) if vocoder in ["RingFormer_v1", "RingFormer_v2"] else None

    # GradScaler for FP16 training
    gradscaler = torch.amp.GradScaler(enabled=(device.type == "cuda" and train_dtype == torch.float16))
    if len(gradscaler_dict) > 0:
        gradscaler.load_state_dict(gradscaler_dict)
        print("[INIT] Loading gradscaler state dict - FP16")


    # Reference sample for live-infer
    reference = get_reference_sample(train_loader, device, config)

    # Cache for training with " cache " enabled
    cache = []

    # GAN Loss debug
    if adversarial_loss == "tprls":
        print("------ GAN LOSS VARIANT: TPRLS ------")
    elif adversarial_loss == "hinge":
        print("------ GAN LOSS VARIANT: HINGE ------")
    elif adversarial_loss == "lsgan":
        print("------ GAN LOSS VARIANT: LSGAN ------")
    else:
        print(f"------ {adversarial_loss} LOSS VARIANT IS NOT SUPPORTED. Exiting .. ------")
        sys.exit(1)

    for epoch in range(epoch_str, total_epoch_count + 1):
        training_loop(
            rank,
            epoch,
            config,
            [net_g, net_d],
            [optim_g, optim_d],
            [scheduler_g, scheduler_d],
            train_loader,
            val_loader if use_validation else None,
            [writer_eval],
            cache,
            total_epoch_count,
            epoch_save_frequency,
            save_weight_models,
            save_only_latest_net_models,
            device,
            device_id,
            reference,
            fn_spectral_loss,
            n_gpus,
            gradscaler,
            fn_hinge_loss,
            hann_window,
        )

        if use_warmup and epoch <= warmup_duration:
            if warmup_scheduler_g:
                warmup_scheduler_g.step()
            if warmup_scheduler_d:
                warmup_scheduler_d.step()

            # Logging of finished warmup
            if epoch == warmup_duration:
                warmup_completed = True
                print(f"    ██████  Warmup completed at epochs: {warmup_duration}")
                print(f"    ██████  LR G: {optim_g.param_groups[0]['lr']}")
                print(f"    ██████  LR D: {optim_d.param_groups[0]['lr']}")
                # scheduler:
                if lr_scheduler == "exp decay epoch":
                    print(f"    ██████  Starting the per-epoch exponential lr decay with gamma of {exp_decay_gamma}")
                elif lr_scheduler == "cosine annealing epoch":
                    print("    ██████  Starting per-epoch cosine annealing scheduler " )

        if use_lr_scheduler and (not use_warmup or warmup_completed):
            # Once the warmup phase is completed, uses exponential lr decay
            if lr_scheduler in ["exp decay epoch", "cosine annealing epoch"]:
                scheduler_g.step()
                scheduler_d.step()

def training_loop(
    rank,
    epoch,
    config,
    nets,
    optims,
    schedulers,
    train_loader,
    val_loader,
    writers,
    cache,
    total_epoch_count,
    epoch_save_frequency,
    save_weight_models,
    save_only_latest_net_models,
    device,
    device_id,
    reference,
    fn_spectral_loss,
    n_gpus,
    gradscaler,
    fn_hinge_loss=None,
    hann_window=None,
):
    """
    Trains and evaluates the model for one epoch.

    Args:
        rank (int): Rank of the current process.
        epoch (int): Current epoch number.
        config (object): Configuration object containing training parameters.
        nets (list): List of models [net_g, net_d].
        optims (list): List of optimizers [optim_g, net_d].
        train_loader: training dataloader.
        val_loader: validation dataloader.
        writers (list): List of TensorBoard writers [writer_eval].
        cache (list): List to cache data in GPU memory.
        total_epoch_count (int): The total number of epochs for training.
        epoch_save_frequency (int): Frequency of saving epochs.
        save_weight_models (int): Whether to save small weight models. 0 for no, 1 for yes.
        save_only_latest_net_models (int): Whether to save only latest G/D or for each epoch.
        device (torch.device): The device to use for training (CPU or GPU).
        reference (list): Contains reference sample. Either custom or from train loader.
        fn_spectral_loss: spectral loss;  can be l1, multi-scale or ms-stft.
        gradscaler: gradscaler for fp16
        hann_window: hann window used for RingFormer
    """
    global global_step, warmup_completed, use_lr_scheduler, lr_scheduler, use_warmup

    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers if schedulers is not None else (None, None)

    train_loader = train_loader if train_loader is not None else None
    if not benchmark_mode and use_validation:
        val_loader = val_loader if val_loader is not None else None

    if writers is not None:
        writer = writers[0]

    fn_hinge_loss = fn_hinge_loss if fn_hinge_loss is not None else None
    
    train_loader.batch_sampler.set_epoch(epoch)

    net_g.train()
    net_d.train()

    # Data caching
    if device.type == "cuda" and cache_data_in_gpu:
        data_iterator = cache
        if cache == []:
            for batch_idx, info in enumerate(train_loader):
                # phone, phone_lengths, pitch, pitchf, spec, spec_lengths, y, y_lengths, sid
                info = [tensor.cuda(device_id, non_blocking=True) for tensor in info]
                cache.append((batch_idx, info))
        else:
            shuffle(cache)
    else:
        data_iterator = enumerate(train_loader)

    epoch_recorder = EpochRecorder()

    # if not from_scratch:
        # # Tensors init for averaged losses:
        # tensor_count = 7 if vocoder in ["RingFormer_v1", "RingFormer_v2"] else 6
        # epoch_loss_tensor = torch.zeros(tensor_count, device=device)
        # num_batches_in_epoch = 0

    if not from_scratch:
        # Tensors init for averaged losses:
        if vocoder in ["RingFormer_v1", "RingFormer_v2"]:
            tensor_count = 7
        elif vocoder == "PCPH-GAN":
            tensor_count = 8
        else:
            tensor_count = 6
        epoch_loss_tensor = torch.zeros(tensor_count, device=device)
        num_batches_in_epoch = 0

    avg_50_cache = {
        "grad_norm_d_50": deque(maxlen=50),
        "grad_norm_g_50": deque(maxlen=50),
        "loss_disc_50": deque(maxlen=50),
        "loss_adv_50": deque(maxlen=50),
        "loss_gen_total_50": deque(maxlen=50),
        "loss_fm_50": deque(maxlen=50),
        "loss_mel_50": deque(maxlen=50),
        "loss_kl_50": deque(maxlen=50),

    }
    if vocoder in ["RingFormer_v1", "RingFormer_v2"]:
        avg_50_cache.update({
            "loss_sd_50": deque(maxlen=50),
        })

    if vocoder in "PCPH-GAN":
        avg_50_cache.update({
            "loss_env_50": deque(maxlen=50),
        })

    use_amp = (config.train.bf16_run or config.train.fp16_run) and device.type == "cuda"

    with tqdm(total=len(train_loader), leave=False) as pbar:
        for batch_idx, info in data_iterator:

            global_step += 1

            if not from_scratch:
                num_batches_in_epoch += 1

            if device.type == "cuda" and not cache_data_in_gpu:
                info = [tensor.cuda(device_id, non_blocking=True) for tensor in info]
            elif device.type != "cuda":
                info = [tensor.to(device) for tensor in info]
            (
                phone,
                phone_lengths,
                pitch,
                pitchf,
                spec,
                spec_lengths,
                y,
                y_lengths,
                sid,
            ) = info

            # Generator forward pass:
            with autocast(device_type="cuda", enabled=use_amp, dtype=train_dtype):
                model_output = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
                # Unpacking:
                if vocoder in ["RingFormer_v1", "RingFormer_v2"]:
                    y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q), (mag, phase) = (model_output)
                else:
                    y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = (model_output)

                # Slice the original waveform ( y ) to match the generated slice:
                if randomized:
                    y = commons.slice_segments(
                        y,
                        ids_slice * config.data.hop_length,
                        config.train.segment_size,
                        dim=3,
                    )

            if vocoder in ["RingFormer_v1", "RingFormer_v2"]:
                reshaped_y = y.view(-1, y.size(-1))
                reshaped_y_hat = y_hat.view(-1, y_hat.size(-1))
                y_stft = torch.stft(reshaped_y, n_fft=config.model.gen_istft_n_fft, hop_length=config.model.gen_istft_hop_size, win_length=config.model.gen_istft_n_fft, window=hann_window, return_complex=True)
                y_hat_stft = torch.stft(reshaped_y_hat, n_fft=config.model.gen_istft_n_fft, hop_length=config.model.gen_istft_hop_size, win_length=config.model.gen_istft_n_fft, window=hann_window, return_complex=True)
                target_magnitude = torch.abs(y_stft)  # shape: [B, F, T]

            # Discriminator forward pass:
            with autocast(device_type="cuda", enabled=use_amp, dtype=train_dtype):
                y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())

            with autocast(device_type="cuda", enabled=False):
                # Compute discriminator loss:
                if adversarial_loss == "lsgan":
                    loss_disc = discriminator_loss(y_d_hat_r, y_d_hat_g)
                elif adversarial_loss == "tprls":
                    loss_disc = discriminator_loss_v2(y_d_hat_r, y_d_hat_g)
                elif adversarial_loss == "hinge":
                    loss_fake, loss_real = fn_hinge_loss(y_d_hat_g, y_d_hat_r)
                    loss_disc = loss_fake + loss_real

            # Discriminator backward and update:
            optim_d.zero_grad(set_to_none=True)
            if train_dtype == torch.float16:
                gradscaler.scale(loss_disc).backward() # Scale and backward of the loss
                gradscaler.unscale_(optim_d) # Unscale
                scale = gradscaler.get_scale() # To retrieve current gradscaler's scaling
                grad_norm_d = torch.nn.utils.clip_grad_norm_(net_d.parameters(), max_norm=float("inf")) # Grad clipping
                gradscaler.step(optim_d) # Optim step
            else:
                loss_disc.backward() # Loss backward
                grad_norm_d = torch.nn.utils.clip_grad_norm_(net_d.parameters(), max_norm=float("inf")) # Grad clipping
                optim_d.step() # Optim step

            # Run discriminator on generated output
            with autocast(device_type="cuda", enabled=use_amp, dtype=train_dtype):
                _, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)

            # Compute generator losses:
            with autocast(device_type="cuda", enabled=False):

                # Spectral loss ( In code kept referenced as "loss_mel" to avoid confusion in old logs / graphs):
                if spectral_loss == "L1 Mel Loss":
                    y_mel = wave_to_mel(config, y, half=train_dtype)
                    y_hat_mel = wave_to_mel(config, y_hat, half=train_dtype)
                    loss_mel = fn_spectral_loss(y_mel, y_hat_mel) * config.train.c_mel
                elif spectral_loss == "Multi-Scale Mel Loss":
                    loss_mel = fn_spectral_loss(y, y_hat) * config.train.c_mel / 3.0
                elif spectral_loss == "Multi-Res STFT Loss":
                    loss_mel = fn_spectral_loss(y_hat.float(), y.float()) * c_stft

                # Feature Matching loss
                loss_fm = feature_loss(fmap_r, fmap_g)
     
                # Generator loss
                if adversarial_loss == "lsgan":
                    loss_adv = generator_loss(y_d_hat_g)
                elif adversarial_loss == "tprls":
                    y_d_hat_r_detached = [i.detach() for i in y_d_hat_r]
                    loss_adv = generator_loss_v2(y_d_hat_g, y_d_hat_r_detached)
                elif adversarial_loss == "hinge":
                    loss_adv = fn_hinge_loss(y_d_hat_g)

                # KL ( Kullback–Leibler divergence ) loss
                if use_kl_annealing:
                    annealing_cycle_steps = len(train_loader) * kl_annealing_cycle_duration
                    kl_beta = 0.5 * (1 - math.cos((global_step % annealing_cycle_steps) * (math.pi / annealing_cycle_steps)))
                else:
                    kl_beta = 1.0

                loss_kl = kl_loss_clamped(z_p, logs_q, m_p, logs_p, z_mask) * config.train.c_kl

                if vocoder in ["RingFormer_v1", "RingFormer_v2"]:
                    # RingFormer related;  Phase, Magnitude and SD:
                    loss_magnitude = torch.nn.functional.l1_loss(mag, target_magnitude)
                    loss_phase = phase_loss(y_stft, y_hat_stft)
                    loss_sd = (loss_magnitude + loss_phase) * 0.7

                if vocoder == "PCPH-GAN":
                    loss_env = envelope_loss(y, y_hat)

                # Total generator loss
                if vocoder in ["RingFormer_v1", "RingFormer_v2"]:
                    loss_gen_total = loss_adv + loss_fm + loss_mel + loss_kl * kl_beta + loss_sd
                elif vocoder == "PCPH-GAN":
                    loss_gen_total = loss_adv + loss_fm + loss_mel + loss_env * 1.0 + loss_kl * kl_beta
                else:
                    loss_gen_total = loss_adv + loss_fm + loss_mel + loss_kl * kl_beta


            # Generator backward and update:
            optim_g.zero_grad(set_to_none=True)
            if train_dtype == torch.float16:
                gradscaler.scale(loss_gen_total).backward() # Scale and backward of the loss
                gradscaler.unscale_(optim_g) # Unscale
                grad_norm_g = torch.nn.utils.clip_grad_norm_(net_g.parameters(), max_norm=float("inf")) # Grad clipping
                gradscaler.step(optim_g) # Optim step
                gradscaler.update() # Scaler update, to prepare the scaling for the next iteration
                skip_lr_sched = (scale > gradscaler.get_scale())
            else:
                loss_gen_total.backward() # Loss backward
                grad_norm_g = torch.nn.utils.clip_grad_norm_(net_g.parameters(), max_norm=float("inf")) # Grad clipping
                optim_g.step() # Optim step
                skip_lr_sched = False

            # Per step exp lr decay for generator
            if not skip_lr_sched: # We skip lr scheduler step if there were nans / infs due to gradscaler's scaling.
                if use_lr_scheduler and (not use_warmup or warmup_completed) and lr_scheduler == "exp decay step":
                    scheduler_d.step()
                    scheduler_g.step()

            if not from_scratch:
                # Loss accumulation In the epoch_loss_tensor
                epoch_loss_tensor[0].add_(loss_disc.detach())
                epoch_loss_tensor[1].add_(loss_adv.detach())
                epoch_loss_tensor[2].add_(loss_gen_total.detach())
                epoch_loss_tensor[3].add_(loss_fm.detach())
                epoch_loss_tensor[4].add_(loss_mel.detach())
                epoch_loss_tensor[5].add_(loss_kl.detach())
                if vocoder in ["RingFormer_v1", "RingFormer_v2"]:
                    epoch_loss_tensor[6].add_(loss_sd.detach())
                if vocoder == "PCPH-GAN":
                    epoch_loss_tensor[6].add_(loss_env.detach())

            # queue for rolling losses / grads over 50 steps
            # Grads:
            avg_50_cache["grad_norm_d_50"].append(grad_norm_d)
            avg_50_cache["grad_norm_g_50"].append(grad_norm_g)
            # Losses:
            avg_50_cache["loss_disc_50"].append(loss_disc.detach())
            avg_50_cache["loss_adv_50"].append(loss_adv.detach())
            avg_50_cache["loss_gen_total_50"].append(loss_gen_total.detach())
            avg_50_cache["loss_fm_50"].append(loss_fm.detach())
            avg_50_cache["loss_mel_50"].append(loss_mel.detach())
            avg_50_cache["loss_kl_50"].append(loss_kl.detach())
            if vocoder in ["RingFormer_v1", "RingFormer_v2"]:
                avg_50_cache["loss_sd_50"].append(loss_sd.detach())
            if vocoder == "PCPH-GAN":
                avg_50_cache["loss_env_50"].append(loss_env.detach())

            if rank == 0 and global_step % 50 == 0:
                scalar_dict_50 = {}
                # Learning rate retrieval for avg-50 variation:
                if from_scratch:
                    lr_d = optim_d.param_groups[0]["lr"]
                    lr_g = optim_g.param_groups[0]["lr"]
                    scalar_dict_50.update({
                    "learning_rate/lr_d": lr_d,
                    "learning_rate/lr_g": lr_g,
                    })
                if optimizer_choice == "Prodigy":
                    prodigy_lr_g = optim_g.param_groups[0].get('d', 0)
                    prodigy_lr_d = optim_d.param_groups[0].get('d', 0)
                    scalar_dict_50.update({
                        "learning_rate/prodigy_lr_g": prodigy_lr_g,
                        "learning_rate/prodigy_lr_d": prodigy_lr_d,
                    })
                # logging rolling averages
                scalar_dict_50.update({
                    # Grads:
                    "grad_avg_50/norm_d_50": sum(avg_50_cache["grad_norm_d_50"])
                    / len(avg_50_cache["grad_norm_d_50"]),
                    "grad_avg_50/norm_g_50": sum(avg_50_cache["grad_norm_g_50"])
                    / len(avg_50_cache["grad_norm_g_50"]),
                    # Losses:
                    "loss_avg_50/loss_disc_50": torch.mean(
                        torch.stack(list(avg_50_cache["loss_disc_50"]))).item(),
                    "loss_avg_50/loss_adv_50": torch.mean(
                        torch.stack(list(avg_50_cache["loss_adv_50"]))).item(),
                    "loss_avg_50/loss_gen_total_50": torch.mean(
                        torch.stack(list(avg_50_cache["loss_gen_total_50"]))).item(),
                    "loss_avg_50/loss_fm_50": torch.mean(
                        torch.stack(list(avg_50_cache["loss_fm_50"]))).item(),
                    "loss_avg_50/loss_mel_50": torch.mean(
                        torch.stack(list(avg_50_cache["loss_mel_50"]))).item(),
                    "loss_avg_50/loss_kl_50": torch.mean(
                        torch.stack(list(avg_50_cache["loss_kl_50"]))).item(),
                })
                if vocoder in ["RingFormer_v1", "RingFormer_v2"]:
                    scalar_dict_50.update({
                        # Losses:
                        "loss_avg_50/loss_sd_50": torch.mean(
                            torch.stack(list(avg_50_cache["loss_sd_50"]))),
                    })
                if vocoder == "PCPH-GAN":
                    scalar_dict_50.update({
                        # Losses:
                        "loss_avg_50/loss_env_50": torch.mean(
                            torch.stack(list(avg_50_cache["loss_env_50"]))),
                    })

                summarize(writer=writer, global_step=global_step, scalars=scalar_dict_50)
                flush_writer(writer, rank)

            if from_scratch and pretrain_preview and rank == 0 and global_step % 50 == 0:
                print(f"    ██████  Generating pretrain-preview at step: {global_step}...")
                o = eval_infer(net_g, reference)
                audio_dict = {f"gen/audio_pretrain_{global_step}s": o[0, :, :]}
                summarize(
                    writer=writer,
                    global_step=global_step,
                    audios=audio_dict,
                    audio_sample_rate=config.data.sample_rate,
                )
                flush_writer(writer, rank)
                torch.cuda.empty_cache()

            pbar.update(1)
        # end of batch train
    # end of tqdm

    if n_gpus > 1 and device.type == 'cuda':
        dist.barrier()

    with torch.no_grad():
        torch.cuda.empty_cache()

    # Logging and checkpointing
    if rank == 0:
        # Used for tensorboard chart - all/mel
        mel = spec_to_mel_torch(
            spec,
            config.data.filter_length,
            config.data.n_mel_channels,
            config.data.sample_rate,
            config.data.mel_fmin,
            config.data.mel_fmax,
        )

        # For fp16 we need to .half() the mel spec
        if train_dtype == torch.float16:
            mel = mel.half()

        # Used for tensorboard chart - slice/mel_org
        if randomized:
            y_mel = commons.slice_segments(
                mel,
                ids_slice,
                config.train.segment_size // config.data.hop_length,
                dim=3,
            )
        else:
            y_mel = mel

        # used for tensorboard chart - slice/mel_gen
        y_hat_mel = wave_to_mel(config, y_hat, half=train_dtype)

        # Mel similarity metric:
        mel_similarity = mel_spec_similarity(y_hat_mel, y_mel)
        print(f'Mel Spectrogram Similarity: {mel_similarity:.2f}%')
        writer.add_scalar('Metric/Mel_Spectrogram_Similarity', mel_similarity, global_step)

        # Learning rate retrieval for avg-epoch variation:
        lr_d = optim_d.param_groups[0]["lr"]
        lr_g = optim_g.param_groups[0]["lr"]

        # Calculate the avg epoch loss:
        if global_step % len(train_loader) == 0 and not from_scratch: # At each epoch completion
            avg_epoch_loss = epoch_loss_tensor / num_batches_in_epoch

            scalar_dict_avg = {
            "loss_avg/loss_disc": avg_epoch_loss[0].item(),
            "loss_avg/loss_adv": avg_epoch_loss[1].item(),
            "loss_avg/loss_gen_total": avg_epoch_loss[2].item(),
            "loss_avg/loss_fm": avg_epoch_loss[3].item(),
            "loss_avg/loss_mel": avg_epoch_loss[4].item(),
            "loss_avg/loss_kl": avg_epoch_loss[5].item(),
            "learning_rate/lr_d": lr_d,
            "learning_rate/lr_g": lr_g,
            }
            if optimizer_choice == "Prodigy":
                prodigy_lr_g = optim_g.param_groups[0].get('d', 0)
                prodigy_lr_d = optim_d.param_groups[0].get('d', 0)
                scalar_dict_avg.update({
                    "learning_rate/prodigy_lr_g": prodigy_lr_g,
                    "learning_rate/prodigy_lr_d": prodigy_lr_d,
                })
            if vocoder in ["RingFormer_v1", "RingFormer_v2"]:
                scalar_dict_avg.update({
                    "loss_avg/loss_sd": avg_epoch_loss[6].item(),
                })
            if vocoder == "PCPH-GAN":
                scalar_dict_avg.update({
                    "loss_avg/loss_env": avg_epoch_loss[6].item(),
                })


            summarize(writer=writer, global_step=global_step, scalars=scalar_dict_avg)
            flush_writer(writer, rank)
            num_batches_in_epoch = 0
            epoch_loss_tensor.zero_()

        # Determine the plot data type
        if train_dtype == torch.float16:
            plot_dtype = torch.float16
        else:
            plot_dtype = torch.float32

        image_dict = {
            "slice/mel_org": plot_spectrogram_to_numpy(y_mel[0].detach().cpu().to(plot_dtype).numpy()),
            "slice/mel_gen": plot_spectrogram_to_numpy(y_hat_mel[0].detach().cpu().to(plot_dtype).numpy()),
            "all/mel": plot_spectrogram_to_numpy(mel[0].detach().cpu().to(plot_dtype).numpy()),
        }

        # At each epoch save point:
        if epoch % epoch_save_frequency == 0:
            if not benchmark_mode:
                if use_validation:
                # Run validation
                    validation_loop(
                        net_g.module if hasattr(net_g, "module") else net_g,
                        val_loader,
                        device,
                        config,
                        writer,
                        global_step,
                    )
            # Inferencing on reference sample
            o = eval_infer(net_g, reference)
            audio_dict = {f"gen/audio_{epoch}e_{global_step}s": o[0, :, :]} # Eval-infer samples
            # Logging
            summarize(
                writer=writer,
                global_step=global_step,
                images=image_dict,
                audios=audio_dict,
                audio_sample_rate=config.data.sample_rate,
            )
            flush_writer(writer, rank)
        else:
            summarize(
                writer=writer,
                global_step=global_step,
                images=image_dict,
            )
            flush_writer(writer, rank)

    # Save checkpoint
    model_add = []
    done = False

    if rank == 0:
        # Print training progress
        record = f"{model_name} | epoch={epoch} | step={global_step} | {epoch_recorder.record()}"
        print(record)

        # Save weights every N epochs
        if epoch % epoch_save_frequency == 0:
            checkpoint_suffix = f"{2333333 if save_only_latest_net_models else global_step}.pth"
            # Save Generator checkpoint
            save_checkpoint(
                architecture,
                net_g,
                optim_g,
                config.train.learning_rate,
                epoch,
                os.path.join(experiment_dir, "G_" + checkpoint_suffix),
                gradscaler,
            )
            # Save Discriminator checkpoint
            save_checkpoint(
                architecture,
                net_d,
                optim_d,
                config.train.learning_rate,
                epoch,
                os.path.join(experiment_dir, "D_" + checkpoint_suffix),
                gradscaler,
            )
            # Save small weight model
            if save_weight_models:
                weight_model_name = small_model_naming(model_name, epoch, global_step)
                model_add.append(os.path.join(experiment_dir, weight_model_name))

        # Check completion
        if epoch >= total_epoch_count:
            print(
                f"Training has been successfully completed with {epoch} epoch, {global_step} steps and {round(loss_gen_total.item(), 3)} loss gen."
            )
            # Final model
            weight_model_name = small_model_naming(model_name, epoch, global_step)
            model_add.append(os.path.join(experiment_dir, weight_model_name))

            done = True

        if model_add:
            ckpt = (
                net_g.module.state_dict()
                if hasattr(net_g, "module")
                else net_g.state_dict()
            )
            for m in model_add:
                if not os.path.exists(m):
                    extract_model(
                        ckpt=ckpt,
                        sr=sample_rate,
                        name=model_name,
                        model_path=m,
                        epoch=epoch,
                        step=global_step,
                        hps=config,
                        vocoder=vocoder,
                        architecture=architecture,
                        vits2_mode=vits2_mode,
                    )
        if done:
            # Clean-up process IDs from memory
            pid_data["process_pids"].clear()  # Clear the PID list when done

            if rank == 0:
                writer.flush()
                writer.close()

            os._exit(2333333)

        with torch.no_grad():
            torch.cuda.empty_cache()

def validation_loop(net_g, val_loader, device, config, writer, global_step):
    net_g.eval()
    torch.cuda.empty_cache()

    total_mel_error = 0.0
    total_mrstft_loss = 0.0
    total_pesq = 0.0
    valid_pesq_count = 0
    total_si_sdr = 0.0
    count = 0

    mrstft = auraloss.freq.MultiResolutionSTFTLoss(device=device)
    resample_to_16k = torchaudio.transforms.Resample(orig_freq=config.data.sample_rate, new_freq=16000).to(device)

    hop_length = config.data.hop_length
    sample_rate = config.data.sample_rate

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            phone, phone_lengths, pitch, pitchf, spec, spec_lengths, y, _, sid = [t.to(device) for t in batch]

        # Infer
            y_hat, x_mask, _ = net_g.infer(phone, phone_lengths, pitch, pitchf, sid)

        # Get reference min-length ( according to gt wave's length )
            y_len = y.shape[-1]

        # Obtaining mel specs
            y_hat_mel = wave_to_mel(config, y_hat, half=train_dtype) # generator-source mel
            mel = wave_to_mel(config, y, half=train_dtype) # gt-source mel

        # Mel loss:
            y_hat_mel_len = y_hat_mel.shape[-1]
            mel_len = mel.shape[-1]

            min_t = min(y_hat_mel_len, mel_len)

            mel_loss = F.l1_loss(y_hat_mel[..., :min_t], mel[..., :min_t])
            total_mel_error += mel_loss.item()

        # STFT loss:
            y_hat_len = y_hat.shape[-1]

            min_samples = min_t * hop_length
            min_samples = min(min_samples, y_len, y_hat_len)

            stft_loss = mrstft(y_hat[..., :min_samples], y[..., :min_samples])
            total_mrstft_loss += stft_loss.item()

        # si_sdr:
            si_sdr_score = si_sdr(y_hat.squeeze(1), y.squeeze(1))
            total_si_sdr += si_sdr_score.item()

        # PESQ:
            try:
                y_16k_batch = resample_to_16k(y).cpu().numpy()          # (B, T)
                y_hat_16k_batch = resample_to_16k(y_hat.squeeze(1)).cpu().numpy()  # (B, T)

                for i in range(y_16k_batch.shape[0]):
                    y_16k_f = np.squeeze(y_16k_batch[i]).astype(np.float32)
                    y_hat_16k_f = np.squeeze(y_hat_16k_batch[i]).astype(np.float32)

                    try:
                        pesq_score = pesq(16000, y_16k_f, y_hat_16k_f, mode="wb")
                        total_pesq += pesq_score
                        valid_pesq_count += 1
                    except Exception as e:
                        print(f"[PESQ skipped] {e}")

            except Exception as e:
                print(f"[PESQ skipped outer] {e}")

            count += 1

    avg_mel = total_mel_error / count
    avg_mrstft = total_mrstft_loss / count
    avg_pesq = total_pesq / max(valid_pesq_count, 1)
    avg_si_sdr = total_si_sdr / count

    if writer is not None:
        writer.add_scalar("validation/loss/mel_l1", avg_mel, global_step)
        writer.add_scalar("validation/loss/mrstft", avg_mrstft, global_step)
        writer.add_scalar("validation/score/pesq", avg_pesq, global_step)
        writer.add_scalar("validation/score/si_sdr", avg_si_sdr, global_step)

    net_g.train()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()

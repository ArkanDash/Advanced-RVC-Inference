import os
import sys
import glob
import json
import torch
import logging
import argparse
import datetime
import warnings

import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm
from collections import deque
from contextlib import nullcontext
from random import randint, shuffle
from distutils.util import strtobool
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from time import time as ttime
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append(os.getcwd())
os.environ["USE_LIBUV"] = "0" if sys.platform == "win32" else "1"

from main.library.utils import clear_gpu_cache
from main.library.backends import directml, opencl
from main.app.variables import logger, translations

from main.library.algorithm import commons
from main.inference.training import losses

from main.inference.training.extract_model import extract_model

from main.inference.training.mel_processing import (
    MultiScaleMelSpectrogramLoss, 
    mel_spectrogram_torch,
    spec_to_mel_torch
)

from main.inference.training.utils import (
    HParams, 
    summarize, 
    load_checkpoint, 
    save_checkpoint, 
    load_wav_to_torch,
    latest_checkpoint_path, 
    plot_spectrogram_to_numpy
)

from main.app.variables import config as main_config
from main.app.variables import configs as main_configs

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

    return parser.parse_args()

d_lr_coeff = 1.0
g_lr_coeff = 1.0
d_step_per_g_step = 1

args = parse_arguments()
model_name, save_every_epoch, total_epoch, pretrainG, pretrainD, version, gpus, batch_size, pitch_guidance, save_only_latest, save_every_weights, cache_data_in_gpu, overtraining_detector, overtraining_threshold, cleanup, model_author, vocoder, checkpointing, optimizer_choice, energy_use, use_custom_reference, reference_path, multiscale_mel_loss = args.model_name, args.save_every_epoch, args.total_epoch, args.g_pretrained_path, args.d_pretrained_path, args.rvc_version, args.gpu, args.batch_size, args.pitch_guidance, args.save_only_latest, args.save_every_weights, args.cache_data_in_gpu, args.overtraining_detector, args.overtraining_threshold, args.cleanup, args.model_author, args.vocoder, args.checkpointing, args.optimizer, args.energy_use, args.use_custom_reference, args.reference_path, args.multiscale_mel_loss

experiment_dir = os.path.join(main_configs["logs_path"], model_name)
training_file_path = os.path.join(experiment_dir, "training_data.json")
config_save_path = os.path.join(experiment_dir, "config.json")

torch.backends.cudnn.deterministic = args.deterministic if not main_config.device.startswith("ocl") else False
torch.backends.cudnn.benchmark = args.benchmark if not main_config.device.startswith("ocl") else False

lowest_value = {"step": 0, "value": float("inf"), "epoch": 0}
global_step, last_loss_gen_all, overtrain_save_epoch = 0, 0, 0
loss_gen_history, smoothed_loss_gen_history, loss_disc_history, smoothed_loss_disc_history = [], [], [], []
avg_losses = {"grad_d_50": deque(maxlen=50), "grad_g_50": deque(maxlen=50), "disc_loss_50": deque(maxlen=50), "adv_loss_50": deque(maxlen=50), "fm_loss_50": deque(maxlen=50), "kl_loss_50": deque(maxlen=50), "mel_loss_50": deque(maxlen=50), "gen_loss_50": deque(maxlen=50)}

with open(config_save_path, "r") as f:
    config = json.load(f)

config = HParams(**config)
config.data.training_files = os.path.join(experiment_dir, "filelist.txt")

def main():
    global training_file_path, last_loss_gen_all, smoothed_loss_gen_history, loss_gen_history, loss_disc_history, smoothed_loss_disc_history, overtrain_save_epoch, model_author, vocoder, checkpointing, gpus, energy_use

    log_data = {
        translations['modelname']: model_name, 
        translations["save_every_epoch"]: save_every_epoch, 
        translations["total_e"]: total_epoch, 
        translations["dorg"].format(pretrainG=pretrainG, pretrainD=pretrainD): "", 
        translations['training_version']: version, 
        "Gpu": gpus, 
        translations['batch_size']: batch_size, 
        translations['training_f0']: pitch_guidance, 
        translations['save_only_latest']: save_only_latest, 
        translations['save_every_weights']: save_every_weights, 
        translations['cache_in_gpu']: cache_data_in_gpu, 
        translations['overtraining_detector']: overtraining_detector, 
        translations['threshold']: overtraining_threshold, 
        translations['cleanup_training']: cleanup, 
        translations['memory_efficient_training']: checkpointing, 
        translations["optimizer"]: optimizer_choice, 
        translations["train&energy"]: energy_use,
        translations["multiscale_mel_loss"]: multiscale_mel_loss
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

        if torch.cuda.is_available() and main_config.device.startswith("cuda"):
            device, gpus = torch.device("cuda"), [int(item) for item in gpus.split("-")]
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

        def start():
            children = []
            pid_data = {"process_pids": []}

            with open(config_save_path, "r") as pid_file:
                try:
                    pid_data.update(json.load(pid_file))
                except json.JSONDecodeError:
                    pass

            with open(config_save_path, "w") as pid_file:
                for rank, device_id in enumerate(gpus):
                    subproc = mp.Process(target=run, args=(rank, n_gpus, experiment_dir, pretrainG, pretrainD, pitch_guidance, total_epoch, save_every_weights, config, device, device_id, model_author, vocoder, checkpointing, energy_use))
                    children.append(subproc)
                    subproc.start()
                    pid_data["process_pids"].append(subproc.pid)

                json.dump(pid_data, pid_file, indent=4)

            for i in range(n_gpus):
                children[i].join()

        def load_from_json(file_path):
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    data = json.load(f)
                    return (data.get("loss_disc_history", []), data.get("smoothed_loss_disc_history", []), data.get("loss_gen_history", []), data.get("smoothed_loss_gen_history", []))
            
            return [], [], [], []

        def continue_overtrain_detector(training_file_path):
            if overtraining_detector and os.path.exists(training_file_path): (loss_disc_history, smoothed_loss_disc_history, loss_gen_history, smoothed_loss_gen_history) = load_from_json(training_file_path)

        if cleanup:
            for root, dirs, files in os.walk(experiment_dir, topdown=False):
                for name in files:
                    file_path = os.path.join(root, name)
                    file_name, file_extension = os.path.splitext(name)
                    if (file_extension == ".0" or (file_name.startswith(("D_", "G_")) and file_extension == ".pth") or (file_name.startswith(("added", "trained")) and file_extension == ".index")): os.remove(file_path)

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
        return translations["time_or_speed_training"].format(current_time=datetime.datetime.now().strftime("%H:%M:%S"), elapsed_time_str=str(datetime.timedelta(seconds=int(round(elapsed_time, 1)))))

def run(rank, n_gpus, experiment_dir, pretrainG, pretrainD, pitch_guidance, custom_total_epoch, custom_save_every_weights, config, device, device_id, model_author, vocoder, checkpointing, energy_use):
    global global_step, smoothed_value_gen, smoothed_value_disc, optimizer_choice

    smoothed_value_gen, smoothed_value_disc = 0, 0
    dist.init_process_group(backend="gloo" if sys.platform == "win32" or device.type != "cuda" else "nccl", init_method="env://", world_size=n_gpus if device.type == "cuda" else 1, rank=rank if device.type == "cuda" else 0)

    torch.manual_seed(config.train.seed)
    if device.type == "cuda": torch.cuda.manual_seed(config.train.seed)
    elif device.type == "ocl": opencl.pytorch_ocl.manual_seed_all(config.train.seed)

    if torch.cuda.is_available(): torch.cuda.set_device(device_id)

    writer_eval = SummaryWriter(log_dir=os.path.join(experiment_dir, "eval")) if rank == 0 else None

    from main.inference.training.data_utils import (
        DistributedBucketSampler,
        TextAudioCollate,
        TextAudioLoader
    )

    train_dataset = TextAudioLoader(config.data, pitch_guidance=pitch_guidance, energy=energy_use)
    train_loader = DataLoader(train_dataset, num_workers=4, shuffle=False, pin_memory=True, collate_fn=TextAudioCollate(pitch_guidance=pitch_guidance, energy=energy_use), batch_sampler=DistributedBucketSampler(train_dataset, batch_size * n_gpus, [50, 100, 200, 300, 400, 500, 600, 700, 800, 900], num_replicas=n_gpus, rank=rank, shuffle=True), persistent_workers=True, prefetch_factor=8)

    if len(train_loader) < 3:
        logger.warning(translations["not_enough_data"])
        sys.exit(1)

    from main.library.algorithm.synthesizers import Synthesizer
    from main.library.algorithm.discriminators import MultiPeriodDiscriminator

    net_g, net_d = (
        Synthesizer(
            config.data.filter_length // 2 + 1, 
            config.train.segment_size // config.data.hop_length, 
            **config.model, 
            use_f0=pitch_guidance, 
            sr=config.data.sample_rate, 
            vocoder=vocoder, 
            checkpointing=checkpointing, 
            energy=energy_use
        ), 
        MultiPeriodDiscriminator(
            version, 
            config.model.use_spectral_norm, 
            checkpointing=checkpointing
        )
    )

    net_g, net_d = (net_g.cuda(device_id), net_d.cuda(device_id)) if torch.cuda.is_available() else (net_g.to(device), net_d.to(device))

    if optimizer_choice == "AnyPrecisionAdamW" and main_config.brain:
        from main.inference.training.anyprecision_optimizer import AnyPrecisionAdamW
        optimizer_optim = AnyPrecisionAdamW
    elif optimizer_choice == "RAdam":
        optimizer_optim = torch.optim.RAdam
    else:
        optimizer_optim = torch.optim.AdamW

    optim_g, optim_d = optimizer_optim(net_g.parameters(), config.train.learning_rate * g_lr_coeff, betas=config.train.betas, eps=config.train.eps), optimizer_optim(net_d.parameters(), config.train.learning_rate * d_lr_coeff, betas=config.train.betas, eps=config.train.eps)
    fn_mel_loss = MultiScaleMelSpectrogramLoss(sample_rate=config.data.sample_rate) if multiscale_mel_loss else torch.nn.L1Loss()

    if not device.type.startswith(("privateuseone", "ocl")): 
        net_g, net_d = (DDP(net_g, device_ids=[device_id]), DDP(net_d, device_ids=[device_id])) if torch.cuda.is_available() else (DDP(net_g), DDP(net_d))

    scaler_dict = {}
    try:
        logger.info(translations["start_training"])

        _, _, _, epoch_str, scaler_dict = load_checkpoint(logger, (os.path.join(experiment_dir, "D_latest.pth") if save_only_latest else latest_checkpoint_path(experiment_dir, "D_*.pth")), net_d, optim_d)
        _, _, _, epoch_str, _ = load_checkpoint(logger, (os.path.join(experiment_dir, "G_latest.pth") if save_only_latest else latest_checkpoint_path(experiment_dir, "G_*.pth")), net_g, optim_g)
        
        epoch_str += 1
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        check = ["", "None"]
        epoch_str, global_step = 1, 0
        strict = main_configs.get("pretrain_strict", True)
        try:
            if pretrainG not in check:
                if rank == 0: logger.info(translations["import_pretrain"].format(dg="G", pretrain=pretrainG))

                ckptG = torch.load(pretrainG, map_location="cpu", weights_only=True)["model"]
                net_g.module.load_state_dict(ckptG, strict=strict) if hasattr(net_g, "module") else net_g.load_state_dict(ckptG, strict=strict)
                del ckptG

            if pretrainD not in check:
                if rank == 0: logger.info(translations["import_pretrain"].format(dg="D", pretrain=pretrainD))

                ckptD = torch.load(pretrainD, map_location="cpu", weights_only=True)["model"]
                net_d.module.load_state_dict(ckptD, strict=strict) if hasattr(net_d, "module") else net_d.load_state_dict(ckptD, strict=strict)
                del ckptD
        except Exception as e:
            logger.warning(translations["checkpointing_err"])
            logger.error(e)
            sys.exit(1)

    scheduler_g, scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=config.train.lr_decay, last_epoch=epoch_str - 2), torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=config.train.lr_decay, last_epoch=epoch_str - 2)
    scaler = GradScaler(device=device, enabled=main_config.is_half and device.type == "cuda")
    cache = []

    if len(scaler_dict) > 0: scaler.load_state_dict(scaler_dict)

    if use_custom_reference and os.path.isfile(os.path.join(reference_path, "feats.npy")):
        import numpy as np

        logger.info(translations["using_reference"].format(reference_name=reference_path))
        phone = np.repeat(np.load(os.path.join(reference_path, "feats.npy")), 2, axis=0)

        reference = (
            torch.FloatTensor(phone).unsqueeze(0).to(device),
            torch.LongTensor([phone.shape[0]]).to(device),
            torch.LongTensor(np.load(os.path.join(reference_path, "pitch_coarse.npy"))[:-1]).unsqueeze(0).to(device) if pitch_guidance else None,
            torch.FloatTensor(np.load(os.path.join(reference_path, "pitch_fine.npy"))[:-1]).unsqueeze(0).to(device) if pitch_guidance else None,
            torch.LongTensor([0]).to(device),
            torch.FloatTensor(np.load(os.path.join(reference_path, "energy.npy"))[:-1]).unsqueeze(0).to(device) if energy_use else None
        )
    else:
        info = next(iter(train_loader))
        reference = (info[0].to(device), info[1].to(device))

        if pitch_guidance:
            reference += (info[2].to(device), info[3].to(device), info[8].to(device))
            reference += (info[9].to(device),) if energy_use else (None,)
        else:
            reference += (None, None, info[6].to(device))
            reference += (info[7].to(device),) if energy_use else (None,)

    for epoch in range(epoch_str, total_epoch + 1):
        train_and_evaluate(rank, epoch, config, [net_g, net_d], [optim_g, optim_d], scaler, train_loader, writer_eval, cache, custom_save_every_weights, custom_total_epoch, device, device_id, reference, model_author, vocoder, energy_use, fn_mel_loss)
        scheduler_g.step(); scheduler_d.step()

def train_and_evaluate(rank, epoch, hps, nets, optims, scaler, train_loader, writer, cache, custom_save_every_weights, custom_total_epoch, device, device_id, reference, model_author, vocoder, energy_use, fn_mel_loss):
    global global_step, lowest_value, loss_disc, consecutive_increases_gen, consecutive_increases_disc, smoothed_value_gen, smoothed_value_disc

    if epoch == 1:
        lowest_value = {"step": 0, "value": float("inf"), "epoch": 0}
        consecutive_increases_gen, consecutive_increases_disc = 0, 0

    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader.batch_sampler.set_epoch(epoch)
    net_g.train(); net_d.train()

    if device.type == "cuda" and cache_data_in_gpu:
        data_iterator = cache
        if cache == []:
            for batch_idx, info in enumerate(train_loader):
                cache.append((batch_idx, [tensor.cuda(device_id, non_blocking=True) for tensor in info]))
        else: shuffle(cache)
    elif device.type in ["privateuseone", "ocl"] and cache_data_in_gpu:
        data_iterator = cache
        if cache == []:
            for batch_idx, info in enumerate(train_loader):
                cache.append((batch_idx, [tensor.to(device_id if device.type == "ocl" else device, non_blocking=True) for tensor in info]))
        else: shuffle(cache)
    else: data_iterator = enumerate(train_loader)

    epoch_recorder = EpochRecorder()

    autocast_enabled = main_config.is_half and device.type == "cuda"
    autocast_dtype = torch.float32 if not autocast_enabled else (torch.bfloat16 if main_config.brain else torch.float16)
    autocasts = autocast(device.type, enabled=autocast_enabled, dtype=autocast_dtype) if not device.type.startswith("ocl") else nullcontext()
    
    with tqdm(total=len(train_loader), leave=False) as pbar:
        for batch_idx, info in data_iterator:
            if device.type == "cuda" and not cache_data_in_gpu: info = [tensor.cuda(device_id, non_blocking=True) for tensor in info]  
            elif device.type in ["privateuseone", "ocl"] and not cache_data_in_gpu: info = [tensor.to(device_id if device.type == "ocl" else device, non_blocking=True) for tensor in info]  
            else: info = [tensor.to(device) for tensor in info]

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
                y_hat, ids_slice, _, z_mask, (_, z_p, m_p, logs_p, _, logs_q) = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid, energy)
                wave = commons.slice_segments(wave, ids_slice * config.data.hop_length, config.train.segment_size, dim=3)

            for _ in range(d_step_per_g_step):
                with autocasts:
                    y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
                    loss_disc, losses_disc_r, losses_disc_g = losses.discriminator_loss(y_d_hat_r, y_d_hat_g)

                optim_d.zero_grad()

                if autocast_enabled:
                    scaler.scale(loss_disc).backward()
                    scaler.unscale_(optim_d)
                    grad_norm_d = commons.clip_grad_value(net_d.parameters(), None)
                    scaler.step(optim_d)
                else:
                    loss_disc.backward()
                    grad_norm_d = commons.clip_grad_value(net_d.parameters(), None)
                    optim_d.step()

            with autocasts:
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)

            if multiscale_mel_loss: 
                loss_mel = fn_mel_loss(wave, y_hat) * config.train.c_mel / 3.0
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

            if device.type == "privateuseone": 
                loss_kl = (losses.kl_loss(z_p.detach().cpu(), logs_q.detach().cpu(), m_p.detach().cpu(), logs_p.detach().cpu(), z_mask.detach().cpu()) * config.train.c_kl).to(device)
            else:
                loss_kl = losses.kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config.train.c_kl

            loss_fm = losses.feature_loss(fmap_r, fmap_g)
            loss_gen, losses_gen = losses.generator_loss(y_d_hat_g)

            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
            if loss_gen_all < lowest_value["value"]: lowest_value = {"step": global_step, "value": loss_gen_all, "epoch": epoch}

            optim_g.zero_grad()
            if autocast_enabled:
                scaler.scale(loss_gen_all).backward()
                scaler.unscale_(optim_g)
                grad_norm_g = commons.clip_grad_value(net_g.parameters(), None)
                scaler.step(optim_g)
                scaler.update()
            else:
                loss_gen_all.backward()
                grad_norm_g = commons.clip_grad_value(net_g.parameters(), None)
                optim_g.step()

            global_step += 1

            avg_losses["grad_d_50"].append(grad_norm_d)
            avg_losses["grad_g_50"].append(grad_norm_g)
            avg_losses["disc_loss_50"].append(loss_disc.detach())
            avg_losses["adv_loss_50"].append(loss_gen.detach())
            avg_losses["fm_loss_50"].append(loss_fm.detach())
            avg_losses["kl_loss_50"].append(loss_kl.detach())
            avg_losses["mel_loss_50"].append(loss_mel.detach())
            avg_losses["gen_loss_50"].append(loss_gen_all.detach())

            if rank == 0 and global_step % 50 == 0:
                scalar_dict = {
                    "grad_avg_50/norm_d": sum(avg_losses["grad_d_50"]) / len(avg_losses["grad_d_50"]),
                    "grad_avg_50/norm_g": sum(avg_losses["grad_g_50"]) / len(avg_losses["grad_g_50"]),
                    "loss_avg_50/d/adv": torch.stack(list(avg_losses["disc_loss_50"])).mean(),
                    "loss_avg_50/g/adv": torch.stack(list(avg_losses["adv_loss_50"])).mean(),
                    "loss_avg_50/g/fm": torch.stack(list(avg_losses["fm_loss_50"])).mean(),
                    "loss_avg_50/g/kl": torch.stack(list(avg_losses["kl_loss_50"])).mean(),
                    "loss_avg_50/g/mel": torch.stack(list(avg_losses["mel_loss_50"])).mean(),
                    "loss_avg_50/g/total": torch.stack(list(avg_losses["gen_loss_50"])).mean()
                }

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
        y_mel = commons.slice_segments(
            mel, 
            ids_slice, 
            config.train.segment_size // config.data.hop_length, 
            dim=3
        )
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
            "loss/g/kl": loss_kl
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

    def check_overtraining(smoothed_loss_history, threshold, epsilon=0.004):
        if len(smoothed_loss_history) < threshold + 1: return False

        for i in range(-threshold, -1):
            if smoothed_loss_history[i + 1] > smoothed_loss_history[i]: return True
            if abs(smoothed_loss_history[i + 1] - smoothed_loss_history[i]) >= epsilon: return False

        return True

    def update_exponential_moving_average(smoothed_loss_history, new_value, smoothing=0.987):
        smoothed_value = new_value if not smoothed_loss_history else (smoothing * smoothed_loss_history[-1] + (1 - smoothing) * new_value)      
        smoothed_loss_history.append(smoothed_value)

        return smoothed_value

    def save_to_json(file_path, loss_disc_history, smoothed_loss_disc_history, loss_gen_history, smoothed_loss_gen_history):
        with open(file_path, "w") as f:
            json.dump({"loss_disc_history": loss_disc_history, "smoothed_loss_disc_history": smoothed_loss_disc_history, "loss_gen_history": loss_gen_history, "smoothed_loss_gen_history": smoothed_loss_gen_history}, f)
    
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
                os.path.join(experiment_dir, "G_" + checkpoint_suffix), 
                scaler
            )
            save_checkpoint(
                logger, 
                net_d, 
                optim_d, 
                config.train.learning_rate, 
                epoch, 
                os.path.join(experiment_dir, "D_" + checkpoint_suffix), 
                scaler
            )

            if custom_save_every_weights: model_add.append(os.path.join(main_configs["weights_path"], f"{model_name}_{epoch}e_{global_step}s.pth"))

        if overtraining_detector and epoch > 1:
            current_loss_disc, current_loss_gen = float(loss_disc), float(lowest_value["value"])
            
            loss_disc_history.append(current_loss_disc)
            loss_gen_history.append(current_loss_gen)
            
            smoothed_value_disc = update_exponential_moving_average(smoothed_loss_disc_history, current_loss_disc)
            smoothed_value_gen = update_exponential_moving_average(smoothed_loss_gen_history, current_loss_gen)
            
            is_overtraining_disc = check_overtraining(smoothed_loss_disc_history, overtraining_threshold * 2)
            is_overtraining_gen = check_overtraining(smoothed_loss_gen_history, overtraining_threshold, 0.01)
            
            consecutive_increases_disc = (consecutive_increases_disc + 1) if is_overtraining_disc else 0
            consecutive_increases_gen = (consecutive_increases_gen + 1) if is_overtraining_gen else 0

            if epoch % save_every_epoch == 0: save_to_json(training_file_path, loss_disc_history, smoothed_loss_disc_history, loss_gen_history, smoothed_loss_gen_history)

            if (is_overtraining_gen and consecutive_increases_gen == overtraining_threshold or is_overtraining_disc and consecutive_increases_disc == (overtraining_threshold * 2)):
                logger.info(translations["overtraining_find"].format(epoch=epoch, smoothed_value_gen=f"{smoothed_value_gen:.3f}", smoothed_value_disc=f"{smoothed_value_disc:.3f}"))
                done = True
            else:
                logger.info(translations["best_epoch"].format(epoch=epoch, smoothed_value_gen=f"{smoothed_value_gen:.3f}", smoothed_value_disc=f"{smoothed_value_disc:.3f}"))
                for file in glob.glob(os.path.join(main_configs["weights_path"], f"{model_name}_*e_*s_best_epoch.pth")):
                    model_del.append(file)

                model_add.append(os.path.join(main_configs["weights_path"], f"{model_name}_{epoch}e_{global_step}s_best_epoch.pth"))
        
        if epoch >= custom_total_epoch:
            logger.info(translations["success_training"].format(epoch=epoch, global_step=global_step, loss_gen_all=round(loss_gen_all.item(), 3)))
            logger.info(translations["training_info"].format(lowest_value_rounded=round(float(lowest_value["value"]), 3), lowest_value_epoch=lowest_value['epoch'], lowest_value_step=lowest_value['step']))
            model_add.append(os.path.join(main_configs["weights_path"], f"{model_name}_{epoch}e_{global_step}s.pth"))
            done = True
            
        for m in model_del:
            os.remove(m)
        
        if model_add:
            ckpt = (net_g.module.state_dict() if hasattr(net_g, "module") else net_g.state_dict())
            for m in model_add:
                extract_model(ckpt=ckpt, sr=config.data.sample_rate, pitch_guidance=pitch_guidance == True, name=model_name, model_path=m, epoch=epoch, step=global_step, version=version, hps=hps, model_author=model_author, vocoder=vocoder, energy_use=energy_use)

        lowest_value_rounded = round(float(lowest_value["value"]), 3)

        if epoch > 1 and overtraining_detector: logger.info(translations["model_training_info"].format(model_name=model_name, epoch=epoch, global_step=global_step, epoch_recorder=epoch_recorder.record(), lowest_value_rounded=lowest_value_rounded, lowest_value_epoch=lowest_value['epoch'], lowest_value_step=lowest_value['step'], remaining_epochs_gen=(overtraining_threshold - consecutive_increases_gen), remaining_epochs_disc=((overtraining_threshold * 2) - consecutive_increases_disc), smoothed_value_gen=f"{smoothed_value_gen:.3f}", smoothed_value_disc=f"{smoothed_value_disc:.3f}"))
        elif epoch > 1 and overtraining_detector == False: logger.info(translations["model_training_info_2"].format(model_name=model_name, epoch=epoch, global_step=global_step, epoch_recorder=epoch_recorder.record(), lowest_value_rounded=lowest_value_rounded, lowest_value_epoch=lowest_value['epoch'], lowest_value_step=lowest_value['step']))
        else: logger.info(translations["model_training_info_3"].format(model_name=model_name, epoch=epoch, global_step=global_step, epoch_recorder=epoch_recorder.record()))

        logger.debug(f"loss_gen_all: {loss_gen_all} loss_gen: {loss_gen} loss_fm: {loss_fm} loss_mel: {loss_mel} loss_kl: {loss_kl}")
        last_loss_gen_all = loss_gen_all

        if done: 
            pid_file_path = os.path.join(experiment_dir, "config.json")
            with open(pid_file_path, "r") as pid_file:
                pid_data = json.load(pid_file)

            with open(pid_file_path, "w") as pid_file:
                pid_data.pop("process_pids", None)
                json.dump(pid_data, pid_file, indent=4)

            if os.path.exists(os.path.join(experiment_dir, "train_pid.txt")): os.remove(os.path.join(experiment_dir, "train_pid.txt"))
            sys.exit(0)

        with torch.no_grad():
            clear_gpu_cache()

if __name__ == "__main__": 
    mp.set_start_method("spawn")
    main()
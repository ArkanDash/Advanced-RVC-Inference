import os
import sys
import logging
import argparse
import warnings

import torch.multiprocessing as mp

# FIX: Ensure project root is in sys.path BEFORE any arvc imports
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from arvc.utils import strtobool
from arvc.rvc.models.utils import check_assets
from arvc.rvc.training.extract.rms import run_rms_extraction
from arvc.rvc.training.extract.feature import run_pitch_extraction
from arvc.utils.variables import config, logger, translations, configs
from arvc.rvc.training.extract.embedding import run_embedding_extraction
from arvc.rvc.training.extract.preparing_files import generate_config, generate_filelist

if not getattr(config, 'debug_mode', False):
    warnings.filterwarnings("ignore")
    for l in ["torch", "faiss", "httpx", "httpcore", "faiss.loader", "numba.core", "urllib3", "matplotlib"]:
        logging.getLogger(l).setLevel(logging.ERROR)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action='store_true')
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--rvc_version", type=str, default="v2")
    parser.add_argument("--f0_method", type=str, default="rmvpe")
    parser.add_argument("--pitch_guidance", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--hop_length", type=int, default=128)
    parser.add_argument("--cpu_cores", type=int, default=2)
    parser.add_argument("--gpu", type=str, default="-")
    parser.add_argument("--sample_rate", type=int, required=True)
    parser.add_argument("--embedder_model", type=str, default="hubert_base")
    parser.add_argument("--f0_onnx", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--embedders_mode", type=str, default="fairseq")
    parser.add_argument("--f0_autotune", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--f0_autotune_strength", type=float, default=1)
    parser.add_argument("--rms_extract", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--alpha", type=float, default=0.5)
    # VRVC additions
    parser.add_argument("--include_mutes", type=int, default=2, help="Number of mute entries per speaker in filelist (from Vietnamese-RVC)")
    parser.add_argument("--embedders_mix", type=lambda x: bool(strtobool(x)), default=False, help="Enable embedder layer mixing (from Vietnamese-RVC)")
    parser.add_argument("--embedders_mix_layers", type=int, default=9, required=False, help="Number of layers for embedder mixing")
    parser.add_argument("--embedders_mix_ratio", type=float, default=0.5, help="Mix ratio for embedder layer blending")
    parser.add_argument("--architecture", type=str, default="RVC", help="Model architecture: RVC or SVC (from Vietnamese-RVC)")
    parser.add_argument("--predictor_onnx", type=lambda x: bool(strtobool(x)), default=False, help="Alias for --f0_onnx (from training service)")

    return parser.parse_args()

def main():
    args = parse_arguments()

    (
        f0_method, hop_length, num_processes, gpus, version, pitch_guidance,
        sample_rate, embedder_model, f0_onnx, embedders_mode, f0_autotune,
        f0_autotune_strength, rms_extract, alpha, include_mutes,
        embedders_mix, embedders_mix_layers, embedders_mix_ratio, architecture
    ) = (
        args.f0_method, args.hop_length, args.cpu_cores, args.gpu,
        args.rvc_version, args.pitch_guidance, args.sample_rate,
        args.embedder_model, args.predictor_onnx or args.f0_onnx, args.embedders_mode,
        args.f0_autotune, args.f0_autotune_strength, args.rms_extract,
        args.alpha, args.include_mutes, args.embedders_mix,
        args.embedders_mix_layers, args.embedders_mix_ratio, args.architecture
    )
    check_assets(f0_method, embedder_model, f0_onnx=f0_onnx, embedders_mode=embedders_mode)
    exp_dir = os.path.join(configs["logs_path"], args.model_name)

    num_processes = max(1, num_processes)

    # ── Validate / build device list ─────────────────────────────────────────
    # BUG FIX: The original code at this point did:
    #
    #     devices = ["cpu"] if gpus == "-" else [
    #         (f"cuda:{idx}") if config.device.startswith("cuda") else ...
    #         for idx in gpus.split("-")
    #     ]
    #
    # which blindly constructed device strings like "cuda:0", "cuda:1" from
    # the raw `--gpu` argument WITHOUT checking:
    #   1. Whether CUDA is actually available (`torch.cuda.is_available()`)
    #   2. Whether the requested index exists (`idx < torch.cuda.device_count()`)
    #   3. Whether `config.device` (the auto-detected backend) actually matches
    #      the user's request (e.g. user on CPU-only machine sending `--gpu 0`
    #      was given `privateuseone:0`, which doesn't exist either).
    #
    # This caused the notorious:
    #   "CUDA error: invalid device ordinal
    #    GPU device may be out of range, do you have enough GPUs?"
    #
    # which silently killed the embedding-extraction subprocess and produced
    # 0 output files, which then cascaded into "File matching failed" in
    # preparing_files.py. Users saw the dataset had been preprocessed
    # ("200 wavs") but training refused to start because the embedding step
    # had silently failed.
    #
    # This fix:
    #   - Forces `["cpu"]` whenever the auto-detected backend is CPU
    #     (regardless of `--gpu`, because the embedder/predictor models are
    #     loaded via `config.device` semantics elsewhere — mixing CPU backend
    #     with a `cuda:N` device string here causes device-mismatch errors).
    #   - For each requested GPU index, checks `torch.cuda.is_available()` and
    #     `idx < torch.cuda.device_count()`. Invalid indices are dropped with
    #     a clear warning. If ALL indices are invalid, falls back to CPU.
    #   - For XPU / OCL / privateuseone backends, performs equivalent checks
    #     (using `torch.xpu.device_count()` / DirectML availability).
    #   - Logs the final device list at INFO so the user can see what was
    #     actually used.
    devices = _build_device_list(gpus, config.device)
    logger.info(
        f"Extraction will use {len(devices)} device(s): {devices} "
        f"(requested --gpu={gpus!r}, auto-detected backend={config.device!r})"
    )

    log_data = {
        translations['modelname']: args.model_name,
        translations['export_process']: exp_dir,
        translations['f0_method']: f0_method,
        translations['pretrain_sr']: sample_rate,
        translations['cpu_core']: num_processes,
        "Gpu": gpus,
        translations['hop_length']: hop_length,
        translations['training_version']: version,
        translations['extract_f0']: pitch_guidance,
        translations['hubert_model']: embedder_model,
        translations.get("f0_onnx_mode", "F0 ONNX"): f0_onnx,
        translations.get("embed_mode", "Embedder mode"): embedders_mode,
        translations.get("train&energy", "Energy"): rms_extract,
        translations.get("alpha_label", "Alpha"): alpha,
        translations.get("include_mutes", "Include mutes"): include_mutes,
        translations.get("embedders_mix", "Embedders mix"): embedders_mix,
        translations.get("embedders_mix_layers", "Mix layers"): embedders_mix_layers,
        translations.get("embedders_mix_ratio", "Mix ratio"): embedders_mix_ratio,
        translations.get("architecture", "Architecture"): architecture,
    }

    for key, value in log_data.items():
        logger.debug(f"{key}: {value}")

    pid_path = os.path.join(exp_dir, "extract_pid.txt")
    with open(pid_path, "w") as pid_file:
        pid_file.write(str(os.getpid()))

    success = False
    try:
        run_pitch_extraction(
            exp_dir, f0_method, hop_length, num_processes, devices, f0_onnx,
            config.is_half, f0_autotune, f0_autotune_strength, alpha
        )
        run_embedding_extraction(
            exp_dir, version, num_processes, devices, embedder_model,
            embedders_mode, config.is_half,
            embedders_mix, embedders_mix_layers, embedders_mix_ratio
        )
        run_rms_extraction(exp_dir, num_processes, devices, rms_extract)
        generate_config(version, sample_rate, exp_dir, architecture)
        generate_filelist(
            pitch_guidance, exp_dir, version, sample_rate, embedders_mode,
            embedder_model, rms_extract, include_mutes
        )
        success = True
    except Exception as e:
        logger.error(f"{translations.get('extract_error', 'Extraction error')}: {e}")

    if os.path.exists(pid_path): os.remove(pid_path)
    if success:
        logger.info(f"{translations.get('extract_success', 'Extraction complete')} {args.model_name}.")


def _build_device_list(gpus, auto_device):
    """Build a list of valid device strings for the extract subprocess pool.

    Args:
        gpus:        Raw value of the `--gpu` CLI argument. Use "-" for CPU,
                     or a hyphen-separated list of indices ("0", "0-1", "0-0-0").
        auto_device:  The auto-detected device string from `config.device`
                     (e.g. "cuda:0", "cpu", "mps", "privateuseone:0").

    Returns:
        List of device strings (e.g. ["cuda:0"], ["cpu"], ["cuda:0","cuda:1"]).
        Invalid GPU indices are dropped. If all indices are invalid (or CUDA
        is unavailable), falls back to ["cpu"] so extraction still runs.
    """
    # CPU requested explicitly
    if gpus is None or gpus == "" or gpus == "-":
        return ["cpu"]

    # If the auto-detected backend is CPU/MPS (no CUDA/XPU/OCL), force CPU.
    # Mixing `--gpu 0` (interpreted as CUDA) with a CPU/MPS backend causes
    # device-mismatch errors in the embedder loader.
    if auto_device == "cpu" or auto_device == "mps" or not auto_device:
        if gpus not in ("-", "", None):
            logger.warning(
                f"--gpu={gpus!r} was requested but the active backend is "
                f"{auto_device!r} (no CUDA/XPU/OCL detected). Falling back to CPU. "
                f"Extract will run on CPU and be slower than GPU."
            )
        return ["cpu"]

    # Determine backend prefix and device-count function based on auto_device
    if auto_device.startswith("cuda"):
        backend = "cuda"
        try:
            import torch as _torch
            device_count_fn = _torch.cuda.device_count
            is_available_fn = _torch.cuda.is_available
        except Exception:
            logger.warning(
                "CUDA backend selected but torch.cuda unavailable. "
                "Falling back to CPU."
            )
            return ["cpu"]
    elif auto_device.startswith("xpu"):
        backend = "xpu"
        try:
            import torch as _torch
            device_count_fn = getattr(_torch.xpu, "device_count", lambda: 0)
            is_available_fn = getattr(_torch.xpu, "is_available", lambda: False)
        except Exception:
            return ["cpu"]
    elif auto_device.startswith("ocl") or auto_device.startswith("privateuseone"):
        backend = "privateuseone"
        # DirectML/OpenCL don't expose a stable device_count in torch; assume 1
        device_count_fn = lambda: 1
        is_available_fn = lambda: True
    else:
        # Unknown backend; fall back to CPU
        logger.warning(
            f"Unknown auto-detected backend {auto_device!r}. Falling back to CPU."
        )
        return ["cpu"]

    # Validate CUDA/XPU availability
    try:
        if not is_available_fn():
            logger.warning(
                f"Backend {backend!r} reported as unavailable. "
                f"Falling back to CPU for extraction."
            )
            return ["cpu"]
    except Exception:
        pass

    # Parse the requested GPU indices
    try:
        indices = [int(s.strip()) for s in str(gpus).split("-") if s.strip() != ""]
    except ValueError:
        logger.warning(
            f"Could not parse --gpu={gpus!r} as a list of integer indices. "
            f"Falling back to CPU."
        )
        return ["cpu"]

    if not indices:
        return ["cpu"]

    # Validate each index against device_count
    try:
        n_devices = device_count_fn()
    except Exception:
        n_devices = 0

    valid_devices = []
    invalid_indices = []
    for idx in indices:
        if 0 <= idx < n_devices:
            valid_devices.append(f"{backend}:{idx}")
        else:
            invalid_indices.append(idx)

    if invalid_indices:
        logger.warning(
            f"GPU index(es) {invalid_indices} are out of range for backend "
            f"{backend!r} (only {n_devices} device(s) available: indices 0..{n_devices-1 if n_devices > 0 else 'none'}). "
            f"Dropping invalid indices. Valid devices kept: {valid_devices}"
        )

    if not valid_devices:
        logger.warning(
            f"All requested GPU indices were invalid. Falling back to CPU "
            f"for extraction (will be slower)."
        )
        return ["cpu"]

    return valid_devices

if __name__ == "__main__": 
    mp.set_start_method("spawn", force=True)
    main()

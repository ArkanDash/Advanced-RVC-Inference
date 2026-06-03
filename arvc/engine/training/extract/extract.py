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
from arvc.engine.models.utils import check_assets
from arvc.engine.training.extract.rms import run_rms_extraction
from arvc.engine.training.extract.feature import run_pitch_extraction
from arvc.utils.variables import config, logger, translations, configs
from arvc.engine.training.extract.embedding import run_embedding_extraction
from arvc.engine.training.extract.preparing_files import generate_config, generate_filelist

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
        args.embedder_model, args.f0_onnx, args.embedders_mode,
        args.f0_autotune, args.f0_autotune_strength, args.rms_extract,
        args.alpha, args.include_mutes, args.embedders_mix,
        args.embedders_mix_layers, args.embedders_mix_ratio, args.architecture
    )
    check_assets(f0_method, embedder_model, f0_onnx=f0_onnx, embedders_mode=embedders_mode)
    exp_dir = os.path.join(configs["logs_path"], args.model_name)

    num_processes = max(1, num_processes)

    # VRVC: XPU device support in device list
    devices = ["cpu"] if gpus == "-" else [
        (
            f"cuda:{idx}"
        ) if config.device.startswith("cuda") else (
            f"xpu:{idx}" if config.device.startswith("xpu") else f"{'ocl' if config.device.startswith('ocl') else 'privateuseone'}:{idx}"
        ) 
        for idx in gpus.split("-")
    ]

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

if __name__ == "__main__": 
    mp.set_start_method("spawn", force=True)
    main()

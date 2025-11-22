import os
import sys
import logging
import argparse
import warnings

import torch.multiprocessing as mp

from distutils.util import strtobool

sys.path.append(os.getcwd())

from main.library.utils import check_assets
from main.inference.extracting.rms import run_rms_extraction
from main.inference.extracting.feature import run_pitch_extraction
from main.app.variables import config, logger, translations, configs
from main.inference.extracting.embedding import run_embedding_extraction
from main.inference.extracting.preparing_files import generate_config, generate_filelist

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

    return parser.parse_args()

def main():
    args = parse_arguments()

    f0_method, hop_length, num_processes, gpus, version, pitch_guidance, sample_rate, embedder_model, f0_onnx, embedders_mode, f0_autotune, f0_autotune_strength, rms_extract, alpha = args.f0_method, args.hop_length, args.cpu_cores, args.gpu, args.rvc_version, args.pitch_guidance, args.sample_rate, args.embedder_model, args.f0_onnx, args.embedders_mode, args.f0_autotune, args.f0_autotune_strength, args.rms_extract, args.alpha
    check_assets(f0_method, embedder_model, f0_onnx=f0_onnx, embedders_mode=embedders_mode)
    exp_dir = os.path.join(configs["logs_path"], args.model_name)

    num_processes = max(1, num_processes)
    devices = ["cpu"] if gpus == "-" else [(f"cuda:{idx}" if config.device.startswith("cuda") else f"{'ocl' if config.device.startswith('ocl') else 'privateuseone'}:{idx}") for idx in gpus.split("-")]

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
        translations["f0_onnx_mode"]: f0_onnx, 
        translations["embed_mode"]: embedders_mode, 
        translations["train&energy"]: rms_extract,
        translations["alpha_label"]: alpha
    }

    for key, value in log_data.items():
        logger.debug(f"{key}: {value}")

    pid_path = os.path.join(exp_dir, "extract_pid.txt")
    with open(pid_path, "w") as pid_file:
        pid_file.write(str(os.getpid()))
    
    try:
        run_pitch_extraction(exp_dir, f0_method, hop_length, num_processes, devices, f0_onnx, config.is_half, f0_autotune, f0_autotune_strength, alpha)
        run_embedding_extraction(exp_dir, version, num_processes, devices, embedder_model, embedders_mode, config.is_half)
        run_rms_extraction(exp_dir, num_processes, devices, rms_extract)
        generate_config(version, sample_rate, exp_dir)
        generate_filelist(pitch_guidance, exp_dir, version, sample_rate, embedders_mode, embedder_model, rms_extract)
    except Exception as e:
        logger.error(f"{translations['extract_error']}: {e}")

    if os.path.exists(pid_path): os.remove(pid_path)
    logger.info(f"{translations['extract_success']} {args.model_name}.")

if __name__ == "__main__": 
    mp.set_start_method("spawn", force=True)
    main()
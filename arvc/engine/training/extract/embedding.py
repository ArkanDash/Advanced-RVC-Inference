import os
import gc
import sys
import time
import threading
import tqdm
import torch
import traceback
import concurrent.futures

import numpy as np

sys.path.append(os.getcwd())

from arvc.utils.variables import logger, translations, config
from arvc.engine.training.extract.setup_path import setup_paths
from arvc.engine.models.utils import load_audio, load_embedders_model, extract_features

def process_file_embedding(files, embedder_model, embedders_mode, device, version, is_half, threads):
    model = load_embedders_model(embedder_model, embedders_mode)
    if isinstance(model, torch.nn.Module): model = model.to(device).to(torch.float16 if is_half else torch.float32).eval()

    failed = 0
    saved = 0
    lock = threading.Lock()

    def worker(file_info):
        nonlocal failed, saved
        try:
            file, out_path = file_info
            out_file_path = os.path.join(out_path, os.path.splitext(os.path.basename(file))[0] + ".npy") if os.path.isdir(out_path) else out_path

            if os.path.exists(out_file_path):
                with lock:
                    saved += 1
                return
            feats = torch.from_numpy(load_audio(file, 16000)).to(device).to(torch.float16 if is_half else torch.float32)

            with torch.no_grad():
                feats = extract_features(model, feats.view(1, -1), version, device)

            feats = feats.squeeze(0).float().cpu().numpy()
            if not np.isnan(feats).any():
                np.save(out_file_path, feats, allow_pickle=False)
                with lock:
                    saved += 1
            else:
                logger.warning(f"{file} {translations.get('NaN', 'contains NaN')}")
                with lock:
                    failed += 1
        except Exception as e:
            with lock:
                failed += 1
            logger.warning(f"Embedding extraction failed for {file_info[0]}: {e}")

    with tqdm.tqdm(total=len(files), ncols=100, unit="p", leave=True) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            for _ in concurrent.futures.as_completed([executor.submit(worker, f) for f in files]):
                pbar.update(1)

    if failed > 0:
        logger.warning(f"Embedding: {failed}/{len(files)} files failed, {saved} saved")

    return saved

def run_embedding_extraction(exp_dir, version, num_processes, devices, embedder_model, embedders_mode, is_half):
    wav_path, out_path = setup_paths(exp_dir, version)

    logger.info(translations.get("start_extract_hubert", "Starting embedding extraction"))
    num_processes = 1 if (config.device.startswith("ocl") and embedders_mode == "onnx") or config.device.startswith("privateuseone") else num_processes

    # Verify source directory exists and has files
    if not os.path.exists(wav_path):
        raise FileNotFoundError(
            f"Embedding source directory not found: {wav_path}\n"
            f"Make sure preprocessing (Step 1) completed successfully."
        )

    paths = sorted([(os.path.join(wav_path, file), out_path) for file in os.listdir(wav_path) if file.endswith(".wav")])

    if not paths:
        # Fall back: try using sliced_audios directly if sliced_audios_16k is empty
        fallback_path = os.path.join(exp_dir, "sliced_audios")
        if os.path.exists(fallback_path) and os.listdir(fallback_path):
            logger.warning(
                f"No wav files found in {wav_path}, falling back to {fallback_path}. "
                f"This may reduce embedding quality."
            )
            paths = sorted([(os.path.join(fallback_path, file), out_path) for file in os.listdir(fallback_path) if file.endswith(".wav")])

    if not paths:
        raise FileNotFoundError(
            f"No wav files found for embedding extraction.\n"
            f"Checked: {wav_path}\n"
            f"Make sure preprocessing (Step 1) completed successfully."
        )

    logger.info(f"Extracting embeddings from {len(paths)} audio files -> {out_path}")

    start_time = time.time()
    total_saved = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(devices)) as executor:
        futures = [executor.submit(process_file_embedding, paths[i::len(devices)], embedder_model, embedders_mode, devices[i], version, is_half, num_processes // len(devices)) for i in range(len(devices))]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    total_saved += result
            except Exception as e:
                logger.error(f"Embedding extraction subprocess failed: {e}")

    # Verify output
    if os.path.exists(out_path):
        output_files = [f for f in os.listdir(out_path) if f.endswith(".npy")]
    else:
        output_files = []

    if len(output_files) == 0:
        logger.error(
            f"Embedding extraction produced 0 output files! "
            f"Input: {len(paths)} wavs, Output dir: {out_path}\n"
            f"Possible causes: model loading failure, device mismatch, or permission error."
        )
    else:
        logger.info(f"Embedding extraction saved {len(output_files)} feature files to {out_path}")

    gc.collect()
    logger.info(translations.get("extract_hubert_success", "Embedding extraction completed in {elapsed_time} seconds").format(elapsed_time=f"{(time.time() - start_time):.2f}"))

def create_mute_file(version, embedder_model, embedders_mode, is_half):
    start_time = time.time()
    logger.info(translations["start_extract_hubert"])

    process_file_embedding([(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logs", "mute", "sliced_audios_16k", "mute.wav"), os.path.join("assets", "logs", "mute", f"{version}_extracted", f"mute_{embedder_model}.npy"))], embedder_model, embedders_mode, config.device, version, is_half, 1)

    gc.collect()
    logger.info(translations["extract_hubert_success"].format(elapsed_time=f"{(time.time() - start_time):.2f}"))

import os
import sys
import time
import torch
import logging
import librosa
import argparse

import numpy as np
import torch.multiprocessing as mp

from tqdm import tqdm
from scipy import signal
from scipy.io import wavfile
from distutils.util import strtobool
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.getcwd())

from main.library.utils import load_audio
from main.inference.preprocess.slicer2 import Slicer
from main.app.variables import config, logger, translations, configs

for l in ["numba.core.byteflow", "numba.core.ssa", "numba.core.interpreter"]:
    logging.getLogger(l).setLevel(logging.ERROR)

OVERLAP, MAX_AMPLITUDE, ALPHA, HIGH_PASS_CUTOFF, SAMPLE_RATE_16K = 0.3, 0.9, 0.75, 48, 16000

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess", action='store_true')
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="./dataset")
    parser.add_argument("--sample_rate", type=int, required=True)
    parser.add_argument("--cpu_cores", type=int, default=2)
    parser.add_argument("--cut_preprocess", type=str, default="Automatic")
    parser.add_argument("--process_effects", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--clean_dataset", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--clean_strength", type=float, default=0.7)
    parser.add_argument("--chunk_len", type=float, default=3.0, required=False)
    parser.add_argument("--overlap_len", type=float, default=0.3, required=False)
    parser.add_argument("--normalization_mode", type=str, default="none", required=False)

    return parser.parse_args()

class PreProcess:
    def __init__(self, sr, exp_dir, per):
        self.slicer = Slicer(sr=sr, threshold=-42, min_length=1500, min_interval=400, hop_size=15, max_sil_kept=500)
        self.sr = sr
        self.b_high, self.a_high = signal.butter(N=5, Wn=HIGH_PASS_CUTOFF, btype="high", fs=self.sr)
        self.per = per
        self.exp_dir = exp_dir
        self.device = "cpu"
        self.gt_wavs_dir = os.path.join(exp_dir, "sliced_audios")
        self.wavs16k_dir = os.path.join(exp_dir, "sliced_audios_16k")
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

    def _normalize_audio(self, audio):
        tmp_max = np.abs(audio).max()
        if tmp_max > 2.5: return None
        return (audio / tmp_max * (MAX_AMPLITUDE * ALPHA)) + (1 - ALPHA) * audio

    def process_audio_segment(self, normalized_audio, sid, idx0, idx1, normalization_mode):
        if normalized_audio is None:
            logger.debug(f"{sid}-{idx0}-{idx1}-filtered")
            return
        
        if normalization_mode == "post": normalized_audio = self._normalize_audio(normalized_audio)
        
        wavfile.write(os.path.join(self.gt_wavs_dir, f"{sid}_{idx0}_{idx1}.wav"), self.sr, normalized_audio.astype(np.float32))
        wavfile.write(os.path.join(self.wavs16k_dir, f"{sid}_{idx0}_{idx1}.wav"), SAMPLE_RATE_16K, librosa.resample(normalized_audio, orig_sr=self.sr, target_sr=SAMPLE_RATE_16K, res_type="soxr_vhq").astype(np.float32))

    def simple_cut(self, audio, sid, idx0, chunk_len, overlap_len, normalization_mode):
        chunk_length = int(self.sr * chunk_len)
        overlap_length = int(self.sr * overlap_len)
        i = 0

        while i < len(audio):
            chunk = audio[i : i + chunk_length]
            if normalization_mode == "post": chunk = self._normalize_audio(chunk)

            if len(chunk) == chunk_length:
                wavfile.write(os.path.join(self.gt_wavs_dir, f"{sid}_{idx0}_{i // (chunk_length - overlap_length)}.wav"), self.sr, chunk.astype(np.float32))
                wavfile.write(os.path.join(self.wavs16k_dir, f"{sid}_{idx0}_{i // (chunk_length - overlap_length)}.wav"), SAMPLE_RATE_16K, librosa.resample(chunk, orig_sr=self.sr, target_sr=SAMPLE_RATE_16K, res_type="soxr_vhq").astype(np.float32))

            i += chunk_length - overlap_length

    def process_audio(self, path, idx0, sid, cut_preprocess, process_effects, clean_dataset, clean_strength, chunk_len, overlap_len, normalization_mode):
        try:
            audio = load_audio(path, self.sr)

            if process_effects: audio = signal.lfilter(self.b_high, self.a_high, audio)
            if normalization_mode == "pre": audio = self._normalize_audio(audio)

            if clean_dataset: 
                if not hasattr(self, "tg"): 
                    from main.tools.noisereduce import TorchGate
                    self.tg = TorchGate(self.sr, prop_decrease=clean_strength).to(self.device)
                audio = self.tg(torch.from_numpy(audio).unsqueeze(0).to(self.device).float()).squeeze(0).cpu().detach().numpy()

            if cut_preprocess == "Skip":
                self.process_audio_segment(
                    audio,
                    sid,
                    idx0,
                    0,
                    normalization_mode,
                )
            elif cut_preprocess == "Simple":
                self.simple_cut(
                    audio,
                    sid,
                    idx0,
                    chunk_len,
                    overlap_len,
                    normalization_mode,
                )
            elif cut_preprocess == "Automatic":
                idx1 = 0
                for audio_segment in self.slicer.slice(audio):
                    i = 0

                    while 1:
                        start = int(self.sr * (self.per - OVERLAP) * i)
                        i += 1

                        if len(audio_segment[start:]) > (self.per + OVERLAP) * self.sr:
                            self.process_audio_segment(audio_segment[start : start + int(self.per * self.sr)], sid, idx0, idx1, normalization_mode)
                            idx1 += 1
                        else:
                            self.process_audio_segment(audio_segment[start:], sid, idx0, idx1, normalization_mode)
                            idx1 += 1
                            break
        except Exception as e:
            raise RuntimeError(f"{translations['process_audio_error']}: {e}")

def process_file(args):
    pp, file, cut_preprocess, process_effects, clean_dataset, clean_strength, chunk_len, overlap_len, normalization_mode = args
    file_path, idx0, sid = file
    pp.process_audio(file_path, idx0, sid, cut_preprocess, process_effects, clean_dataset, clean_strength, chunk_len, overlap_len, normalization_mode)

def preprocess_training_set(input_root, sr, num_processes, exp_dir, per, cut_preprocess, process_effects, clean_dataset, clean_strength, chunk_len, overlap_len, normalization_mode):
    start_time = time.time()
    pp = PreProcess(sr, exp_dir, per)
    logger.info(translations["start_preprocess"].format(num_processes=num_processes))
    files = []
    idx = 0

    for root, _, filenames in os.walk(input_root):
        try:
            sid = 0 if root == input_root else int(os.path.basename(root))
            for f in filenames:
                if f.lower().endswith(("wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3")):
                    files.append((os.path.join(root, f), idx, sid))
                    idx += 1
        except ValueError:
            raise ValueError(f"{translations['not_integer']} '{os.path.basename(root)}'.")

    with tqdm(total=len(files), ncols=100, unit="f") as pbar:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(process_file, (pp, file, cut_preprocess, process_effects, clean_dataset, clean_strength, chunk_len, overlap_len, normalization_mode)) for file in files]
            for future in as_completed(futures):
                try:
                    future.result() 
                except Exception as e:
                    raise RuntimeError(f"{translations['process_error']}: {e}")
                pbar.update(1)

    elapsed_time = time.time() - start_time
    logger.info(translations["preprocess_success"].format(elapsed_time=f"{elapsed_time:.2f}"))

def main():
    args = parse_arguments()
    experiment_directory = os.path.join(configs["logs_path"], args.model_name)

    num_processes = args.cpu_cores
    num_processes = 2 if num_processes is None else int(num_processes)

    dataset, sample_rate, cut_preprocess, preprocess_effects, clean_dataset, clean_strength, chunk_len, overlap_len, normalization_mode = args.dataset_path, args.sample_rate, args.cut_preprocess, args.process_effects, args.clean_dataset, args.clean_strength, args.chunk_len, args.overlap_len, args.normalization_mode
    os.makedirs(experiment_directory, exist_ok=True)

    log_data = {translations['modelname']: args.model_name, translations['export_process']: experiment_directory, translations['dataset_folder']: dataset, translations['pretrain_sr']: sample_rate, translations['cpu_core']: num_processes, translations['split_audio']: cut_preprocess, translations['preprocess_effect']: preprocess_effects, translations['clear_audio']: clean_dataset}
    if clean_dataset: log_data[translations['clean_strength']] = clean_strength

    for key, value in log_data.items():
        logger.debug(f"{key}: {value}")

    pid_path = os.path.join(experiment_directory, "preprocess_pid.txt")
    with open(pid_path, "w") as pid_file:
        pid_file.write(str(os.getpid()))
    
    try:
        preprocess_training_set(dataset, sample_rate, num_processes, experiment_directory, config.per_preprocess, cut_preprocess, preprocess_effects, clean_dataset, clean_strength, chunk_len, overlap_len, normalization_mode)
    except Exception as e:
        logger.error(f"{translations['process_audio_error']} {e}")
        import traceback
        logger.debug(traceback.format_exc())
        
    if os.path.exists(pid_path): os.remove(pid_path)
    logger.info(f"{translations['preprocess_model_success']} {args.model_name}")

if __name__ == "__main__": 
    mp.set_start_method("spawn", force=True)
    main()
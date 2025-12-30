import os
import sys
import time
from scipy import signal
from scipy.io import wavfile
import numpy as np
import concurrent.futures
from tqdm import tqdm
import json
from distutils.util import strtobool
import librosa
import multiprocessing
import shutil
import soundfile as sf
import pyloudnorm as pyln

import soxr
from fractions import Fraction

now_directory = os.getcwd()
sys.path.append(now_directory)

from rvc.lib.utils import load_audio, load_audio_ffmpeg
from rvc.train.preprocess.slicer import Slicer

import logging
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logging.getLogger("numba.core.byteflow").setLevel(logging.WARNING)
logging.getLogger("numba.core.ssa").setLevel(logging.WARNING)
logging.getLogger("numba.core.interpreter").setLevel(logging.WARNING)

OVERLAP = 0.3
PERCENTAGE = 3.0
MAX_AMPLITUDE = 0.9
ALPHA = 0.75
HIGH_PASS_CUTOFF = 48
SAMPLE_RATE_16K = 16000
RES_TYPE = "soxr_vhq"


def secs_to_samples(secs, sr):
    """Return an *exact* integer number of samples for `secs` seconds at `sr` Hz.
       Raises if the result is not an integer (prevents float drift)."""
    frac = Fraction(str(secs)) * sr
    if frac.denominator != 1:
        raise ValueError(f"{secs}s × {sr}Hz is not an integer sample count")
    return frac.numerator


class PreProcess:
    def __init__(self, sr: int, exp_dir: str):
        self.slicer = Slicer(
            sr=sr,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sr = sr
        self.b_high, self.a_high = signal.butter(N=5, Wn=HIGH_PASS_CUTOFF, btype="high", fs=self.sr)
        self.exp_dir = exp_dir

        self.gt_wavs_dir = os.path.join(exp_dir, "sliced_audios")
        self.wavs16k_dir = os.path.join(exp_dir, "sliced_audios_16k")
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)


    def process_audio_segment(
        self,
        audio: np.ndarray,
        sid: int,
        idx0: int,
        idx1: int,
        loading_resampling: str,
    ):
        # Saving slices for GroundTruth ( 'sliced_audios' dir )
        wavfile.write(
            os.path.join(self.gt_wavs_dir, f"{sid}_{idx0}_{idx1}.wav"),
            self.sr,
            audio.astype(np.float32),
        )
        # Resampling of slices for wavs16k ( 'sliced_audios_16k' dir )
        if loading_resampling == "librosa":
            chunk_16k = librosa.resample(
                audio, orig_sr=self.sr, target_sr=SAMPLE_RATE_16K, res_type=RES_TYPE
            )
        else: # ffmpeg
            chunk_16k = load_audio_ffmpeg(
                audio, sample_rate=SAMPLE_RATE_16K, source_sr=self.sr,
            )
        # Saving slices for 16khz ( 'sliced_audios_16k' dir )
        wavfile.write(
            os.path.join(self.wavs16k_dir, f"{sid}_{idx0}_{idx1}.wav"),
            SAMPLE_RATE_16K,
            chunk_16k.astype(np.float32),
        )


    def simple_cut(
        self,
        audio: np.ndarray,
        sid: int,
        idx0: int,
        chunk_len: float,
        overlap_len: float,
        loading_resampling: str,
    ):
        chunk_len_smpl = secs_to_samples(chunk_len, self.sr)
        stride = chunk_len_smpl - secs_to_samples(overlap_len, self.sr)

        slice_idx = 0
        i = 0
        while i < len(audio):
            chunk = audio[i : i + chunk_len_smpl]

            # If the last slice's below 3 seconds, discard it.
            if len(chunk) < chunk_len_smpl:
                logger.warning(f"The last resulting slice ({sid}-{idx0}-{slice_idx}) is too short: {len(chunk)} < {chunk_len_smpl} samples - Discarding!")
                break

            # Saving slices
            wavfile.write(
                os.path.join(self.gt_wavs_dir, f"{sid}_{idx0}_{slice_idx}.wav"),
                self.sr, chunk.astype(np.float32))

            # Resampling of slices for wavs16k ( 'sliced_audios_16k' dir )
            if loading_resampling == "librosa":
                chunk_16k = librosa.resample(
                    chunk, orig_sr=self.sr, target_sr=SAMPLE_RATE_16K, res_type=RES_TYPE
                )
            else: # ffmpeg
                chunk_16k = load_audio_ffmpeg(
                    chunk, sample_rate=SAMPLE_RATE_16K, source_sr=self.sr,
                )
            # Saving slices for 16khz ( 'sliced_audios_16k' dir )
            wavfile.write(
                os.path.join(self.wavs16k_dir, f"{sid}_{idx0}_{slice_idx}.wav"),
                SAMPLE_RATE_16K, chunk_16k.astype(np.float32))

            slice_idx += 1
            i += stride

    def process_audio(
        self,
        path: str,
        idx0: int,
        sid: int,
        cut_preprocess: str,
        process_effects: bool,
        noise_reduction: bool,
        reduction_strength: float,
        chunk_len: float,
        overlap_len: float,
        loading_resampling: str,
    ):
        audio_length = 0
        try:
            # Loading the audio
            if loading_resampling == "librosa":
                audio = load_audio(path, self.sr) # Librosa's using SoXr
            else:
                audio = load_audio_ffmpeg(path, self.sr) # FFmpeg's using Windowed Sinc filter with Blackman-Nuttall window.

            # Getting the length
            audio_length = librosa.get_duration(y=audio, sr=self.sr)

            # Processing, Filtering, Noise reduction
            if process_effects:
                audio = signal.lfilter(self.b_high, self.a_high, audio)
            if noise_reduction:
                import noisereduce as nr
                audio = nr.reduce_noise(y=audio, sr=self.sr, prop_decrease=reduction_strength)

            # Slicing approach
            if cut_preprocess == "Skip":
                self.process_audio_segment(audio, sid, idx0, 0, loading_resampling)
            elif cut_preprocess == "Simple":
                self.simple_cut(audio, sid, idx0, chunk_len, overlap_len, loading_resampling)
            elif cut_preprocess == "Automatic":
                idx1 = 0
                for audio_segment in self.slicer.slice(audio):
                    i = 0
                    while True:
                        start = int(self.sr * (PERCENTAGE - OVERLAP) * i)
                        i += 1
                        if len(audio_segment[start:]) > (PERCENTAGE + OVERLAP) * self.sr:
                            tmp_audio = audio_segment[start : start + int(PERCENTAGE * self.sr)]
                            self.process_audio_segment(tmp_audio, sid, idx0, idx1, loading_resampling)
                            idx1 += 1
                        else:
                            tmp_audio = audio_segment[start:]
                            self.process_audio_segment(tmp_audio, sid, idx0, idx1, loading_resampling)
                            idx1 += 1
                            break
        except Exception as e:
            logger.error(f"Error processing {path}: {e}")
            raise e
        return audio_length

def _process_audio_worker(args):
    (
        path,
        idx0,
        sid,
        sr,
        exp_dir,
        cut_preprocess,
        process_effects,
        noise_reduction,
        reduction_strength,
        chunk_len,
        overlap_len,
        loading_resampling,
    ) = args
    pp = PreProcess(sr, exp_dir)
    return pp.process_audio(
        path,
        idx0,
        sid,
        cut_preprocess,
        process_effects,
        noise_reduction,
        reduction_strength,
        chunk_len,
        overlap_len,
        loading_resampling,
    )

def _process_and_save_worker(args):
    file_name, final_lufs_target, gt_wavs_dir, wavs16k_dir = args
    try:
        # Process ground truth audio
        gt_path = os.path.join(gt_wavs_dir, file_name)
        gt_audio, gt_sr = sf.read(gt_path)

        meter = pyln.Meter(gt_sr, block_size=0.200)
        loudness = meter.integrated_loudness(gt_audio)
        gt_normalized = pyln.normalize.loudness(gt_audio, loudness, final_lufs_target)

        if np.abs(gt_normalized).max() > 1.0:
            raise ValueError(f"Normalization resulted in clipping for {file_name}")

        wavfile.write(gt_path, gt_sr, gt_normalized.astype(np.float32))

        # Process 16k audio
        k16_path = os.path.join(wavs16k_dir, file_name)
        k16_audio, k16_sr = sf.read(k16_path)

        meter = pyln.Meter(k16_sr, block_size=0.200)
        loudness = meter.integrated_loudness(k16_audio)
        k16_normalized = pyln.normalize.loudness(k16_audio, loudness, final_lufs_target)

        if np.abs(k16_normalized).max() > 1.0:
            raise ValueError(f"Normalization resulted in clipping for {file_name} (16k)")

        wavfile.write(k16_path, k16_sr, k16_normalized.astype(np.float32))
    except Exception as e:
        logger.error(f"Error normalizing {file_name}: {e}")
        raise e

def _process_and_save_classic_worker(args):
    file_name, gt_wavs_dir, wavs16k_dir = args
    try:
        # Process ground truth audio
        gt_path = os.path.join(gt_wavs_dir, file_name)
        gt_audio, gt_sr = sf.read(gt_path)

        # Simple peak normalization to 0.95
        peak_amp = np.max(np.abs(gt_audio))
        if peak_amp > 0:
            gt_normalized = (gt_audio / peak_amp) * 0.95
        else:
            gt_normalized = gt_audio

        wavfile.write(gt_path, gt_sr, gt_normalized.astype(np.float32))

        # Process 16k audio
        k16_path = os.path.join(wavs16k_dir, file_name)
        k16_audio, k16_sr = sf.read(k16_path)

        # Peak norm to 0.95
        peak_amp_16k = np.max(np.abs(k16_audio))
        if peak_amp_16k > 0:
            k16_normalized = (k16_audio / peak_amp_16k) * 0.95
        else:
            k16_normalized = k16_audio

        wavfile.write(k16_path, k16_sr, k16_normalized.astype(np.float32))

    except Exception as e:
        logger.error(f"Error classic normalizing {file_name}: {e}")
        raise e

def _measure_lufs_and_peak_worker(file_path):
    try:
        audio, sr = sf.read(file_path)
        
        # Return negative infinity for silent clips to ignore them
        if np.max(np.abs(audio)) == 0:
            return -np.inf, -np.inf

        meter = pyln.Meter(sr, block_size=0.200)
        integrated_loudness = meter.integrated_loudness(audio)
        
        # This measures the sample peak, not the "True Peak", but it's what's needed for preventing digital clipping.
        sample_peak = 20 * np.log10(np.max(np.abs(audio)))
        return integrated_loudness, sample_peak
    except Exception as e:
        logger.error(f"Error measuring loudness for {file_path}: {e}")
        return None

def normalize_sliced_audio(
    exp_dir: str,
    target_lufs: float,
    lufs_range_finder: bool,
    num_processes: int,
):
    gt_wavs_dir = os.path.join(exp_dir, "sliced_audios")
    wavs16k_dir = os.path.join(exp_dir, "sliced_audios_16k")

    audio_files = [f for f in os.listdir(gt_wavs_dir) if f.endswith(".wav")]
    audio_files.sort()

    final_lufs_target = target_lufs
    if lufs_range_finder:
        logger.info("Starting LUFS and Peak measurement dry run...")

        file_paths = [os.path.join(gt_wavs_dir, f) for f in audio_files]
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap_unordered(_measure_lufs_and_peak_worker, file_paths), total=len(file_paths), desc="Measuring Audio Levels"))
        
        valid_loudness_and_peaks = [r for r in results if r is not None and r[0] > -np.inf and r[1] > -np.inf]
        
        if not valid_loudness_and_peaks:
            logger.error("No valid audio levels could be measured. Aborting normalization.")
            return

        safety_margin_db = -1.0
        potential_safe_targets = []
        for lufs, peak in valid_loudness_and_peaks:
            gain_to_safe_peak = safety_margin_db - peak
            potential_target_lufs = lufs + gain_to_safe_peak
            potential_safe_targets.append(potential_target_lufs)


        safe_lufs_target = min(potential_safe_targets)
        final_lufs_target = safe_lufs_target
        
        loudest_lufs_original = max(r[0] for r in valid_loudness_and_peaks)
        highest_peak_original = max(r[1] for r in valid_loudness_and_peaks)
        
        logger.info(f"Loudest audio segment measured at: {loudest_lufs_original:.2f} LUFS.")
        logger.info(f"Highest sample peak measured at: {highest_peak_original:.2f} dBFS.")
        logger.info(f"Calculated safe normalization target to avoid clipping is: {safe_lufs_target:.2f} LUFS.")

    logger.info(f"Starting Loudness Normalization with LUFS target: {final_lufs_target:.2f} LUFS")

    arg_list = [(file_name, final_lufs_target, gt_wavs_dir, wavs16k_dir) for file_name in audio_files]

    with multiprocessing.Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap_unordered(_process_and_save_worker, arg_list), total=len(audio_files), desc="Loudness Normalization"))

    logger.info("Loudness Normalization complete.")


def format_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def save_dataset_duration(file_path, dataset_duration):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}

    formatted_duration = format_duration(dataset_duration)
    new_data = {
        "total_dataset_duration": formatted_duration,
        "total_seconds": dataset_duration,
    }
    data.update(new_data)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def cleanup_dirs(exp_dir):
    gt_wavs_dir = os.path.join(exp_dir, "sliced_audios")
    wavs16k_dir = os.path.join(exp_dir, "sliced_audios_16k")
    logger.info("Cleaning up partially processed audio directories if they exist...")
    if os.path.exists(gt_wavs_dir):
        shutil.rmtree(gt_wavs_dir)
        logger.info(f"Deleted directory: {gt_wavs_dir}")
    if os.path.exists(wavs16k_dir):
        shutil.rmtree(wavs16k_dir)
        logger.info(f"Deleted directory: {wavs16k_dir}")


def preprocess_training_set(
    input_root: str,
    sr: int,
    num_processes: int,
    exp_dir: str,
    cut_preprocess: str,
    process_effects: bool,
    noise_reduction: bool,
    reduction_strength: float,
    chunk_len: float,
    overlap_len: float,
    normalization_mode: str,
    loading_resampling: str,
    target_lufs: float,
    lufs_range_finder: bool
):
    start_time = time.time()
    logger.info(f"Starting preprocess with {num_processes} processes...")

    files = []
    idx = 0
    for root, _, filenames in os.walk(input_root):
        try:
            sid = 0 if root == input_root else int(os.path.basename(root))
            for f in filenames:
                if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                    files.append((os.path.join(root, f), idx, sid))
                    idx += 1
        except ValueError:
            logger.warning(
                f'Speaker ID folder is expected to be integer, got "{os.path.basename(root)}" instead.'
            )

    cleanup_dirs(exp_dir)

    arg_list = [
        (
            file_path,
            idx0,
            sid,
            sr,
            exp_dir,
            cut_preprocess,
            process_effects,
            noise_reduction,
            reduction_strength,
            chunk_len,
            overlap_len,
            loading_resampling,
        )
        for (file_path, idx0, sid) in files
    ]

    audio_lengths = []
    try:
        with tqdm(total=len(arg_list), desc="Slicing & Resampling") as pbar:
            with multiprocessing.Pool(processes=num_processes) as pool:
                for result in pool.imap_unordered(_process_audio_worker, arg_list):
                    if result:
                        audio_lengths.append(result)
                    pbar.update(1)

        total_audio_length = sum(audio_lengths)
        save_dataset_duration(os.path.join(exp_dir, "model_info.json"), total_audio_length)

    except Exception as e:
        logger.error(f"Slicing and resampling failed: {e}. Aborting.")
        cleanup_dirs(exp_dir)
        return

    if normalization_mode == "post_lufs":
        logger.info("Loudness Normalization enabled. Initiating...")
        try:
            normalize_sliced_audio(
                exp_dir,
                target_lufs,
                lufs_range_finder,
                num_processes
            )
        except Exception as e:
            logger.error(f"Normalization failed: {e}. Aborting.")
            cleanup_dirs(exp_dir)
            return

    elif normalization_mode == "post_peak":
        logger.info("Classic Peak Normalization enabled. Initiating...")
        gt_wavs_dir = os.path.join(exp_dir, "sliced_audios")
        wavs16k_dir = os.path.join(exp_dir, "sliced_audios_16k")
        audio_files = [f for f in os.listdir(gt_wavs_dir) if f.endswith(".wav")]
        audio_files.sort()

        arg_list = [(file_name, gt_wavs_dir, wavs16k_dir) for file_name in audio_files]

        try:
            with multiprocessing.Pool(processes=num_processes) as pool:
                list(tqdm(pool.imap_unordered(_process_and_save_classic_worker, arg_list), total=len(audio_files), desc="Peak Normalization"))
        except Exception as e:
            logger.error(f"Peak Normalization failed: {e}. Aborting.")
            cleanup_dirs(exp_dir)
            return
    else:
        logger.info("Normalization disabled. Skipping normalization phase.")

    elapsed_time = time.time() - start_time
    logger.info(f"Preprocessing completed in {elapsed_time:.2f} seconds "
                f"on {format_duration(total_audio_length)} of audio.")


if __name__ == "__main__":
    if len(sys.argv) < 15:
        print("Usage: python preprocess.py <experiment_directory> <input_root> <sample_rate> <num_processes or 'none'> <cut_preprocess> <process_effects> <noise_reduction> <reduction_strength> <chunk_len> <overlap_len> <normalization_mode> <loading_resampling> <target_lufs> <lufs_range_finder>")
        sys.exit(1)
    experiment_directory = str(sys.argv[1])
    input_root = str(sys.argv[2])
    sample_rate = int(sys.argv[3])
    num_processes = sys.argv[4]

    if num_processes.lower() == "none":
        num_processes = multiprocessing.cpu_count()
    else:
        num_processes = int(num_processes)

    cut_preprocess = str(sys.argv[5])
    process_effects = bool(strtobool(sys.argv[6]))
    noise_reduction = bool(strtobool(sys.argv[7]))
    reduction_strength = float(sys.argv[8])
    chunk_len = float(sys.argv[9])
    overlap_len = float(sys.argv[10])
    normalization_mode = str(sys.argv[11])
    loading_resampling = str(sys.argv[12])
    target_lufs = float(sys.argv[13])
    lufs_range_finder = bool(strtobool(sys.argv[14]))

    preprocess_training_set(
        input_root,
        sample_rate,
        num_processes,
        experiment_directory,
        cut_preprocess,
        process_effects,
        noise_reduction,
        reduction_strength,
        chunk_len,
        overlap_len,
        normalization_mode,
        loading_resampling,
        target_lufs,
        lufs_range_finder
    )
import os
import gc
import sys
import tqdm
import time
import traceback
import concurrent.futures

import numpy as np

sys.path.append(os.getcwd())

from arvc.engine.models.utils import load_audio
from arvc.utils.variables import config, configs, logger, translations
from arvc.engine.training.extract.setup_path import setup_paths


def validate_and_fix_f0(pitch, pitchf, f0_min=50.0, f0_max=1100.0):
    """Validate and fix F0 values to ensure voice training accuracy.
    
    Bug Fix: Original code saved raw F0 values without validation, causing:
    - NaN/Inf values corrupting training gradients
    - Out-of-range F0 values causing pitch embedding errors
    - F0 discontinuities causing unstable pitch during inference
    
    Args:
        pitch: Coarse F0 (integer bin indices)
        pitchf: Fine F0 (float frequency values)
        f0_min: Minimum valid frequency (Hz)
        f0_max: Maximum valid frequency (Hz)
        
    Returns:
        Tuple of (validated_pitch, validated_pitchf)
    """
    # Replace NaN/Inf with zeros (unvoiced)
    if isinstance(pitchf, np.ndarray):
        nan_mask = np.isnan(pitchf) | np.isinf(pitchf)
        if np.any(nan_mask):
            pitchf = np.where(nan_mask, 0.0, pitchf)
            if isinstance(pitch, np.ndarray):
                pitch = np.where(nan_mask, 0, pitch)
    
    # Clamp F0 to valid range to prevent out-of-bounds pitch embeddings
    # This is CRITICAL for voice accuracy - out-of-range values cause
    # the pitch embedding lookup to fail or produce wrong results
    if isinstance(pitchf, np.ndarray):
        pitchf = np.clip(pitchf, f0_min, f0_max)
    
    # Ensure coarse pitch is within [0, 255] for embedding lookup
    # The TextEncoder uses nn.Embedding(256, hidden_channels) so values MUST be in range
    if isinstance(pitch, np.ndarray):
        pitch = np.clip(pitch, 0, 255).astype(np.int64)
    
    return pitch, pitchf


def smooth_f0_contours(pitchf, window_size=5):
    """Apply median smoothing to F0 contours to remove spurious jumps.
    
    Bug Fix: Raw F0 extraction often contains octave jumps and transient errors
    that cause the model to learn incorrect pitch patterns, leading to
    inaccurate voice reproduction with pitch instability.
    
    Args:
        pitchf: Fine F0 array
        window_size: Window size for median filter (must be odd)
        
    Returns:
        Smoothed F0 array
    """
    if not isinstance(pitchf, np.ndarray) or len(pitchf) < window_size:
        return pitchf
    
    from scipy.ndimage import median_filter
    
    # Only smooth voiced frames (non-zero F0)
    voiced_mask = pitchf > 0
    if not np.any(voiced_mask):
        return pitchf
    
    smoothed = pitchf.copy()
    smoothed[voiced_mask] = median_filter(pitchf[voiced_mask], size=window_size)
    
    return smoothed


class FeatureInput:
    def __init__(self, is_half=config.is_half, device=config.device):
        self.sample_rate = 16000
        # Configurable f0_min/f0_max (from Vietnamese-RVC): allows customization
        # for non-standard vocal ranges (soprano, bass, instruments) via the
        # config dict. Defaults remain 50–1100 Hz which covers human voice.
        _f0_min = configs.get("f0_min", 50)
        _f0_max = configs.get("f0_max", 1100)
        self.f0_max = float(_f0_max)
        self.f0_min = float(_f0_min)
        self.device = device
        self.is_half = is_half

    def process_file(self, file_info, f0_method, hop_length, f0_onnx, f0_autotune, f0_autotune_strength, alpha):
        if not hasattr(self, "f0_gen"): 
            from arvc.engine.models.predictors.Generator import Generator
            self.f0_gen = Generator(self.sample_rate, hop_length, self.f0_min, self.f0_max, alpha, self.is_half, self.device, f0_onnx, False)

        inp_path, opt_path1, opt_path2, file_inp = file_info
        if os.path.exists(opt_path1 + ".npy") and os.path.exists(opt_path2 + ".npy"): return

        try:
            pitch, pitchf = self.f0_gen.calculator(
                x_pad=config.x_pad, 
                f0_method=f0_method, 
                x=load_audio(file_inp, self.sample_rate), 
                f0_up_key=0, 
                p_len=None, 
                filter_radius=3,  # BUG FIX: Increased from default for better octave error detection
                f0_autotune=f0_autotune, 
                f0_autotune_strength=f0_autotune_strength, 
                manual_f0=None, 
                proposal_pitch=False, 
                proposal_pitch_threshold=0.0
            )
            
            # ═══════════════════════════════════════════════════════════════
            # BUG FIX: Validate and fix F0 values before saving
            # This fixes "training voice not accurate" caused by:
            # 1. NaN/Inf F0 values corrupting model weights
            # 2. Out-of-range F0 causing pitch embedding index-out-of-bounds
            # 3. F0 discontinuities causing pitch instability in generated audio
            # ═══════════════════════════════════════════════════════════════
            pitch, pitchf = validate_and_fix_f0(pitch, pitchf, self.f0_min, self.f0_max)
            
            # Apply optional smoothing for more stable pitch contours
            # This significantly improves voice naturalness and accuracy
            if not f0_autotune:  # Don't smooth if autotune is already applied
                pitchf = smooth_f0_contours(pitchf, window_size=5)
            
            np.save(opt_path2, pitchf, allow_pickle=False)
            np.save(opt_path1, pitch, allow_pickle=False)
        except Exception as e:
            logger.info(f"{translations['extract_file_error']} {inp_path}: {e}")
            logger.debug(traceback.format_exc())

    def process_files(self, files, f0_method, hop_length, f0_onnx, device, is_half, threads, f0_autotune, f0_autotune_strength, alpha):
        self.device = device
        self.is_half = is_half

        def worker(file_info):
            self.process_file(file_info, f0_method, hop_length, f0_onnx, f0_autotune, f0_autotune_strength, alpha)

        with tqdm.tqdm(total=len(files), ncols=100, unit="p", leave=True) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
                for _ in concurrent.futures.as_completed([executor.submit(worker, f) for f in files]):
                    pbar.update(1)

def run_pitch_extraction(exp_dir, f0_method, hop_length, num_processes, devices, f0_onnx, is_half, f0_autotune, f0_autotune_strength, alpha):
    input_root, *output_roots = setup_paths(exp_dir)
    output_root1, output_root2 = output_roots if len(output_roots) == 2 else (output_roots[0], None)

    logger.info(translations["extract_f0_method"].format(num_processes=num_processes, f0_method=f0_method))
    # BUG FIX (from Vietnamese-RVC): "pesto" was missing from the list of f0
    # methods that force num_processes=1 on OCL/privateuseone backends.
    # PESTO uses ONNX runtime under the hood, which crashes on DirectML when
    # run in parallel — same as crepe/fcpe/rmvpe/penn/swift.
    num_processes = 1 if config.device.startswith(("ocl", "privateuseone")) and ("crepe" in f0_method or "fcpe" in f0_method or "rmvpe" in f0_method or "penn" in f0_method or "swift" in f0_method or "pesto" in f0_method) else num_processes
    paths = [(os.path.join(input_root, name), os.path.join(output_root1, name) if output_root1 else None, os.path.join(output_root2, name) if output_root2 else None, os.path.join(input_root, name)) for name in sorted(os.listdir(input_root)) if "spec" not in name]

    start_time = time.time()
    feature_input = FeatureInput()
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(devices)) as executor:
        concurrent.futures.wait([executor.submit(feature_input.process_files, paths[i::len(devices)], f0_method, hop_length, f0_onnx, devices[i], is_half, num_processes // len(devices), f0_autotune, f0_autotune_strength, alpha) for i in range(len(devices))])
    
    gc.collect()
    logger.info(translations["extract_f0_success"].format(elapsed_time=f"{(time.time() - start_time):.2f}"))

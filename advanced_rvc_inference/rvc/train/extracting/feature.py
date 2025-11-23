import os
import gc
import sys
import tqdm
import time
import traceback
import concurrent.futures

import numpy as np

sys.path.append(os.getcwd())

from main.library.utils import load_audio
from main.app.variables import config, logger, translations
from main.inference.extracting.setup_path import setup_paths

class FeatureInput:
    def __init__(self, is_half=config.is_half, device=config.device):
        self.sample_rate = 16000
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.device = device
        self.is_half = is_half

    def process_file(self, file_info, f0_method, hop_length, f0_onnx, f0_autotune, f0_autotune_strength, alpha):
        if not hasattr(self, "f0_gen"): 
            from main.library.predictors.Generator import Generator
            self.f0_gen = Generator(self.sample_rate, hop_length, self.f0_min, self.f0_max, alpha, self.is_half, self.device, f0_onnx, False)

        inp_path, opt_path1, opt_path2, file_inp = file_info
        if os.path.exists(opt_path1 + ".npy") and os.path.exists(opt_path2 + ".npy"): return

        try:
            pitch, pitchf = self.f0_gen.calculator(x_pad=config.x_pad, f0_method=f0_method, x=load_audio(file_inp, self.sample_rate), f0_up_key=0, p_len=None, filter_radius=3, f0_autotune=f0_autotune, f0_autotune_strength=f0_autotune_strength, manual_f0=None, proposal_pitch=False, proposal_pitch_threshold=0.0)
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
    num_processes = 1 if config.device.startswith(("ocl", "privateuseone")) and ("crepe" in f0_method or "fcpe" in f0_method or "rmvpe" in f0_method or "penn" in f0_method or "swift" in f0_method) else num_processes
    paths = [(os.path.join(input_root, name), os.path.join(output_root1, name) if output_root1 else None, os.path.join(output_root2, name) if output_root2 else None, os.path.join(input_root, name)) for name in sorted(os.listdir(input_root)) if "spec" not in name]

    start_time = time.time()
    feature_input = FeatureInput()
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(devices)) as executor:
        concurrent.futures.wait([executor.submit(feature_input.process_files, paths[i::len(devices)], f0_method, hop_length, f0_onnx, devices[i], is_half, num_processes // len(devices), f0_autotune, f0_autotune_strength, alpha) for i in range(len(devices))])
    
    gc.collect()
    logger.info(translations["extract_f0_success"].format(elapsed_time=f"{(time.time() - start_time):.2f}"))
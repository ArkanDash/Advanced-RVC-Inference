import os
import sys
import time
import tqdm
import torch
import librosa
import traceback
import concurrent.futures

import numpy as np
import torch.nn as nn

sys.path.append(os.getcwd())

from main.library.utils import load_audio
from main.app.variables import logger, translations
from main.inference.extracting.setup_path import setup_paths

class RMSEnergyExtractor(nn.Module):
    def __init__(self, frame_length=2048, hop_length=512, center=True, pad_mode = "reflect"):
        super().__init__()
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.center = center
        self.pad_mode = pad_mode

    def forward(self, x):
        assert x.ndim == 2
        assert x.shape[0] == 1

        if str(x.device).startswith(("ocl", "privateuseone")): x = x.contiguous()

        rms = torch.from_numpy(
            librosa.feature.rms(
                y=x.squeeze(0).cpu().numpy(), 
                frame_length=self.frame_length, 
                hop_length=self.hop_length, 
                center=self.center, 
                pad_mode=self.pad_mode
            )
        )

        if str(x.device).startswith(("ocl", "privateuseone")): rms = rms.contiguous()
        return rms.squeeze(-2).to(x.device)
    
def process_file_rms(files, device, threads):
    threads = max(1, threads)

    module = RMSEnergyExtractor(
        frame_length=2048, hop_length=160, center=True, pad_mode = "reflect"
    ).to(device).eval().float()

    def worker(file_info):
        try:
            file, out_path = file_info
            out_file_path = os.path.join(out_path, os.path.basename(file))

            if os.path.exists(out_file_path + ".npy"): return
            feats = torch.from_numpy(load_audio(file, 16000)).unsqueeze(0)

            with torch.no_grad():
                feats = module(feats if device.startswith(("ocl", "privateuseone")) else feats.to(device))
                
            np.save(out_file_path, feats.float().cpu().numpy(), allow_pickle=False)
        except:
            logger.debug(traceback.format_exc())

    with tqdm.tqdm(total=len(files), ncols=100, unit="p", leave=True) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            for _ in concurrent.futures.as_completed([executor.submit(worker, f) for f in files]):
                pbar.update(1)

def run_rms_extraction(exp_dir, num_processes, devices, rms_extract):
    if rms_extract:
        wav_path, out_path = setup_paths(exp_dir, rms_extract=rms_extract)
        paths = sorted([(os.path.join(wav_path, file), out_path) for file in os.listdir(wav_path) if file.endswith(".wav")])

        start_time = time.time()
        logger.info(translations["rms_start_extract"].format(num_processes=num_processes))

        with concurrent.futures.ProcessPoolExecutor(max_workers=len(devices)) as executor:
            concurrent.futures.wait([executor.submit(process_file_rms, paths[i::len(devices)], devices[i], num_processes // len(devices)) for i in range(len(devices))])

        logger.info(translations["rms_success_extract"].format(elapsed_time=f"{(time.time() - start_time):.2f}"))
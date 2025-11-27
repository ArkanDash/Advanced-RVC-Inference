import os
import re
import sys
import math
import torch
import warnings
import numba as nb
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np

from scipy.signal import medfilt
from librosa import yin, pyin, piptrack

sys.path.append(os.getcwd())

from assets.config.variables import config, configs, logger, translations
from ..utils import autotune_f0, proposal_f0_up_key, circular_write
from .CREPE.filter import mean, median
from .WORLD.SWIPE import swipe, stonemask

def post_process(tf0, f0, f0_up_key, manual_x_pad, f0_mel_min, f0_mel_max, manual_f0 = None):
    f0 = np.multiply(f0, pow(2, f0_up_key / 12))

    if manual_f0 is not None:
        replace_f0 = np.interp(
            list(
                range(
                    np.round(
                        (manual_f0[:, 0].max() - manual_f0[:, 0].min()) * tf0 + 1
                    ).astype(np.int16)
                )
            ), 
            manual_f0[:, 0] * 100, 
            manual_f0[:, 1]
        )
        f0[manual_x_pad * tf0 : manual_x_pad * tf0 + len(replace_f0)] = replace_f0[:f0[manual_x_pad * tf0 : manual_x_pad * tf0 + len(replace_f0)].shape[0]]

    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255

    return np.rint(f0_mel).astype(np.int32), f0

@nb.jit(nopython=True)
def post_process(tf0, f0, f0_up_key, manual_x_pad, f0_mel_min, f0_mel_max, manual_f0 = None):
    f0 = np.multiply(f0, pow(2, f0_up_key / 12))

    if manual_f0 is not None:
        replace_f0 = np.interp(
            list(
                range(
                    np.round(
                        (manual_f0[:, 0].max() - manual_f0[:, 0].min()) * tf0 + 1
                    ).astype(np.int16)
                )
            ), 
            manual_f0[:, 0] * 100, 
            manual_f0[:, 1]
        )
        f0[manual_x_pad * tf0 : manual_x_pad * tf0 + len(replace_f0)] = replace_f0[:f0[manual_x_pad * tf0 : manual_x_pad * tf0 + len(replace_f0)].shape[0]]

    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255

    return np.rint(f0_mel).astype(np.int32), f0

def realtime_post_process(f0, pitch, pitchf, f0_up_key = 0, f0_mel_min = 50.0, f0_mel_max = 1100.0):
    f0 *= 2 ** (f0_up_key / 12)

    f0_mel = 1127.0 * torch.log(1.0 + f0 / 700.0)
    f0_mel = torch.clip((f0_mel - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1, 1, 255, out=f0_mel)
    f0_coarse = torch.round(f0_mel, out=f0_mel).long()

    if pitch is not None and pitchf is not None:
        circular_write(f0_coarse, pitch)
        circular_write(f0, pitchf)
    else:
        pitch = f0_coarse
        pitchf = f0

    return pitch.unsqueeze(0), pitchf.unsqueeze(0)

class Generator:
    def __init__(self, sample_rate = 16000, hop_length = 160, f0_min = 50, f0_max = 1100, alpha = 0.5, is_half = False, device = "cpu", f0_onnx_mode = False, del_onnx_model = True, auto_download_models = True):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.is_half = is_half
        self.device = device
        self.providers = getattr(config, 'providers', ['CPUExecutionProvider'])
        self.f0_onnx_mode = f0_onnx_mode
        self.del_onnx_model = del_onnx_model
        self.auto_download_models = auto_download_models
        self.window = 160
        self.batch_size = 512
        self.alpha = alpha
        self.ref_freqs = [49.00, 51.91, 55.00, 58.27, 61.74, 65.41, 69.30, 73.42, 77.78, 82.41, 87.31, 92.50, 98.00, 103.83, 110.00, 116.54, 123.47, 130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.00, 196.00,  207.65, 220.00, 233.08, 246.94, 261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88, 523.25, 554.37, 587.33, 622.25, 659.25, 698.46, 739.99, 783.99, 830.61, 880.00, 932.33, 987.77, 1046.50]
        
        # Import the downloader functions
        from ..utils import ensure_f0_model_available, get_f0_model_path
        
    def _ensure_model_available(self, method, model_suffix=""):
        """Ensure F0 model is available, download if necessary"""
        if not self.auto_download_models:
            return True
            
        # Map method names to internal names
        method_mapping = {
            'crepe': 'crepe',
            'fcpe': 'fcpe', 
            'rmvpe': 'rmvpe',
            'djcm': 'djcm'
        }
        
        internal_method = method_mapping.get(method, method)
        
        # Try to get model path
        model_path = get_f0_model_path(internal_method, self.f0_onnx_mode)
        
        if model_path:
            return True
            
        # If auto-download is enabled, download the model
        if self.auto_download_models:
            print(f"Auto-downloading {internal_method} model...")
            from ..utils import ensure_f0_model_available
            downloaded_path = ensure_f0_model_available(internal_method, auto_download=True)
            return downloaded_path is not None
            
        return False

    def calculator(self, x_pad, f0_method, x, f0_up_key = 0, p_len = None, filter_radius = 3, f0_autotune = False, f0_autotune_strength = 1, manual_f0 = None, proposal_pitch = False, proposal_pitch_threshold = 255.0):
        if p_len is None: p_len = x.shape[0] // self.window
        if "hybrid" in f0_method: logger.debug(translations["hybrid_calc"].format(f0_method=f0_method))

        model = self.get_f0_hybrid if "hybrid" in f0_method else self.compute_f0
        f0 = model(f0_method, x, p_len, filter_radius if filter_radius % 2 != 0 else filter_radius + 1)
        
        if proposal_pitch: 
            up_key = proposal_f0_up_key(f0, proposal_pitch_threshold, configs["limit_f0"])
            logger.debug(translations["proposal_f0"].format(up_key=up_key))
            f0_up_key += up_key

        if f0_autotune: 
            logger.debug(translations["startautotune"])
            f0 = autotune_f0(self.ref_freqs, f0, f0_autotune_strength)

        return post_process(
            self.sample_rate // self.window, 
            f0, 
            f0_up_key, 
            x_pad, 
            1127 * math.log(1 + self.f0_min / 700), 
            1127 * math.log(1 + self.f0_max / 700), 
            manual_f0
        )
    
    def realtime_calculator(self, audio, f0_method, pitch, pitchf, f0_up_key = 0, filter_radius = 3, f0_autotune = False, f0_autotune_strength = 1, proposal_pitch = False, proposal_pitch_threshold = 255.0):
        if isinstance(audio, np.ndarray) and audio.ndim > 1:
            audio = audio.mean(axis=0)
        elif hasattr(audio, 'cpu'):  # torch tensor
            audio = audio.cpu().numpy()
        
        p_len = audio.shape[0] // self.window

        f0 = self.compute_f0(
            f0_method,
            audio,
            p_len,
            filter_radius if filter_radius % 2 != 0 else filter_radius + 1
        )

        if f0_autotune: f0 = autotune_f0(self.ref_freqs, f0, f0_autotune_strength)

        if proposal_pitch: 
            up_key = proposal_f0_up_key(f0, proposal_pitch_threshold, configs["limit_f0"])
            f0_up_key += up_key

        # Convert to torch tensor for realtime_post_process
        f0_tensor = torch.from_numpy(f0.astype(np.float32))
        
        return realtime_post_process(
            f0_tensor, 
            pitch, 
            pitchf,
            f0_up_key, 
            self.f0_min, 
            self.f0_max
        )

    def _resize_f0(self, x, target_len):
        source = np.array(x)
        source[source < 0.001] = np.nan

        return np.nan_to_num(
            np.interp(
                np.arange(0, len(source) * target_len, len(source)) / target_len, 
                np.arange(0, len(source)), 
                source
            )
        )
    
    def compute_f0(self, f0_method, x, p_len, filter_radius):
        if "pm" in f0_method:
            f0 = self.get_f0_pm(x, p_len, filter_radius=filter_radius, mode=f0_method.split("-")[1])
        elif f0_method in ["harvest", "dio"]:
            f0 = self.get_f0_pyworld(x, p_len, filter_radius, f0_method)
        elif "crepe" in f0_method:
            split_f0 = f0_method.split("-")
            f0 = self.get_f0_mangio_crepe(x, p_len, split_f0[2]) if split_f0[0] == "mangio" else self.get_f0_crepe(x, p_len, split_f0[1], filter_radius=filter_radius)
        elif "fcpe" in f0_method:
            f0 = self.get_f0_fcpe(x, p_len, legacy="legacy" in f0_method and "previous" not in f0_method, previous="previous" in f0_method, filter_radius=filter_radius)
        elif "rmvpe" in f0_method:
            f0 = self.get_f0_rmvpe(x, p_len, clipping="clipping" in f0_method, filter_radius=filter_radius)
        elif f0_method in ["yin", "pyin", "piptrack"]:
            f0 = self.get_f0_librosa(x, p_len, mode=f0_method)
        elif "swipe" in f0_method:
            f0 = self.get_f0_swipe(x, p_len, filter_radius=filter_radius)
        elif "penn" in f0_method:
            f0 = self.get_f0_mangio_penn(x, p_len) if f0_method.split("-")[0] == "mangio" else self.get_f0_penn(x, p_len, filter_radius=filter_radius)
        elif "djcm" in f0_method:
            f0 = self.get_f0_djcm(x, p_len, clipping="clipping" in f0_method, filter_radius=filter_radius)
        elif "pesto" in f0_method:
            f0 = self.get_f0_pesto(x, p_len)
        elif "swift" in f0_method:
            f0 = self.get_f0_swift(x, p_len, filter_radius=filter_radius)
        else:
            raise ValueError(translations["option_not_valid"])
        
        if isinstance(f0, tuple): f0 = f0[0]
        if "medfilt" in f0_method: f0 = medfilt(f0, kernel_size=5)

        return f0
    
    def get_f0_hybrid(self, methods_str, x, p_len, filter_radius):
        methods_str = re.search(r"hybrid\[(.+)\]", methods_str)
        if methods_str: methods = [method.strip() for method in methods_str.group(1).split("+")]

        n = len(methods)
        f0_stack = []

        for method in methods:
            f0_stack.append(
                self._resize_f0(
                    self.compute_f0(method, x, p_len, filter_radius),
                    p_len
                )
            )
        
        f0_mix = np.zeros(p_len)

        if not f0_stack: return f0_mix
        if len(f0_stack) == 1: return f0_stack[0]

        weights = (1 - np.abs(np.arange(n) / (n - 1) - (1 - self.alpha))) ** 2
        weights /= weights.sum()

        stacked = np.vstack(f0_stack)
        voiced_mask = np.any(stacked > 0, axis=0)
        f0_mix[voiced_mask] = np.exp(np.nansum(np.log(stacked + 1e-6) * weights[:, None], axis=0)[voiced_mask])

        return f0_mix

    def get_f0_librosa(self, x, p_len, mode="yin"):
        if mode != "piptrack":
            self.if_yin = mode == "yin"
            self.yin = yin if self.if_yin else pyin

            f0 = self.yin(
                x.astype(np.float32), 
                sr=self.sample_rate, 
                fmin=self.f0_min, 
                fmax=self.f0_max, 
                hop_length=self.hop_length
            )

            if not self.if_yin: f0 = f0[0]
        else:
            pitches, magnitudes = piptrack(
                y=x.astype(np.float32),
                sr=self.sample_rate,
                fmin=self.f0_min,
                fmax=self.f0_max,
                hop_length=self.hop_length,
            )

            max_indexes = np.argmax(magnitudes, axis=0)
            f0 = pitches[max_indexes, range(magnitudes.shape[1])]

        return self._resize_f0(f0, p_len)
    
    def get_f0_crepe(self, x, p_len, model="full", filter_radius=3):
        if not hasattr(self, "crepe"):
            # Ensure model is available before loading
            if not self._ensure_model_available("crepe"):
                raise FileNotFoundError("CREPE model not available. Please enable auto-download or manually download the model.")
                
            from .CREPE.CREPE import CREPE

            self.crepe = CREPE(
                os.path.join(
                    configs["predictors_path"], 
                    f"crepe_{model}.{'onnx' if self.f0_onnx_mode else 'pth'}"
                ), 
                model_size=model, 
                hop_length=self.window, 
                batch_size=self.batch_size, 
                f0_min=self.f0_min, 
                f0_max=self.f0_max, 
                device=self.device, 
                sample_rate=self.sample_rate, 
                providers=self.providers, 
                onnx=self.f0_onnx_mode, 
                return_periodicity=True
            )

        f0, pd = self.crepe.compute_f0(torch.tensor(np.copy(x))[None].float(), pad=True)
        if self.f0_onnx_mode and self.del_onnx_model: del self.crepe.model, self.crepe

        f0, pd = mean(f0, filter_radius), median(pd, filter_radius)
        f0[pd < 0.1] = 0

        return self._resize_f0(f0[0].cpu().numpy(), p_len)
    
    def get_f0_fcpe(self, x, p_len, legacy=False, previous=False, filter_radius=3):
        if not hasattr(self, "fcpe"): 
            # Ensure model is available before loading
            if not self._ensure_model_available("fcpe"):
                raise FileNotFoundError("FCPE model not available. Please enable auto-download or manually download the model.")
                
            from .FCPE.FCPE import FCPE

            self.fcpe = FCPE(
                configs, 
                os.path.join(
                    configs["predictors_path"], 
                    ("fcpe_legacy" if legacy else ("fcpe" if previous else "ddsp_200k")) + (".onnx" if self.f0_onnx_mode else ".pt")
                ), 
                hop_length=self.hop_length, 
                f0_min=self.f0_min, 
                f0_max=self.f0_max, 
                dtype=torch.float32, 
                device=self.device, 
                sample_rate=self.sample_rate, 
                threshold=(filter_radius / 100) if legacy else (filter_radius / 1000 * 2), 
                providers=self.providers, 
                onnx=self.f0_onnx_mode, 
                legacy=legacy
            )
        
        f0 = self.fcpe.compute_f0(x, p_len)
        if self.f0_onnx_mode and self.del_onnx_model: del self.fcpe.fcpe.model, self.fcpe

        return f0
    
    def get_f0_rmvpe(self, x, p_len, clipping=False, filter_radius=3):
        if not hasattr(self, "rmvpe"): 
            # Ensure model is available before loading
            if not self._ensure_model_available("rmvpe"):
                raise FileNotFoundError("RMVPE model not available. Please enable auto-download or manually download the model.")
                
            from .RMVPE.RMVPE import RMVPE

            self.rmvpe = RMVPE(
                os.path.join(
                    configs["predictors_path"], 
                    "rmvpe" + (".onnx" if self.f0_onnx_mode else ".pt")
                ), 
                is_half=self.is_half, 
                device=self.device, 
                onnx=self.f0_onnx_mode, 
                providers=self.providers
            )

        filter_radius = filter_radius / 100
        f0 = self.rmvpe.infer_from_audio_with_pitch(x, thred=filter_radius, f0_min=self.f0_min, f0_max=self.f0_max) if clipping else self.rmvpe.infer_from_audio(x, thred=filter_radius)
        
        if self.f0_onnx_mode and self.del_onnx_model: del self.rmvpe.model, self.rmvpe
        return self._resize_f0(f0, p_len)
    
    def get_f0_pyworld(self, x, p_len, filter_radius, model="harvest"):
        if not hasattr(self, "pw"): 
            from .WORLD.WORLD import PYWORLD

            self.pw = PYWORLD(os.path.join(configs["predictors_path"], "world"), os.path.join(configs["binary_path"], "world.bin"))

        x = x.astype(np.double)
        pw = self.pw.harvest if model == "harvest" else self.pw.dio

        f0, t = pw(
            x, 
            fs=self.sample_rate, 
            f0_ceil=self.f0_max, 
            f0_floor=self.f0_min, 
            frame_period=1000 * self.window / self.sample_rate
        )

        f0 = self.pw.stonemask(
            x, 
            self.sample_rate, 
            t, 
            f0
        )

        if filter_radius > 2 and model == "harvest": f0 = medfilt(f0, filter_radius)
        elif model == "dio":
            for index, pitch in enumerate(f0):
                f0[index] = round(pitch, 1)

        return self._resize_f0(f0, p_len)
    
    def get_f0_swipe(self, x, p_len, filter_radius=3):
        f0, t = swipe(
            x.astype(np.float32), 
            self.sample_rate, 
            f0_floor=self.f0_min, 
            f0_ceil=self.f0_max, 
            frame_period=1000 * self.window / self.sample_rate,
            sTHR=filter_radius / 10
        )

        return self._resize_f0(
            stonemask(
                x, 
                self.sample_rate, 
                t, 
                f0
            ), 
            p_len
        )
    
    def get_f0_djcm(self, x, p_len, clipping=False, filter_radius=3):
        if not hasattr(self, "djcm"): 
            # Ensure model is available before loading
            if not self._ensure_model_available("djcm"):
                raise FileNotFoundError("DJCM model not available. Please enable auto-download or manually download the model.")
                
            from .DJCM.DJCM import DJCM
            
            self.djcm = DJCM(
                os.path.join(
                    configs["predictors_path"], 
                    "djcm" + (".onnx" if self.f0_onnx_mode else ".pt")
                ), 
                is_half=self.is_half, 
                device=self.device, 
                onnx=self.f0_onnx_mode, 
                providers=self.providers
            )

        filter_radius /= 10
        f0 = self.djcm.infer_from_audio_with_pitch(x, thred=filter_radius, f0_min=self.f0_min, f0_max=self.f0_max) if clipping else self.djcm.infer_from_audio(x, thred=filter_radius)
        
        if self.f0_onnx_mode and self.del_onnx_model: del self.djcm.model, self.djcm
        return self._resize_f0(f0, p_len)
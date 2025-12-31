import os
import sys
import torch

import numpy as np
import torch.nn.functional as F

from librosa.filters import mel

sys.path.append(os.getcwd())

class STFT:
    def __init__(self, sr=22050, n_mels=80, n_fft=1024, win_size=1024, hop_length=256, fmin=20, fmax=11025, clip_val=1e-5):
        self.target_sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.clip_val = clip_val
        self.mel_basis = {}
        self.hann_window = {}

    def get_mel(self, y, keyshift=0, speed=1, center=False, train=False):
        n_fft = self.n_fft
        win_size = self.win_size
        hop_length = self.hop_length
        fmax = self.fmax
        factor = 2 ** (keyshift / 12)
        win_size_new = int(np.round(win_size * factor))
        hop_length_new = int(np.round(hop_length * speed))
        mel_basis = self.mel_basis if not train else {}
        hann_window = self.hann_window if not train else {}
        mel_basis_key = str(fmax) + "_" + str(y.device)

        if mel_basis_key not in mel_basis: mel_basis[mel_basis_key] = torch.from_numpy(mel(sr=self.target_sr, n_fft=n_fft, n_mels=self.n_mels, fmin=self.fmin, fmax=fmax)).float().to(y.device)
        keyshift_key = str(keyshift) + "_" + str(y.device)
        if keyshift_key not in hann_window: hann_window[keyshift_key] = torch.hann_window(win_size_new).to(y.device)

        pad_left = (win_size_new - hop_length_new) // 2
        pad_right = max((win_size_new - hop_length_new + 1) // 2, win_size_new - y.size(-1) - pad_left)

        pad = F.pad(y.unsqueeze(1), (pad_left, pad_right), mode="reflect" if pad_right < y.size(-1) else "constant").squeeze(1)
        n_fft = int(np.round(n_fft * factor))

        if str(y.device).startswith(("ocl", "privateuseone")):
            if not hasattr(self, "stft"): 
                from main.library.backends.utils import STFT as _STFT
                self.stft = _STFT(filter_length=n_fft, hop_length=hop_length_new, win_length=win_size_new).to(y.device)
            spec = self.stft.transform(pad, 1e-9)
        else:
            spec = torch.stft(pad, n_fft, hop_length=hop_length_new, win_length=win_size_new, window=hann_window[keyshift_key], center=center, pad_mode="reflect", normalized=False, onesided=True, return_complex=True)
            spec = (spec.real.pow(2) + spec.imag.pow(2) + 1e-9).sqrt()

        if keyshift != 0:
            size = n_fft // 2 + 1
            resize = spec.size(1)
            spec = (F.pad(spec, (0, 0, 0, size - resize)) if resize < size else spec[:, :size, :]) * win_size / win_size_new

        return ((mel_basis[mel_basis_key] @ spec).clamp(min=self.clip_val) * 1).log()
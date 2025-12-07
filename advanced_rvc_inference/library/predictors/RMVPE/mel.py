import os
import sys
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from librosa.filters import mel

sys.path.append(os.getcwd())

class MelSpectrogram(nn.Module):
    def __init__(self, n_mel_channels, sample_rate, win_length, hop_length, n_fft=None, mel_fmin=0, mel_fmax=None, clamp=1e-5):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.hann_window = {}
        mel_basis = mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax, htk=True)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.n_fft = win_length if n_fft is None else n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp

    def forward(self, audio, keyshift=0, speed=1, center=True):
        factor = 2 ** (keyshift / 12)
        win_length_new = int(np.round(self.win_length * factor))
        keyshift_key = str(keyshift) + "_" + str(audio.device)
        if keyshift_key not in self.hann_window: self.hann_window[keyshift_key] = torch.hann_window(win_length_new).to(audio.device)

        n_fft = int(np.round(self.n_fft * factor))
        hop_length = int(np.round(self.hop_length * speed))

        if str(audio.device).startswith(("ocl", "privateuseone")):
            if not hasattr(self, "stft"): 
                from main.library.backends.utils import STFT
                self.stft = STFT(filter_length=n_fft, hop_length=hop_length, win_length=win_length_new).to(audio.device)
            magnitude = self.stft.transform(audio, 1e-9)
        else:
            fft = torch.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length_new, window=self.hann_window[keyshift_key], center=center, return_complex=True)
            magnitude = (fft.real.pow(2) + fft.imag.pow(2)).sqrt()

        if keyshift != 0:
            size = self.n_fft // 2 + 1
            resize = magnitude.size(1)
            if resize < size: magnitude = F.pad(magnitude, (0, 0, 0, size - resize))
            magnitude = magnitude[:, :size, :] * self.win_length / win_length_new

        mel_output = self.mel_basis @ magnitude
        return mel_output.clamp(min=self.clamp).log()
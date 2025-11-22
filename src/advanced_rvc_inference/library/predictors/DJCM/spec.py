import os
import sys
import torch

import numpy as np
import torch.nn as nn

sys.path.append(os.getcwd())

class Spectrogram(nn.Module):
    def __init__(self, hop_length, win_length, n_fft=None, clamp=1e-10):
        super(Spectrogram, self).__init__()
        self.n_fft = win_length if n_fft is None else n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.clamp = clamp
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def forward(self, audio, center=True):
        bs, c, segment_samples = audio.shape
        audio = audio.reshape(bs * c, segment_samples)

        if str(audio.device).startswith(("ocl", "privateuseone")):
            if not hasattr(self, "stft"): 
                from main.library.backends.utils import STFT
                self.stft = STFT(filter_length=self.n_fft, hop_length=self.hop_length, win_length=self.win_length).to(audio.device)
            magnitude = self.stft.transform(audio, 1e-9)
        else:
            fft = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window, center=center, pad_mode="reflect", return_complex=True)
            magnitude = (fft.real.pow(2) + fft.imag.pow(2)).sqrt()

        mag = magnitude.transpose(1, 2).clamp(self.clamp, np.inf)
        mag = mag.reshape(bs, c, mag.shape[1], mag.shape[2])

        return mag
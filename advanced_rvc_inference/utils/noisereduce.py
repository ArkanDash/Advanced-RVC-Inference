import os
import sys
import torch

from torch.nn.functional import conv1d, conv2d

sys.path.append(os.getcwd())

@torch.no_grad()
def temperature_sigmoid(x, x0, temp_coeff):
    return ((x - x0) / temp_coeff).sigmoid()

@torch.no_grad()
def linspace(start, stop, num = 50, endpoint = True, **kwargs):
    return torch.linspace(start, stop, num, **kwargs) if endpoint else torch.linspace(start, stop, num + 1, **kwargs)[:-1]

@torch.no_grad()
def amp_to_db(x, eps=torch.finfo(torch.float32).eps, top_db=40):
    x_db = 20 * (x + eps).log10()
    return x_db.max((x_db.max(-1).values - top_db).unsqueeze(-1))

class TorchGate(torch.nn.Module):
    @torch.no_grad()
    def __init__(self, sr, nonstationary = False, n_std_thresh_stationary = 1.5, n_thresh_nonstationary = 1.3, temp_coeff_nonstationary = 0.1, n_movemean_nonstationary = 20, prop_decrease = 1.0, n_fft = 1024, win_length = None, hop_length = None, freq_mask_smooth_hz = 500, time_mask_smooth_ms = 50):
        super().__init__()
        self.sr = sr
        self.nonstationary = nonstationary
        assert 0.0 <= prop_decrease <= 1.0
        self.prop_decrease = prop_decrease
        self.n_fft = n_fft
        self.win_length = self.n_fft if win_length is None else win_length
        self.hop_length = self.win_length // 4 if hop_length is None else hop_length
        self.n_std_thresh_stationary = n_std_thresh_stationary
        self.temp_coeff_nonstationary = temp_coeff_nonstationary
        self.n_movemean_nonstationary = n_movemean_nonstationary
        self.n_thresh_nonstationary = n_thresh_nonstationary
        self.freq_mask_smooth_hz = freq_mask_smooth_hz
        self.time_mask_smooth_ms = time_mask_smooth_ms
        self.register_buffer("smoothing_filter", self._generate_mask_smoothing_filter())

    @torch.no_grad()
    def _generate_mask_smoothing_filter(self):
        if self.freq_mask_smooth_hz is None and self.time_mask_smooth_ms is None: return None
        n_grad_freq = (1 if self.freq_mask_smooth_hz is None else int(self.freq_mask_smooth_hz / (self.sr / (self.n_fft / 2))))
        if n_grad_freq < 1: raise ValueError

        n_grad_time = (1 if self.time_mask_smooth_ms is None else int(self.time_mask_smooth_ms / ((self.hop_length / self.sr) * 1000)))
        if n_grad_time < 1: raise ValueError
        if n_grad_time == 1 and n_grad_freq == 1: return None

        smoothing_filter = torch.outer(torch.cat([linspace(0, 1, n_grad_freq + 1, endpoint=False), linspace(1, 0, n_grad_freq + 2)])[1:-1], torch.cat([linspace(0, 1, n_grad_time + 1, endpoint=False), linspace(1, 0, n_grad_time + 2)])[1:-1]).unsqueeze(0).unsqueeze(0)
        return smoothing_filter / smoothing_filter.sum()

    @torch.no_grad()
    def _stationary_mask(self, X_db):
        std_freq_noise, mean_freq_noise = torch.std_mean(X_db, dim=-1)
        return X_db > (mean_freq_noise + std_freq_noise * self.n_std_thresh_stationary).unsqueeze(2)

    @torch.no_grad()
    def _nonstationary_mask(self, X_abs):
        X_smoothed = (conv1d(X_abs.reshape(-1, 1, X_abs.shape[-1]), torch.ones(self.n_movemean_nonstationary, dtype=X_abs.dtype, device=X_abs.device).view(1, 1, -1), padding="same").view(X_abs.shape) / self.n_movemean_nonstationary)
        return temperature_sigmoid(((X_abs - X_smoothed) / X_smoothed), self.n_thresh_nonstationary, self.temp_coeff_nonstationary)

    def forward(self, x):
        assert x.ndim == 2
        if x.shape[-1] < self.win_length * 2: raise Exception

        if str(x.device).startswith(("ocl", "privateuseone")):
            if not hasattr(self, "stft"): 
                from main.library.backends.utils import STFT
                self.stft = STFT(filter_length=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, pad_mode="constant").to(x.device)
            X, phase = self.stft.transform(x, eps=1e-9, return_phase=True)
        else:
            X = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, return_complex=True, pad_mode="constant", center=True, window=torch.hann_window(self.win_length).to(x.device))
            
        sig_mask = self.prop_decrease * ((self._nonstationary_mask(X.abs()) if self.nonstationary else self._stationary_mask(amp_to_db(X.abs()))).float() * 1.0 - 1.0) + 1.0
        if self.smoothing_filter is not None: sig_mask = conv2d(sig_mask.unsqueeze(1), self.smoothing_filter.to(sig_mask.dtype), padding="same")
        Y = X * sig_mask.squeeze(1)

        return self.stft.inverse(Y, phase) if hasattr(self, "stft") else torch.istft(Y, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, center=True, window=torch.hann_window(self.win_length).to(Y.device)).to(dtype=x.dtype)
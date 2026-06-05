import os
import sys
import torch
import librosa


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return (x.clamp(min=clip_val) * C).log()

def dynamic_range_decompression_torch(x, C=1):
    return x.exp() / C

def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)

def spectral_de_normalize_torch(magnitudes):
    return dynamic_range_decompression_torch(magnitudes)

stft = None
mel_basis, hann_window = {}, {}

def spectrogram_torch(
    y, 
    n_fft, 
    hop_size, 
    win_size, 
    center=False
):
    global hann_window, stft

    wnsize_dtype_device = str(win_size) + "_" + str(y.dtype) + "_" + str(y.device)
    if wnsize_dtype_device not in hann_window: 
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    pad = torch.nn.functional.pad(
        y.unsqueeze(1), 
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), 
        mode="reflect"
    ).squeeze(1)

    if str(y.device).startswith(("ocl", "privateuseone")):
        if stft is None: 
            from arvc.engine.models.backends.utils import STFT

            stft = STFT(
                filter_length=n_fft, 
                hop_length=hop_size, 
                win_length=n_fft
            ).to(y.device)

        spec = stft.transform(
            pad.to(y.device), 
            eps=1e-6, 
            center=center
        )
    else:
        spec = torch.stft(
            pad, 
            n_fft, 
            hop_length=hop_size, 
            win_length=win_size, 
            window=hann_window[wnsize_dtype_device].to(pad.device), 
            center=center, 
            pad_mode="reflect", 
            normalized=False, 
            onesided=True, 
            return_complex=True
        )

        spec = spec.abs().clamp_min_(1e-6)

    return spec.to(y.device)

def spec_to_mel_torch(
    spec, 
    n_fft, 
    num_mels, 
    sample_rate, 
    fmin, 
    fmax
):
    global mel_basis

    fmax_dtype_device = str(fmax) + "_" + str(spec.dtype) + "_" + str(spec.device)
    if fmax_dtype_device not in mel_basis: 
        mel_basis[fmax_dtype_device] = torch.from_numpy(
            librosa.filters.mel(
                sr=sample_rate, 
                n_fft=n_fft, 
                n_mels=num_mels, 
                fmin=fmin, 
                fmax=fmax
            )
        ).to(dtype=spec.dtype, device=spec.device)
    
    return spectral_normalize_torch(mel_basis[fmax_dtype_device] @ spec)

def mel_spectrogram_torch(
    y, 
    n_fft, 
    num_mels, 
    sample_rate, 
    hop_size, 
    win_size, 
    fmin, 
    fmax, 
    center=False
):
    return spec_to_mel_torch(
        spectrogram_torch(
            y, 
            n_fft, 
            hop_size, 
            win_size, 
            center
        ), 
        n_fft, 
        num_mels, 
        sample_rate, 
        fmin, 
        fmax
    )

def compute_window_length(n_mels: int, sample_rate: int):
    """Compute optimal STFT window length for a given mel band count and sample rate.

    Derived from the relationship between frequency resolution and mel band count.
    Returns the nearest power-of-2 window length for FFT efficiency.
    (From Applio — avoids hardcoded window lengths that may be suboptimal
    at non-standard sample rates.)
    """
    f_min = 0
    f_max = sample_rate / 2
    window_length_seconds = 8 * n_mels / (f_max - f_min)
    window_length = int(window_length_seconds * sample_rate)
    return 2 ** (window_length.bit_length() - 1)


class MultiScaleMelSpectrogramLoss(torch.nn.Module):
    """Multi-scale mel spectrogram loss for improved audio quality.

    Uses 8 mel scales with dynamically computed window lengths and hop lengths,
    following the PolTrain approach. This captures both fine spectral detail
    and broad spectral shape across multiple resolutions, producing
    significantly better audio quality than single-scale mel loss.

    Dynamic window length computation adapts to the sample rate, ensuring
    proper frequency resolution at each scale. The hop length is set to
    sample_rate // 100 for consistent temporal resolution across scales.
    """

    def __init__(
        self,
        sample_rate=24000,
        n_mels=None,
        loss_fn=torch.nn.L1Loss()
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.loss_fn = loss_fn
        self.log_base = torch.tensor(10.0).log()
        self.hann_window = {}
        self.mel_banks = {}

        # 8 scales following PolTrain (adds 480 mel bands vs Applio's 7)
        if n_mels is None:
            n_mels = [5, 10, 20, 40, 80, 160, 320, 480]

        # Dynamic window lengths and hop lengths (PolTrain approach)
        # hop_length = sample_rate // 100 gives consistent temporal resolution
        # across all sample rates, unlike window_length // 4
        self.stft_params = [
            (mel, compute_window_length(mel, sample_rate), sample_rate // 100)
            for mel in n_mels
        ]

    def mel_spectrogram(self, wav, n_mels, window_length, hop_length):
        dtype_device = str(wav.dtype) + "_" + str(wav.device)
        win_dtype_device = str(window_length) + "_" + dtype_device
        mel_dtype_device = str(n_mels) + "_" + dtype_device

        if win_dtype_device not in self.hann_window:
            self.hann_window[win_dtype_device] = torch.hann_window(window_length, device=wav.device, dtype=torch.float32)

        wav = wav.float().squeeze(1)

        if str(wav.device).startswith(("ocl", "privateuseone")):
            stft = torch.stft(
                wav.cpu(),
                n_fft=window_length,
                hop_length=hop_length,
                window=self.hann_window[win_dtype_device].cpu(),
                return_complex=True
            )

            magnitude = torch.sqrt(stft.real.pow(2) + stft.imag.pow(2) + 1e-6).to(wav.device, dtype=torch.float32)
        else:
            stft = torch.stft(
                wav,
                n_fft=window_length,
                hop_length=hop_length,
                window=self.hann_window[win_dtype_device],
                return_complex=True
            )

            magnitude = torch.sqrt(stft.real.pow(2) + stft.imag.pow(2) + 1e-6)

        if mel_dtype_device not in self.mel_banks:
            self.mel_banks[mel_dtype_device] = torch.from_numpy(
                librosa.filters.mel(
                    sr=self.sample_rate,
                    n_mels=n_mels,
                    n_fft=window_length,
                    fmin=0,
                    fmax=None
                )
            ).to(device=wav.device, dtype=torch.float32)

        mel_spec = torch.matmul(self.mel_banks[mel_dtype_device], magnitude)
        return mel_spec

    def forward(self, real, fake):
        loss = 0.0

        for p in self.stft_params:
            real_mels = self.mel_spectrogram(real, *p)
            fake_mels = self.mel_spectrogram(fake, *p)
            real_logmels = real_mels.clamp(min=1e-5).log() / self.log_base
            fake_logmels = fake_mels.clamp(min=1e-5).log() / self.log_base
            loss += self.loss_fn(real_logmels, fake_logmels)

        return loss

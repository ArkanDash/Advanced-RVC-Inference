import os
import sys
import torch
import librosa


# ────────────────────────────────────────────────────────────────────────────
# Pure-torch Mel filter bank (backported from Vietnamese-RVC)
#
# Replaces librosa.filters.mel + torch.from_numpy with a native torch
# implementation. This eliminates the CPU round-trip when computing mel
# spectrograms on GPU, which is significant during training (called every
# iteration for the mel loss) and on non-CUDA devices (DirectML/OpenCL/XPU).
#
# The output is bit-for-bit equivalent to librosa.filters.mel(norm="slaney")
# for the same (sr, n_fft, n_mels, fmin, fmax) arguments.
# ────────────────────────────────────────────────────────────────────────────

def _hz_to_mel(frequencies, htk=False, dtype=torch.float32, device=None):
    """Convert frequencies from Hertz to the Mel scale (pure-torch)."""
    frequencies = torch.as_tensor(frequencies, dtype=dtype, device=device)
    if htk:
        return 2595.0 * (1.0 + frequencies / 700.0).log10()

    f_min = 0.0
    f_sp = 200.0 / 3
    mels = (frequencies - f_min) / f_sp

    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = torch.tensor(6.4, dtype=dtype, device=device).log() / 27.0

    return torch.where(
        frequencies >= min_log_hz,
        min_log_mel + (frequencies / min_log_hz).log() / logstep,
        mels,
    )


def _mel_to_hz(mels, htk=False, dtype=torch.float32, device=None):
    """Convert Mel values back to Hertz (pure-torch)."""
    mels = torch.as_tensor(mels, dtype=dtype, device=device)
    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = torch.tensor(6.4, dtype=dtype, device=device).log() / 27.0

    return torch.where(
        mels >= min_log_mel,
        min_log_hz * (logstep * (mels - min_log_mel)).exp(),
        freqs,
    )


def mel_filter_bank(
    sr,
    n_fft,
    n_mels=128,
    fmin=0.0,
    fmax=None,
    htk=False,
    norm="slaney",
    dtype=torch.float32,
    device=None,
):
    """Generate a Mel filter bank tensor (pure-torch, no librosa).

    Equivalent to librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax, norm="slaney")
    but runs natively on the target device/dtype — no CPU round-trip.

    Args:
        sr: Sample rate.
        n_fft: FFT size.
        n_mels: Number of Mel bands.
        fmin: Minimum frequency (Hz).
        fmax: Maximum frequency (Hz). Defaults to Nyquist.
        htk: Use the HTK Mel scale.
        norm: Filter normalization ("slaney" only; numeric norms not supported).
        dtype: Tensor dtype.
        device: Target device.

    Returns:
        torch.Tensor: Mel filter bank with shape (n_mels, n_fft // 2 + 1).
    """
    if fmax is None:
        fmax = float(sr) / 2
    n_mels = int(n_mels)

    weights = torch.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype, device=device)

    # FFT bin center frequencies
    fftfreqs = torch.fft.rfftfreq(n_fft, d=1.0 / sr, device=device).to(dtype)

    # Mel band edge frequencies
    mel_f = _mel_to_hz(
        torch.linspace(
            _hz_to_mel(fmin, htk=htk, dtype=dtype, device=device),
            _hz_to_mel(fmax, htk=htk, dtype=dtype, device=device),
            n_mels + 2,
            dtype=dtype,
            device=device,
        ),
        htk=htk,
        dtype=dtype,
        device=device,
    )

    # Triangular filter slopes
    fdiff = mel_f.diff()
    ramps = mel_f.unsqueeze(1) - fftfreqs.unsqueeze(0)

    lower = -ramps[:-2] / fdiff[:-1].unsqueeze(1)
    weights = lower.minimum(ramps[2:] / fdiff[1:].unsqueeze(1)).clamp(min=0)

    # Slaney normalization (area of each filter = 1)
    if isinstance(norm, str):
        if norm == "slaney":
            weights *= (2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])).unsqueeze(1)
        else:
            raise ValueError(f"Unsupported normalization: {norm!r}. Use 'slaney'.")
    else:
        raise ValueError(
            "Numeric norm not supported in pure-torch mel_filter_bank. Use 'slaney'."
        )

    return weights


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
        # BACKPORT (Vietnamese-RVC): use pure-torch mel filter bank instead
        # of librosa.filters.mel + torch.from_numpy. Eliminates CPU round-trip
        # on every distinct (fmax, dtype, device) combination.
        try:
            mel_basis[fmax_dtype_device] = mel_filter_bank(
                sr=sample_rate,
                n_fft=n_fft,
                n_mels=num_mels,
                fmin=fmin,
                fmax=fmax if fmax is not None else float(sample_rate) / 2,
                dtype=spec.dtype,
                device=spec.device,
            )
        except Exception:
            # Fallback to librosa if pure-torch version fails (e.g. very old torch)
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

    @property
    def num_scales(self):
        """Number of mel scales used by this loss function.

        BUG FIX: train.py uses getattr(fn_mel_loss, 'num_scales', 3) to normalize
        the mel loss. Without this property, the default of 3 was always returned,
        causing the mel loss to be ~2.67x larger than intended (8 scales / 3 default).
        This skewed the loss balance and made mel loss dominate over KL/FM/adv losses.
        """
        return len(self.stft_params)

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
            # BACKPORT (Vietnamese-RVC): use pure-torch mel filter bank.
            # This eliminates the librosa CPU round-trip that was happening
            # for every distinct (n_mels, dtype, device) combination in the
            # multi-scale mel loss. Since multi-scale calls this with 8
            # different n_mels values, the savings are significant on GPU.
            try:
                self.mel_banks[mel_dtype_device] = mel_filter_bank(
                    sr=self.sample_rate,
                    n_fft=window_length,
                    n_mels=n_mels,
                    fmin=0,
                    fmax=None,
                    dtype=torch.float32,
                    device=wav.device,
                )
            except Exception:
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

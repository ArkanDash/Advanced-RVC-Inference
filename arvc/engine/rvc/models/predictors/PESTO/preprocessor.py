import torch

import numpy as np

from scipy.signal import get_window

class Preprocessor(torch.nn.Module):
    def __init__(self, hop_size, sampling_rate = None, **hcqt_kwargs):
        super(Preprocessor, self).__init__()
        self.hcqt_sr = None
        self.hcqt_kernels = None
        self.hop_size = hop_size
        self.hcqt_kwargs = hcqt_kwargs
        self.to_log = ToLogMagnitude()
        self.register_buffer("_device", torch.zeros(()), persistent=False)
        if sampling_rate is not None:
            self.hcqt_sr = sampling_rate
            self._reset_hcqt_kernels()

    def forward(self, x, sr = None):
        return self.to_log(self.hcqt(x, sr=sr).permute(0, 3, 1, 2, 4))

    def hcqt(self, audio, sr = None) :
        if sr is not None and sr != self.hcqt_sr:
            self.hcqt_sr = sr
            self._reset_hcqt_kernels()

        return self.hcqt_kernels(audio)

    def _reset_hcqt_kernels(self):
        self.hcqt_kernels = HarmonicCQT(sr=self.hcqt_sr, hop_length=int(self.hop_size * self.hcqt_sr / 1000 + 0.5), **self.hcqt_kwargs).to(self._device.device)

class ToLogMagnitude(torch.nn.Module):
    def __init__(self):
        super(ToLogMagnitude, self).__init__()
        self.eps = torch.finfo(torch.float32).eps

    def forward(self, x):
        x = (x[..., 0] ** 2 + x[..., 1] ** 2).sqrt() if x.shape[-1] == 2 else x.abs()
        x.clamp_(min=self.eps).log10_().mul_(20)

        return x
    
class HarmonicCQT(torch.nn.Module):
    def __init__(self, harmonics, sr = 22050, hop_length = 512, fmin = 32.7, fmax = None, bins_per_semitone = 1, n_bins = 84, center_bins = True, gamma = 0, center = True, streaming = False, mirror = 0, max_batch_size = 1):
        super(HarmonicCQT, self).__init__()
        if center_bins: fmin = fmin / 2 ** ((bins_per_semitone - 1) / (24 * bins_per_semitone))
        self.cqt_kernels = torch.nn.ModuleList([CQT(sr=sr, hop_length=hop_length, fmin=h * fmin, fmax=fmax, n_bins=n_bins, bins_per_octave=12*bins_per_semitone, gamma=gamma, center=center, streaming=streaming, mirror=mirror, max_batch_size=max_batch_size, output_format="Complex") for h in harmonics])

    def forward(self, audio_waveforms):
        return torch.stack([cqt(audio_waveforms) for cqt in self.cqt_kernels], dim=1)
    
class BaseCQT(torch.nn.Module):
    def __init__(self, sr=22050, hop_length=512, fmin=32.70, fmax=None, n_bins=84, bins_per_octave=12, gamma=0, filter_scale=1, norm=1, window="hann", center = True, trainable=False, output_format="Magnitude"):
        super(BaseCQT, self).__init__()
        self.trainable = trainable
        self.n_bins = n_bins
        self.hop_length = hop_length
        self.center = center
        self.output_format = output_format
        cqt_kernels, self.kernel_width, lengths, freqs = self.create_cqt_kernels(float(filter_scale) / (2 ** (1 / bins_per_octave) - 1), sr, fmin, n_bins, bins_per_octave, norm, window, fmax, gamma=gamma)
        self.sqrt_lengths = lengths.sqrt_().unsqueeze_(-1)
        self.frequencies = freqs
        self.cqt_kernels = torch.from_numpy(cqt_kernels).unsqueeze(1)
    
    def create_cqt_kernels(self, Q, fs, fmin, n_bins=84, bins_per_octave=12, norm=1, window="hann", fmax=None, topbin_check=True, gamma=0):
        fftLen = 2 ** int(np.ceil(np.log2(np.ceil(Q * fs / fmin))))

        if (fmax != None) and (n_bins == None):
            n_bins = np.ceil(bins_per_octave * np.log2(fmax / fmin))
            freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.double(bins_per_octave))
        elif (fmax == None) and (n_bins != None): freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.double(bins_per_octave))
        else:
            n_bins = np.ceil(bins_per_octave * np.log2(fmax / fmin))
            freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.double(bins_per_octave))

        if np.max(freqs) > fs / 2 and topbin_check == True: raise ValueError

        lengths = np.ceil(Q * fs / (freqs + gamma / (2.0 ** (1.0 / bins_per_octave) - 1.0)))
        fftLen = int(2 ** (np.ceil(np.log2(int(max(lengths))))))
        tempKernel = np.zeros((int(n_bins), int(fftLen)), dtype=np.complex64)

        for k in range(0, int(n_bins)):
            l = lengths[k]
            start = (int(np.ceil(fftLen / 2.0 - l / 2.0)) - 1) if l % 2 == 1 else int(np.ceil(fftLen / 2.0 - l / 2.0))
            N = int(l)

            if isinstance(window, str):
                sig = get_window(window, N, fftbins=True)
            elif isinstance(window, tuple):
                if window[0] == "gaussian":
                    assert window[1] >= 0
                    sig = get_window(("gaussian", np.floor(-N / 2 / np.sqrt(-2 * np.log(10 ** (-window[1] / 20))))), N, fftbins=True)
            else: raise Exception

            sig = sig * np.exp(np.r_[-l // 2: l // 2] * 1j * 2 * np.pi * freqs[k] / fs) / l

            if norm:
                tempKernel[k, start: start + int(l)] = sig / np.linalg.norm(sig, norm)
            else:
                tempKernel[k, start: start + int(l)] = sig

        return tempKernel, fftLen, torch.tensor(lengths).float(), freqs

    @torch.no_grad()
    def init_weights(self):
        self.conv.weight.copy_(torch.cat((self.cqt_kernels.real, -self.cqt_kernels.imag), dim=0))
        self.conv.weight.requires_grad = self.trainable

    def forward(self, x, output_format=None, normalization_type="librosa"):
        output_format = output_format or self.output_format
        x = self.broadcast_dim(x)
        cqt = self.conv(x).view(x.size(0), 2, self.n_bins, -1)

        if normalization_type == "librosa": cqt *= self.sqrt_lengths.to(cqt.device)
        elif normalization_type == "convolutional": pass
        elif normalization_type == "wrap": cqt *= 2
        else: raise ValueError

        if output_format == "Magnitude": return cqt.pow(2).sum(-3).add(1e-8 if self.trainable else 0).sqrt()
        if output_format == "Complex": return cqt.permute(0, 2, 3, 1)

        cqt_real, cqt_imag = cqt.split(self.n_bins, dim=-2)
        if output_format == "Phase": return torch.stack((cqt_imag.atan2(cqt_real).cos(), cqt_imag.atan2(cqt_real).sin()), -1)

        raise ValueError

    def broadcast_dim(self, x):
        if x.dim() == 2: x = x[:, None, :]
        elif x.dim() == 1: x = x[None, None, :]
        elif x.dim() == 3: pass
        else: raise ValueError

        return x

class RegularCQT(BaseCQT):
    def __init__(self, *args, pad_mode="reflect", **kwargs):
        super().__init__(*args, **kwargs)
        padding = self.kernel_width // 2 if self.center else 0
        self.conv = torch.nn.Conv1d(1, 2 * self.n_bins, kernel_size=self.kernel_width, stride=self.hop_length, padding=padding, padding_mode=pad_mode, bias=False)
        self.init_weights()

class StreamingCQT(BaseCQT):
    def __init__(self, *args, mirror = 0, max_batch_size = 1, **kwargs):
        super(StreamingCQT, self).__init__(*args, **kwargs)
        if self.center:
            mirrored_samples = int(mirror * (self.kernel_width - self.hop_length) / 2)
            padding = self.kernel_width - self.hop_length - mirrored_samples
        else:
            mirrored_samples = 0
            padding = 0

        self.conv = CachedConv1d(1, 2 * self.n_bins, kernel_size=self.kernel_width, stride=self.hop_length, padding=padding, mirror=mirrored_samples, max_batch_size=max_batch_size, bias=False)
        self.init_weights()

class CQT:
    regular_only_kwargs = ["pad_mode"]
    streaming_only_kwargs = ["mirror", "max_batch_size"]

    def __new__(cls, *args, **kwargs):
        streaming = kwargs.pop("streaming", False)

        if streaming:
            for kwarg in cls.regular_only_kwargs:
                kwargs.pop(kwarg, None)

            return StreamingCQT(*args, **kwargs)

        for kwarg in cls.streaming_only_kwargs:
            kwargs.pop(kwarg, None)

        return RegularCQT(*args, **kwargs)
    
class CachedConv1d(torch.nn.Conv1d):
    def __init__(self, *args, **kwargs):
        kwargs["padding"] = 0
        super(CachedConv1d, self).__init__(*args, **kwargs)
        padding = kwargs.get("padding", 0)
        max_batch_size = kwargs.pop("max_batch_size", 1)
        mirror = kwargs.pop("mirror", 0)
        mirror_fn = kwargs.pop("mirror_fn", "zeros")
        cumulative_delay = kwargs.pop("cumulative_delay", 0)

        if isinstance(padding, int): r_pad = padding
        elif isinstance(padding, list) or isinstance(padding, tuple):
            r_pad = padding[1]
            padding = padding[0] + padding[1]
        else: raise TypeError

        s = self.stride[0]
        cd = cumulative_delay

        self.cumulative_delay = (r_pad + ((s - ((r_pad + cd) % s)) % s) + cd) // s
        self.cache = CachedPadding1d(padding, max_batch_size=max_batch_size)

        if mirror == 0:
            mirroring_fn = torch.nn.Identity
        elif mirror_fn == "reflection":
            mirroring_fn = torch.nn.ReflectionPad1d
        elif mirror_fn == "zeros":
            mirroring_fn = torch.nn.ZeroPad1d
        elif mirror_fn == "refill":
            mirroring_fn = RefillPad1d
        else:
            mirroring_fn = torch.nn.Identity

        self.mirror = mirroring_fn((0, mirror))

    def forward(self, x):
        return super(CachedConv1d, self).forward(self.mirror(self.cache(x)))
    
class RefillPad1d(torch.nn.Module):
    def __init__(self, padding):
        super(RefillPad1d, self).__init__()
        self.right_padding = padding[1]

    def forward(self, x):
        return torch.cat((x, x[..., -self.right_padding:]), dim=-1)
    
class CachedPadding1d(torch.nn.Module):
    def __init__(self, padding, max_batch_size = 1, crop=False):
        super().__init__()
        self.padding = padding
        self.max_batch_size = max_batch_size
        self.crop = crop
        self.init_cache()

    @torch.jit.unused
    @torch.no_grad()
    def init_cache(self):
        self.register_buffer("pad", torch.zeros(self.max_batch_size, 1, self.padding), persistent=False)

    def forward(self, x):
        bs = x.size(0)
        if self.padding:
            x = torch.cat((self.pad[:bs], x), -1)
            self.pad[:bs].copy_(x[..., -self.padding:])

        return x
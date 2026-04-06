"""
Vocos vocoder implementation.

A Fourier-based vocoder that replaces traditional transposed convolutions
with 1D convolutions + inverse STFT for waveform reconstruction.
Uses Snake activations and supports f0 harmonic injection.

Reference: https://github.com/gemelo-ai/vocos
"""

import math
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.checkpoint import checkpoint


LRELU_SLOPE = 0.1


class SnakeActivation(nn.Module):
    """Snake activation function: x + (1/alpha) * sin^2(alpha * x)."""

    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return x + (1.0 / (self.alpha + 1e-9)) * (x * self.alpha).sin().pow(2)


class SnakeConvBlock(nn.Module):
    """Conv1d block with Snake activation and weight normalization."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.act = SnakeActivation(out_channels)

    def forward(self, x):
        return self.act(self.conv(x))

    def remove_weight_norm(self):
        if hasattr(self.conv, "parametrizations") and "weight" in self.conv.parametrizations:
            parametrize.remove_parametrizations(self.conv, "weight", leave_parametrized=True)
        else:
            remove_weight_norm(self.conv)


class VocosResBlock(nn.Module):
    """Residual block with dilated convolutions and Snake activation."""

    def __init__(self, channels, kernel_size=3, dilations=(1, 3, 5)):
        super().__init__()
        self.layers = nn.ModuleList()

        for d in dilations:
            self.layers.append(
                nn.ModuleList([
                    weight_norm(
                        nn.Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            padding=(kernel_size * d - d) // 2,
                            dilation=d,
                        )
                    ),
                    SnakeActivation(channels),
                    weight_norm(
                        nn.Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            padding=kernel_size // 2,
                        )
                    ),
                ])
            )

    def forward(self, x):
        for conv1, act, conv2 in self.layers:
            residual = x
            x = act(conv1(x))
            x = conv2(x)
            x = x + residual
        return x

    def remove_weight_norm(self):
        for conv1, _, conv2 in self.layers:
            if hasattr(conv1, "parametrizations") and "weight" in conv1.parametrizations:
                parametrize.remove_parametrizations(conv1, "weight", leave_parametrized=True)
            else:
                remove_weight_norm(conv1)

            if hasattr(conv2, "parametrizations") and "weight" in conv2.parametrizations:
                parametrize.remove_parametrizations(conv2, "weight", leave_parametrized=True)
            else:
                remove_weight_norm(conv2)


class SineGen(nn.Module):
    """Sine wave generator for harmonic source excitation."""

    def __init__(self, samp_rate, harmonic_num=0, sine_amp=0.1, noise_std=0.003, voiced_threshold=0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        uv = torch.ones_like(f0) * (f0 > self.voiced_threshold)
        if uv.device.type == "privateuseone":
            uv = uv.float()
        return uv

    def _f02sine(self, f0, upp):
        rad = f0 / self.sampling_rate * torch.arange(1, upp + 1, dtype=f0.dtype, device=f0.device)
        rad += F.pad(
            (torch.fmod(rad[:, :-1, -1:].float() + 0.5, 1.0) - 0.5).cumsum(dim=1).fmod(1.0).to(f0),
            (0, 0, 1, 0),
            mode='constant',
        )
        rad = rad.reshape(f0.shape[0], -1, 1)
        rad *= torch.arange(1, self.dim + 1, dtype=f0.dtype, device=f0.device).reshape(1, 1, -1)
        rand_ini = torch.rand(1, 1, self.dim, device=f0.device)
        rand_ini[..., 0] = 0
        rad += rand_ini
        return (2 * np.pi * rad).sin()

    def forward(self, f0, upp):
        with torch.no_grad():
            f0 = f0.unsqueeze(-1)
            sine_waves = self._f02sine(f0, upp) * self.sine_amp
            uv = F.interpolate(
                self._f02uv(f0).transpose(2, 1), scale_factor=float(upp), mode="nearest"
            ).transpose(2, 1)
            sine_waves = sine_waves * uv + (
                (uv * self.noise_std + (1 - uv) * self.sine_amp / 3) * torch.randn_like(sine_waves)
            )
        return sine_waves


class SourceModuleHnNSF(nn.Module):
    """Source module for harmonic noise generation."""

    def __init__(self, sample_rate, harmonic_num=0, sine_amp=0.1, add_noise_std=0.003, voiced_threshold=0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen(sample_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshold)
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x, upsample_factor=1):
        return self.l_tanh(self.l_linear(self.l_sin_gen(x, upsample_factor).to(dtype=self.l_linear.weight.dtype)))


class ISTFTHead(nn.Module):
    """Inverse STFT head for waveform reconstruction from complex spectrogram."""

    def __init__(self, dim, n_fft, hop_size, padding="center"):
        super().__init__()
        out_dim = n_fft + 2
        self.conv_pre = weight_norm(nn.Conv1d(dim, out_dim, kernel_size=7, padding=3))
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.padding = padding
        self.window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)

    def forward(self, x):
        x = self.conv_pre(x)
        x = F.pad(x, (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        x = x.squeeze(1)

        mag = x.abs()
        phase = torch.atan2(x[:, 1, :], x[:, 0, :])

        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)

        # Pad if needed to match hop alignment
        pad_len = (self.hop_size - (x.shape[-1] % self.hop_size)) % self.hop_size
        real = F.pad(real, (0, pad_len))
        imag = F.pad(imag, (0, pad_len))

        window = self.window.to(x.device).to(x.dtype)
        spec = torch.stack([real, imag], dim=1)
        spec = torch.view_as_complex(spec)

        # Overlap-add via ISTFT
        waveform = torch.istft(spec, self.n_fft, self.hop_size, window=window, center=True)

        # Trim to match expected length from mel input
        # The ratio between time-domain and mel-domain
        expected_len = self.hop_size * (x.shape[-1] // self.hop_size)
        if waveform.shape[-1] > expected_len:
            waveform = waveform[..., :expected_len]

        return waveform.unsqueeze(1)

    def remove_weight_norm(self):
        if hasattr(self.conv_pre, "parametrizations") and "weight" in self.conv_pre.parametrizations:
            parametrize.remove_parametrizations(self.conv_pre, "weight", leave_parametrized=True)
        else:
            remove_weight_norm(self.conv_pre)


class VocosGenerator(nn.Module):
    """
    Vocos generator using 1D convolutions + inverse STFT.

    This Fourier-based vocoder replaces transposed convolutions with
    Conv1d upsampling + inverse STFT for cleaner waveform reconstruction.

    Args:
        in_channel: Number of input mel channels.
        upsample_initial_channel: Initial hidden channel dimension.
        upsample_rates: Tuple of upsampling factors per layer.
        upsample_kernel_sizes: Tuple of kernel sizes per upsampling layer.
        resblock_kernel_sizes: Tuple of kernel sizes for residual blocks.
        resblock_dilations: Tuple of dilation tuples for residual blocks.
        gin_channels: Speaker embedding channels.
        sr: Sample rate.
        harmonic_num: Number of harmonics for source excitation.
        checkpointing: Enable gradient checkpointing for memory efficiency.
    """

    def __init__(
        self,
        in_channel=128,
        upsample_initial_channel=512,
        upsample_rates=(8, 8, 2, 2),
        upsample_kernel_sizes=(16, 16, 4, 4),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilations=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        gin_channels=256,
        sr=48000,
        harmonic_num=0,
        checkpointing=False,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.upp = math.prod(upsample_rates)
        self.checkpointing = checkpointing
        self.sr = sr

        # F0 harmonic source
        self.f0_upsamp = nn.Upsample(scale_factor=self.upp)
        self.m_source = SourceModuleHnNSF(sample_rate=sr, harmonic_num=harmonic_num)

        # Initial convolution
        self.conv_pre = weight_norm(
            nn.Conv1d(in_channel, upsample_initial_channel, kernel_size=7, padding=3)
        )

        # Upsampling layers using ConvTranspose1d
        self.ups = nn.ModuleList()
        self.noise_convs = nn.ModuleList()

        channels = [upsample_initial_channel // (2 ** (i + 1)) for i in range(self.num_upsamples)]
        stride_f0s = [
            math.prod(upsample_rates[i + 1:]) if i + 1 < self.num_upsamples else 1
            for i in range(self.num_upsamples)
        ]

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2 ** i),
                        channels[i],
                        k,
                        u,
                        padding=((k - u) // 2) if u % 2 == 0 else (u // 2 + u % 2),
                        output_padding=u % 2,
                    )
                )
            )
            stride = stride_f0s[i]
            kernel = 1 if stride == 1 else stride * 2 - stride % 2
            self.noise_convs.append(
                nn.Conv1d(1, channels[i], kernel_size=kernel, stride=stride,
                          padding=0 if stride == 1 else (kernel - stride) // 2)
            )

        # Residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = channels[i]
            self.resblocks.append(
                nn.ModuleList([
                    VocosResBlock(ch, kernel_size=k, dilations=d)
                    for k, d in zip(resblock_kernel_sizes, resblock_dilations)
                ])
            )

        # ISTFT head for waveform reconstruction
        n_fft = self.upp * 4  # 4x oversampled STFT
        self.istft_head = ISTFTHead(channels[-1], n_fft=n_fft, hop_size=self.upp)

        # Speaker conditioning
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, f0, g=None):
        """Forward pass: mel spectrogram + f0 -> waveform."""
        har_source = self.m_source(f0, self.upp).transpose(1, 2)
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i, (ups, noise_conv) in enumerate(zip(self.ups, self.noise_convs)):
            x = F.leaky_relu(x, LRELU_SLOPE)

            if self.training and self.checkpointing:
                x = checkpoint(ups, x, use_reentrant=False) + noise_conv(har_source)
                xs = sum(
                    [
                        checkpoint(rb, x, use_reentrant=False)
                        for rb in self.resblocks[i]
                    ]
                )
            else:
                x = ups(x) + noise_conv(har_source)
                xs = sum([rb(x) for rb in self.resblocks[i]])

            x = xs / self.num_kernels

        # Use ISTFT head for waveform reconstruction
        return self.istft_head(F.leaky_relu(x)).squeeze(1)

    def remove_weight_norm(self):
        if hasattr(self.conv_pre, "parametrizations") and "weight" in self.conv_pre.parametrizations:
            parametrize.remove_parametrizations(self.conv_pre, "weight", leave_parametrized=True)
        else:
            remove_weight_norm(self.conv_pre)

        for up in self.ups:
            if hasattr(up, "parametrizations") and "weight" in up.parametrizations:
                parametrize.remove_parametrizations(up, "weight", leave_parametrized=True)
            else:
                remove_weight_norm(up)

        for res_list in self.resblocks:
            for rb in res_list:
                rb.remove_weight_norm()

        self.istft_head.remove_weight_norm()

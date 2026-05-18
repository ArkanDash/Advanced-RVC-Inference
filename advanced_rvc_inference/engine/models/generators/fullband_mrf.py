"""
FullBand Multi-Receptive Field HiFi-GAN (FullBand-MRF) vocoder.

An enhanced MRF vocoder with wider receptive fields, full-band mel
processing, and Snake activations for improved high-frequency detail.

Key improvements over standard MRF:
- Wider receptive fields via larger dilation rates
- Snake activations for better high-frequency modeling
- Full-band mel processing without sub-band splitting
- Improved MRF block design with channel attention

Args:
    in_channel: Number of input mel channels.
    upsample_initial_channel: Initial hidden channel dimension.
    upsample_rates: Tuple of upsampling factors per layer.
    upsample_kernel_sizes: Tuple of kernel sizes per upsampling layer.
    resblock_kernel_sizes: Tuple of kernel sizes for residual blocks.
    resblock_dilations: Tuple of dilation tuples for residual blocks.
    gin_channels: Speaker embedding channels.
    sr/sample_rate: Sample rate.
    harmonic_num: Number of harmonics for source excitation.
    checkpointing: Enable gradient checkpointing.
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
    """Snake activation: x + (1/alpha) * sin^2(alpha * x)."""

    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return x + (1.0 / (self.alpha + 1e-9)) * (x * self.alpha).sin().pow(2)


class ChannelAttention(nn.Module):
    """Channel attention mechanism for feature recalibration."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.conv(self.avg_pool(x))


class FullBandMRFLayer(nn.Module):
    """Enhanced MRF layer with wider receptive field and Snake activation."""

    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                channels, channels, kernel_size,
                padding=(kernel_size * dilation - dilation) // 2,
                dilation=dilation,
            )
        )
        self.act1 = SnakeActivation(channels)
        self.conv2 = weight_norm(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        )
        self.act2 = SnakeActivation(channels)

    def forward(self, x):
        residual = x
        x = self.act1(x)
        x = self.conv1(x)
        x = self.act2(x)
        x = self.conv2(x)
        return residual + x

    def remove_weight_norm(self):
        for conv in [self.conv1, self.conv2]:
            if hasattr(conv, "parametrizations") and "weight" in conv.parametrizations:
                parametrize.remove_parametrizations(conv, "weight", leave_parametrized=True)
            else:
                remove_weight_norm(conv)


class FullBandMRFBlock(nn.Module):
    """Enhanced MRF block with channel attention.

    Uses wider dilations (up to 32) for larger receptive fields
    and channel attention for feature recalibration.
    """

    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.layers = nn.ModuleList()
        for dilation in dilations:
            self.layers.append(FullBandMRFLayer(channels, kernel_size, dilation))
        # Channel attention for feature recalibration
        self.attention = ChannelAttention(channels)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.attention(x)

    def remove_weight_norm(self):
        for layer in self.layers:
            layer.remove_weight_norm()


class SineGen(nn.Module):
    """Sine wave generator for harmonic excitation."""

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

    def _f02sine(self, f0_values):
        rad_values = (f0_values / self.sampling_rate) % 1
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], dtype=f0_values.dtype, device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        tmp_over_one = torch.cumsum(rad_values, 1) % 1
        tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
        cumsum_shift = torch.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

        return (torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi).sin()

    def forward(self, f0):
        with torch.no_grad():
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, dtype=f0.dtype, device=f0.device)
            f0_buf[:, :, 0] = f0[:, :, 0]
            for idx in np.arange(self.harmonic_num):
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)

            sine_waves = self._f02sine(f0_buf) * self.sine_amp
            uv = self._f02uv(f0)
            sine_waves = sine_waves * uv + (
                (uv * self.noise_std + (1 - uv) * self.sine_amp / 3) * torch.randn_like(sine_waves)
            )
        return sine_waves


class SourceModuleHnNSF(nn.Module):
    """Neural Sine Filter source excitation module."""

    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1, add_noise_std=0.003, voiced_threshold=0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshold)
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x):
        return self.l_tanh(self.l_linear(self.l_sin_gen(x).to(dtype=self.l_linear.weight.dtype)))


class FullBandMRFGenerator(nn.Module):
    """
    FullBand Multi-Receptive Field HiFi-GAN Generator.

    Enhanced MRF vocoder with:
    - Wider receptive fields (larger dilation rates up to 32)
    - Snake activations for better high-frequency modeling
    - Channel attention for feature recalibration
    - Full-band mel processing

    Args:
        in_channel: Input mel channels.
        upsample_initial_channel: Initial channel dimension.
        upsample_rates: Upsampling rates per layer.
        upsample_kernel_sizes: Kernel sizes for upsampling.
        resblock_kernel_sizes: Kernel sizes for MRF blocks.
        resblock_dilations: Dilation configs for MRF blocks.
        gin_channels: Speaker embedding channels.
        sr/sample_rate: Sample rate.
        harmonic_num: Number of harmonics.
        checkpointing: Enable gradient checkpointing.
    """

    def __init__(
        self,
        in_channel=128,
        upsample_initial_channel=512,
        upsample_rates=(8, 8, 2, 2),
        upsample_kernel_sizes=(16, 16, 4, 4),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilations=((1, 3, 5, 7, 11, 15), (1, 3, 5, 7, 11, 15), (1, 3, 5, 7, 11, 15)),
        gin_channels=256,
        sr=48000,
        sample_rate=None,
        harmonic_num=0,
        checkpointing=False,
    ):
        super().__init__()
        if sample_rate is not None:
            sr = sample_rate

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.upp = math.prod(upsample_rates)
        self.checkpointing = checkpointing

        # F0 harmonic source
        self.f0_upsample = nn.Upsample(scale_factor=self.upp)
        self.m_source = SourceModuleHnNSF(sr, harmonic_num)

        # Pre-convolution with Snake activation
        self.conv_pre = weight_norm(
            nn.Conv1d(in_channel, upsample_initial_channel, kernel_size=7, padding=3)
        )
        self.act_pre = SnakeActivation(upsample_initial_channel)

        # Upsampling layers
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

        # Enhanced MRF blocks with wider receptive fields
        self.mrfs = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = channels[i]
            self.mrfs.append(
                nn.ModuleList([
                    FullBandMRFBlock(ch, kernel_size=k, dilations=d)
                    for k, d in zip(resblock_kernel_sizes, resblock_dilations)
                ])
            )

        # Post-processing with Snake activation
        self.act_post = SnakeActivation(channels[-1])
        self.conv_post = weight_norm(
            nn.Conv1d(channels[-1], 1, kernel_size=7, padding=3)
        )

        # Speaker conditioning
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, f0, g=None):
        """Forward: mel + f0 -> waveform."""
        har_source = self.m_source(self.f0_upsample(f0[:, None, :]).transpose(-1, -2)).transpose(-1, -2)
        x = self.act_pre(self.conv_pre(x))
        if g is not None:
            x = x + self.cond(g)

        for i, (ups, mrf, noise_conv) in enumerate(
            zip(self.ups, self.mrfs, self.noise_convs)
        ):
            x = F.leaky_relu(x, LRELU_SLOPE)

            if self.training and self.checkpointing:
                x = checkpoint(ups, x, use_reentrant=False) + noise_conv(har_source)
                xs = sum([checkpoint(block, x, use_reentrant=False) for block in mrf])
            else:
                x = ups(x) + noise_conv(har_source)
                xs = sum([block(x) for block in mrf])

            x = xs / self.num_kernels

        return self.conv_post(self.act_post(x)).tanh()

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

        for mrf in self.mrfs:
            for block in mrf:
                block.remove_weight_norm()

        if hasattr(self.conv_post, "parametrizations") and "weight" in self.conv_post.parametrizations:
            parametrize.remove_parametrizations(self.conv_post, "weight", leave_parametrized=True)
        else:
            remove_weight_norm(self.conv_post)

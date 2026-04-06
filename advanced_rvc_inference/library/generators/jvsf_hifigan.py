"""
Joint-Variable Source-Filter HiFi-GAN (JVSF-HiFi-GAN) vocoder implementation.

A variant that models source (glottal excitation) and filter (vocal tract)
separately, then combines them through a joint synthesis path.

Source module generates multiple harmonics from f0.
Filter module shapes spectral envelope with Conv1d layers.
Joint path merges both for final waveform synthesis.

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


class JVSourceModule(nn.Module):
    """Joint-Variable source module with multiple harmonics.

    Generates a rich harmonic source signal from F0 with
    adjustable harmonic amplitudes.
    """

    def __init__(self, sample_rate, harmonic_num=8, sine_amp=0.1, noise_std=0.003):
        super().__init__()
        self.harmonic_num = harmonic_num
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.sampling_rate = sample_rate
        self.dim = harmonic_num + 1

        # Learnable harmonic amplitude weights
        self.harmonic_weights = nn.Parameter(torch.ones(self.dim) / self.dim)

        # Merge harmonics to single source
        self.merge = nn.Linear(self.dim, 1)
        self.tanh = nn.Tanh()

    def _f02sine(self, f0, upp):
        """Generate sine waves from f0."""
        rad = f0 / self.sampling_rate * torch.arange(
            1, upp + 1, dtype=f0.dtype, device=f0.device
        )
        rad += F.pad(
            (torch.fmod(rad[:, :-1, -1:].float() + 0.5, 1.0) - 0.5)
            .cumsum(dim=1)
            .fmod(1.0)
            .to(f0),
            (0, 0, 1, 0),
            mode='constant',
        )
        rad = rad.reshape(f0.shape[0], -1, 1)
        # Apply harmonic multipliers
        rad *= torch.arange(1, self.dim + 1, dtype=f0.dtype, device=f0.device).reshape(1, 1, -1)
        rand_ini = torch.rand(1, 1, self.dim, device=f0.device)
        rand_ini[..., 0] = 0
        rad += rand_ini

        return (2 * np.pi * rad).sin()

    def forward(self, f0, upp):
        """Generate harmonic source signal from f0."""
        with torch.no_grad():
            f0 = f0.unsqueeze(-1)
            sine_waves = self._f02sine(f0, upp) * self.sine_amp

            # Apply learnable harmonic weighting
            weights = F.softmax(self.harmonic_weights, dim=0)
            sine_waves = sine_waves * weights.reshape(1, 1, -1)

            # Voiced/unvoiced detection
            uv = torch.ones_like(f0) * (f0 > 0)
            if uv.device.type == "privateuseone":
                uv = uv.float()
            uv = F.interpolate(
                uv.transpose(2, 1), scale_factor=float(upp), mode="nearest"
            ).transpose(2, 1)

            # Mix sine waves with noise for unvoiced regions
            noise = (uv * self.noise_std + (1 - uv) * self.sine_amp / 3) * torch.randn_like(sine_waves)
            sine_waves = sine_waves * uv + noise

        return self.tanh(self.merge(sine_waves.to(dtype=self.merge.weight.dtype)))


class JVFilterModule(nn.Module):
    """Filter module that shapes the spectral envelope.

    Uses stacked Conv1d layers with residual connections
    to model the vocal tract filter response.
    """

    def __init__(self, channels, num_layers=4, kernel_size=3):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            in_ch = channels if i == 0 else channels
            self.layers.append(
                nn.ModuleList([
                    weight_norm(
                        nn.Conv1d(
                            in_ch, channels, kernel_size,
                            padding=kernel_size // 2,
                            dilation=2 ** i,
                        )
                    ),
                    nn.LeakyReLU(LRELU_SLOPE),
                ])
            )

        # Learnable filter gain
        self.gain = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        """Apply spectral filter to input."""
        residual = x
        for conv, act in self.layers:
            x = act(conv(x))
        return (residual + x) * self.gain

    def remove_weight_norm(self):
        for conv, _ in self.layers:
            if hasattr(conv, "parametrizations") and "weight" in conv.parametrizations:
                parametrize.remove_parametrizations(conv, "weight", leave_parametrized=True)
            else:
                remove_weight_norm(conv)


class JVSFResBlock(nn.Module):
    """Residual block for joint source-filter synthesis."""

    def __init__(self, channels, kernel_size=3, dilations=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for d in dilations:
            self.convs1.append(
                weight_norm(
                    nn.Conv1d(
                        channels, channels, kernel_size,
                        padding=(kernel_size * d - d) // 2,
                        dilation=d,
                    )
                )
            )
            self.convs2.append(
                weight_norm(
                    nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
                )
            )

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            residual = x
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = F.leaky_relu(c1(x), LRELU_SLOPE)
            x = c2(x)
            x = x + residual
        return x

    def remove_weight_norm(self):
        for conv_list in [self.convs1, self.convs2]:
            for conv in conv_list:
                if hasattr(conv, "parametrizations") and "weight" in conv.parametrizations:
                    parametrize.remove_parametrizations(conv, "weight", leave_parametrized=True)
                else:
                    remove_weight_norm(conv)


class JVSFHiFiGANGenerator(nn.Module):
    """
    Joint-Variable Source-Filter HiFi-GAN Generator.

    Models source and filter separately:
    - Source module generates harmonics from f0
    - Filter module with Conv1d layers shapes spectral envelope
    - Joint synthesis path combines both representations

    Args:
        in_channel: Input mel channels.
        upsample_initial_channel: Initial channel dimension.
        upsample_rates: Upsampling rates per layer.
        upsample_kernel_sizes: Kernel sizes for upsampling.
        resblock_kernel_sizes: Kernel sizes for residual blocks.
        resblock_dilations: Dilation configs for residual blocks.
        gin_channels: Speaker embedding channels.
        sr/sample_rate: Sample rate.
        harmonic_num: Number of harmonics (default 8).
        checkpointing: Enable gradient checkpointing.
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
        sample_rate=None,
        harmonic_num=8,
        checkpointing=False,
    ):
        super().__init__()
        if sample_rate is not None:
            sr = sample_rate

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.upp = math.prod(upsample_rates)
        self.checkpointing = checkpointing

        # Source module
        self.source = JVSourceModule(sr, harmonic_num=harmonic_num)

        # Pre-convolution
        self.conv_pre = weight_norm(
            nn.Conv1d(in_channel, upsample_initial_channel, kernel_size=7, padding=3)
        )

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

        # Filter module per upsampling stage
        self.filters = nn.ModuleList()
        for i in range(len(self.ups)):
            self.filters.append(JVFilterModule(channels[i]))

        # Residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = channels[i]
            self.resblocks.append(
                nn.ModuleList([
                    JVSFResBlock(ch, kernel_size=k, dilations=d)
                    for k, d in zip(resblock_kernel_sizes, resblock_dilations)
                ])
            )

        # Post-convolution
        self.conv_post = weight_norm(
            nn.Conv1d(channels[-1], 1, kernel_size=7, padding=3)
        )

        # Speaker conditioning
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, f0, g=None):
        """Forward: mel + f0 -> waveform."""
        # Generate harmonic source
        har_source = self.source(f0, self.upp).transpose(1, 2)

        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i, (ups, noise_conv, filt) in enumerate(
            zip(self.ups, self.noise_convs, self.filters)
        ):
            x = F.leaky_relu(x, LRELU_SLOPE)

            if self.training and self.checkpointing:
                x = checkpoint(ups, x, use_reentrant=False) + noise_conv(har_source)
                # Apply filter module
                x = checkpoint(filt, x, use_reentrant=False)
                xs = sum(
                    [checkpoint(rb, x, use_reentrant=False) for rb in self.resblocks[i]]
                )
            else:
                x = ups(x) + noise_conv(har_source)
                # Apply filter module
                x = filt(x)
                xs = sum([rb(x) for rb in self.resblocks[i]])

            x = xs / self.num_kernels

        return self.conv_post(F.leaky_relu(x, LRELU_SLOPE)).tanh()

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

        for filt in self.filters:
            filt.remove_weight_norm()

        for res_list in self.resblocks:
            for rb in res_list:
                rb.remove_weight_norm()

        if hasattr(self.conv_post, "parametrizations") and "weight" in self.conv_post.parametrizations:
            parametrize.remove_parametrizations(self.conv_post, "weight", leave_parametrized=True)
        else:
            remove_weight_norm(self.conv_post)

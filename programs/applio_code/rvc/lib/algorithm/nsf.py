import math
import torch
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from typing import Optional

from programs.applio_code.rvc.lib.algorithm.generators import SineGen
from programs.applio_code.rvc.lib.algorithm.residuals import (
    LRELU_SLOPE,
    ResBlock1,
    ResBlock2,
)
from programs.applio_code.rvc.lib.algorithm.commons import init_weights


class SourceModuleHnNSF(torch.nn.Module):
    """
    Source Module for harmonic-plus-noise excitation.

    Args:
        sample_rate (int): Sampling rate in Hz.
        harmonic_num (int, optional): Number of harmonics above F0. Defaults to 0.
        sine_amp (float, optional): Amplitude of sine source signal. Defaults to 0.1.
        add_noise_std (float, optional): Standard deviation of additive Gaussian noise. Defaults to 0.003.
        voiced_threshod (float, optional): Threshold to set voiced/unvoiced given F0. Defaults to 0.
        is_half (bool, optional): Whether to use half precision. Defaults to True.
    """

    def __init__(
        self,
        sample_rate,
        harmonic_num=0,
        sine_amp=0.1,
        add_noise_std=0.003,
        voiced_threshod=0,
        is_half=True,
    ):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.is_half = is_half

        self.l_sin_gen = SineGen(
            sample_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod
        )
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x: torch.Tensor, upsample_factor: int = 1):
        sine_wavs, uv, _ = self.l_sin_gen(x, upsample_factor)
        sine_wavs = sine_wavs.to(dtype=self.l_linear.weight.dtype)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge, None, None


class GeneratorNSF(torch.nn.Module):
    """
    Generator for synthesizing audio using the NSF (Neural Source Filter) approach.

    Args:
        initial_channel (int): Number of channels in the initial convolutional layer.
        resblock (str): Type of residual block to use (1 or 2).
        resblock_kernel_sizes (list): Kernel sizes of the residual blocks.
        resblock_dilation_sizes (list): Dilation rates of the residual blocks.
        upsample_rates (list): Upsampling rates.
        upsample_initial_channel (int): Number of channels in the initial upsampling layer.
        upsample_kernel_sizes (list): Kernel sizes of the upsampling layers.
        gin_channels (int): Number of channels for the global conditioning input.
        sr (int): Sampling rate.
        is_half (bool, optional): Whether to use half precision. Defaults to False.
    """

    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels,
        sr,
        is_half=False,
    ):
        super(GeneratorNSF, self).__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.f0_upsamp = torch.nn.Upsample(scale_factor=math.prod(upsample_rates))
        self.m_source = SourceModuleHnNSF(
            sample_rate=sr, harmonic_num=0, is_half=is_half
        )

        self.conv_pre = torch.nn.Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock_cls = ResBlock1 if resblock == "1" else ResBlock2

        self.ups = torch.nn.ModuleList()
        self.noise_convs = torch.nn.ModuleList()

        channels = [
            upsample_initial_channel // (2 ** (i + 1))
            for i in range(len(upsample_rates))
        ]
        stride_f0s = [
            math.prod(upsample_rates[i + 1 :]) if i + 1 < len(upsample_rates) else 1
            for i in range(len(upsample_rates))
        ]

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    torch.nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        channels[i],
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

            self.noise_convs.append(
                torch.nn.Conv1d(
                    1,
                    channels[i],
                    kernel_size=(stride_f0s[i] * 2 if stride_f0s[i] > 1 else 1),
                    stride=stride_f0s[i],
                    padding=(stride_f0s[i] // 2 if stride_f0s[i] > 1 else 0),
                )
            )

        self.resblocks = torch.nn.ModuleList(
            [
                resblock_cls(channels[i], k, d)
                for i in range(len(self.ups))
                for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ]
        )

        self.conv_post = torch.nn.Conv1d(channels[-1], 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = torch.nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        self.upp = math.prod(upsample_rates)
        self.lrelu_slope = LRELU_SLOPE

    def forward(self, x, f0, g: Optional[torch.Tensor] = None):
        har_source, _, _ = self.m_source(f0, self.upp)
        har_source = har_source.transpose(1, 2)
        x = self.conv_pre(x)

        if g is not None:
            x = x + self.cond(g)

        for i, (ups, noise_convs) in enumerate(zip(self.ups, self.noise_convs)):
            x = torch.nn.functional.leaky_relu(x, self.lrelu_slope)
            x = ups(x)
            x = x + noise_convs(har_source)

            xs = sum(
                [
                    resblock(x)
                    for j, resblock in enumerate(self.resblocks)
                    if j in range(i * self.num_kernels, (i + 1) * self.num_kernels)
                ]
            )
            x = xs / self.num_kernels

        x = torch.nn.functional.leaky_relu(x)
        x = torch.tanh(self.conv_post(x))
        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()

    def __prepare_scriptable__(self):
        for l in self.ups:
            for hook in l._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    remove_weight_norm(l)
        for l in self.resblocks:
            for hook in l._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    remove_weight_norm(l)
        return self

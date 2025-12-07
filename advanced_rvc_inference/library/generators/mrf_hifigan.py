import math
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from torch.nn.utils import remove_weight_norm
from torch.utils.checkpoint import checkpoint
from torch.nn.utils.parametrizations import weight_norm

LRELU_SLOPE = 0.1

class MRFLayer(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size * dilation - dilation) // 2, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, dilation=1))

    def forward(self, x):
        return x + self.conv2(F.leaky_relu(self.conv1(F.leaky_relu(x, LRELU_SLOPE)), LRELU_SLOPE))

    def remove_weight_norm(self):
        if hasattr(self.conv1, "parametrizations") and "weight" in self.conv1.parametrizations: parametrize.remove_parametrizations(self.conv1, "weight", leave_parametrized=True)
        else: remove_weight_norm(self.conv1)

        if hasattr(self.conv2, "parametrizations") and "weight" in self.conv2.parametrizations: parametrize.remove_parametrizations(self.conv2, "weight", leave_parametrized=True)
        else: remove_weight_norm(self.conv2)

class MRFBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.layers = nn.ModuleList()

        for dilation in dilations:
            self.layers.append(MRFLayer(channels, kernel_size, dilation))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def remove_weight_norm(self):
        for layer in self.layers:
            layer.remove_weight_norm()

class SineGenerator(nn.Module):
    def __init__(self, samp_rate, harmonic_num = 0, sine_amp = 0.1, noise_std = 0.003, voiced_threshold = 0):
        super(SineGenerator, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        uv = torch.ones_like(f0) * (f0 > self.voiced_threshold)
        if uv.device.type == "privateuseone": uv = uv.float()

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
            sine_waves = sine_waves * uv + ((uv * self.noise_std + (1 - uv) * self.sine_amp / 3) * torch.randn_like(sine_waves))

        return sine_waves

class SourceModuleHnNSF(nn.Module):
    def __init__(self, sampling_rate, harmonic_num = 0, sine_amp = 0.1, add_noise_std = 0.003, voiced_threshold = 0):
        super(SourceModuleHnNSF, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGenerator(sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshold)
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x):
        return self.l_tanh(self.l_linear(self.l_sin_gen(x).to(dtype=self.l_linear.weight.dtype)))

class HiFiGANMRFGenerator(nn.Module):
    def __init__(self, in_channel, upsample_initial_channel, upsample_rates, upsample_kernel_sizes, resblock_kernel_sizes, resblock_dilations, gin_channels, sample_rate, harmonic_num, checkpointing = False):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.checkpointing = checkpointing
        self.f0_upsample = nn.Upsample(scale_factor=np.prod(upsample_rates))
        self.m_source = SourceModuleHnNSF(sample_rate, harmonic_num)
        self.conv_pre = weight_norm(nn.Conv1d(in_channel, upsample_initial_channel, kernel_size=7, stride=1, padding=3))
        self.upsamples = nn.ModuleList()
        self.noise_convs = nn.ModuleList()
        stride_f0s = [math.prod(upsample_rates[i + 1 :]) if i + 1 < len(upsample_rates) else 1 for i in range(len(upsample_rates))]

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.upsamples.append(weight_norm(nn.ConvTranspose1d(upsample_initial_channel // (2**i), upsample_initial_channel // (2 ** (i + 1)), kernel_size=k, stride=u, padding=((k - u) // 2) if u % 2 == 0 else (u // 2 + u % 2), output_padding=u % 2)))
            stride = stride_f0s[i]
            kernel = 1 if stride == 1 else stride * 2 - stride % 2
            self.noise_convs.append(nn.Conv1d(1, upsample_initial_channel // (2 ** (i + 1)), kernel_size=kernel, stride=stride, padding=0 if stride == 1 else (kernel - stride) // 2))

        self.mrfs = nn.ModuleList()
        for i in range(len(self.upsamples)):
            channel = upsample_initial_channel // (2 ** (i + 1))
            self.mrfs.append(nn.ModuleList([MRFBlock(channel, kernel_size=k, dilations=d) for k, d in zip(resblock_kernel_sizes, resblock_dilations)]))

        self.conv_post = weight_norm(nn.Conv1d(channel, 1, kernel_size=7, stride=1, padding=3))
        if gin_channels != 0: self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, f0, g = None):
        har_source = self.m_source(self.f0_upsample(f0[:, None, :]).transpose(-1, -2)).transpose(-1, -2)
        x = self.conv_pre(x)
        if g is not None: x += self.cond(g)

        for ups, mrf, noise_conv in zip(self.upsamples, self.mrfs, self.noise_convs):
            x = F.leaky_relu(x, LRELU_SLOPE)

            if self.training and self.checkpointing:
                x = checkpoint(ups, x, use_reentrant=False) + noise_conv(har_source)
                xs = sum([checkpoint(layer, x, use_reentrant=False) for layer in mrf])
            else:
                x = ups(x) + noise_conv(har_source)
                xs = sum([layer(x) for layer in mrf])

            x = xs / self.num_kernels

        return self.conv_post(F.leaky_relu(x)).tanh()

    def remove_weight_norm(self):
        if hasattr(self.conv_pre, "parametrizations") and "weight" in self.conv_pre.parametrizations: parametrize.remove_parametrizations(self.conv_pre, "weight", leave_parametrized=True)
        else: remove_weight_norm(self.conv_pre)

        for up in self.upsamples:
            if hasattr(up, "parametrizations") and "weight" in up.parametrizations: parametrize.remove_parametrizations(up, "weight", leave_parametrized=True)
            else: remove_weight_norm(up)

        for mrf in self.mrfs:
            for block in mrf:
                block.remove_weight_norm()

        if hasattr(self.conv_post, "parametrizations") and "weight" in self.conv_post.parametrizations: parametrize.remove_parametrizations(self.conv_post, "weight", leave_parametrized=True)
        else: remove_weight_norm(self.conv_post)
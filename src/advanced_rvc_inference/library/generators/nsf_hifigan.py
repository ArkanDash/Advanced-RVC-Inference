import os
import sys
import math
import torch

import numpy as np
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from torch.nn.utils import remove_weight_norm
from torch.utils.checkpoint import checkpoint
from torch.nn.utils.parametrizations import weight_norm

sys.path.append(os.getcwd())

from main.library.algorithm.commons import init_weights
from main.library.algorithm.residuals import ResBlock, LRELU_SLOPE

class SineGen(torch.nn.Module):
    def __init__(self, samp_rate, harmonic_num=0, sine_amp=0.1, noise_std=0.003, voiced_threshold=0, flag_for_pulse=False):
        super(SineGen, self).__init__()
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
    
    def _f02sine(self, f0, upp):
        rad = f0 / self.sampling_rate * torch.arange(1, upp + 1, dtype=f0.dtype, device=f0.device)
        rad += F.pad((torch.fmod(rad[:, :-1, -1:].float() + 0.5, 1.0) - 0.5).cumsum(dim=1).fmod(1.0).to(f0), (0, 0, 1, 0), mode='constant')
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
            uv = F.interpolate(self._f02uv(f0).transpose(2, 1), scale_factor=float(upp), mode="nearest").transpose(2, 1)
            sine_waves = sine_waves * uv + ((uv * self.noise_std + (1 - uv) * self.sine_amp / 3) * torch.randn_like(sine_waves))

        return sine_waves

class SourceModuleHnNSF(torch.nn.Module):
    def __init__(self, sample_rate, harmonic_num=0, sine_amp=0.1, add_noise_std=0.003, voiced_threshod=0):
        super(SourceModuleHnNSF, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen(sample_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod)
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x, upsample_factor = 1):
        return self.l_tanh(self.l_linear(self.l_sin_gen(x, upsample_factor).to(dtype=self.l_linear.weight.dtype)))

class HiFiGANNRFGenerator(torch.nn.Module):
    def __init__(self, initial_channel, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels, sr, checkpointing = False):
        super(HiFiGANNRFGenerator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.upp = math.prod(upsample_rates)
        self.f0_upsamp = torch.nn.Upsample(scale_factor=self.upp)
        self.m_source = SourceModuleHnNSF(sample_rate=sr, harmonic_num=0)

        self.conv_pre = torch.nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        self.checkpointing = checkpointing

        self.ups = torch.nn.ModuleList()
        self.noise_convs = torch.nn.ModuleList()

        channels = [upsample_initial_channel // (2 ** (i + 1)) for i in range(self.num_upsamples)]
        stride_f0s = [math.prod(upsample_rates[i + 1 :]) if i + 1 < self.num_upsamples else 1 for i in range(self.num_upsamples)]

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(torch.nn.ConvTranspose1d(upsample_initial_channel // (2**i), channels[i], k, u, padding=((k - u) // 2) if u % 2 == 0 else (u // 2 + u % 2), output_padding=u % 2)))
            stride = stride_f0s[i]
            kernel = 1 if stride == 1 else stride * 2 - stride % 2
            self.noise_convs.append(torch.nn.Conv1d(1, channels[i], kernel_size=kernel, stride=stride, padding=0 if stride == 1 else (kernel - stride) // 2))

        self.resblocks = torch.nn.ModuleList([ResBlock(channels[i], k, d) for i in range(len(self.ups)) for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes)])
        self.conv_post = torch.nn.Conv1d(channels[-1], 1, 7, 1, padding=3, bias=False)

        self.ups.apply(init_weights)
        if gin_channels != 0: self.cond = torch.nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, f0, g = None):
        har_source = self.m_source(f0, self.upp).transpose(1, 2)
        x = self.conv_pre(x)
        if g is not None: x += self.cond(g)

        for i, (ups, noise_convs) in enumerate(zip(self.ups, self.noise_convs)):
            x = F.leaky_relu(x, LRELU_SLOPE)

            if self.training and self.checkpointing:
                x = checkpoint(ups, x, use_reentrant=False) + noise_convs(har_source)
                xs = sum([checkpoint(resblock, x, use_reentrant=False) for j, resblock in enumerate(self.resblocks) if j in range(i * self.num_kernels, (i + 1) * self.num_kernels)])
            else:
                x = ups(x) + noise_convs(har_source)
                xs = sum([resblock(x) for j, resblock in enumerate(self.resblocks) if j in range(i * self.num_kernels, (i + 1) * self.num_kernels)])

            x = xs / self.num_kernels

        return self.conv_post(F.leaky_relu(x)).tanh()

    def remove_weight_norm(self):
        for l in self.ups:
            if hasattr(l, "parametrizations") and "weight" in l.parametrizations: parametrize.remove_parametrizations(l, "weight", leave_parametrized=True)
            else: remove_weight_norm(l)

        for l in self.resblocks:
            l.remove_weight_norm()
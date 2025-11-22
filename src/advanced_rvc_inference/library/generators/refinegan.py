import os
import sys
import torch
import torchaudio

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from torch.utils.checkpoint import checkpoint
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm

sys.path.append(os.getcwd())

from main.library.algorithm.commons import init_weights, get_padding

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size = 7, dilation = (1, 3, 5), leaky_relu_slope = 0.2):
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope
        self.convs1 = nn.ModuleList([weight_norm(nn.Conv1d(channels, channels, kernel_size, stride=1, dilation=d, padding=get_padding(kernel_size, d))) for d in dilation])
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList([weight_norm(nn.Conv1d(channels, channels, kernel_size, stride=1, dilation=1, padding=get_padding(kernel_size, 1))) for _ in dilation])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            x = c2(F.leaky_relu(c1(F.leaky_relu(x, self.leaky_relu_slope)), self.leaky_relu_slope)) + x

        return x

    def remove_weight_norm(self):
        for c1, c2 in zip(self.convs1, self.convs2):
            if hasattr(c1, "parametrizations") and "weight" in c1.parametrizations: parametrize.remove_parametrizations(c1, "weight", leave_parametrized=True)
            else: remove_weight_norm(c1)

            if hasattr(c2, "parametrizations") and "weight" in c2.parametrizations: parametrize.remove_parametrizations(c2, "weight", leave_parametrized=True)
            else: remove_weight_norm(c2)

class AdaIN(nn.Module):
    def __init__(self, *, channels, leaky_relu_slope = 0.2):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels) * 1e-4)
        self.activation = nn.LeakyReLU(leaky_relu_slope)

    def forward(self, x):
        return self.activation(x + (torch.randn_like(x) * self.weight[None, :, None]))
    
class ParallelResBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels, kernel_sizes = (3, 7, 11), dilation = (1, 3, 5), leaky_relu_slope = 0.2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=1, padding=3)
        self.input_conv.apply(init_weights)
        self.blocks = nn.ModuleList([nn.Sequential(AdaIN(channels=out_channels), ResBlock(out_channels, kernel_size=kernel_size, dilation=dilation, leaky_relu_slope=leaky_relu_slope), AdaIN(channels=out_channels)) for kernel_size in kernel_sizes])

    def forward(self, x):
        x = self.input_conv(x)
        return torch.stack([block(x) for block in self.blocks], dim=0).mean(dim=0)

    def remove_weight_norm(self):
        remove_weight_norm(self.input_conv)
        for block in self.blocks:
            block[1].remove_weight_norm()

class SineGenerator(nn.Module):
    def __init__(self, samp_rate, harmonic_num=0, sine_amp=0.1, noise_std=0.003, voiced_threshold=0):
        super(SineGenerator, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.merge = nn.Sequential(nn.Linear(self.dim, 1, bias=False), nn.Tanh())

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

        return self.merge(sine_waves)
    
class RefineGANGenerator(nn.Module):
    def __init__(self, *, sample_rate = 44100, upsample_rates = (8, 8, 2, 2), leaky_relu_slope = 0.2, num_mels = 128, start_channels = 16, gin_channels = 256, checkpointing = False, upsample_initial_channel = 512):
        super().__init__()
        self.upsample_rates = upsample_rates
        self.checkpointing = checkpointing
        self.leaky_relu_slope = leaky_relu_slope
        self.upp = np.prod(upsample_rates)
        self.m_source = SineGenerator(sample_rate)
        self.pre_conv = weight_norm(nn.Conv1d(1, 16, 7, 1, padding=3))
        channels = start_channels
        size = self.upp
        self.downsample_blocks = nn.ModuleList([])
        self.df0 = []

        for i, _ in enumerate(upsample_rates):
            new_size = int(size / upsample_rates[-i - 1])
            self.df0.append([size, new_size])
            size = new_size

            new_channels = channels * 2
            self.downsample_blocks.append(weight_norm(nn.Conv1d(channels, new_channels, 7, 1, padding=3)))
            channels = new_channels

        channels = upsample_initial_channel
        self.mel_conv = weight_norm(nn.Conv1d(num_mels, channels // 2, 7, 1, padding=3))
        self.mel_conv.apply(init_weights)

        if gin_channels != 0: self.cond = nn.Conv1d(256, channels // 2, 1)

        self.upsample_blocks = nn.ModuleList([])
        self.upsample_conv_blocks = nn.ModuleList([])

        for rate in upsample_rates:
            new_channels = channels // 2
            self.upsample_blocks.append(nn.Upsample(scale_factor=rate, mode="linear"))
            self.upsample_conv_blocks.append(ParallelResBlock(in_channels=channels + channels // 4, out_channels=new_channels, kernel_sizes=(3, 7, 11), dilation=(1, 3, 5), leaky_relu_slope=leaky_relu_slope))
            channels = new_channels

        self.conv_post = weight_norm(nn.Conv1d(channels, 1, 7, 1, padding=3, bias=False))
        self.conv_post.apply(init_weights)

    def forward(self, mel, f0, g = None):
        f0_size = mel.shape[-1]
        har_source = self.m_source(F.interpolate(f0.unsqueeze(1), size=f0_size * self.upp, mode="linear").transpose(1, 2)).transpose(1, 2)
        x = self.pre_conv(har_source)
        downs = []

        for block, (old_size, new_size) in zip(self.downsample_blocks, self.df0):
            x = F.leaky_relu(x, self.leaky_relu_slope)
            downs.append(x)
            x = torchaudio.functional.resample(x.contiguous(), orig_freq=int(f0_size * old_size), new_freq=int(f0_size * new_size), lowpass_filter_width=64, rolloff=0.9475937167399596, resampling_method="sinc_interp_kaiser", beta=14.769656459379492)
            x = block(x)

        mel = self.mel_conv(mel)
        if g is not None: mel += self.cond(g)

        x = torch.cat([mel, x], dim=1)
        for ups, res, down in zip(self.upsample_blocks, self.upsample_conv_blocks, reversed(downs)):
            x = F.leaky_relu(x, self.leaky_relu_slope)
            x = checkpoint(res, torch.cat([checkpoint(ups, x, use_reentrant=False), down], dim=1), use_reentrant=False) if self.training and self.checkpointing else res(torch.cat([ups(x), down], dim=1))

        return self.conv_post(F.leaky_relu(x, self.leaky_relu_slope)).tanh()

    def remove_weight_norm(self):
        if hasattr(self.pre_conv, "parametrizations") and "weight" in self.pre_conv.parametrizations: parametrize.remove_parametrizations(self.pre_conv, "weight", leave_parametrized=True)
        else: remove_weight_norm(self.pre_conv)

        if hasattr(self.mel_conv, "parametrizations") and "weight" in self.mel_conv.parametrizations: parametrize.remove_parametrizations(self.mel_conv, "weight", leave_parametrized=True)
        else: remove_weight_norm(self.mel_conv)

        if hasattr(self.conv_post, "parametrizations") and "weight" in self.conv_post.parametrizations: parametrize.remove_parametrizations(self.conv_post, "weight", leave_parametrized=True)
        else: remove_weight_norm(self.conv_post)

        for block in self.downsample_blocks:
            block.remove_weight_norm()

        for block in self.upsample_conv_blocks:
            block.remove_weight_norm()
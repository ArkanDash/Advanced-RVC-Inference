import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm

from torch.nn import Conv1d, ConvTranspose1d

from torch.amp import autocast # guard
from torch.utils.checkpoint import checkpoint

import einops
import numpy as np

from rvc.lib.algorithm.residuals import ResBlock, ResBlock_Snake_Fused, ResBlock_SnakeBeta
from rvc.lib.algorithm.conformer.conformer import Conformer

from rvc.lib.algorithm.commons import init_weights
from rvc.lib.algorithm.conformer.stft import TorchSTFT

# DEBUG
import torchaudio
import sys

class SineGenerator(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: int = 0,
        f0_upsample_factor: int = 1,
    ):
        super(SineGenerator, self).__init__()

        self.sampling_rate = sample_rate
        self.harmonic_num = harmonic_num
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.voiced_threshold = voiced_threshold
        self.f0_upsample_factor = f0_upsample_factor
        self.dim = self.harmonic_num + 1

    def _f0_to_voiced_mask(self, f0):
        # generate voiced_mask signal
        voiced_mask = (f0 > self.voiced_threshold).type(torch.float32)
        return voiced_mask

    def _f0_to_sines(self, f0_values):
        """ f0_values: (batchsize, length, dim)
            where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The interger part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1

        # initial phase noise (no noise for fundamental component)
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        rad_values = torch.nn.functional.interpolate(
            rad_values.transpose(1, 2), 
            scale_factor=1/self.f0_upsample_factor, 
            mode="linear"
        ).transpose(1, 2)

        phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi
        phase = torch.nn.functional.interpolate(
            phase.transpose(1, 2) * self.f0_upsample_factor, 
            scale_factor=self.f0_upsample_factor,
            mode="linear"
        ).transpose(1, 2)

        sines = torch.sin(phase)

        return sines

    def forward(self, f0):
        with autocast(device_type="cuda", enabled=False):
            f0_buf = torch.zeros(
                f0.shape[0],
                f0.shape[1],
                self.dim,
                device=f0.device
            )
            # fundamental component
            fn = torch.multiply(f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device))

            # generate sine waveforms
            sine_waves = self._f0_to_sines(fn) * self.sine_amp

            # generate voiced_mask signal
            voiced_mask = self._f0_to_voiced_mask(f0)

            # noise: for unvoiced should be similar to sine_amp
            noise_amp = voiced_mask * self.noise_std + (1 - voiced_mask) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)

            # first: set the unvoiced part to 0 by voiced_mask
            # then: additive noise
            sine_waves = sine_waves * voiced_mask + noise

        return sine_waves

class SourceModuleHnNSF(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshold: int = 0,
        f0_upsample_factor: int = 1,
    ):
        super(SourceModuleHnNSF, self).__init__()

        # to produce sine waveforms
        self.l_sin_gen = SineGenerator(
            sample_rate,
            harmonic_num,
            sine_amp,
            add_noise_std,
            voiced_threshold,
            f0_upsample_factor,
        )

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, f0: torch.Tensor):
        # source for harmonic branch
        with torch.no_grad():
            sine_wavs = self.l_sin_gen(f0)

        # Merge of the harmonics
        with autocast(device_type="cuda", enabled=False):
            sine_merge = self.l_tanh(self.l_linear(sine_wavs))

        return sine_merge


class RingFormerGenerator(nn.Module):
    def __init__(
        self,
        initial_channel, # 192
        resblock_kernel_sizes, # [3,7,11]
        resblock_dilation_sizes, # [[1,3,5], [1,3,5], [1,3,5]]
        upsample_rates, # [4, 4]
        upsample_initial_channel, # 512
        upsample_kernel_sizes, # [8, 8]
        gen_istft_n_fft, # 24khz: 60,  48khz: 120
        gen_istft_hop_size, # 24khz: 15, 48khz: 30
        gin_channels, # 256
        sr, # 24000, 48000, 
        harmonic_num = 8,
        checkpointing: bool = False,
    ):
        super(RingFormerGenerator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.gen_istft_n_fft = gen_istft_n_fft
        self.gen_istft_hop_size = gen_istft_hop_size

        self.checkpointing = checkpointing
        self.conv_pre = weight_norm(Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3))

        '''
        Available ResBlock types:

            - ResBlock:  HiFi-Gan's 'ResBlock1' without any extras
            - ResBlock_Snake_Fused:  'ResBlock1' I modified by adding in Snake activation with triton fused kernel ( Forward, Backward ) ; https://github.com/falkaer/pytorch-snake
            - ResBlock_SnakeBeta: 'ResBlock1' which is using Snake-Beta instead of Snake. Has learnable both alphas and betas ; https://github.com/NVIDIA/BigVGAN/blob/main/activations.py
        '''
        ResBlock_Type = ResBlock_SnakeBeta


        f0_upsample_factor = math.prod(upsample_rates) * gen_istft_hop_size

        self.f0_upsampler = torch.nn.Upsample(scale_factor = f0_upsample_factor)

        self.m_source = SourceModuleHnNSF(
            sample_rate = sr,
            harmonic_num = harmonic_num,
            voiced_threshold = 0,
            f0_upsample_factor = f0_upsample_factor,
        )

        self.noise_convs = nn.ModuleList()
        self.noise_res = nn.ModuleList()
        self.ups = nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2 ** i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2
                    )
                )
            )

            c_cur = upsample_initial_channel // (2 ** (i + 1))

            if i + 1 < len(upsample_rates):
                stride_f0 = math.prod(upsample_rates[i + 1:])
                kernel = stride_f0 * 2 - stride_f0 % 2
                padding = 0 if stride_f0 == 1 else (kernel - stride_f0) // 2

                self.noise_convs.append(Conv1d(
                    self.gen_istft_n_fft + 2, c_cur, kernel_size=kernel, stride=stride_f0, padding=padding))

                self.noise_res.append(ResBlock_Type(c_cur, 7, [1, 3, 5]))
            else:
                self.noise_convs.append(Conv1d(self.gen_istft_n_fft + 2, c_cur, kernel_size=1))
                self.noise_res.append(ResBlock_Type(c_cur, 11, [1, 3, 5]))

        self.alphas = nn.ParameterList()
        self.alphas.append(nn.Parameter(torch.ones(1, upsample_initial_channel, 1)))

        self.resblocks = nn.ModuleList()

        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            self.alphas.append(nn.Parameter(torch.ones(1, ch, 1)))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock_Type(ch, k, d))

        self.conformers = nn.ModuleList()
        self.post_n_fft = self.gen_istft_n_fft
        self.conv_post = weight_norm(Conv1d(128, self.post_n_fft + 2, 7, 1, padding=3))

        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** i)
            self.conformers.append(
                Conformer(
                    dim=ch,
                    depth=2,
                    dim_head=64,
                    heads=8,
                    ff_mult=4,
                    conv_expansion_factor = 2,
                    conv_kernel_size=31,
                    attn_dropout=0.1,
                    ff_dropout=0.1,
                    conv_dropout=0.1,
                )
            )

        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.reflection_pad = nn.ReflectionPad1d((1, 0))

        self.stft = TorchSTFT(
            "cuda",
            filter_length=self.gen_istft_n_fft,
            hop_length=self.gen_istft_hop_size,
            win_length=self.gen_istft_n_fft
        )

        if gin_channels != 0:
            self.cond = Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x: torch.Tensor, f0: torch.Tensor, g: Optional[torch.Tensor] = None):

        f0 = self.f0_upsampler(f0[:, None]).transpose(1, 2)

        har_source = self.m_source(f0)
        har_source = har_source.squeeze(-1)

        har_spec, har_phase = self.stft.transform(har_source)

        har = torch.cat([har_spec, har_phase], dim=1)

        x = self.conv_pre(x)

        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = x + (1 / self.alphas[i]) * (torch.sin(self.alphas[i] * x) ** 2)

            x = einops.rearrange(x, 'b f t -> b t f')
            x = self.conformers[i](x)
            x = einops.rearrange(x, 'b t f -> b f t')

            x_source = self.noise_convs[i](har)
            x_source = self.noise_res[i](x_source)

            if self.training and self.checkpointing:
                x = checkpoint(self.ups[i], x, use_reentrant=False)

                if i == self.num_upsamples - 1:
                    x = self.reflection_pad(x)

                x = x + x_source

                xs = None
                for j in range(self.num_kernels):
                    blk = self.resblocks[i * self.num_kernels + j]
                    cur = checkpoint(blk, x, use_reentrant=False)
                    xs = cur if xs is None else xs + cur
                x = xs / self.num_kernels
            else:
                x = self.ups[i](x)
 
                if i == self.num_upsamples - 1:
                    x = self.reflection_pad(x)

                x = x + x_source

                xs = None
                for j in range(self.num_kernels):
                    if xs is None:
                        xs = self.resblocks[i * self.num_kernels + j](x)
                    else:
                        xs += self.resblocks[i * self.num_kernels + j](x)
                x = xs / self.num_kernels

        x = x + (1 / self.alphas[i + 1]) * (torch.sin(self.alphas[i + 1] * x) ** 2)

        with autocast(device_type="cuda", enabled=False):
            x = x.to(torch.float32)
            x = self.conv_post(x)

            spec = torch.exp(x[:, :self.post_n_fft // 2 + 1, :]).to(x.device)
            phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :]).to(x.device)
            out = self.stft.inverse(spec, phase).to(x.device)

        return out, spec, phase

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

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
        # conv_pre, conv_post
        for hook in self.conv_pre._forward_pre_hooks.values():
            if (
                hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                and hook.__class__.__name__ == "WeightNorm"
            ):
                remove_weight_norm(self.conv_pre)
        for hook in self.conv_post._forward_pre_hooks.values():
            if (
                hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                and hook.__class__.__name__ == "WeightNorm"
            ):
                remove_weight_norm(self.conv_post)

        return self

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from torch.nn import Conv1d, ConvTranspose1d
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint
import einops
import numpy as np

from advanced_rvc_inference.library.algorithm.normalization import ResBlock, ResBlock_SnakeBeta
from advanced_rvc_inference.library.algorithm.conformer import Conformer


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
        super().__init__()
        self.sampling_rate = sample_rate
        self.harmonic_num = harmonic_num
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.voiced_threshold = voiced_threshold
        self.f0_upsample_factor = f0_upsample_factor
        self.dim = self.harmonic_num + 1

    def _f0_to_voiced_mask(self, f0):
        voiced_mask = (f0 > self.voiced_threshold).type(torch.float32)
        return voiced_mask

    def _f0_to_sines(self, f0_values):
        rad_values = (f0_values / self.sampling_rate) % 1
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

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
            fn = torch.multiply(f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device))
            sine_waves = self._f0_to_sines(fn) * self.sine_amp
            voiced_mask = self._f0_to_voiced_mask(f0)
            noise_amp = voiced_mask * self.noise_std + (1 - voiced_mask) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)
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
        super().__init__()
        self.l_sin_gen = SineGenerator(
            sample_rate,
            harmonic_num,
            sine_amp,
            add_noise_std,
            voiced_threshold,
            f0_upsample_factor,
        )
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, f0: torch.Tensor):
        with torch.no_grad():
            sine_wavs = self.l_sin_gen(f0)
        with autocast(device_type="cuda", enabled=False):
            sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge


def init_weights(m, mean=0.0, std=0.01):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        m.weight.data.normal_(mean, std)
    if hasattr(m, 'bias') and m.bias is not None:
        m.bias.data.zero_()


class TorchSTFT:
    def __init__(self, device, filter_length, hop_length, win_length):
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.device = device
        
        import librosa
        window = torch.hann_window(win_length).to(device)
        self.register_buffer('window', window)
        
        self.forward_basis = torch.nn.functional.pad(
            torch.eye(filter_length // 2 + 1, win_length, device=device),
            (0, win_length - filter_length),
            "constant",
            0,
        ).float()
        
        self.register_buffer('forward_basis_eye', self.forward_basis)
        basis = torch.fft.rfft(torch.eye(win_length), n=filter_length)
        on = (basis.real ** 2 + basis.imag ** 2) ** 0.5
        on = on / on.mean()
        on = on[:, : filter_length // 2 + 1]
        
        self.register_buffer('inverse_spectrogram_basis', torch.fft.irfft(on, n=filter_length), persistent=False)

    def transform(self, x):
        x = F.pad(x, (int((self.filter_length - self.win_length) / 2), int((self.filter_length - self.win_length) / 2), 0, 0), mode="reflect")
        forward_basis = self.forward_basis_eye
        magnitude = torch.sqrt((x @ forward_basis) ** 2 + (x @ forward_basis.t()) ** 2 + 1e-6)
        phase = torch.atan2((x @ forward_basis.t()), (x @ forward_basis) + 1e-6)
        return magnitude, phase

    def inverse(self, magnitude, phase):
        complex_stft = magnitude * torch.exp(1j * phase)
        x = torch.fft.irfft(complex_stft, n=self.filter_length)
        x = F.pad(x, (int((self.filter_length - self.win_length) / 2), int((self.filter_length - self.win_length) / 2)), mode="constant", constant_values=0)
        return x


class RingFormerGenerator(nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gen_istft_n_fft,
        gen_istft_hop_size,
        gin_channels,
        sr,
        harmonic_num=8,
        checkpointing=False,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.gen_istft_n_fft = gen_istft_n_fft
        self.gen_istft_hop_size = gen_istft_hop_size
        self.checkpointing = checkpointing
        
        self.conv_pre = weight_norm(Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3))
        ResBlock_Type = ResBlock_SnakeBeta

        f0_upsample_factor = math.prod(upsample_rates) * gen_istft_hop_size
        self.f0_upsampler = torch.nn.Upsample(scale_factor=f0_upsample_factor)

        self.m_source = SourceModuleHnNSF(
            sample_rate=sr,
            harmonic_num=harmonic_num,
            voiced_threshold=0,
            f0_upsample_factor=f0_upsample_factor,
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
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
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
                    conv_expansion_factor=2,
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
            "cuda" if torch.cuda.is_available() else "cpu",
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
                x = self.ups[i]
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
            out = self.stft.inverse(spec, phase)

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
                if hook.__module__ == "torch.nn.utils.parametrizations.weight_norm" and hook.__class__.__name__ == "WeightNorm":
                    remove_weight_norm(l)
        for l in self.resblocks:
            for hook in l._forward_pre_hooks.values():
                if hook.__module__ == "torch.nn.utils.parametrizations.weight_norm" and hook.__class__.__name__ == "WeightNorm":
                    l.remove_weight_norm()
        for hook in self.conv_pre._forward_pre_hooks.values():
            if hook.__module__ == "torch.nn.utils.parametrizations.weight_norm" and hook.__class__.__name__ == "WeightNorm":
                remove_weight_norm(self.conv_pre)
        for hook in self.conv_post._forward_pre_hooks.values():
            if hook.__module__ == "torch.nn.utils.parametrizations.weight_norm" and hook.__class__.__name__ == "WeightNorm":
                remove_weight_norm(self.conv_post)
        return self

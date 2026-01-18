import math
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm

class AntiAliasActivation(nn.Module):
    def __init__(
        self, 
        channels, 
        up=2, 
        down=2, 
        up_k=12, 
        down_k=12
    ):
        super().__init__()
        self.up = UpSample1d(up, up_k)
        self.act = SnakeBeta(channels)
        self.down = DownSample1d(down, down_k)

    def forward(self, x):
        return self.down(
            self.act(self.up(x))
        )

class Snake1(nn.Module):
    def __init__(
        self, 
        channels
    ):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x):
        alpha = self.alpha.exp()
        return x + (1.0 / (alpha + 1e-9)) * (x * alpha).sin().pow(2)

class SnakeBeta(nn.Module):
    def __init__(
        self, 
        channels
    ):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x):
        return x + (1.0 / (self.beta.exp() + 1e-9)) * (x * self.alpha.exp()).sin().pow(2)

def kaiser_sinc_filter1d(
    cutoff, 
    half_width, 
    kernel_size
):
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width

    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95

    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0

    window = torch.kaiser_window(
        kernel_size, 
        beta=beta, 
        periodic=False
    )

    time = (
        torch.arange(-half_size, half_size) + 0.5
    ) if even else (
        torch.arange(kernel_size) - half_size
    )

    if cutoff == 0:
        filter = torch.zeros_like(time)
    else:
        filter = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
        filter /= filter.sum()

    return filter.view(1, 1, kernel_size)

class UpSample1d(nn.Module):
    def __init__(
        self, 
        ratio=2, 
        kernel_size=None
    ):
        super().__init__()
        self.ratio = ratio
        self.stride = ratio

        kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size

        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (kernel_size - self.stride + 1) // 2

        filter = kaiser_sinc_filter1d(
            cutoff=0.5 / ratio, 
            half_width=0.6 / ratio, 
            kernel_size=kernel_size
        )
        self.register_buffer("filter", filter)

    def forward(self, x):
        x = self.ratio * F.conv_transpose1d(
            F.pad(
                x, 
                (self.pad, self.pad), 
                mode="replicate"
            ), 
            self.filter.expand(x.size(1), -1, -1), 
            stride=self.stride, 
            groups=x.size(1)
        )

        return x[..., self.pad_left : -self.pad_right]

class LowPassFilter1d(nn.Module):
    def __init__(
        self, 
        cutoff=0.5, 
        half_width=0.6, 
        stride=1, 
        kernel_size=12
    ):
        super().__init__()
        if cutoff < -0.0 or cutoff > 0.5:
            raise ValueError

        even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(even)
        self.pad_right = kernel_size // 2
        self.stride = stride

        filter = kaiser_sinc_filter1d(
            cutoff, 
            half_width, 
            kernel_size
        )
        self.register_buffer("filter", filter)

    def forward(self, x):
        return F.conv1d(
            F.pad(
                x, 
                (self.pad_left, self.pad_right), 
                mode="replicate"
            ), 
            self.filter.expand(x.size(1), -1, -1), 
            stride=self.stride, 
            groups=x.size(1)
        )

class DownSample1d(nn.Module):
    def __init__(
        self, 
        ratio=2, 
        kernel_size=None
    ):
        super().__init__()
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size,
        )

    def forward(self, x):
        return self.lowpass(x)

class AMPLayer(nn.Module):
    def __init__(
        self, 
        channels, 
        kernel_size, 
        dilation
    ):
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=(kernel_size * dilation - dilation) // 2,
                dilation=dilation,
            )
        )
        self.conv2 = weight_norm(
            nn.Conv1d(
                channels, 
                channels, 
                kernel_size, 
                padding=kernel_size // 2, 
                dilation=1
            )
        )

        self.act1 = AntiAliasActivation(channels)
        self.act2 = AntiAliasActivation(channels)

    def forward(self, x):
        y = self.conv1(self.act1(x))
        y = self.conv2(self.act2(y))

        return x + y

    def remove_weight_norm(self):
        if hasattr(self.conv1, "parametrizations") and "weight" in self.conv1.parametrizations: parametrize.remove_parametrizations(self.conv1, "weight", leave_parametrized=True)
        else: remove_weight_norm(self.conv1)

        if hasattr(self.conv2, "parametrizations") and "weight" in self.conv2.parametrizations: parametrize.remove_parametrizations(self.conv2, "weight", leave_parametrized=True)
        else: remove_weight_norm(self.conv2)

class AMPBlock(nn.Module):
    def __init__(
        self, 
        channels, 
        kernel_size, 
        dilations
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            AMPLayer(channels, kernel_size, dilation) 
            for dilation in dilations
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def remove_weight_norm(self):
        for layer in self.layers:
            layer.remove_weight_norm()

class SineGen(nn.Module):
    def __init__(
        self,
        sampling_rate,
        harmonic_num=0,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0,
        flag_for_pulse=False,
    ):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = sampling_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse

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
    def __init__(
        self,
        sampling_rate,
        harmonic_num=0,
        sine_amp=0.1,
        add_noise_std=0.003,
        voiced_threshod=0,
    ):
        super(SourceModuleHnNSF, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod)
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x):
        sine_merge = self.l_tanh(
            self.l_linear(
                self.l_sin_gen(x).to(dtype=self.l_linear.weight.dtype)
            )
        )

        return sine_merge

class BigVGANGenerator(nn.Module):
    def __init__(
        self,
        in_channel,
        upsample_initial_channel,
        upsample_rates,
        upsample_kernel_sizes,
        resblock_kernel_sizes,
        resblock_dilations,
        gin_channels,
        sample_rate,
        harmonic_num,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len (upsample_rates)
        self.f0_upsample = nn.Upsample(scale_factor=np.prod(upsample_rates))
        self.m_source = SourceModuleHnNSF(sample_rate, harmonic_num)

        self.conv_pre = weight_norm(
            nn.Conv1d(
                in_channel, 
                upsample_initial_channel, 
                kernel_size=7, 
                stride=1, 
                padding=3
            )
        )

        self.amps = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.noise_convs = nn.ModuleList()

        stride_f0s = [
            math.prod(upsample_rates[i + 1 :]) if i + 1 < self.num_upsamples else 1 
            for i in range(self.num_upsamples)
        ]

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            padding = ((k - u) // 2) if u % 2 == 0 else (u // 2 + u % 2)
                
            self.upsamples.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size=k,
                        stride=u,
                        padding=padding,
                        output_padding=u % 2,
                    )
                )
            )

            stride = stride_f0s[i]
            kernel = (1 if stride == 1 else stride * 2 - stride % 2)
            padding = (0 if stride == 1 else (kernel - stride) // 2)
            
            self.noise_convs.append(
                nn.Conv1d(
                    1,
                    upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                )
            )

        for i in range(len(self.upsamples)):
            channel = upsample_initial_channel // (2 ** (i + 1))

            self.amps.append(
                nn.ModuleList([
                    AMPBlock(channel, kernel_size=k, dilations=d)
                    for k, d in zip(resblock_kernel_sizes, resblock_dilations)
                ])
            )

        self.act_post = AntiAliasActivation(channel)
        self.conv_post = weight_norm(
            nn.Conv1d(
                channel, 
                1, 
                kernel_size=7, 
                stride=1, 
                padding=3
            )
        )

        if gin_channels != 0:
            self.cond = nn.Conv1d(
                gin_channels, 
                upsample_initial_channel, 
                1
            )

    def forward(self, x, f0, g=None):
        har_source = self.m_source(self.f0_upsample(f0[:, None, :]).transpose(-1, -2)).transpose(-1, -2)
        x = self.conv_pre(x)
        if g is not None: x += self.cond(g)  
        
        for up, amp, noise_conv in zip(self.upsamples, self.amps, self.noise_convs):
            xs = 0

            x = up(x)
            x += noise_conv(har_source)

            for layer in amp:
                xs += layer(x)

            x = xs / self.num_kernels

        return self.conv_post(self.act_post(x)).tanh()

    def remove_weight_norm(self):
        if hasattr(self.conv_pre, "parametrizations") and "weight" in self.conv_pre.parametrizations: parametrize.remove_parametrizations(self.conv_pre, "weight", leave_parametrized=True)
        else: remove_weight_norm(self.conv_pre)

        for up in self.upsamples:
            if hasattr(up, "parametrizations") and "weight" in up.parametrizations: parametrize.remove_parametrizations(up, "weight", leave_parametrized=True)
            else: remove_weight_norm(up)

        for amp in self.amps:
            amp.remove_weight_norm()

        if hasattr(self.conv_post, "parametrizations") and "weight" in self.conv_post.parametrizations: parametrize.remove_parametrizations(self.conv_post, "weight", leave_parametrized=True)
        else: remove_weight_norm(self.conv_post)

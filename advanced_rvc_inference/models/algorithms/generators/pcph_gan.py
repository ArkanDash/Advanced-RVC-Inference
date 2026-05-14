import math
from typing import Optional, Tuple
from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint


LRELU_SLOPE = 0.1


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def create_conv1d_layer(channels, kernel_size, dilation):
    return weight_norm(torch.nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation, padding=get_padding(kernel_size, dilation)))


class SnakeBeta(nn.Module):
    def __init__(self, num_channels, init=1.0, beta_init=1.0, log_scale=True):
        super().__init__()
        self.num_channels = num_channels
        self.log_scale = log_scale
        self.log_scale_factor = nn.Parameter(torch.zeros(num_channels))
        if beta_init is not None:
            self.beta = nn.Parameter(torch.ones(num_channels) * beta_init)
        else:
            self.beta = None

    def forward(self, x):
        if self.beta is not None:
            x = x + (1.0 / (self.beta + 0.000000001)) * (torch.sin(x * self.beta) ** 2)
        return x


class PchipF0UpsamplerTorch(nn.Module):
    def __init__(self, scale_factor: int = 480):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input [B, 1, T], got {x.dim()}D")

        if x.size(1) != 1:
            raise ValueError(f"Expected second dim to be 1, got {x.size(1)}")

        B, _, T_in = x.shape
        x_flat = x.squeeze(1)
        T_out = T_in * self.scale_factor

        try:
            from scipy.interpolate import PchipInterpolator
            x_np = x_flat.cpu().numpy()
            t_in = np.arange(T_in)
            t_out = np.linspace(0, 1, T_out)
            pchip = PchipInterpolator(t_in, x_np, axis=1)
            x_up = torch.from_numpy(pchip(t_out)).float().to(x.device)
            x_up = x_up.view(B, 1, T_out)
        except Exception:
            x_up = F.interpolate(x, size=T_out, mode='linear', align_corners=False)

        return x_up


class KaiserSincFilter1d(nn.Module):
    def __init__(self, channels, kernel_size=61, stride=1, rolloff=0.90, beta=10.0):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.channels = channels
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        cutoff = 0.5 * rolloff
        t = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        sinc_wave = torch.sinc(2 * cutoff * t)
        window = torch.kaiser_window(kernel_size, periodic=False, beta=beta)
        filter_kernel = sinc_wave * window
        filter_kernel = filter_kernel / filter_kernel.sum()
        self.register_buffer("kernel", filter_kernel.view(1, 1, -1).repeat(channels, 1, 1))

    def forward(self, x):
        return F.conv1d(x, self.kernel, stride=self.stride, padding=self.padding, groups=self.channels)


class pu_downsampler(nn.Module):
    def __init__(self, out_channels, downsample_factor):
        super().__init__()
        self.factor = downsample_factor
        self.mixer = nn.Conv1d(
            in_channels=downsample_factor,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=True
        )

    def forward(self, x):
        b, c, t = x.size()
        if t % self.factor != 0:
            pad_amt = self.factor - (t % self.factor)
            x = F.pad(x, (0, pad_amt))
            t = x.shape[-1]
        x = x.view(b, c, t // self.factor, self.factor)
        x = x.permute(0, 3, 2, 1)
        x = x.reshape(b, self.factor, t // self.factor)
        return self.mixer(x)


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilations: Tuple[int] = (1, 3, 5)):
        super().__init__()
        self.convs1 = self._create_convs(channels, kernel_size, dilations)
        self.convs2 = self._create_convs(channels, kernel_size, [1] * len(dilations))
        self.activation = SnakeBeta(channels)
        self.acts1 = nn.ModuleList([SnakeBeta(num_channels=channels, init=1.0, beta_init=1.0, log_scale=True) for _ in dilations])
        self.acts2 = nn.ModuleList([SnakeBeta(num_channels=channels, init=1.0, beta_init=1.0, log_scale=True) for _ in dilations])

    @staticmethod
    def _create_convs(channels: int, kernel_size: int, dilations: Tuple[int]):
        layers = torch.nn.ModuleList([create_conv1d_layer(channels, kernel_size, d) for d in dilations])
        layers.apply(init_weights)
        return layers

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None):
        for i, (conv1, conv2) in enumerate(zip(self.convs1, self.convs2)):
            x_residual = x
            xt = self.acts1[i](x)
            if x_mask is not None:
                xt = xt * x_mask
            xt = conv1(xt)
            xt = self.acts2[i](xt)
            if x_mask is not None:
                xt = xt * x_mask
            xt = conv2(xt)
            x = xt + x_residual
            if x_mask is not None:
                x = x * x_mask
        return x

    def remove_weight_norm(self):
        for conv in chain(self.convs1, self.convs2):
            remove_weight_norm(conv)


def pcph_generator_v2(
    f0: torch.Tensor,
    hop_length: int,
    sample_rate: int,
    random_init_phase: Optional[bool] = True,
    power_factor: Optional[float] = 0.1,
    max_frequency: Optional[float] = None,
    epsilon: float = 1e-6,
    pchip_upsampler: bool = True,
    use_modulo: bool = True
) -> torch.Tensor:
    batch, _, frames = f0.size()
    device = f0.device

    if pchip_upsampler:
        pchip_f0_upsampler = PchipF0UpsamplerTorch(scale_factor=hop_length)
        f0_upsampled = pchip_f0_upsampler(f0)
    else:
        f0_upsampled = F.interpolate(f0, scale_factor=hop_length, mode='linear', align_corners=False)

    if torch.all(f0_upsampled < 1.0):
        _, _, total_length = f0_upsampled.size()
        zeros = torch.zeros((batch, 1, total_length), device=device, dtype=f0_upsampled.dtype)
        return zeros, zeros

    voiced_mask = (f0_upsampled > 1.0).float()
    phase_increment = f0_upsampled / sample_rate

    if random_init_phase:
        init_phase = torch.rand((batch, 1, 1), device=device)
        phase_increment[:, :, :1] += init_phase

    phase = torch.cumsum(phase_increment.double(), dim=2) * 2.0 * torch.pi
    if use_modulo:
        phase = torch.fmod(phase, 2.0 * torch.pi)
    phase = phase.float()

    nyquist = sample_rate / 2.0
    limit_freq = max_frequency if max_frequency is not None else nyquist
    safe_f0 = torch.clamp(f0_upsampled, min=1e-5)
    N = torch.floor(limit_freq / safe_f0)

    harmonics = torch.sin(phase) * torch.sqrt(2.0 / torch.clamp(N, min=1.0))
    amp_scale = power_factor * torch.sqrt(2.0 / torch.clamp(N, min=1.0))
    pcph_harmonic_signal = harmonics * amp_scale * voiced_mask

    return pcph_harmonic_signal, voiced_mask


class SourceModulePCPH(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        hop_length: int = 480,
        random_init_phase: bool = True,
        power_factor: float = 0.1,
        add_noise_std: float = 0.003,
        use_pchip: bool = True,
        stable_init: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.random_init_phase = random_init_phase
        self.power_factor = power_factor
        self.noise_std = add_noise_std
        self.use_pchip = use_pchip
        self.stable_init = stable_init
        self.l_linear = torch.nn.Linear(1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, f0: torch.Tensor, upsample_factor: int = None):
        if f0.dim() == 2:
            f0 = f0.unsqueeze(1)
        hop = upsample_factor if upsample_factor is not None else self.hop_length

        with autocast('cuda', enabled=False):
            f0 = f0.float()
            with torch.no_grad():
                pcph_harmonic_signal, voiced_mask = pcph_generator_v2(
                    f0,
                    hop_length=hop,
                    sample_rate=self.sample_rate,
                    pchip_upsampler=self.use_pchip,
                    random_init_phase=self.random_init_phase,
                    power_factor=self.power_factor
                )

            is_fully_unvoiced = torch.all(voiced_mask == 0.0)

            if is_fully_unvoiced:
                noise = torch.randn_like(pcph_harmonic_signal)
                excitation_signal = noise * (self.power_factor / 3.0)
            else:
                noise_amp = voiced_mask * self.noise_std + (1.0 - voiced_mask) * (self.power_factor / 3.0)
                noise = torch.randn_like(pcph_harmonic_signal) * noise_amp
                excitation_signal = pcph_harmonic_signal + noise

        excitation_signal = excitation_signal.to(dtype=self.l_linear.weight.dtype)
        excitation_signal = excitation_signal.transpose(1, 2)
        excitation = self.l_tanh(self.l_linear(excitation_signal))
        excitation = excitation.transpose(1, 2)

        return excitation


class PCPH_GAN_Generator(nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels,
        sr,
        checkpointing: bool = False,
    ):
        super().__init__()
        self.leaky_relu_slope = LRELU_SLOPE
        self.checkpointing = checkpointing
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        total_ups_factor = math.prod(upsample_rates)

        self.m_source = SourceModulePCPH(
            sample_rate=sr,
            hop_length=total_ups_factor,
            random_init_phase=True,
            power_factor=0.1,
            add_noise_std=0.003,
            use_pchip=True,
            stable_init=True
        )

        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        self.har_convs = nn.ModuleList()

        ch = upsample_initial_channel

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            ch //= 2
            self.ups.append(weight_norm(ConvTranspose1d(2 * ch, ch, k, u, padding=(k - u) // 2)))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d))

            if i + 1 < len(upsample_rates):
                s_c = int(math.prod(upsample_rates[i + 1:]))
                self.har_convs.append(pu_downsampler(out_channels=ch, downsample_factor=s_c))
            else:
                self.har_convs.append(Conv1d(1, ch, kernel_size=1))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3, bias=False))

        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

        if gin_channels != 0:
            self.cond = Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x: torch.Tensor, f0: torch.Tensor, g: Optional[torch.Tensor] = None):
        har_prior = self.m_source(f0)
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.silu(x, inplace=True)
            x = self.ups[i](x)
            har_prior_injection = self.har_convs[i](har_prior)
            x = x + har_prior_injection
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.silu(x, inplace=True)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
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
        for hook in self.conv_post._forward_pre_hooks.values():
            if hook.__module__ == "torch.nn.utils.parametrizations.weight_norm" and hook.__class__.__name__ == "WeightNorm":
                remove_weight_norm(self.conv_post)
        return self

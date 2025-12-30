import math
from typing import Optional, Tuple
from itertools import chain

import torch
from torch import Tensor
import numpy as np

import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm

import torch.nn.functional as F

from torch.amp import autocast # guard
from torch.utils.checkpoint import checkpoint

from rvc.lib.algorithm.generators.pcph_gan_modules.pcph_dirichlet_fused import FusedDirichlet
from rvc.lib.algorithm.generators.pcph_gan_modules.snake_beta_fused_triton import SnakeBeta as SnakeBetaFused
from rvc.lib.algorithm.generators.pcph_gan_modules.PchipF0UpsamplerTorch import PchipF0UpsamplerTorch

# DEBUG
import torchaudio
import sys

LRELU_SLOPE = 0.1

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

def create_conv1d_layer(channels, kernel_size, dilation):
    return weight_norm(torch.nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation, padding=get_padding(kernel_size, dilation)))


class KaiserSincFilter1d(nn.Module):
    def __init__(
        self, 
        channels, 
        kernel_size=61,
        stride=1, 
        rolloff=0.90,
        beta=10.0,
    ):
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
    '''
    Space-to-Depth (Pixel Unshuffle) style downsampler for dense PCPH harmonic signal
    '''
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
        # x: [B, 1, T]
        b, c, t = x.size()

        if t % self.factor != 0:
            pad_amt = self.factor - (t % self.factor)
            x = F.pad(x, (0, pad_amt))
            t = x.shape[-1]

        x = x.view(b, c, t // self.factor, self.factor)
        x = x.permute(0, 3, 2, 1) 
        x = x.reshape(b, self.factor, t // self.factor)

        return self.mixer(x)

# Residual block ( Similar to HiFi-GAN but introduces masking as seen in VITS1 ) 
class ResBlock(torch.nn.Module):
    """
    A residual block module that applies a series of 1D convolutional layers
    with residual connections.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: Tuple[int] = (1, 3, 5),
    ):
        """
        Initializes the ResBlock.

        Args:
            channels (int): Number of input and output channels for the convolution layers.
            kernel_size (int): Size of the convolution kernel. Defaults to 3.
            dilations (Tuple[int]): Tuple of dilation rates for the convolution layers in the first set.
        """
        super().__init__()
        self.convs1 = self._create_convs(channels, kernel_size, dilations)
        self.convs2 = self._create_convs(channels, kernel_size, [1] * len(dilations))

        self.activation = SnakeBetaFused(channels)

        self.acts1 = nn.ModuleList([
            SnakeBetaFused(num_channels=channels, init=1.0, beta_init=1.0, log_scale=True) for _ in dilations])
        self.acts2 = nn.ModuleList([
            SnakeBetaFused(num_channels=channels, init=1.0, beta_init=1.0, log_scale=True) for _ in dilations])

        #self.filters1 = nn.ModuleList([KaiserSincFilter1d(channels) for _ in dilations])
        #self.filters2 = nn.ModuleList([KaiserSincFilter1d(channels) for _ in dilations])

    @staticmethod
    def _create_convs(channels: int, kernel_size: int, dilations: Tuple[int]):
        """
        Creates a list of 1D convolutional layers with specified dilations.

        Args:
            channels (int): Number of input and output channels for the convolution layers.
            kernel_size (int): Size of the convolution kernel.
            dilations (Tuple[int]): Tuple of dilation rates for each convolution layer.
        """
        layers = torch.nn.ModuleList(
            [create_conv1d_layer(channels, kernel_size, d) for d in dilations]
        )
        layers.apply(init_weights)
        return layers

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None):
        for i, (conv1, conv2) in enumerate(zip(self.convs1, self.convs2)):

            x_residual = x # Residual store

            xt = self.acts1[i](x) # Activation 1
            #xt = self.filters1[i](xt) # Anti-Aliasing 1 - Disabled, needs testing in other trial.

            if x_mask is not None: # Masking 1
                xt = xt * x_mask
            xt = conv1(xt) # Conv 1

            xt = self.acts2[i](xt) # Activation 2
            #xt = self.filters2[i](xt) # Anti-Aliasing 2 - Disabled, needs testing in other trial.

            if x_mask is not None: # Masking 2
                xt = xt * x_mask
            xt = conv2(xt) # Conv 2

            x = xt + x_residual # Residual connection

            if x_mask is not None: # Final mask
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
    """
    An optimized O(1) generator for Pseudo-Constant-Power Harmonic waveforms.
    Now using a fused Triton kernel for the Dirichlet summation.
    """
    batch, _, frames = f0.size()
    device = f0.device

    # F0 upsampling
    if pchip_upsampler:
        pchip_f0_upsampler = PchipF0UpsamplerTorch(scale_factor=hop_length)
        f0_upsampled = pchip_f0_upsampler(f0) 
    else:
        f0_upsampled = F.interpolate(
            f0, scale_factor=hop_length, mode='linear', align_corners=False
        )

    # Early return for mute / silent / all unvoiced
    if torch.all(f0_upsampled < 1.0):
        # Return all zeros for harmonic signal and for voiced mask
        _, _, total_length = f0_upsampled.size()
        zeros = torch.zeros((batch, 1, total_length), device=device, dtype=f0_upsampled.dtype)
        return zeros, zeros

    # Preparation
    # Create masks
    voiced_mask = (f0_upsampled > 1.0).float()

    # Calculate Phase (Theta)
    # phase = 2 * pi * integral(f0 / sr)
    phase_increment = f0_upsampled / sample_rate

    # Randomize initial phase
    if random_init_phase:
        init_phase = torch.rand((batch, 1, 1), device=device)
        # phase_increment[:, :, :1] = phase_increment[:, :, :1] + init_phase # Out of place
        phase_increment[:, :, :1] += init_phase # In-place

    # Cumsum
    # Multiplying by 2pi at the end to save ops during the cumsum
    phase = torch.cumsum(phase_increment.double(), dim=2) * 2.0 * torch.pi
    if use_modulo:
        phase = torch.fmod(phase, 2.0 * torch.pi)
    phase = phase.float()

    # Dynamic harmonic count (N)
    # N is the max harmonic index before aliasing (Nyquist)
    # N(t) = floor( MaxFreq / f0(t) )
    nyquist = sample_rate / 2.0
    limit_freq = max_frequency if max_frequency is not None else nyquist

    # Zero-Division safety for unvoiced segments
    safe_f0 = torch.clamp(f0_upsampled, min=1e-5)
    N = torch.floor(limit_freq / safe_f0)

    # Uses fused Triton dirichlet summation to calculate the raw harmonics
    harmonics = FusedDirichlet.apply(phase, N, epsilon)

    # Amplitude Normalization (Pseudo-Constant-Power)
    # Power Factor Normalization: amp = P * sqrt(2/N)
    amp_scale = power_factor * torch.sqrt(2.0 / torch.clamp(N, min=1.0))

    # Apply masks and scale
    pcph_harmonic_signal = harmonics * amp_scale * voiced_mask

    return pcph_harmonic_signal, voiced_mask


class SourceModulePCPH(torch.nn.Module):
    """
    Source Module using PCPH harmonics + Noise with learnable mixing.
    """
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
        super(SourceModulePCPH, self).__init__()
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
        """
        f0: (Batch, Frames) or (Batch, 1, Frames)
        """
        if f0.dim() == 2:
            f0 = f0.unsqueeze(1)

        hop = upsample_factor if upsample_factor is not None else self.hop_length

        with autocast('cuda', enabled=False):
            f0 = f0.float()

            # Generate pcph harm signal and voiced mask
            with torch.no_grad():
                pcph_harmonic_signal, voiced_mask = pcph_generator_v2(
                    f0,
                    hop_length=hop,
                    sample_rate=self.sample_rate,
                    pchip_upsampler=self.use_pchip,
                    random_init_phase=self.random_init_phase,
                    power_factor=self.power_factor
                )

            # Generate Noise
            is_fully_unvoiced = torch.all(voiced_mask == 0.0)

            if is_fully_unvoiced:
                # Generate gaussian noise
                noise = torch.randn_like(pcph_harmonic_signal)
                # Excitation is only noise
                excitation_signal = noise * (self.power_factor / 3.0)
            else:
                # Voiced: small texture noise (std). Unvoiced: louder noise (amp / 3).
                noise_amp = voiced_mask * self.noise_std + (1.0 - voiced_mask) * (self.power_factor / 3.0)
                # Generate Gaussian noise
                noise = torch.randn_like(pcph_harmonic_signal) * noise_amp
                # Merge harmonic signal and noise
                excitation_signal = pcph_harmonic_signal + noise

        # Ensure matching dtype
        excitation_signal = excitation_signal.to(dtype=self.l_linear.weight.dtype)

        # Trainable projection ( linear -> tanh )
        excitation_signal = excitation_signal.transpose(1, 2)
        excitation = self.l_tanh(self.l_linear(excitation_signal))
        excitation = excitation.transpose(1, 2)

        return excitation

class PCPH_GAN_Generator(nn.Module):
    def __init__(
        self,
        initial_channel, # 192
        resblock_kernel_sizes, # [3,7,11]
        resblock_dilation_sizes, # [[1,3,5], [1,3,5], [1,3,5]]
        upsample_rates, # [12, 10, 2, 2]
        upsample_initial_channel, # 512
        upsample_kernel_sizes, # [24, 20, 4, 4]
        gin_channels, # 256
        sr, # 48000,
        checkpointing: bool = False, # not implemented yet.
    ):
        super(PCPH_GAN_Generator, self).__init__()
        self.leaky_relu_slope = LRELU_SLOPE
        self.checkpointing = checkpointing
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        total_ups_factor = math.prod(upsample_rates)

        # PCPH handler
        self.m_source = SourceModulePCPH(
            sample_rate=sr,
            hop_length=total_ups_factor,
            random_init_phase=True,
            power_factor=0.1,
            add_noise_std=0.003,
            use_pchip=True,
            stable_init=True
        )

        # Initial feats conv, projection: 192 -> 512
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        #self.conv_pre = weight_norm(Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)) # Weight Norm, not applicable in this arcg.

        # Module containers init
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        self.har_convs = nn.ModuleList()


        ch = upsample_initial_channel # 512

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):

            # 512:  256 --> 128 --> 64 --> 32
            ch //= 2

            # Features upsampling convs
            self.ups.append(weight_norm(ConvTranspose1d(2 * ch, ch, k, u, padding=(k - u) // 2)))

            # Resblocks
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d))

            # Harmonic prior downsampling convs
            if i + 1 < len(upsample_rates):
                s_c = int(math.prod(upsample_rates[i + 1:]))
                # Space-to-depth downsampling
                self.har_convs.append(pu_downsampler(out_channels=ch, downsample_factor=s_c))
            else:
                # Projecting 1 channel -> ch channels
                self.har_convs.append(Conv1d(1, ch, kernel_size=1))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3, bias=False)) # Weight Norm

        # init weights
        #self.conv_pre.apply(init_weights) # Legacy, not needed in this architecture. IGNORE.
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

        # embedding / spk conditioning layer
        if gin_channels != 0:
            self.cond = Conv1d(gin_channels, upsample_initial_channel, 1)


    def forward(self, x: torch.Tensor, f0: torch.Tensor, g: Optional[torch.Tensor] = None):
        # x: [B, 192, Frames]
        # f0: [B, Frames]

        # Generate the prior
        har_prior = self.m_source(f0) # Output: B, 1, F

        # Pre-convolution ( Channel expansion: 192 -> 512 )
        x = self.conv_pre(x)

        # Apply spk emb conditioning
        if g is not None:
            x = x + self.cond(g)

        # Main loop
        for i in range(self.num_upsamples):

            # pre-upsampling activation
            x = F.silu(x, inplace=True)

            # Upsample features
            x = self.ups[i](x)

            # pcph harmonic injection
            har_prior_injection = self.har_convs[i](har_prior)
            x = x + har_prior_injection

            # Resblocks processing
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # Activation before post conv
        x = F.silu(x, inplace=True)

        # Post convolution
        x = self.conv_post(x)

        # Tanh
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        # Upsamplers
        for l in self.ups:
            remove_weight_norm(l)
        # ResBlocks
        for l in self.resblocks:
            l.remove_weight_norm()
        # Pre Convolution
        #remove_weight_norm(self.conv_pre)
        # Post Convolution
        remove_weight_norm(self.conv_post)

    def __prepare_scriptable__(self):
        # Upsamplers
        for l in self.ups:
            for hook in l._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    remove_weight_norm(l)
        # ResBlocks
        for l in self.resblocks:
            for hook in l._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    remove_weight_norm(l)
        # conv_pre
        # for hook in self.conv_pre._forward_pre_hooks.values():
            # if (
                # hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                # and hook.__class__.__name__ == "WeightNorm"
            # ):
                # remove_weight_norm(self.conv_pre)

        # conv_post
        for hook in self.conv_post._forward_pre_hooks.values():
            if (
                hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                and hook.__class__.__name__ == "WeightNorm"
            ):
                remove_weight_norm(self.conv_post)

        return self




# def debug_inplace_safety():
    # model = PCPH_GAN_Generator(
        # initial_channel=192,
        # resblock_kernel_sizes=[3, 7, 11],
        # resblock_dilation_sizes=[[1, 3, 5], [[1, 3, 5], [[1, 3, 5]],
        # upsample_rates=[12, 10, 2, 2],
        # upsample_initial_channel=512,
        # upsample_kernel_sizes=[24, 20, 4, 4],
        # gin_channels=256,
        # sr=48000
    # ).cuda()

    # with torch.autograd.set_detect_anomaly(True):
        # # Create dummy inputs
        # x = torch.randn(1, 192, 36).cuda().requires_grad_(True)
        # f0 = torch.randn(1, 36).cuda()
        # g = torch.randn(1, 256, 36).cuda()

        # try:
            # # Forward pass
            # output = model(x, f0, g)

            # # Backward pass
            # # This is where an inplace error would trigger a crash
            # loss = output.sum()
            # loss.backward()

            # print("✅ 100% SAFE: Backward pass completed without inplace errors.")

        # except RuntimeError as e:
            # print(f"❌ INPLACE ERROR DETECTED: {e}")

# if __name__ == "__main__":
    # debug_inplace_safety()
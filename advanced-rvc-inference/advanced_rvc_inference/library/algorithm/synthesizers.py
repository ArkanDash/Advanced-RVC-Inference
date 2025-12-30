import torch
from typing import Optional, List
import random
import sys
import os

sys.path.append(os.getcwd())

from advanced_rvc_inference.library.algorithm.commons import slice_segments, rand_slice_segments
from advanced_rvc_inference.library.algorithm.normalizing_flows import ResidualCouplingBlock, ResidualCouplingTransformersBlock
from advanced_rvc_inference.library.algorithm.encoders import PosteriorEncoder # Posterior encoder, shared between Vits1 and Vits2
from advanced_rvc_inference.library.algorithm.encoders_vits2 import TextEncoder_VITS2
from advanced_rvc_inference.library.algorithm.encoders import TextEncoder as TextEncoder_VITS1


debug_shapes = False


class Synthesizer(torch.nn.Module):
    def __init__(
        self,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        upsample_rates: list,
        upsample_initial_channel: int,
        upsample_kernel_sizes: list,
        spk_embed_dim: int,
        gin_channels: int,
        sr: int,
        use_f0: bool,
        text_enc_hidden_dim: int = 768,
        vocoder: str = "HiFi-GAN",
        checkpointing: bool = False,
        randomized: bool = True,
        vits2_mode: bool = False,
        gen_istft_n_fft: int = 120,
        gen_istft_hop_size: int = 30,
        **kwargs,
    ):
        super().__init__()
        self.segment_size = segment_size
        self.use_f0 = use_f0
        self.vocoder = vocoder
        self.randomized = randomized
        self.vits2_mode = vits2_mode

        if vits2_mode:
            self.enc_p = TextEncoder_VITS2(
                inter_channels,
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
                text_enc_hidden_dim,
                f0=use_f0,
                gin_channels=gin_channels,
            )
        else:
            self.enc_p = TextEncoder_VITS1(
                inter_channels,
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
                text_enc_hidden_dim,
                f0=use_f0,
            )
        if use_f0:
            if vocoder == "MRF HiFi-GAN":
                from advanced_rvc_inference.library.algorithm.generators import HiFiGANMRFGenerator
                self.dec = HiFiGANMRFGenerator(
                    in_channel=inter_channels,
                    upsample_initial_channel=upsample_initial_channel,
                    upsample_rates=upsample_rates,
                    upsample_kernel_sizes=upsample_kernel_sizes,
                    resblock_kernel_sizes=resblock_kernel_sizes,
                    resblock_dilations=resblock_dilation_sizes,
                    gin_channels=gin_channels,
                    sample_rate=sr,
                    harmonic_num=8,
                    checkpointing=checkpointing,
                )
                print("    ██████  Vocoder: NSF-HiFi-GAN ( MRF VARIANT )")
            elif vocoder == "RefineGAN":
                from advanced_rvc_inference.library.algorithm.generators import RefineGANGenerator
                self.dec = RefineGANGenerator(
                    sample_rate=sr,
                    downsample_rates=upsample_rates[::-1],
                    upsample_rates=upsample_rates,
                    start_channels=16,
                    num_mels=inter_channels,
                    checkpointing=checkpointing,
                )
                print("    ██████  Vocoder: RefineGAN")
            elif vocoder in ["RingFormer_v1", "RingFormer_v2"]:
                from advanced_rvc_inference.library.algorithm.generators import RingFormerGenerator
                self.dec = RingFormerGenerator(
                    initial_channel=inter_channels,
                    resblock_kernel_sizes=resblock_kernel_sizes,
                    resblock_dilation_sizes=resblock_dilation_sizes,
                    upsample_rates=upsample_rates,
                    upsample_initial_channel=upsample_initial_channel,
                    upsample_kernel_sizes=upsample_kernel_sizes,
                    gen_istft_n_fft=gen_istft_n_fft,
                    gen_istft_hop_size=gen_istft_hop_size,
                    gin_channels=gin_channels,
                    sr=sr,
                    checkpointing=checkpointing,
                )
                print(f"    ██████  Vocoder: {vocoder}")
            elif vocoder == "PCPH-GAN":
                from advanced_rvc_inference.library.algorithm.generators import PCPH_GAN_Generator
                self.dec = PCPH_GAN_Generator(
                    inter_channels,
                    resblock_kernel_sizes,
                    resblock_dilation_sizes,
                    upsample_rates,
                    upsample_initial_channel,
                    upsample_kernel_sizes,
                    gin_channels=gin_channels,
                    sr=sr,
                    checkpointing=checkpointing,
                )
                print("    ██████  Vocoder: PCPH-GAN")
            else:  # vocoder == "HiFi-GAN"
                from advanced_rvc_inference.library.algorithm.generators import HiFiGANNSFGenerator
                self.dec = HiFiGANNSFGenerator(
                    inter_channels,
                    resblock_kernel_sizes,
                    resblock_dilation_sizes,
                    upsample_rates,
                    upsample_initial_channel,
                    upsample_kernel_sizes,
                    gin_channels=gin_channels,
                    sr=sr,
                    checkpointing=checkpointing,
                )
                print("    ██████  Vocoder: NSF-HiFi-GAN")
        else:
            if vocoder in ["MRF HiFi-GAN", "RefineGAN", "RingFormer_v1", "RingFormer_v2"]:
                print(f"{vocoder} does not support training without pitch guidance.")
                self.dec = None
            else: # vocoder == "HiFi-GAN"
                from advanced_rvc_inference.library.algorithm.generators import HiFiGANGenerator
                self.dec = HiFiGANGenerator(
                    inter_channels,
                    resblock_kernel_sizes,
                    resblock_dilation_sizes,
                    upsample_rates,
                    upsample_initial_channel,
                    upsample_kernel_sizes,
                    gin_channels=gin_channels,
                    checkpointing=checkpointing,
                )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        if vits2_mode:
            self.flow = ResidualCouplingTransformersBlock(
                inter_channels,
                hidden_channels,
                5,
                1,
                3,
                gin_channels=gin_channels,
            )
        else:
            self.flow = ResidualCouplingBlock(
                inter_channels,
                hidden_channels,
                5,
                1,
                3,
                gin_channels=gin_channels,
            )

        self.emb_g = torch.nn.Embedding(spk_embed_dim, gin_channels)

    def _remove_weight_norm_from(self, module):
        """Utility to remove weight normalization from a module."""
        for hook in module._forward_pre_hooks.values():
            if getattr(hook, "__class__", None).__name__ == "WeightNorm":
                torch.nn.utils.remove_weight_norm(module)

    def remove_weight_norm(self):
        """Removes weight normalization from the model."""
        for module in [self.dec, self.flow, self.enc_q]:
            self._remove_weight_norm_from(module)

    def __prepare_scriptable__(self):
        self.remove_weight_norm()
        return self

    def forward(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: Optional[torch.Tensor] = None,
        pitchf: Optional[torch.Tensor] = None,
        spec: Optional[torch.Tensor] = None, # y
        spec_lengths: Optional[torch.Tensor] = None, # y_lengths
        ds: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the model.

        Args:
            phone (torch.Tensor): Phoneme sequence.
            phone_lengths (torch.Tensor): Lengths of the phoneme sequences.
            pitch (torch.Tensor, optional): Pitch sequence.
            pitchf (torch.Tensor, optional): Fine-grained pitch sequence.
            spek (torch.Tensor, optional): Target spectrogram.  - y
            spek_lengths (torch.Tensor, optional): Lengths of the target spectrograms. - y_lengths
            ds (torch.Tensor, optional): Speaker embedding.
        """
        g = self.emb_g(ds).unsqueeze(-1)

        if self.vits2_mode:
            m_p, logs_p, x_mask = self.enc_p(phone=phone, pitch=pitch, lengths=phone_lengths, g=g)
        else:
            m_p, logs_p, x_mask = self.enc_p(phone=phone, pitch=pitch, lengths=phone_lengths)

        if spec is not None:
            if debug_shapes:
                print(f"[DEBUG PRE-DECODER] spec_lengths shape: {spec_lengths}")
                print(f"[DEBUG PRE-DECODER] spec shape: {spec.shape}")

            z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g)

            if debug_shapes:
                print(f"[DEBUG PRE-DECODER] z shape: {z.shape}")

            z_p = self.flow(z, spec_mask, g=g)

            if self.vocoder in ["RingFormer_v1", "RingFormer_v2"]:
                if self.randomized:
                    z_slice, ids_slice = rand_slice_segments(z, spec_lengths, self.segment_size)
                    pitchf = slice_segments(pitchf, ids_slice, self.segment_size, 2)
                    o, spec, phase = self.dec(z_slice, pitchf, g=g) # f0 output

                    return o, ids_slice, x_mask, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q), (spec, phase)
                else:
                    o, spec, phase = self.dec(z, pitchf, g=g) # f0 output

                    return o, None, x_mask, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q), (spec, phase)

            else: # For HiFi-Gan, PCPH-Gan, MRF-HiFi-Gan and RefineGan training
                if self.randomized:
                    z_slice, ids_slice = rand_slice_segments(z, spec_lengths, self.segment_size)

                    if self.use_f0:
                        pitchf = slice_segments(pitchf, ids_slice, self.segment_size, 2)
                        o = self.dec(z_slice, pitchf, g=g)
                    else:
                        o = self.dec(z_slice, g=g)

                    return o, ids_slice, x_mask, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q)
                else:
                    if self.use_f0:
                        o = self.dec(z, pitchf, g=g)
                    else:
                        o = self.dec(z, g=g)

                    return o, None, x_mask, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q)
        else:
            print(" NONE SPEC ")
            return None, None, x_mask, None, (None, None, m_p, logs_p, None, None)

    @torch.jit.export
    def infer(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: Optional[torch.Tensor] = None,
        nsff0: Optional[torch.Tensor] = None,
        sid: torch.Tensor = None,
        seed: int = 0,
        rate: Optional[torch.Tensor] = None,
    ):
        """
        Inference of the model.

        Args:
            phone (torch.Tensor): Phoneme sequence.
            phone_lengths (torch.Tensor): Lengths of the phoneme sequences.
            pitch (torch.Tensor, optional): Pitch sequence.
            nsff0 (torch.Tensor, optional): Fine-grained pitch sequence.
            sid (torch.Tensor): Speaker embedding.
            rate (torch.Tensor, optional): Rate for time-stretching.
            seed (int, optional): Seed for randomization of noise.
        """
        g = self.emb_g(sid).unsqueeze(-1)

        if self.vits2_mode:
            m_p, logs_p, x_mask = self.enc_p(phone=phone, pitch=pitch, lengths=phone_lengths, g=g)
        else:
            m_p, logs_p, x_mask = self.enc_p(phone=phone, pitch=pitch, lengths=phone_lengths)

        if seed != 0:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            print(f"[INFER] Seed specified; Inference is performed in deterministic mode using seed:{seed}.")
        else:
            print(f"[INFER] Seed kept at: {seed}; Inference is performed in randomized mode.")
            rand_seed = random.randint(0, 2**32 - 1)  # 32-bit seed range
            random.seed(rand_seed)
            torch.manual_seed(rand_seed)
            torch.cuda.manual_seed_all(rand_seed)
            print(f"[INFER] Randomized seed: {rand_seed}; Exposed for reproduction.")

        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask

        if rate is not None:
            head = int(z_p.shape[2] * (1.0 - rate.item()))
            z_p, x_mask = z_p[:, :, head:], x_mask[:, :, head:]

            if self.use_f0 and nsff0 is not None:
                nsff0 = nsff0[:, head:]

        z = self.flow(z_p, x_mask, g=g, reverse=True)

        if self.vocoder in ["RingFormer_v1", "RingFormer_v2"]:
            o, _, _ = self.dec(z * x_mask, nsff0, g=g)
        else:
            o = (self.dec(z * x_mask, nsff0, g=g) if self.use_f0 else self.dec(z * x_mask, g=g))

        return o, x_mask, (z, z_p, m_p, logs_p)


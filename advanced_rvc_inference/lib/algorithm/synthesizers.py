import os
import sys
import torch

sys.path.append(os.getcwd())

from main.library.algorithm.residuals import ResidualCouplingBlock
from main.library.algorithm.encoders import TextEncoder, PosteriorEncoder
from main.library.algorithm.commons import slice_segments, rand_slice_segments

class Synthesizer(torch.nn.Module):
    def __init__(self, spec_channels, segment_size, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, spk_embed_dim, gin_channels, sr, use_f0, text_enc_hidden_dim=768, vocoder="Default", checkpointing=False, onnx=False, energy=False, **kwargs):
        super(Synthesizer, self).__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.spk_embed_dim = spk_embed_dim
        self.use_f0 = use_f0
        self.enc_p = TextEncoder(inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, float(p_dropout), text_enc_hidden_dim, f0=use_f0, energy=energy, onnx=onnx)

        if use_f0:
            if vocoder == "RefineGAN": 
                from main.library.generators.refinegan import RefineGANGenerator
                self.dec = RefineGANGenerator(sample_rate=sr, upsample_rates=upsample_rates, num_mels=inter_channels, checkpointing=checkpointing)
            elif vocoder in ["MRF-HiFi-GAN", "MRF HiFi-GAN"]: 
                from main.library.generators.mrf_hifigan import HiFiGANMRFGenerator
                self.dec = HiFiGANMRFGenerator(in_channel=inter_channels, upsample_initial_channel=upsample_initial_channel, upsample_rates=upsample_rates, upsample_kernel_sizes=upsample_kernel_sizes, resblock_kernel_sizes=resblock_kernel_sizes, resblock_dilations=resblock_dilation_sizes, gin_channels=gin_channels, sample_rate=sr, harmonic_num=8, checkpointing=checkpointing)
            else: 
                from main.library.generators.nsf_hifigan import HiFiGANNRFGenerator
                self.dec = HiFiGANNRFGenerator(inter_channels, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels, sr=sr, checkpointing=checkpointing)
        else: 
            from main.library.generators.hifigan import HiFiGANGenerator
            self.dec = HiFiGANGenerator(inter_channels, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)

        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels)
        self.emb_g = torch.nn.Embedding(self.spk_embed_dim, gin_channels)

    def remove_weight_norm(self):
        for module in [self.dec, self.flow, self.enc_q]:
            module.remove_weight_norm()

    @torch.jit.ignore
    def forward(self, phone, phone_lengths, pitch = None, pitchf = None, y = None, y_lengths = None, ds = None, energy = None):
        g = self.emb_g(ds).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths, energy)

        if y is not None:
            z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
            z_slice, ids_slice = rand_slice_segments(z, y_lengths, self.segment_size)

            return (self.dec(z_slice, slice_segments(pitchf, ids_slice, self.segment_size, 2), g=g) if self.use_f0 else self.dec(z_slice, g=g)), ids_slice, x_mask, y_mask, (z, self.flow(z, y_mask, g=g), m_p, logs_p, m_q, logs_q)
        else: return None, None, x_mask, None, (None, None, m_p, logs_p, None, None)

    @torch.jit.export
    def infer(self, phone, phone_lengths, pitch = None, nsff0 = None, sid = None, energy = None):
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths, energy)
        z_p = (m_p + logs_p.exp() * torch.randn_like(m_p) * 0.66666) * x_mask

        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec(z * x_mask, nsff0, g=g) if self.use_f0 else self.dec(z * x_mask, g=g)

        return o, x_mask, (z, z_p, m_p, logs_p)
    
class SynthesizerONNX(Synthesizer):
    def __init__(self, spec_channels, segment_size, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, spk_embed_dim, gin_channels, sr, use_f0, text_enc_hidden_dim=768, vocoder="Default", checkpointing=False, energy=False, **kwargs):
        super().__init__(spec_channels, segment_size, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, spk_embed_dim, gin_channels, sr, use_f0, text_enc_hidden_dim, vocoder, checkpointing, True, energy)

    def forward(self, phone, phone_lengths, g=None, rnd=None, pitch=None, nsff0=None, energy=None):
        g = self.emb_g(g).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths, energy)
        z_p = (m_p + logs_p.exp() * rnd) * x_mask

        z = self.flow(z_p, x_mask, g=g, reverse=True)

        return self.dec(
            (z * x_mask)[:, :, :None], 
            nsff0, 
            g=g
        ) if self.use_f0 else self.dec(
            (z * x_mask)[:, :, :None], 
            g=g
        )
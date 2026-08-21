import torch

from arvc.engine.models.algorithms.residuals import ResidualCouplingBlock
from arvc.engine.models.algorithms.encoders import TextEncoder, PosteriorEncoder, TextEncoderSVC
from arvc.engine.models.algorithms.commons import slice_segments, rand_slice_segments, sequence_mask

class Synthesizer(torch.nn.Module):
    def __init__(self, spec_channels, segment_size, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, spk_embed_dim, gin_channels, sr, use_f0, text_enc_hidden_dim=768, vocoder="Default", randomized=True, checkpointing=False, onnx=False, energy=False, **kwargs):
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
        self.vocoder = vocoder
        self.randomized = randomized
        self.enc_p = TextEncoder(inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, float(p_dropout), text_enc_hidden_dim, f0=use_f0, energy=energy, onnx=onnx)

        if use_f0:
            if vocoder == "RefineGAN": 
                from arvc.engine.models.generators.refinegan import RefineGANGenerator
                self.dec = RefineGANGenerator(
                    sample_rate=sr,
                    upsample_rates=upsample_rates,
                    num_mels=inter_channels, 
                    checkpointing=checkpointing
                )
            elif vocoder == "BigVGAN":
                from arvc.engine.models.generators.bigvgan import BigVGANGenerator
                self.dec = BigVGANGenerator(
                    in_channel=inter_channels,
                    upsample_initial_channel=upsample_initial_channel,
                    upsample_rates=upsample_rates,
                    upsample_kernel_sizes=upsample_kernel_sizes,
                    resblock_kernel_sizes=resblock_kernel_sizes,
                    resblock_dilations=resblock_dilation_sizes,
                    gin_channels=gin_channels,
                    sample_rate=sr,
                    harmonic_num=0, 
                )
            elif vocoder in ["MRF-HiFi-GAN", "MRF HiFi-GAN"]: 
                from arvc.engine.models.generators.mrf_hifigan import HiFiGANMRFGenerator
                self.dec = HiFiGANMRFGenerator(in_channel=inter_channels, upsample_initial_channel=upsample_initial_channel, upsample_rates=upsample_rates, upsample_kernel_sizes=upsample_kernel_sizes, resblock_kernel_sizes=resblock_kernel_sizes, resblock_dilations=resblock_dilation_sizes, gin_channels=gin_channels, sample_rate=sr, harmonic_num=8, checkpointing=checkpointing)
            else:
                # Default: HiFi-GAN NSF (matches VRVC)
                from arvc.engine.models.generators.nsf_hifigan import HiFiGANNRFGenerator
                self.dec = HiFiGANNRFGenerator(inter_channels, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels, sr=sr, checkpointing=checkpointing)
        else: 
            # No pitch guidance: plain HiFi-GAN (matches VRVC)
            from arvc.engine.models.generators.hifigan import HiFiGANGenerator
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
            z_p = self.flow(z, y_mask, g=g)

            if self.randomized:
                z_slice, ids_slice = rand_slice_segments(z, y_lengths, self.segment_size)
                pitch_slice = slice_segments(pitchf, ids_slice, self.segment_size, 2) if self.use_f0 else None
                if self.use_f0:
                    dec_out = self.dec(z_slice, pitch_slice, g=g)
                else:
                    dec_out = self.dec(z_slice, g=g)
                return dec_out, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)
            else:
                if self.use_f0:
                    dec_out = self.dec(z, pitchf, g=g)
                else:
                    dec_out = self.dec(z, g=g)
                return dec_out, None, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)
        else: return None, None, x_mask, None, (None, None, m_p, logs_p, None, None)

    @torch.jit.export
    def infer(self, phone, phone_lengths, pitch = None, nsff0 = None, sid = None, energy = None):
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths, energy)
        z_p = (m_p + logs_p.exp() * torch.randn_like(m_p) * 0.66666) * x_mask

        z = self.flow(z_p, x_mask, g=g, reverse=True)
        if self.use_f0:
            o = self.dec(z * x_mask, nsff0, g=g)
        else:
            o = self.dec(z * x_mask, g=g)

        return o, x_mask, (z, z_p, m_p, logs_p)
    
class SynthesizerONNX(Synthesizer):
    def __init__(self, spec_channels, segment_size, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, spk_embed_dim, gin_channels, sr, use_f0, text_enc_hidden_dim=768, vocoder="Default", checkpointing=False, energy=False, **kwargs):
        super().__init__(spec_channels, segment_size, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, spk_embed_dim, gin_channels, sr, use_f0, text_enc_hidden_dim, vocoder, checkpointing, True, energy)

    def forward(self, phone, phone_lengths, g=None, rnd=None, pitch=None, nsff0=None, energy=None):
        g = self.emb_g(g).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths, energy)
        z_p = (m_p + logs_p.exp() * rnd) * x_mask

        z = self.flow(z_p, x_mask, g=g, reverse=True)

        if self.use_f0:
            return self.dec(
                (z * x_mask)[:, :, :None], 
                nsff0, 
                g=g
            )
        else:
            return self.dec(
                (z * x_mask)[:, :, :None], 
                g=g
            )

class SynthesizerSVC(torch.nn.Module):
    def __init__(
        self, 
        spec_channels, 
        segment_size, 
        inter_channels, 
        hidden_channels, 
        filter_channels, 
        n_heads, 
        n_layers, 
        kernel_size, 
        p_dropout, 
        resblock, 
        resblock_kernel_sizes, 
        resblock_dilation_sizes, 
        upsample_rates, 
        upsample_initial_channel, 
        upsample_kernel_sizes, 
        spk_embed_dim, 
        gin_channels, 
        sr, 
        text_enc_hidden_dim=768, 
        vocoder="Default", 
        checkpointing=False, 
        onnx=False, 
        noise_scale=0.35,
        **kwargs
    ):

        super().__init__()
        self.segment_size = segment_size
        self.noise_scale = noise_scale
        self.sr = sr

        self.enc_p = TextEncoderSVC(
            inter_channels,
            hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            onnx=onnx
        )

        if vocoder == "RefineGAN": 
            from arvc.engine.models.generators.refinegan import RefineGANGenerator
            self.dec = RefineGANGenerator(
                sample_rate=sr, 
                upsample_rates=upsample_rates, 
                num_mels=inter_channels, 
                checkpointing=checkpointing
            )
        elif vocoder == "BigVGAN":
            from arvc.engine.models.generators.bigvgan import BigVGANGenerator
            self.dec = BigVGANGenerator(
                in_channel=inter_channels,
                upsample_initial_channel=upsample_initial_channel,
                upsample_rates=upsample_rates,
                upsample_kernel_sizes=upsample_kernel_sizes,
                resblock_kernel_sizes=resblock_kernel_sizes,
                resblock_dilations=resblock_dilation_sizes,
                gin_channels=gin_channels,
                sample_rate=sr,
                harmonic_num=0, 
            )
        elif vocoder in ["MRF-HiFi-GAN", "MRF HiFi-GAN"]: 
            from arvc.engine.models.generators.mrf_hifigan import HiFiGANMRFGenerator
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
                checkpointing=checkpointing
            )
        else:
            # Default: HiFi-GAN NSF (matches VRVC)
            from arvc.engine.models.generators.nsf_hifigan import HiFiGANNRFGenerator
            self.dec = HiFiGANNRFGenerator(
                inter_channels, 
                resblock_kernel_sizes, 
                resblock_dilation_sizes, 
                upsample_rates, 
                upsample_initial_channel, 
                upsample_kernel_sizes, 
                gin_channels=gin_channels, 
                sr=sr, 
                checkpointing=checkpointing,
                harmonic_num=8
            )

        self.emb_uv = torch.nn.Embedding(2, hidden_channels)
        self.emb_g = torch.nn.Embedding(spk_embed_dim, gin_channels)
        self.pre = torch.nn.Conv1d(text_enc_hidden_dim, hidden_channels, kernel_size=5, padding=2)
        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)

    def remove_weight_norm(self):
        for module in [self.dec, self.flow, self.enc_q]:
            module.remove_weight_norm()

    def forward(self, phone, phone_lengths, pitch = None, pitchf = None, spec = None, spec_lengths = None, ds = None):
        g = self.emb_g(ds.unsqueeze(0) if ds.dim() == 1 else ds).transpose(1, 2)
        phone = phone.transpose(1, 2)

        x_mask = sequence_mask(phone_lengths, phone.size(2)).unsqueeze(1).to(phone.dtype)
        x = self.pre(phone) * x_mask + self.emb_uv((pitchf > 0.0).long()).transpose(1, 2)

        _, m_p, logs_p, _ = self.enc_p(x, x_mask, f0=pitch)
        z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g)

        z_p = self.flow(z, spec_mask, g=g)
        z_slice, ids_slice = rand_slice_segments(z, spec_lengths, self.segment_size)

        return (
            self.dec(
                z_slice, 
                g=g, 
                f0=slice_segments(pitchf, ids_slice, self.segment_size, 2)
            ), 
            ids_slice, 
            x_mask, 
            spec_mask, 
            (z, z_p, m_p, logs_p, m_q, logs_q)
        )

    @torch.no_grad()
    def infer(self, phone, phone_lengths, pitch = None, nsff0 = None, sid = None):
        g = self.emb_g(sid.unsqueeze(0) if sid.dim() == 1 else sid).transpose(1, 2)
        phone = phone.transpose(1, 2)
        
        x_mask = sequence_mask(phone_lengths, phone.size(2)).unsqueeze(1).to(phone.dtype)
        x = self.pre(phone) * x_mask + self.emb_uv((nsff0 > 0.0).long()).transpose(1, 2)
        
        z_p, m_p, logs_p, c_mask = self.enc_p(x, x_mask, f0=pitch, noise_scale=self.noise_scale)
        z = self.flow(z_p, c_mask, g=g, reverse=True)
        o = self.dec(z * c_mask, g=g, f0=nsff0)

        target_len = phone_lengths * (self.sr // 100)
        if o.shape[-1] != target_len:
            o = torch.nn.functional.interpolate(
                o,
                size=target_len,
                mode="linear",
                align_corners=False
            )

        return o, x_mask, (z, z_p, m_p, logs_p)

    @torch.no_grad()
    def onnx_infer(self, phone, phone_lengths, pitch = None, nsff0 = None, sid = None):
        g = self.emb_g(sid.unsqueeze(0) if sid.dim() == 1 else sid).transpose(1, 2)
        phone = phone.transpose(1, 2)
        
        x_mask = sequence_mask(phone_lengths, phone.size(2)).unsqueeze(1).to(phone.dtype)

        z_p, _, _, c_mask = self.enc_p(
            self.pre(phone) * x_mask + self.emb_uv((nsff0 > 0.0).long()).transpose(1, 2), 
            x_mask, 
            f0=pitch, 
            noise_scale=self.noise_scale
        )

        o = self.dec(
            self.flow(
                z_p, 
                c_mask, 
                g=g, 
                reverse=True
            ) * c_mask, 
            g=g, 
            f0=nsff0
        )

        target_len = phone_lengths * (self.sr // 100)
        if o.shape[-1] != target_len:
            o = torch.nn.functional.interpolate(
                o,
                size=target_len,
                mode="linear",
                align_corners=False
            )

        return o

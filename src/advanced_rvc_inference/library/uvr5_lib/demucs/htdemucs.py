import os
import sys
import math
import torch
import random

import numpy as np

from torch import nn
from einops import rearrange
from fractions import Fraction
from torch.nn import functional as F

sys.path.append(os.getcwd())

from main.app.variables import translations
from main.library.uvr5_lib.demucs.states import capture_init
from main.library.uvr5_lib.demucs.demucs import rescale_module
from main.library.uvr5_lib.demucs.hdemucs import pad1d, spectro, ispectro, wiener, ScaledEmbedding, HEncLayer, MultiWrap, HDecLayer

def create_sin_embedding(length, dim, shift = 0, device="cpu", max_period=10000):
    assert dim % 2 == 0
    pos = shift + torch.arange(length, device=device).view(-1, 1, 1)
    half_dim = dim // 2
    adim = torch.arange(dim // 2, device=device).view(1, 1, -1)
    phase = pos / (max_period ** ((adim.to(torch.float32) / torch.tensor(half_dim - 1, dtype=torch.float32, device=device)) if str(device).startswith("ocl") else (adim / (half_dim - 1))))
    return torch.cat([phase.cos(), phase.sin()], dim=-1)

def create_2d_sin_embedding(d_model, height, width, device="cpu", max_period=10000):
    if d_model % 4 != 0: raise ValueError(translations["dims"].format(dims=d_model))
    pe = torch.zeros(d_model, height, width)
    d_model = int(d_model / 2)
    div_term = (torch.arange(0.0, d_model, 2) * -(math.log(max_period) / d_model)).exp()
    pos_w = torch.arange(0.0, width).unsqueeze(1)
    pos_h = torch.arange(0.0, height).unsqueeze(1)
    pe[0:d_model:2, :, :] = (pos_w * div_term).sin().transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = (pos_w * div_term).cos().transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = (pos_h * div_term).sin().transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1 :: 2, :, :] = (pos_h * div_term).cos().transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe[None, :].to(device)

def create_sin_embedding_cape(length, dim, batch_size, mean_normalize, augment, max_global_shift = 0.0, max_local_shift = 0.0, max_scale = 1.0, device = "cpu", max_period = 10000.0):
    assert dim % 2 == 0
    pos = 1.0 * torch.arange(length).view(-1, 1, 1) 
    pos = pos.repeat(1, batch_size, 1)  
    if mean_normalize: pos -= torch.nanmean(pos, dim=0, keepdim=True)

    if augment:
        delta = np.random.uniform(-max_global_shift, +max_global_shift, size=[1, batch_size, 1])
        delta_local = np.random.uniform(-max_local_shift, +max_local_shift, size=[length, batch_size, 1])
        log_lambdas = np.random.uniform(-np.log(max_scale), +np.log(max_scale), size=[1, batch_size, 1])
        pos = (pos + delta + delta_local) * np.exp(log_lambdas)

    pos = pos.to(device)
    half_dim = dim // 2
    adim = torch.arange(dim // 2, device=device).view(1, 1, -1)
    phase = pos / (max_period ** ((adim.to(torch.float32) / torch.tensor(half_dim - 1, dtype=torch.float32, device=device)) if str(device).startswith("ocl") else (adim / (half_dim - 1))))
    return torch.cat([phase.cos(), phase.sin()], dim=-1).float()

class MyGroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = x.transpose(1, 2)
        return super().forward(x).transpose(1, 2)
    
class LayerScale(nn.Module):
    def __init__(self, channels, init = 0, channel_last=False):
        super().__init__()
        self.channel_last = channel_last
        self.scale = nn.Parameter(torch.zeros(channels, requires_grad=True))
        self.scale.data[:] = init

    def forward(self, x):
        if self.channel_last: return self.scale * x
        else: return self.scale[:, None] * x

class MyTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu, group_norm=0, norm_first=False, norm_out=False, layer_norm_eps=1e-5, layer_scale=False, init_values=1e-4, device=None, dtype=None, sparse=False, mask_type="diag", mask_random_seed=42, sparse_attn_window=500, global_window=50, auto_sparsity=False, sparsity=0.95, batch_first=False):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first, device=device, dtype=dtype)
        self.auto_sparsity = auto_sparsity

        if group_norm:
            self.norm1 = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs)

        self.norm_out = None
        if self.norm_first & norm_out: self.norm_out = MyGroupNorm(num_groups=int(norm_out), num_channels=d_model)

        self.gamma_1 = LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        self.gamma_2 = LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        T, B, C = x.shape

        if self.norm_first:
            x = x + self.gamma_1(self._sa_block(self.norm1(x), src_mask, src_key_padding_mask))
            x = x + self.gamma_2(self._ff_block(self.norm2(x)))
            if self.norm_out: x = self.norm_out(x)
        else:
            x = self.norm1(x + self.gamma_1(self._sa_block(x, src_mask, src_key_padding_mask)))
            x = self.norm2(x + self.gamma_2(self._ff_block(x)))

        return x

class CrossTransformerEncoder(nn.Module):
    def __init__(self, dim, emb = "sin", hidden_scale = 4.0, num_heads = 8, num_layers = 6, cross_first = False, dropout = 0.0, max_positions = 1000, norm_in = True, norm_in_group = False, group_norm = False, norm_first = False, norm_out = False, max_period = 10000.0, weight_decay = 0.0, lr = None, layer_scale = False, gelu = True, sin_random_shift = 0, weight_pos_embed = 1.0, cape_mean_normalize = True, cape_augment = True, cape_glob_loc_scale = [5000.0, 1.0, 1.4], sparse_self_attn = False, sparse_cross_attn = False, mask_type = "diag", mask_random_seed = 42, sparse_attn_window = 500, global_window = 50, auto_sparsity = False, sparsity = 0.95):
        super().__init__()
        assert dim % num_heads == 0
        hidden_dim = int(dim * hidden_scale)
        self.num_layers = num_layers
        self.classic_parity = int(cross_first)
        self.emb = emb
        self.max_period = max_period
        self.weight_decay = weight_decay
        self.weight_pos_embed = weight_pos_embed
        self.sin_random_shift = sin_random_shift

        if emb == "cape":
            self.cape_mean_normalize = cape_mean_normalize
            self.cape_augment = cape_augment
            self.cape_glob_loc_scale = cape_glob_loc_scale

        if emb == "scaled": self.position_embeddings = ScaledEmbedding(max_positions, dim, scale=0.2)

        self.lr = lr
        activation = F.gelu if gelu else F.relu

        if norm_in:
            self.norm_in = nn.LayerNorm(dim)
            self.norm_in_t = nn.LayerNorm(dim)
        elif norm_in_group:
            self.norm_in = MyGroupNorm(int(norm_in_group), dim)
            self.norm_in_t = MyGroupNorm(int(norm_in_group), dim)
        else:
            self.norm_in = nn.Identity()
            self.norm_in_t = nn.Identity()

        self.layers = nn.ModuleList()
        self.layers_t = nn.ModuleList()

        kwargs_common = {
            "d_model": dim,
            "nhead": num_heads,
            "dim_feedforward": hidden_dim,
            "dropout": dropout,
            "activation": activation,
            "group_norm": group_norm,
            "norm_first": norm_first,
            "norm_out": norm_out,
            "layer_scale": layer_scale,
            "mask_type": mask_type,
            "mask_random_seed": mask_random_seed,
            "sparse_attn_window": sparse_attn_window,
            "global_window": global_window,
            "sparsity": sparsity,
            "auto_sparsity": auto_sparsity,
            "batch_first": True,
        }

        kwargs_classic_encoder = dict(kwargs_common)
        kwargs_classic_encoder.update({"sparse": sparse_self_attn})
        kwargs_cross_encoder = dict(kwargs_common)
        kwargs_cross_encoder.update({"sparse": sparse_cross_attn})

        for idx in range(num_layers):
            if idx % 2 == self.classic_parity:
                self.layers.append(MyTransformerEncoderLayer(**kwargs_classic_encoder))
                self.layers_t.append(MyTransformerEncoderLayer(**kwargs_classic_encoder))
            else:
                self.layers.append(CrossTransformerEncoderLayer(**kwargs_cross_encoder))
                self.layers_t.append(CrossTransformerEncoderLayer(**kwargs_cross_encoder))

    def forward(self, x, xt):
        B, C, Fr, T1 = x.shape

        pos_emb_2d = create_2d_sin_embedding(C, Fr, T1, x.device, self.max_period) 
        pos_emb_2d = rearrange(pos_emb_2d, "b c fr t1 -> b (t1 fr) c")

        x = rearrange(x, "b c fr t1 -> b (t1 fr) c")
        x = self.norm_in(x)
        x = x + self.weight_pos_embed * pos_emb_2d

        B, C, T2 = xt.shape
        xt = rearrange(xt, "b c t2 -> b t2 c")  

        pos_emb = self._get_pos_embedding(T2, B, C, x.device)
        pos_emb = rearrange(pos_emb, "t2 b c -> b t2 c")

        xt = self.norm_in_t(xt)
        xt = xt + self.weight_pos_embed * pos_emb

        for idx in range(self.num_layers):
            if idx % 2 == self.classic_parity:
                x = self.layers[idx](x)
                xt = self.layers_t[idx](xt)
            else:
                old_x = x
                x = self.layers[idx](x, xt)
                xt = self.layers_t[idx](xt, old_x)

        x = rearrange(x, "b (t1 fr) c -> b c fr t1", t1=T1)
        xt = rearrange(xt, "b t2 c -> b c t2")
        return x, xt

    def _get_pos_embedding(self, T, B, C, device):
        if self.emb == "sin":
            shift = random.randrange(self.sin_random_shift + 1)
            pos_emb = create_sin_embedding(T, C, shift=shift, device=device, max_period=self.max_period)
        elif self.emb == "cape":
            if self.training: pos_emb = create_sin_embedding_cape(T, C, B, device=device, max_period=self.max_period, mean_normalize=self.cape_mean_normalize, augment=self.cape_augment, max_global_shift=self.cape_glob_loc_scale[0], max_local_shift=self.cape_glob_loc_scale[1], max_scale=self.cape_glob_loc_scale[2])
            else: pos_emb = create_sin_embedding_cape(T, C, B, device=device, max_period=self.max_period, mean_normalize=self.cape_mean_normalize, augment=False)
        elif self.emb == "scaled":
            pos = torch.arange(T, device=device)
            pos_emb = self.position_embeddings(pos)[:, None]

        return pos_emb

    def make_optim_group(self):
        group = {"params": list(self.parameters()), "weight_decay": self.weight_decay}
        if self.lr is not None: group["lr"] = self.lr
        return group

class CrossTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward = 2048, dropout = 0.1, activation=F.relu, layer_norm_eps = 1e-5, layer_scale = False, init_values = 1e-4, norm_first = False, group_norm = False, norm_out = False, sparse=False, mask_type="diag", mask_random_seed=42, sparse_attn_window=500, global_window=50, sparsity=0.95, auto_sparsity=None, device=None, dtype=None, batch_first=False):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.auto_sparsity = auto_sparsity
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        self.norm_first = norm_first

        if group_norm:
            self.norm1 = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm3 = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs)
        else:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.norm_out = None
        if self.norm_first & norm_out:
            self.norm_out = MyGroupNorm(num_groups=int(norm_out), num_channels=d_model)

        self.gamma_1 = LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        self.gamma_2 = LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str): self.activation = self._get_activation_fn(activation)
        else: self.activation = activation

    def forward(self, q, k, mask=None):
        if self.norm_first:
            x = q + self.gamma_1(self._ca_block(self.norm1(q), self.norm2(k), mask))
            x = x + self.gamma_2(self._ff_block(self.norm3(x)))

            if self.norm_out: x = self.norm_out(x)
        else:
            x = self.norm1(q + self.gamma_1(self._ca_block(q, k, mask)))
            x = self.norm2(x + self.gamma_2(self._ff_block(x)))

        return x

    def _ca_block(self, q, k, attn_mask=None):
        x = self.cross_attn(q, k, k, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def _get_activation_fn(self, activation):
        if activation == "relu": return F.relu
        elif activation == "gelu": return F.gelu
        raise RuntimeError(translations["activation"].format(activation=activation))

class HTDemucs(nn.Module):
    @capture_init
    def __init__(self, sources, audio_channels=2, channels=48, channels_time=None, growth=2, nfft=4096, wiener_iters=0, end_iters=0, wiener_residual=False, cac=True, depth=4, rewrite=True, multi_freqs=None, multi_freqs_depth=3, freq_emb=0.2, emb_scale=10, emb_smooth=True, kernel_size=8, time_stride=2, stride=4, context=1, context_enc=0, norm_starts=4, norm_groups=4, dconv_mode=1, dconv_depth=2, dconv_comp=8, dconv_init=1e-3, bottom_channels=0, t_layers=5, t_emb="sin", t_hidden_scale=4.0, t_heads=8, t_dropout=0.0, t_max_positions=10000, t_norm_in=True, t_norm_in_group=False, t_group_norm=False, t_norm_first=True, t_norm_out=True, t_max_period=10000.0, t_weight_decay=0.0, t_lr=None, t_layer_scale=True, t_gelu=True, t_weight_pos_embed=1.0, t_sin_random_shift=0, t_cape_mean_normalize=True, t_cape_augment=True, t_cape_glob_loc_scale=[5000.0, 1.0, 1.4], t_sparse_self_attn=False, t_sparse_cross_attn=False, t_mask_type="diag", t_mask_random_seed=42, t_sparse_attn_window=500, t_global_window=100, t_sparsity=0.95, t_auto_sparsity=False, t_cross_first=False, rescale=0.1, samplerate=44100, segment=4 * 10, use_train_segment=True):
        super().__init__()
        self.cac = cac
        self.wiener_residual = wiener_residual
        self.audio_channels = audio_channels
        self.sources = sources
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.depth = depth
        self.bottom_channels = bottom_channels
        self.channels = channels
        self.samplerate = samplerate
        self.segment = segment
        self.use_train_segment = use_train_segment
        self.nfft = nfft
        self.hop_length = nfft // 4
        self.wiener_iters = wiener_iters
        self.end_iters = end_iters
        self.freq_emb = None
        assert wiener_iters == end_iters
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.tencoder = nn.ModuleList()
        self.tdecoder = nn.ModuleList()
        chin = audio_channels
        chin_z = chin 
        if self.cac: chin_z *= 2
        chout = channels_time or channels
        chout_z = channels
        freqs = nfft // 2

        for index in range(depth):
            norm = index >= norm_starts
            freq = freqs > 1
            stri = stride
            ker = kernel_size

            if not freq:
                assert freqs == 1
                ker = time_stride * 2
                stri = time_stride

            pad = True
            last_freq = False

            if freq and freqs <= kernel_size:
                ker = freqs
                pad = False
                last_freq = True

            kw = {
                "kernel_size": ker,
                "stride": stri,
                "freq": freq,
                "pad": pad,
                "norm": norm,
                "rewrite": rewrite,
                "norm_groups": norm_groups,
                "dconv_kw": {"depth": dconv_depth, "compress": dconv_comp, "init": dconv_init, "gelu": True},
            }

            kwt = dict(kw)
            kwt["freq"] = 0
            kwt["kernel_size"] = kernel_size
            kwt["stride"] = stride
            kwt["pad"] = True
            kw_dec = dict(kw)
            multi = False

            if multi_freqs and index < multi_freqs_depth:
                multi = True
                kw_dec["context_freq"] = False

            if last_freq:
                chout_z = max(chout, chout_z)
                chout = chout_z

            enc = HEncLayer(chin_z, chout_z, dconv=dconv_mode & 1, context=context_enc, **kw)
            if freq:
                tenc = HEncLayer(chin, chout, dconv=dconv_mode & 1, context=context_enc, empty=last_freq, **kwt)
                self.tencoder.append(tenc)

            if multi: enc = MultiWrap(enc, multi_freqs)

            self.encoder.append(enc)
            if index == 0:
                chin = self.audio_channels * len(self.sources)
                chin_z = chin
                if self.cac: chin_z *= 2

            dec = HDecLayer(chout_z, chin_z, dconv=dconv_mode & 2, last=index == 0, context=context, **kw_dec)
            if multi: dec = MultiWrap(dec, multi_freqs)

            if freq:
                tdec = HDecLayer(chout, chin, dconv=dconv_mode & 2, empty=last_freq, last=index == 0, context=context, **kwt)
                self.tdecoder.insert(0, tdec)

            self.decoder.insert(0, dec)
            chin = chout
            chin_z = chout_z
            chout = int(growth * chout)
            chout_z = int(growth * chout_z)

            if freq:
                if freqs <= kernel_size: freqs = 1
                else: freqs //= stride

            if index == 0 and freq_emb:
                self.freq_emb = ScaledEmbedding(freqs, chin_z, smooth=emb_smooth, scale=emb_scale)
                self.freq_emb_scale = freq_emb

        if rescale: rescale_module(self, reference=rescale)
        transformer_channels = channels * growth ** (depth - 1)

        if bottom_channels:
            self.channel_upsampler = nn.Conv1d(transformer_channels, bottom_channels, 1)
            self.channel_downsampler = nn.Conv1d(bottom_channels, transformer_channels, 1)
            self.channel_upsampler_t = nn.Conv1d(transformer_channels, bottom_channels, 1)
            self.channel_downsampler_t = nn.Conv1d(bottom_channels, transformer_channels, 1)
            transformer_channels = bottom_channels

        if t_layers > 0: self.crosstransformer = CrossTransformerEncoder(dim=transformer_channels, emb=t_emb, hidden_scale=t_hidden_scale, num_heads=t_heads, num_layers=t_layers, cross_first=t_cross_first, dropout=t_dropout, max_positions=t_max_positions, norm_in=t_norm_in, norm_in_group=t_norm_in_group, group_norm=t_group_norm, norm_first=t_norm_first, norm_out=t_norm_out, max_period=t_max_period, weight_decay=t_weight_decay, lr=t_lr, layer_scale=t_layer_scale, gelu=t_gelu, sin_random_shift=t_sin_random_shift, weight_pos_embed=t_weight_pos_embed, cape_mean_normalize=t_cape_mean_normalize, cape_augment=t_cape_augment, cape_glob_loc_scale=t_cape_glob_loc_scale, sparse_self_attn=t_sparse_self_attn, sparse_cross_attn=t_sparse_cross_attn, mask_type=t_mask_type, mask_random_seed=t_mask_random_seed, sparse_attn_window=t_sparse_attn_window, global_window=t_global_window, sparsity=t_sparsity, auto_sparsity=t_auto_sparsity)
        else: self.crosstransformer = None

    def _spec(self, x):
        hl = self.hop_length
        nfft = self.nfft
        assert hl == nfft // 4
        le = int(math.ceil(x.shape[-1] / hl))
        pad = hl // 2 * 3
        x = pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode="reflect")
        z = spectro(x, nfft, hl)[..., :-1, :]
        assert z.shape[-1] == le + 4, (z.shape, x.shape, le)
        z = z[..., 2 : 2 + le]
        return z

    def _ispec(self, z, length=None, scale=0):
        hl = self.hop_length // (4**scale)
        z = F.pad(z, (0, 0, 0, 1))
        z = F.pad(z, (2, 2))
        pad = hl // 2 * 3
        le = hl * int(math.ceil(length / hl)) + 2 * pad
        x = ispectro(z, hl, length=le)
        x = x[..., pad : pad + length]
        return x

    def _magnitude(self, z):
        if self.cac:
            B, C, Fr, T = z.shape
            m = torch.view_as_real(z).permute(0, 1, 4, 2, 3)
            m = m.reshape(B, C * 2, Fr, T)
        else: m = z.abs()
        return m

    def _mask(self, z, m):
        niters = self.wiener_iters
        if self.cac:
            B, S, C, Fr, T = m.shape
            out = m.view(B, S, -1, 2, Fr, T).permute(0, 1, 2, 4, 5, 3)
            out = torch.view_as_complex(out.contiguous())
            return out
        
        if self.training: niters = self.end_iters

        if niters < 0:
            z = z[:, None]
            return z / (1e-8 + z.abs()) * m
        else: return self._wiener(m, z, niters)

    def _wiener(self, mag_out, mix_stft, niters):
        init = mix_stft.dtype
        wiener_win_len = 300
        residual = self.wiener_residual
        B, S, C, Fq, T = mag_out.shape
        mag_out = mag_out.permute(0, 4, 3, 2, 1)
        mix_stft = torch.view_as_real(mix_stft.permute(0, 3, 2, 1))

        outs = []

        for sample in range(B):
            pos = 0
            out = []

            for pos in range(0, T, wiener_win_len):
                frame = slice(pos, pos + wiener_win_len)
                z_out = wiener(mag_out[sample, frame], mix_stft[sample, frame], niters, residual=residual)
                out.append(z_out.transpose(-1, -2))

            outs.append(torch.cat(out, dim=0))

        out = torch.view_as_complex(torch.stack(outs, 0))
        out = out.permute(0, 4, 3, 2, 1).contiguous()

        if residual: out = out[:, :-1]
        assert list(out.shape) == [B, S, C, Fq, T]
        return out.to(init)

    def valid_length(self, length):
        if not self.use_train_segment: return length
        
        training_length = int(self.segment * self.samplerate)
        if training_length < length: raise ValueError(translations["length_or_training_length"].format(length=length, training_length=training_length))
        
        return training_length

    def forward(self, mix):
        length = mix.shape[-1]
        length_pre_pad = None

        if self.use_train_segment:
            if self.training: self.segment = Fraction(mix.shape[-1], self.samplerate)
            else:
                training_length = int(self.segment * self.samplerate)

                if mix.shape[-1] < training_length:
                    length_pre_pad = mix.shape[-1]
                    mix = F.pad(mix, (0, training_length - length_pre_pad))

        z = self._spec(mix)
        mag = self._magnitude(z).to(mix.device)
        x = mag
        B, C, Fq, T = x.shape
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)
        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)

        saved, saved_t, lengths, lengths_t = [], [], [], []

        for idx, encode in enumerate(self.encoder):
            lengths.append(x.shape[-1])
            inject = None

            if idx < len(self.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = self.tencoder[idx]
                xt = tenc(xt)

                if not tenc.empty: saved_t.append(xt)
                else: inject = xt

            x = encode(x, inject)
            if idx == 0 and self.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + self.freq_emb_scale * emb

            saved.append(x)

        if self.crosstransformer:
            if self.bottom_channels:
                b, c, f, t = x.shape
                x = rearrange(x, "b c f t-> b c (f t)")
                x = self.channel_upsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = self.channel_upsampler_t(xt)

            x, xt = self.crosstransformer(x, xt)

            if self.bottom_channels:
                x = rearrange(x, "b c f t-> b c (f t)")
                x = self.channel_downsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = self.channel_downsampler_t(xt)

        for idx, decode in enumerate(self.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            offset = self.depth - len(self.tdecoder)

            if idx >= offset:
                tdec = self.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)

                if tdec.empty:
                    assert pre.shape[2] == 1, pre.shape
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)

        assert len(saved) == 0
        assert len(lengths_t) == 0
        assert len(saved_t) == 0

        S = len(self.sources)
        x = x.view(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]
        device_type = x.device.type
        device_load = f"{device_type}:{x.device.index}" if not device_type == "mps" else device_type
        x_is_other_gpu = not device_type in ["cuda", "cpu"]
        if x_is_other_gpu: x = x.cpu()
        zout = self._mask(z, x)

        if self.use_train_segment: x = self._ispec(zout, length) if self.training else self._ispec(zout, training_length)
        else: x = self._ispec(zout, length)

        if x_is_other_gpu: x = x.to(device_load)

        if self.use_train_segment: xt = xt.view(B, S, -1, length) if self.training else xt.view(B, S, -1, training_length)
        else: xt = xt.view(B, S, -1, length)

        xt = xt * stdt[:, None] + meant[:, None]
        x = xt + x

        if length_pre_pad: x = x[..., :length_pre_pad]
        return x
import os
import sys
import math
import torch

from torch import nn
from copy import deepcopy

from torch.nn import functional as F

sys.path.append(os.getcwd())

from main.library.uvr5_lib.demucs.states import capture_init
from main.library.uvr5_lib.demucs.demucs import DConv, rescale_module

def spectro(x, n_fft=512, hop_length=None, pad=0):
    *other, length = x.shape
    x = x.reshape(-1, length)
    device_type = x.device.type
    is_other_gpu = not device_type in ["cuda", "cpu"]
    if is_other_gpu: x = x.cpu()
    z = torch.stft(x, n_fft * (1 + pad), hop_length or n_fft // 4, window=torch.hann_window(n_fft).to(x), win_length=n_fft, normalized=True, center=True, return_complex=True, pad_mode="reflect")
    _, freqs, frame = z.shape
    return z.view(*other, freqs, frame)

def ispectro(z, hop_length=None, length=None, pad=0):
    *other, freqs, frames = z.shape
    n_fft = 2 * freqs - 2
    z = z.view(-1, freqs, frames)
    win_length = n_fft // (1 + pad)
    device_type = z.device.type
    is_other_gpu = not device_type in ["cuda", "cpu"]
    if is_other_gpu: z = z.cpu()
    x = torch.istft(z, n_fft, hop_length, window=torch.hann_window(win_length).to(z.real), win_length=win_length, normalized=True, length=length, center=True)
    _, length = x.shape
    return x.view(*other, length)

def atan2(y, x):
    pi = 2 * torch.asin(torch.tensor(1.0))
    x += ((x == 0) & (y == 0)) * 1.0
    out = torch.atan(y / x)
    out += ((y >= 0) & (x < 0)) * pi
    out -= ((y < 0) & (x < 0)) * pi
    out *= 1 - ((y > 0) & (x == 0)) * 1.0
    out += ((y > 0) & (x == 0)) * (pi / 2)
    out *= 1 - ((y < 0) & (x == 0)) * 1.0
    out += ((y < 0) & (x == 0)) * (-pi / 2)
    return out

def _norm(x):
    return x[..., 0].abs() ** 2 + x[..., 1].abs() ** 2

def _mul_add(a, b, out = None):
    target_shape = torch.Size([max(sa, sb) for (sa, sb) in zip(a.shape, b.shape)])
    if out is None or out.shape != target_shape: out = torch.zeros(target_shape, dtype=a.dtype, device=a.device)

    if out is a:
        real_a = a[..., 0]
        out[..., 0] = out[..., 0] + (real_a * b[..., 0] - a[..., 1] * b[..., 1])
        out[..., 1] = out[..., 1] + (real_a * b[..., 1] + a[..., 1] * b[..., 0])
    else:
        out[..., 0] = out[..., 0] + (a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1])
        out[..., 1] = out[..., 1] + (a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0])

    return out

def _mul(a, b, out = None):
    target_shape = torch.Size([max(sa, sb) for (sa, sb) in zip(a.shape, b.shape)])
    if out is None or out.shape != target_shape: out = torch.zeros(target_shape, dtype=a.dtype, device=a.device)

    if out is a:
        real_a = a[..., 0]
        out[..., 0] = real_a * b[..., 0] - a[..., 1] * b[..., 1]
        out[..., 1] = real_a * b[..., 1] + a[..., 1] * b[..., 0]
    else:
        out[..., 0] = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]
        out[..., 1] = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]

    return out

def _inv(z, out = None):
    ez = _norm(z)
    if out is None or out.shape != z.shape: out = torch.zeros_like(z)

    out[..., 0] = z[..., 0] / ez
    out[..., 1] = -z[..., 1] / ez

    return out

def _conj(z, out = None):
    if out is None or out.shape != z.shape: out = torch.zeros_like(z)

    out[..., 0] = z[..., 0]
    out[..., 1] = -z[..., 1]

    return out

def _invert(M, out = None):
    nb_channels = M.shape[-2]
    if out is None or out.shape != M.shape: out = torch.empty_like(M)

    if nb_channels == 1: out = _inv(M, out)
    elif nb_channels == 2:
        det = _mul(M[..., 0, 0, :], M[..., 1, 1, :])
        det = det - _mul(M[..., 0, 1, :], M[..., 1, 0, :])
        invDet = _inv(det)
        out[..., 0, 0, :] = _mul(invDet, M[..., 1, 1, :], out[..., 0, 0, :])
        out[..., 1, 0, :] = _mul(-invDet, M[..., 1, 0, :], out[..., 1, 0, :])
        out[..., 0, 1, :] = _mul(-invDet, M[..., 0, 1, :], out[..., 0, 1, :])
        out[..., 1, 1, :] = _mul(invDet, M[..., 0, 0, :], out[..., 1, 1, :])
    else: raise Exception("Torch == 2 Channels")
    return out

def expectation_maximization(y, x, iterations = 2, eps = 1e-10, batch_size = 200):
    (nb_frames, nb_bins, nb_channels) = x.shape[:-1]
    nb_sources = y.shape[-1]
    regularization = torch.cat((torch.eye(nb_channels, dtype=x.dtype, device=x.device)[..., None], torch.zeros((nb_channels, nb_channels, 1), dtype=x.dtype, device=x.device)), dim=2)
    regularization = (torch.as_tensor(eps)).sqrt() * (regularization[None, None, ...].expand((-1, nb_bins, -1, -1, -1)))
    R = [torch.zeros((nb_bins, nb_channels, nb_channels, 2), dtype=x.dtype, device=x.device) for j in range(nb_sources)]
    weight = torch.zeros((nb_bins,), dtype=x.dtype, device=x.device)
    v = torch.zeros((nb_frames, nb_bins, nb_sources), dtype=x.dtype, device=x.device)

    for _ in range(iterations):
        v = torch.mean(y[..., 0, :].abs() ** 2 + y[..., 1, :].abs() ** 2, dim=-2)
        for j in range(nb_sources):
            R[j] = torch.tensor(0.0, device=x.device)

            weight = torch.tensor(eps, device=x.device)
            pos = 0
            batch_size = batch_size if batch_size else nb_frames

            while pos < nb_frames:
                t = torch.arange(pos, min(nb_frames, pos + batch_size))
                pos = int(t[-1]) + 1

                R[j] = R[j] + _covariance(y[t, ..., j]).sum(dim=0)
                weight = weight + v[t, ..., j].sum(dim=0)

            R[j] = R[j] / weight[..., None, None, None]
            weight = torch.zeros_like(weight)

        if y.requires_grad: y = y.clone()

        pos = 0

        while pos < nb_frames:
            t = torch.arange(pos, min(nb_frames, pos + batch_size))
            pos = int(t[-1]) + 1

            y[t, ...] = torch.tensor(0.0, device=x.device, dtype=x.dtype)

            Cxx = regularization

            for j in range(nb_sources):
                Cxx = Cxx + (v[t, ..., j, None, None, None] * R[j][None, ...].clone())

            inv_Cxx = _invert(Cxx)

            for j in range(nb_sources):
                gain = torch.zeros_like(inv_Cxx)
                indices = torch.cartesian_prod(torch.arange(nb_channels), torch.arange(nb_channels), torch.arange(nb_channels))

                for index in indices:
                    gain[:, :, index[0], index[1], :] = _mul_add(R[j][None, :, index[0], index[2], :].clone(), inv_Cxx[:, :, index[2], index[1], :], gain[:, :, index[0], index[1], :])

                gain = gain * v[t, ..., None, None, None, j]

                for i in range(nb_channels):
                    y[t, ..., j] = _mul_add(gain[..., i, :], x[t, ..., i, None, :], y[t, ..., j])

    return y, v, R

def wiener(targets_spectrograms, mix_stft, iterations = 1, softmask = False, residual = False, scale_factor = 10.0, eps = 1e-10):
    if softmask: y = mix_stft[..., None] * (targets_spectrograms / (eps + targets_spectrograms.sum(dim=-1, keepdim=True).to(mix_stft.dtype)))[..., None, :]
    else:
        angle = atan2(mix_stft[..., 1], mix_stft[..., 0])[..., None]
        nb_sources = targets_spectrograms.shape[-1]
        y = torch.zeros(mix_stft.shape + (nb_sources,), dtype=mix_stft.dtype, device=mix_stft.device)
        y[..., 0, :] = targets_spectrograms * angle.cos()
        y[..., 1, :] = targets_spectrograms * angle.sin()

    if residual: y = torch.cat([y, mix_stft[..., None] - y.sum(dim=-1, keepdim=True)], dim=-1)
    if iterations == 0: return y

    max_abs = torch.as_tensor(1.0, dtype=mix_stft.dtype, device=mix_stft.device).max(_norm(mix_stft).sqrt().max() / scale_factor)
    mix_stft = mix_stft / max_abs
    y = y / max_abs
    y = expectation_maximization(y, mix_stft, iterations, eps=eps)[0]
    y = y * max_abs

    return y

def _covariance(y_j):
    (nb_frames, nb_bins, nb_channels) = y_j.shape[:-1]

    Cj = torch.zeros((nb_frames, nb_bins, nb_channels, nb_channels, 2), dtype=y_j.dtype, device=y_j.device)
    indices = torch.cartesian_prod(torch.arange(nb_channels), torch.arange(nb_channels))

    for index in indices:
        Cj[:, :, index[0], index[1], :] = _mul_add(y_j[:, :, index[0], :], _conj(y_j[:, :, index[1], :]), Cj[:, :, index[0], index[1], :])

    return Cj

def pad1d(x, paddings, mode = "constant", value = 0.0):
    x0 = x
    length = x.shape[-1]
    padding_left, padding_right = paddings

    if mode == "reflect":
        max_pad = max(padding_left, padding_right)

        if length <= max_pad:
            extra_pad = max_pad - length + 1
            extra_pad_right = min(padding_right, extra_pad)
            extra_pad_left = extra_pad - extra_pad_right
            paddings = (padding_left - extra_pad_left, padding_right - extra_pad_right)
            x = F.pad(x, (extra_pad_left, extra_pad_right))

    out = F.pad(x, paddings, mode, value)

    assert out.shape[-1] == length + padding_left + padding_right
    assert (out[..., padding_left : padding_left + length] == x0).all()
    return out

class ScaledEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, scale = 10.0, smooth=False):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        if smooth:
            weight = torch.cumsum(self.embedding.weight.data, dim=0)
            weight = weight / torch.arange(1, num_embeddings + 1).to(weight).sqrt()[:, None]
            self.embedding.weight.data[:] = weight

        self.embedding.weight.data /= scale
        self.scale = scale

    @property
    def weight(self):
        return self.embedding.weight * self.scale

    def forward(self, x):
        return self.embedding(x) * self.scale

class HEncLayer(nn.Module):
    def __init__(self, chin, chout, kernel_size=8, stride=4, norm_groups=1, empty=False, freq=True, dconv=True, norm=True, context=0, dconv_kw={}, pad=True, rewrite=True):
        super().__init__()
        norm_fn = lambda d: nn.Identity()  
        if norm: norm_fn = lambda d: nn.GroupNorm(norm_groups, d)  
        pad = kernel_size // 4 if pad else 0

        klass = nn.Conv1d
        self.freq = freq
        self.kernel_size = kernel_size
        self.stride = stride
        self.empty = empty
        self.norm = norm
        self.pad = pad

        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            pad = [pad, 0]
            klass = nn.Conv2d
            
        self.conv = klass(chin, chout, kernel_size, stride, pad)
        if self.empty: return
        
        self.norm1 = norm_fn(chout)
        self.rewrite = None

        if rewrite:
            self.rewrite = klass(chout, 2 * chout, 1 + 2 * context, 1, context)
            self.norm2 = norm_fn(2 * chout)

        self.dconv = None
        if dconv: self.dconv = DConv(chout, **dconv_kw)

    def forward(self, x, inject=None):
        if not self.freq and x.dim() == 4:
            B, C, Fr, T = x.shape
            x = x.view(B, -1, T)

        if not self.freq:
            le = x.shape[-1]
            if not le % self.stride == 0: x = F.pad(x, (0, self.stride - (le % self.stride)))

        y = self.conv(x)
        if self.empty: return y
        
        if inject is not None:
            assert inject.shape[-1] == y.shape[-1], (inject.shape, y.shape)

            if inject.dim() == 3 and y.dim() == 4: inject = inject[:, :, None]
            y = y + inject
            
        y = F.gelu(self.norm1(y))

        if self.dconv:
            if self.freq:
                B, C, Fr, T = y.shape
                y = y.permute(0, 2, 1, 3).reshape(-1, C, T)

            y = self.dconv(y)
            if self.freq: y = y.view(B, Fr, C, T).permute(0, 2, 1, 3)

        if self.rewrite:
            z = self.norm2(self.rewrite(y))
            z = F.glu(z, dim=1)
        else: z = y

        return z

class MultiWrap(nn.Module):
    def __init__(self, layer, split_ratios):
        super().__init__()
        self.split_ratios = split_ratios
        self.layers = nn.ModuleList()
        self.conv = isinstance(layer, HEncLayer)
        assert not layer.norm
        assert layer.freq
        assert layer.pad

        if not self.conv: assert not layer.context_freq

        for _ in range(len(split_ratios) + 1):
            lay = deepcopy(layer)

            if self.conv: lay.conv.padding = (0, 0)
            else: lay.pad = False

            for m in lay.modules():
                if hasattr(m, "reset_parameters"): m.reset_parameters()

            self.layers.append(lay)

    def forward(self, x, skip=None, length=None):
        B, C, Fr, T = x.shape
        ratios = list(self.split_ratios) + [1]
        start = 0
        outs = []

        for ratio, layer in zip(ratios, self.layers):
            if self.conv:
                pad = layer.kernel_size // 4

                if ratio == 1:
                    limit = Fr
                    frames = -1
                else:
                    limit = int(round(Fr * ratio))
                    le = limit - start

                    if start == 0: le += pad
                        
                    frames = round((le - layer.kernel_size) / layer.stride + 1)
                    limit = start + (frames - 1) * layer.stride + layer.kernel_size
                    
                    if start == 0: limit -= pad

                assert limit - start > 0, (limit, start)
                assert limit <= Fr, (limit, Fr)

                y = x[:, :, start:limit, :]

                if start == 0: y = F.pad(y, (0, 0, pad, 0))
                if ratio == 1: y = F.pad(y, (0, 0, 0, pad))

                outs.append(layer(y))
                start = limit - layer.kernel_size + layer.stride
            else:
                limit = Fr if ratio == 1 else int(round(Fr * ratio))

                last = layer.last
                layer.last = True

                y = x[:, :, start:limit]
                s = skip[:, :, start:limit]
                out, _ = layer(y, s, None)

                if outs:
                    outs[-1][:, :, -layer.stride :] += out[:, :, : layer.stride] - layer.conv_tr.bias.view(1, -1, 1, 1)
                    out = out[:, :, layer.stride :]

                if ratio == 1: out = out[:, :, : -layer.stride // 2, :]
                if start == 0: out = out[:, :, layer.stride // 2 :, :]

                outs.append(out)
                layer.last = last
                start = limit

        out = torch.cat(outs, dim=2)
        if not self.conv and not last: out = F.gelu(out)

        if self.conv: return out
        else: return out, None

class HDecLayer(nn.Module):
    def __init__(self, chin, chout, last=False, kernel_size=8, stride=4, norm_groups=1, empty=False, freq=True, dconv=True, norm=True, context=1, dconv_kw={}, pad=True, context_freq=True, rewrite=True):
        super().__init__()
        norm_fn = lambda d: nn.Identity()  

        if norm: norm_fn = lambda d: nn.GroupNorm(norm_groups, d)  
        pad = kernel_size // 4 if pad else 0
            
        self.pad = pad
        self.last = last
        self.freq = freq
        self.chin = chin
        self.empty = empty
        self.stride = stride
        self.kernel_size = kernel_size
        self.norm = norm
        self.context_freq = context_freq
        klass = nn.Conv1d
        klass_tr = nn.ConvTranspose1d

        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            klass = nn.Conv2d
            klass_tr = nn.ConvTranspose2d

        self.conv_tr = klass_tr(chin, chout, kernel_size, stride)
        self.norm2 = norm_fn(chout)

        if self.empty: return
        self.rewrite = None

        if rewrite:
            if context_freq: self.rewrite = klass(chin, 2 * chin, 1 + 2 * context, 1, context)
            else: self.rewrite = klass(chin, 2 * chin, [1, 1 + 2 * context], 1, [0, context])

            self.norm1 = norm_fn(2 * chin)

        self.dconv = None
        if dconv: self.dconv = DConv(chin, **dconv_kw)

    def forward(self, x, skip, length):
        if self.freq and x.dim() == 3:
            B, C, T = x.shape
            x = x.view(B, self.chin, -1, T)

        if not self.empty:
            x = x + skip

            y = F.glu(self.norm1(self.rewrite(x)), dim=1) if self.rewrite else x
                
            if self.dconv:
                if self.freq:
                    B, C, Fr, T = y.shape
                    y = y.permute(0, 2, 1, 3).reshape(-1, C, T)

                y = self.dconv(y)

                if self.freq: y = y.view(B, Fr, C, T).permute(0, 2, 1, 3)
        else:
            y = x
            assert skip is None

        z = self.norm2(self.conv_tr(y))

        if self.freq:
            if self.pad: z = z[..., self.pad : -self.pad, :]
        else:
            z = z[..., self.pad : self.pad + length]
            assert z.shape[-1] == length, (z.shape[-1], length)

        if not self.last: z = F.gelu(z)
        return z, y

class HDemucs(nn.Module):
    @capture_init
    def __init__(self, sources, audio_channels=2, channels=48, channels_time=None, growth=2, nfft=4096, wiener_iters=0, end_iters=0, wiener_residual=False, cac=True, depth=6, rewrite=True, hybrid=True, hybrid_old=False, multi_freqs=None, multi_freqs_depth=2, freq_emb=0.2, emb_scale=10, emb_smooth=True, kernel_size=8, time_stride=2, stride=4, context=1, context_enc=0, norm_starts=4, norm_groups=4, dconv_mode=1, dconv_depth=2, dconv_comp=4, dconv_attn=4, dconv_lstm=4, dconv_init=1e-4, rescale=0.1, samplerate=44100, segment=4 * 10):
        super().__init__()
        self.cac = cac
        self.wiener_residual = wiener_residual
        self.audio_channels = audio_channels
        self.sources = sources
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.depth = depth
        self.channels = channels
        self.samplerate = samplerate
        self.segment = segment
        self.nfft = nfft
        self.hop_length = nfft // 4
        self.wiener_iters = wiener_iters
        self.end_iters = end_iters
        self.freq_emb = None
        self.hybrid = hybrid
        self.hybrid_old = hybrid_old
        if hybrid_old: assert hybrid
        if hybrid: assert wiener_iters == end_iters
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        if hybrid:
            self.tencoder = nn.ModuleList()
            self.tdecoder = nn.ModuleList()

        chin = audio_channels
        chin_z = chin  

        if self.cac: chin_z *= 2

        chout = channels_time or channels
        chout_z = channels
        freqs = nfft // 2

        for index in range(depth):
            lstm = index >= dconv_lstm
            attn = index >= dconv_attn
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
                "dconv_kw": {"lstm": lstm, "attn": attn, "depth": dconv_depth, "compress": dconv_comp, "init": dconv_init, "gelu": True},
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
            if hybrid and freq:
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

            if hybrid and freq:
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

    def _spec(self, x):
        hl = self.hop_length
        nfft = self.nfft

        if self.hybrid:
            assert hl == nfft // 4
            le = int(math.ceil(x.shape[-1] / hl))
            pad = hl // 2 * 3
            x = pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode="reflect") if not self.hybrid_old else pad1d(x, (pad, pad + le * hl - x.shape[-1]))

        z = spectro(x, nfft, hl)[..., :-1, :]
        if self.hybrid:
            assert z.shape[-1] == le + 4, (z.shape, x.shape, le)
            z = z[..., 2 : 2 + le]

        return z

    def _ispec(self, z, length=None, scale=0):
        hl = self.hop_length // (4**scale)
        z = F.pad(z, (0, 0, 0, 1))

        if self.hybrid:
            z = F.pad(z, (2, 2))
            pad = hl // 2 * 3
            le = hl * int(math.ceil(length / hl)) + 2 * pad if not self.hybrid_old else hl * int(math.ceil(length / hl))
            x = ispectro(z, hl, length=le)
            x = x[..., pad : pad + length] if not self.hybrid_old else x[..., :length]
        else: x = ispectro(z, hl, length)

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

    def forward(self, mix):
        x = mix
        length = x.shape[-1]
        z = self._spec(mix)
        mag = self._magnitude(z).to(mix.device)
        x = mag
        B, C, Fq, T = x.shape
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)

        if self.hybrid:
            xt = mix
            meant = xt.mean(dim=(1, 2), keepdim=True)
            stdt = xt.std(dim=(1, 2), keepdim=True)
            xt = (xt - meant) / (1e-5 + stdt)

        saved, saved_t, lengths, lengths_t = [], [], [], []

        for idx, encode in enumerate(self.encoder):
            lengths.append(x.shape[-1])
            inject = None

            if self.hybrid and idx < len(self.tencoder):
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

        x = torch.zeros_like(x)
        if self.hybrid: xt = torch.zeros_like(x)

        for idx, decode in enumerate(self.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))

            if self.hybrid: offset = self.depth - len(self.tdecoder)

            if self.hybrid and idx >= offset:
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
        x = self._ispec(zout, length)
        if x_is_other_gpu: x = x.to(device_load)

        if self.hybrid:
            xt = xt.view(B, S, -1, length)
            xt = xt * stdt[:, None] + meant[:, None]
            x = xt + x

        return x
import os
import sys
import math
import torch
import inspect

import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.getcwd())

from main.library.uvr5_lib.demucs.utils import center_trim
from main.library.uvr5_lib.demucs.states import capture_init

def unfold(a, kernel_size, stride):
    *shape, length = a.shape
    n_frames = math.ceil(length / stride)
    tgt_length = (n_frames - 1) * stride + kernel_size
    a = F.pad(a, (0, tgt_length - length))
    strides = list(a.stride())
    assert strides[-1] == 1
    strides = strides[:-1] + [stride, 1]
    return a.as_strided([*shape, n_frames, kernel_size], strides)

def rescale_conv(conv, reference):
    scale = (conv.weight.std().detach() / reference) ** 0.5
    conv.weight.data /= scale
    if conv.bias is not None: conv.bias.data /= scale

def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d)): rescale_conv(sub, reference)

class BLSTM(nn.Module):
    def __init__(self, dim, layers=1, max_steps=None, skip=False):
        super().__init__()
        assert max_steps is None or max_steps % 4 == 0
        self.max_steps = max_steps
        self.lstm = nn.LSTM(bidirectional=True, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = nn.Linear(2 * dim, dim)
        self.skip = skip

    def forward(self, x):
        B, C, T = x.shape
        y = x
        framed = False

        if self.max_steps is not None and T > self.max_steps:
            width = self.max_steps
            stride = width // 2
            frames = unfold(x, width, stride)
            nframes = frames.shape[2]
            framed = True
            x = frames.permute(0, 2, 1, 3).reshape(-1, C, width)

        x = x.permute(2, 0, 1)
        x = self.lstm(x)[0]
        x = self.linear(x)
        x = x.permute(1, 2, 0)

        if framed:
            out = []
            frames = x.reshape(B, -1, C, width)
            limit = stride // 2

            for k in range(nframes):
                if k == 0: out.append(frames[:, k, :, :-limit])
                elif k == nframes - 1: out.append(frames[:, k, :, limit:])
                else: out.append(frames[:, k, :, limit:-limit])

            out = torch.cat(out, -1)
            out = out[..., :T]
            x = out

        if self.skip: x = x + y
        return x

class LayerScale(nn.Module):
    def __init__(self, channels, init = 0):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(channels, requires_grad=True))
        self.scale.data[:] = init

    def forward(self, x):
        return self.scale[:, None] * x

class DConv(nn.Module):
    def __init__(self, channels, compress = 4, depth = 2, init = 1e-4, norm=True, attn=False, heads=4, ndecay=4, lstm=False, gelu=True, kernel=3, dilate=True):
        super().__init__()
        assert kernel % 2 == 1
        self.channels = channels
        self.compress = compress
        self.depth = abs(depth)
        dilate = depth > 0
        norm_fn = lambda d: nn.Identity()  
        if norm: norm_fn = lambda d: nn.GroupNorm(1, d)  
        hidden = int(channels / compress)
        act = nn.GELU if gelu else nn.ReLU
        self.layers = nn.ModuleList([])

        for d in range(self.depth):
            dilation = 2**d if dilate else 1
            padding = dilation * (kernel // 2)

            mods = [nn.Conv1d(channels, hidden, kernel, dilation=dilation, padding=padding), norm_fn(hidden), act(), nn.Conv1d(hidden, 2 * channels, 1), norm_fn(2 * channels), nn.GLU(1), LayerScale(channels, init)]

            if attn: mods.insert(3, LocalState(hidden, heads=heads, ndecay=ndecay))
            if lstm: mods.insert(3, BLSTM(hidden, layers=2, max_steps=200, skip=True))
            layer = nn.Sequential(*mods)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)

        return x

class LocalState(nn.Module):
    def __init__(self, channels, heads = 4, nfreqs = 0, ndecay = 4):
        super().__init__()
        assert channels % heads == 0, (channels, heads)
        self.heads = heads
        self.nfreqs = nfreqs
        self.ndecay = ndecay
        self.content = nn.Conv1d(channels, channels, 1)
        self.query = nn.Conv1d(channels, channels, 1)
        self.key = nn.Conv1d(channels, channels, 1)

        if nfreqs: self.query_freqs = nn.Conv1d(channels, heads * nfreqs, 1)

        if ndecay:
            self.query_decay = nn.Conv1d(channels, heads * ndecay, 1)
            self.query_decay.weight.data *= 0.01
            assert self.query_decay.bias is not None  
            self.query_decay.bias.data[:] = -2

        self.proj = nn.Conv1d(channels + heads * nfreqs, channels, 1)

    def forward(self, x):
        B, C, T = x.shape
        heads = self.heads
        indexes = torch.arange(T, device=x.device, dtype=x.dtype)
        delta = indexes[:, None] - indexes[None, :]
        queries = self.query(x).view(B, heads, -1, T)
        keys = self.key(x).view(B, heads, -1, T)
        dots = torch.einsum("bhct,bhcs->bhts", keys, queries)
        dots /= keys.shape[2] ** 0.5

        if self.nfreqs:
            periods = torch.arange(1, self.nfreqs + 1, device=x.device, dtype=x.dtype)
            freq_kernel = (2 * math.pi * delta / periods.view(-1, 1, 1)).cos()
            freq_q = self.query_freqs(x).view(B, heads, -1, T) / self.nfreqs**0.5
            dots += torch.einsum("fts,bhfs->bhts", freq_kernel, freq_q)

        if self.ndecay:
            decays = torch.arange(1, self.ndecay + 1, device=x.device, dtype=x.dtype)
            decay_q = self.query_decay(x).view(B, heads, -1, T)
            decay_q = decay_q.sigmoid() / 2
            decay_kernel = -decays.view(-1, 1, 1) * delta.abs() / self.ndecay**0.5
            dots += torch.einsum("fts,bhfs->bhts", decay_kernel, decay_q)

        dots.masked_fill_(torch.eye(T, device=dots.device, dtype=torch.bool), -100)
        weights = dots.softmax(dim=2)
        content = self.content(x).view(B, heads, -1, T)
        result = torch.einsum("bhts,bhct->bhcs", weights, content)

        if self.nfreqs:
            time_sig = torch.einsum("bhts,fts->bhfs", weights, freq_kernel)
            result = torch.cat([result, time_sig], 2)

        result = result.reshape(B, -1, T)
        return x + self.proj(result)

class Demucs(nn.Module):
    @capture_init
    def __init__(self, sources, audio_channels=2, channels=64, growth=2.0, depth=6, rewrite=True, lstm_layers=0, kernel_size=8, stride=4, context=1, gelu=True, glu=True, norm_starts=4, norm_groups=4, dconv_mode=1, dconv_depth=2, dconv_comp=4, dconv_attn=4, dconv_lstm=4, dconv_init=1e-4, normalize=True, resample=True, rescale=0.1, samplerate=44100, segment=4 * 10):
        super().__init__()
        self.audio_channels = audio_channels
        self.sources = sources
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.depth = depth
        self.resample = resample
        self.channels = channels
        self.normalize = normalize
        self.samplerate = samplerate
        self.segment = segment
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.skip_scales = nn.ModuleList()

        if glu:
            activation = nn.GLU(dim=1)
            ch_scale = 2
        else:
            activation = nn.ReLU()
            ch_scale = 1

        act2 = nn.GELU if gelu else nn.ReLU

        in_channels = audio_channels
        padding = 0

        for index in range(depth):
            norm_fn = lambda d: nn.Identity()  
            if index >= norm_starts: norm_fn = lambda d: nn.GroupNorm(norm_groups, d)  

            encode = []
            encode += [nn.Conv1d(in_channels, channels, kernel_size, stride), norm_fn(channels), act2()]
            attn = index >= dconv_attn
            lstm = index >= dconv_lstm

            if dconv_mode & 1: encode += [DConv(channels, depth=dconv_depth, init=dconv_init, compress=dconv_comp, attn=attn, lstm=lstm)]
            if rewrite: encode += [nn.Conv1d(channels, ch_scale * channels, 1), norm_fn(ch_scale * channels), activation]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            out_channels = in_channels if index > 0 else len(self.sources) * audio_channels
                
            if rewrite: decode += [nn.Conv1d(channels, ch_scale * channels, 2 * context + 1, padding=context), norm_fn(ch_scale * channels), activation]
            if dconv_mode & 2: decode += [DConv(channels, depth=dconv_depth, init=dconv_init, compress=dconv_comp, attn=attn, lstm=lstm)]
            decode += [nn.ConvTranspose1d(channels, out_channels, kernel_size, stride, padding=padding)]

            if index > 0: decode += [norm_fn(out_channels), act2()]
            self.decoder.insert(0, nn.Sequential(*decode))
            in_channels = channels
            channels = int(growth * channels)

        channels = in_channels
        self.lstm = BLSTM(channels, lstm_layers) if lstm_layers else None
        if rescale: rescale_module(self, reference=rescale)

    def valid_length(self, length):
        if self.resample: length *= 2

        for _ in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(1, length)

        for _ in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size

        if self.resample: length = math.ceil(length / 2)
        return int(length)

    def forward(self, mix):
        x = mix
        length = x.shape[-1]

        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            mean = mono.mean(dim=-1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            x = (x - mean) / (1e-5 + std)
        else:
            mean = 0
            std = 1

        delta = self.valid_length(length) - length
        x = F.pad(x, (delta // 2, delta - delta // 2))

        if self.resample: x = resample_frac(x, 1, 2)
        saved = []

        for encode in self.encoder:
            x = encode(x)
            saved.append(x)

        if self.lstm: x = self.lstm(x)

        for decode in self.decoder:
            skip = saved.pop(-1)
            skip = center_trim(skip, x)
            x = decode(x + skip)

        if self.resample: x = resample_frac(x, 2, 1)

        x = x * std + mean
        x = center_trim(x, length)
        x = x.view(x.size(0), len(self.sources), self.audio_channels, x.size(-1))
        return x

    def load_state_dict(self, state, strict=True):
        for idx in range(self.depth):
            for a in ["encoder", "decoder"]:
                for b in ["bias", "weight"]:
                    new = f"{a}.{idx}.3.{b}"
                    old = f"{a}.{idx}.2.{b}"

                    if old in state and new not in state: state[new] = state.pop(old)
        super().load_state_dict(state, strict=strict)

class ResampleFrac(torch.nn.Module):
    def __init__(self, old_sr, new_sr, zeros = 24, rolloff = 0.945):
        super().__init__()
        gcd = math.gcd(old_sr, new_sr)
        self.old_sr = old_sr // gcd
        self.new_sr = new_sr // gcd
        self.zeros = zeros
        self.rolloff = rolloff
        self._init_kernels()

    def _init_kernels(self):
        if self.old_sr == self.new_sr: return

        kernels = []
        sr = min(self.new_sr, self.old_sr)
        sr *= self.rolloff

        self._width = math.ceil(self.zeros * self.old_sr / sr)
        idx = torch.arange(-self._width, self._width + self.old_sr).float()

        for i in range(self.new_sr):
            t = ((-i / self.new_sr + idx / self.old_sr) * sr).clamp_(-self.zeros, self.zeros)
            t *= math.pi

            kernel = sinc(t) * ((t / self.zeros / 2).cos()**2)
            kernel.div_(kernel.sum())
            kernels.append(kernel)

        self.register_buffer("kernel", torch.stack(kernels).view(self.new_sr, 1, -1))

    def forward(self, x, output_length = None, full = False):
        if self.old_sr == self.new_sr: return x
        shape = x.shape
        length = x.shape[-1]
        
        x = x.reshape(-1, length)
        y = F.conv1d(F.pad(x[:, None], (self._width, self._width + self.old_sr), mode='replicate'), self.kernel, stride=self.old_sr).transpose(1, 2).reshape(list(shape[:-1]) + [-1])

        float_output_length = torch.as_tensor(self.new_sr * length / self.old_sr)
        max_output_length = float_output_length.ceil().long()
        default_output_length = float_output_length.floor().long()

        if output_length is None: applied_output_length = max_output_length if full else default_output_length
        elif output_length < 0 or output_length > max_output_length: raise ValueError("output_length < 0 or output_length > max_output_length")
        else:
            applied_output_length = torch.tensor(output_length)
            if full: raise ValueError("full=True")

        return y[..., :applied_output_length] 

    def __repr__(self):
        return simple_repr(self)

def sinc(x):
    return torch.where(x == 0, torch.tensor(1., device=x.device, dtype=x.dtype), x.sin() / x)

def simple_repr(obj, attrs = None, overrides = {}):
    params = inspect.signature(obj.__class__).parameters
    attrs_repr = []

    if attrs is None: attrs = list(params.keys())
    for attr in attrs:
        display = False

        if attr in overrides: value = overrides[attr]
        elif hasattr(obj, attr): value = getattr(obj, attr)
        else: continue

        if attr in params:
            param = params[attr]
            if param.default is inspect._empty or value != param.default: display = True
        else: display = True

        if display: attrs_repr.append(f"{attr}={value}")
    return f"{obj.__class__.__name__}({','.join(attrs_repr)})"

def resample_frac(x, old_sr, new_sr, zeros = 24, rolloff = 0.945, output_length = None, full = False):
    return ResampleFrac(old_sr, new_sr, zeros, rolloff).to(x)(x, output_length, full)
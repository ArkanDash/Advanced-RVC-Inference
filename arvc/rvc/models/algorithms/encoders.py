import math
import torch

from arvc.rvc.models.algorithms.modules import WaveNet
from arvc.rvc.models.algorithms.commons import sequence_mask
from arvc.rvc.models.algorithms.normalization import LayerNorm
from arvc.rvc.models.algorithms.attentions import MultiHeadAttention, FFN


def validate_pitch_tensor(pitch, max_val=255):
    """Validate and clamp pitch tensor to valid embedding range.
    
    Bug Fix: Original code did not validate pitch values before using them
    as indices for nn.Embedding lookup. Out-of-range values cause:
    - IndexError crashes during training
    - Silent corruption of embeddings (PyTorch modulo behavior)
    - Unstable training leading to inaccurate voice models
    
    Args:
        pitch: Pitch tensor (any integer type)
        max_val: Maximum valid value for embedding lookup (default 255 for 256 bins)
        
    Returns:
        Clamped pitch tensor with valid range [0, max_val]
    """
    if pitch is None:
        return pitch
    
    # Clamp to valid range - CRITICAL for embedding lookup safety
    # Without this, out-of-range pitch values either crash or silently corrupt
    # the model's internal representations, causing voice accuracy issues
    pitch = torch.clamp(pitch, 0, max_val).long()
    
    return pitch


class Encoder(torch.nn.Module):
    def __init__(
        self, 
        hidden_channels, 
        filter_channels, 
        n_heads, 
        n_layers, 
        kernel_size=1, 
        p_dropout=0.0, 
        window_size=10, 
        onnx=False, 
        **kwargs
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.drop = torch.nn.Dropout(p_dropout)

        self.attn_layers = torch.nn.ModuleList([
            MultiHeadAttention(
                hidden_channels, 
                hidden_channels, 
                n_heads, 
                p_dropout=p_dropout, 
                window_size=window_size, 
                onnx=onnx
            )
            for _ in range(n_layers)
        ])

        self.norm_layers_1 = torch.nn.ModuleList([
            LayerNorm(
                hidden_channels, 
                onnx=onnx
            )
            for _ in range(n_layers)
        ])

        self.ffn_layers = torch.nn.ModuleList([
            FFN(
                hidden_channels, 
                hidden_channels, 
                filter_channels, 
                kernel_size, 
                p_dropout=p_dropout, 
                onnx=onnx
            ) 
            for _ in range(n_layers)
        ])

        self.norm_layers_2 = torch.nn.ModuleList([
            LayerNorm(
                hidden_channels, 
                onnx=onnx
            ) 
            for _ in range(n_layers)
        ])

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask

        for i in range(self.n_layers):
            x = self.norm_layers_1[i](x + self.drop(self.attn_layers[i](x, x, attn_mask)))
            x = self.norm_layers_2[i](x + self.drop(self.ffn_layers[i](x, x_mask)))

        return x * x_mask
    
class TextEncoder(torch.nn.Module):
    def __init__(
        self, 
        out_channels, 
        hidden_channels, 
        filter_channels, 
        n_heads, 
        n_layers, 
        kernel_size, 
        p_dropout, 
        embedding_dim, 
        f0=True, 
        energy=False, 
        onnx=False
    ):
        super(TextEncoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.lrelu = torch.nn.LeakyReLU(0.1, inplace=True)
        self.emb_phone = torch.nn.Linear(embedding_dim, hidden_channels)
        # BUG FIX: Use 256 pitch bins (standard) but add safe handling for out-of-range values
        self.emb_pitch = torch.nn.Embedding(256, hidden_channels) if f0 else None
        self.emb_energy = torch.nn.Linear(1, hidden_channels) if energy else None
        self.encoder = Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, float(p_dropout), onnx=onnx)
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, phone, pitch, lengths, energy):
        x = self.emb_phone(phone)

        # ═══════════════════════════════════════════════════════════════
        # BUG FIX: Validate pitch before embedding lookup
        # Original code directly used pitch values as embedding indices without
        # validation. This causes:
        # 1. IndexError if pitch > 255 or pitch < 0
        # 2. Silent embedding corruption (PyTorch wraps negative indices)
        # 3. Model learns incorrect pitch representations → inaccurate voice
        # ═══════════════════════════════════════════════════════════════
        if pitch is not None:
            pitch = validate_pitch_tensor(pitch, max_val=255)
            x += self.emb_pitch(pitch)
        
        if energy is not None:
            # Validate energy: replace NaN/Inf with zeros
            energy = torch.where(
                torch.isfinite(energy), 
                energy, 
                torch.zeros_like(energy)
            )
            x += self.emb_energy(energy.unsqueeze(-1))

        x = self.lrelu(x * math.sqrt(self.hidden_channels)).transpose(1, -1)
        x_mask = sequence_mask(lengths, x.size(2)).unsqueeze(1).to(x.dtype)
        m, logs = (self.proj(self.encoder(x * x_mask, x_mask)) * x_mask).split(self.out_channels, dim=1)

        return m, logs, x_mask

class TextEncoderSVC(torch.nn.Module):
    def __init__(
        self,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        onnx=False
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.f0_emb = torch.nn.Embedding(256, hidden_channels)
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels * 2, 1)
        self.encoder = Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, float(p_dropout), window_size=4, onnx=onnx)

    def forward(self, x, x_mask, f0=None, noise_scale=1):
        x = x + self.f0_emb(f0).transpose(1, 2)

        m, logs = (self.proj(self.encoder(x * x_mask, x_mask)) * x_mask).split(self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * logs.exp() * noise_scale) * x_mask

        return z, m, logs, x_mask

class PosteriorEncoder(torch.nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        hidden_channels, 
        kernel_size, 
        dilation_rate, 
        n_layers, 
        gin_channels=0
    ):
        super(PosteriorEncoder, self).__init__()
        self.out_channels = out_channels
        self.pre = torch.nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WaveNet(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g = None):
        x_mask = sequence_mask(x_lengths, x.size(2)).unsqueeze(1).to(x.dtype)

        m, logs = (
            self.proj(
                self.enc(
                    self.pre(x) * x_mask, 
                    x_mask, 
                    g=g
                )
            ) * x_mask
        ).split(self.out_channels, dim=1)

        return (m + torch.randn_like(m) * logs.exp()) * x_mask, m, logs, x_mask

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()

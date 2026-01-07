import torch
import numpy as np
import math
from torch import nn
from torch.nn import functional as F
from typing import Optional

from advanced_rvc_inference.library.algorithm.modules import WaveNet
from advanced_rvc_inference.library.algorithm.commons import sequence_mask
from advanced_rvc_inference.library.algorithm.normalization import LayerNorm
from advanced_rvc_inference.library.algorithm.attentions import MultiHeadAttention, FFN




class Encoder_VITS2(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        window_size: int = 10,
        gin_channels: int = 0,
        **kwargs
    ):
        super(Encoder_VITS2, self).__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = int(n_layers)
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        # Speaker-conditioned text encoder logic
        self.gin_channels = gin_channels
        self.cond_layer_idx = self.n_layers
        if self.gin_channels != 0:
            # Projection ~ ,aps speaker embedding dimension (G) to token dimension (H)
            self.spk_emb_linear = nn.Linear(self.gin_channels, self.hidden_channels)
            # Conditioning Layer Index ( VITS2 uses the 3rd layer by default, index 2 )
            self.cond_layer_idx = 2
            # Safety check
            if self.cond_layer_idx >= self.n_layers:
                raise ValueError("cond_layer_idx must be less than n_layers")

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for i in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask

        for i in range(self.n_layers):
            # Speaker-conditioned text encoder logic ( VITS2 )
            # Projected speaker bias at the target layer
            if i == self.cond_layer_idx and g is not None:
                g_proj = g.transpose(1, 2)
                # [B, 1, G] -> [B, 1, H]
                g_proj = self.spk_emb_linear(g_proj)
                # Transpose back [B, 1, H] -> [B, H, 1]
                g_proj = g_proj.transpose(1, 2)
                # Additive Bias
                x = x + g_proj
                x = x * x_mask # Mask after adding bias

            # Self-Attention block
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            # FFN block
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)

        x = x * x_mask
        return x

class TextEncoder_VITS2(nn.Module):
    """
    Text Encoder.

    Args:
        out_channels (int): Output channels of the encoder.
        hidden_channels (int): Hidden channels of the encoder.
        filter_channels (int): Filter channels of the encoder.
        n_heads (int): Number of attention heads.
        n_layers (int): Number of encoder layers.
        kernel_size (int): Kernel size of the convolutional layers.
        p_dropout (float): Dropout probability.
        embedding_dim (int): Embedding dimension for phone embeddings (v1 = 256, v2 = 768).
        f0 (bool, optional): Whether to use F0 embedding. Defaults to True.
    """
    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        embedding_dim: int,
        f0: bool = True,
        gin_channels: int = 0,
        **kwargs
    ):
        super(TextEncoder_VITS2, self).__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.emb_phone = nn.Linear(embedding_dim, hidden_channels)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.emb_pitch = torch.nn.Embedding(256, hidden_channels) if f0 else None

        self.encoder = Encoder_VITS2(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            gin_channels=gin_channels,
            **kwargs,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        phone: torch.Tensor,
        pitch: torch.Tensor,
        lengths: torch.Tensor,
        skip_head: Optional[torch.Tensor] = None,
        g: Optional[torch.Tensor] = None,
    ):
        if pitch is None:
            x = self.emb_phone(phone)
        else:
            x = self.emb_phone(phone) + self.emb_pitch(pitch)

        x = x * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = self.lrelu(x)
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(sequence_mask(lengths, x.size(2)), 1).to(x.dtype)
        x = self.encoder(x * x_mask, x_mask, g=g)

        if skip_head is not None:
            assert isinstance(skip_head, torch.Tensor)
            head = int(skip_head.item())
            x = x[:, :, head:]
            x_mask = x_mask[:, :, head:]
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)

        return m, logs, x_mask

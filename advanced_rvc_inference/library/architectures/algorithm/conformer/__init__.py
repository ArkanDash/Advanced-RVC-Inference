import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class GLU(nn.Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        x, gate = x.chunk(2, dim=self.dim)
        return x * gate.sigmoid()


class BaseAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.dropout = nn.Dropout(attention_dropout)

    def _reshape_qkv(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        x = x.reshape(B, T, self.num_heads, -1)
        return x.transpose(1, 2)  # (B, H, T, D/H)


class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.pointwise_conv1 = nn.Conv1d(dim, dim, kernel_size=1, stride=1, padding=0, bias=bias)
        self.depthwise_conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size // 2),
            groups=dim,
            bias=bias,
        )
        self.batch_norm = nn.BatchNorm1d(dim)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1, stride=1, padding=0, bias=bias)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 1)
        x = self.pointwise_conv1(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        return x


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=bias),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MultiHeadedSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, -1).transpose(1, 2)
        k = k.view(B, T, self.num_heads, -1).transpose(1, 2)
        v = v.view(B, T, self.num_heads, -1).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -1e9)
        att = att.softmax(dim=-1)
        att = self.dropout(att)
        x = (att @ v).transpose(1, 2).contiguous().view(B, T, D)
        x = self.out_proj(x)
        return x


class ConformerBlock(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 4,
        ff_hidden_dim: int = 1024,
        kernel_size: int = 7,
        attention_dropout: float = 0.1,
        conv_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.ff1 = FeedForward(dim, ff_hidden_dim, ff_dropout, bias)
        self.self_attn = MultiHeadedSelfAttention(dim, num_heads, attention_dropout, bias)
        self.conv_module = ConformerConvModule(dim, kernel_size, bias)
        self.ff2 = FeedForward(dim, ff_hidden_dim, ff_dropout, bias)
        self.batch_norm = nn.BatchNorm1d(dim)
        self.scale = 0.5

    def forward(self, x: Tensor) -> Tensor:
        x = self.ff1(x)
        x = x + self.scale * self.self_attn(x)
        x = x.permute(0, 2, 1)
        x = self.batch_norm(x)
        x = x.permute(0, 2, 1)
        x = self.conv_module(x)
        x = x.permute(0, 2, 1)
        x = self.batch_norm(x)
        x = x.permute(0, 2, 1)
        x = x + self.scale * self.ff2(x)
        return x


class Conformer(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        ff_mult: int = 4,
        conv_expansion_factor: int = 2,
        conv_kernel_size: int = 31,
        attention_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        conv_dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        ff_hidden_dim = int(dim * ff_mult)
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    dim,
                    num_heads,
                    ff_hidden_dim,
                    conv_kernel_size,
                    attention_dropout,
                    conv_dropout,
                    ff_dropout,
                    bias,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x

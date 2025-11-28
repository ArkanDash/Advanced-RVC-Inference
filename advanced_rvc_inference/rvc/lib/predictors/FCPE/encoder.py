import os
import sys

import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.getcwd())

from main.library.predictors.FCPE.attentions import SelfAttention
from main.library.predictors.FCPE.utils import calc_same_padding, Transpose, GLU, Swish

class ConformerConvModule_LEGACY(nn.Module):
    def __init__(self, dim, causal=False, expansion_factor=2, kernel_size=31, dropout=0.0):
        super().__init__()
        inner_dim = dim * expansion_factor
        self.net = nn.Sequential(nn.LayerNorm(dim), Transpose((1, 2)), nn.Conv1d(dim, inner_dim * 2, 1), GLU(dim=1), DepthWiseConv1d_LEGACY(inner_dim, inner_dim, kernel_size=kernel_size, padding=(calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0))), Swish(), nn.Conv1d(inner_dim, dim, 1), Transpose((1, 2)), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)

class ConformerConvModule(nn.Module):
    def __init__(self, dim, expansion_factor=2, kernel_size=31, dropout=0):
        super().__init__()
        inner_dim = dim * expansion_factor
        self.net = nn.Sequential(nn.LayerNorm(dim), Transpose((1, 2)), nn.Conv1d(dim, inner_dim * 2, 1), nn.GLU(dim=1), DepthWiseConv1d(inner_dim, inner_dim, kernel_size=kernel_size, padding=calc_same_padding(kernel_size)[0], groups=inner_dim), nn.SiLU(), nn.Conv1d(inner_dim, dim, 1), Transpose((1, 2)), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)

class DepthWiseConv1d_LEGACY(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)

    def forward(self, x):
        return self.conv(F.pad(x, self.padding))

class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding, groups):
        super().__init__()
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size=kernel_size, padding=padding, groups=groups)

    def forward(self, x):
        return self.conv(x)

class EncoderLayer(nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.conformer = ConformerConvModule_LEGACY(parent.dim_model)
        self.norm = nn.LayerNorm(parent.dim_model)
        self.dropout = nn.Dropout(parent.residual_dropout)
        self.attn = SelfAttention(dim=parent.dim_model, heads=parent.num_heads, causal=False)

    def forward(self, phone, mask=None):
        phone = phone + (self.attn(self.norm(phone), mask=mask))
        return phone + (self.conformer(phone))

class ConformerNaiveEncoder(nn.Module):
    def __init__(self, num_layers, num_heads, dim_model, use_norm = False, conv_only = False, conv_dropout = 0, atten_dropout = 0):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.use_norm = use_norm
        self.residual_dropout = 0.1  
        self.attention_dropout = 0.1  
        self.encoder_layers = nn.ModuleList([CFNEncoderLayer(dim_model, num_heads, use_norm, conv_only, conv_dropout, atten_dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for (_, layer) in enumerate(self.encoder_layers):
            x = layer(x, mask)

        return x 
    
class CFNEncoderLayer(nn.Module):
    def __init__(self, dim_model, num_heads = 8, use_norm = False, conv_only = False, conv_dropout = 0, atten_dropout = 0):
        super().__init__()
        self.conformer = nn.Sequential(ConformerConvModule(dim_model), nn.Dropout(conv_dropout)) if conv_dropout > 0 else ConformerConvModule(dim_model)
        self.norm = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(0.1)  
        self.attn = SelfAttention(dim=dim_model, heads=num_heads, causal=False, use_norm=use_norm, dropout=atten_dropout) if not conv_only else None

    def forward(self, x, mask=None):
        if self.attn is not None: x = x + (self.attn(self.norm(x), mask=mask))
        return x + (self.conformer(x)) 
import os
import sys
import math
import torch

sys.path.append(os.getcwd())

from main.library.algorithm.modules import WaveNet
from main.library.algorithm.commons import sequence_mask
from main.library.algorithm.normalization import LayerNorm
from main.library.algorithm.attentions import MultiHeadAttention, FFN

class Encoder(torch.nn.Module):
    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0.0, window_size=10, onnx=False, **kwargs):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.drop = torch.nn.Dropout(p_dropout)
        self.attn_layers = torch.nn.ModuleList()
        self.norm_layers_1 = torch.nn.ModuleList()
        self.ffn_layers = torch.nn.ModuleList()
        self.norm_layers_2 = torch.nn.ModuleList()

        for _ in range(self.n_layers):
            self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, window_size=window_size, onnx=onnx))
            self.norm_layers_1.append(LayerNorm(hidden_channels, onnx=onnx))

            self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout, onnx=onnx))
            self.norm_layers_2.append(LayerNorm(hidden_channels, onnx=onnx))

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        
        for i in range(self.n_layers):
            x = self.norm_layers_1[i](x + self.drop(self.attn_layers[i](x, x, attn_mask)))
            x = self.norm_layers_2[i](x + self.drop(self.ffn_layers[i](x, x_mask)))

        return x * x_mask
    
class TextEncoder(torch.nn.Module):
    def __init__(self, out_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, embedding_dim, f0=True, energy=False, onnx=False):
        super(TextEncoder, self).__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.lrelu = torch.nn.LeakyReLU(0.1, inplace=True)
        self.emb_phone = torch.nn.Linear(embedding_dim, hidden_channels)
        self.emb_pitch = torch.nn.Embedding(256, hidden_channels) if f0 else None
        self.emb_energy = torch.nn.Linear(1, hidden_channels) if energy else None
        self.encoder = Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, float(p_dropout), onnx=onnx)
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, phone, pitch, lengths, energy):
        x = self.emb_phone(phone)

        if pitch is not None: x += self.emb_pitch(pitch)
        if energy is not None: x += self.emb_energy(energy.unsqueeze(-1))

        x = self.lrelu(x * math.sqrt(self.hidden_channels)).transpose(1, -1)
        x_mask = sequence_mask(lengths, x.size(2)).unsqueeze(1).to(x.dtype)
        m, logs = (self.proj(self.encoder(x * x_mask, x_mask)) * x_mask).split(self.out_channels, dim=1)

        return m, logs, x_mask

class PosteriorEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0):
        super(PosteriorEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.pre = torch.nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WaveNet(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g = None):
        x_mask = sequence_mask(x_lengths, x.size(2)).unsqueeze(1).to(x.dtype)
        m, logs = (self.proj(self.enc((self.pre(x) * x_mask), x_mask, g=g)) * x_mask).split(self.out_channels, dim=1)

        return ((m + torch.randn_like(m) * logs.exp()) * x_mask), m, logs, x_mask

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()
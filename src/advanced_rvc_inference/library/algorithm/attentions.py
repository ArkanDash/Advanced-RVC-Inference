import os
import sys
import math
import torch

import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.getcwd())

from main.library.algorithm.commons import convert_pad_shape

class MultiHeadAttention(nn.Module):
    def __init__(self, channels, out_channels, n_heads, p_dropout=0.0, window_size=None, heads_share=True, block_length=None, proximal_bias=False, proximal_init=False, onnx=False):
        super().__init__()
        assert channels % n_heads == 0
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        self.onnx = onnx
        self.attn = None
        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5

            self.emb_rel_k = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
            self.emb_rel_v = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)
        nn.init.xavier_uniform_(self.conv_o.weight)

        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(self, x, c, attn_mask=None):
        q, k, v = self.conv_q(x), self.conv_k(c), self.conv_v(c)
        x, self.attn = self.attention(q, k, v, mask=attn_mask)

        return self.conv_o(x)

    def attention(self, query, key, value, mask=None):
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        scores = (query / math.sqrt(self.k_channels)) @ key.transpose(-2, -1)
    
        if self.window_size is not None:
            assert (t_s == t_t)
            scores += self._relative_position_to_absolute_position(self._matmul_with_relative_keys(query / math.sqrt(self.k_channels), self._get_relative_embeddings(self.emb_rel_k, t_s, onnx=self.onnx)), onnx=self.onnx)

        if self.proximal_bias:
            assert t_s == t_t
            scores += self._attention_bias_proximal(t_s).to(device=scores.device, dtype=scores.dtype)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.block_length is not None:
                assert (t_s == t_t)
                scores = scores.masked_fill((torch.ones_like(scores).triu(-self.block_length).tril(self.block_length)) == 0, -1e4)

        p_attn = self.drop(F.softmax(scores, dim=-1))
        output = p_attn @ value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        if self.window_size is not None: output += self._matmul_with_relative_values(self._absolute_position_to_relative_position(p_attn, onnx=self.onnx), self._get_relative_embeddings(self.emb_rel_v, t_s, onnx=self.onnx))
        return (output.transpose(2, 3).contiguous().view(b, d, t_t)), p_attn

    def _matmul_with_relative_values(self, x, y):
        return x @ y.unsqueeze(0)

    def _matmul_with_relative_keys(self, x, y):
        return x @ y.unsqueeze(0).transpose(-2, -1)

    def _get_relative_embeddings(self, relative_embeddings, length, onnx=False):
        if onnx:
            pad_length = (length - (self.window_size + 1)).clamp(min=0)
            slice_start_position = ((self.window_size + 1) - length).clamp(min=0)

            return (F.pad(relative_embeddings, [0, 0, pad_length, pad_length, 0, 0]) if pad_length > 0 else relative_embeddings)[:, slice_start_position:(slice_start_position + 2 * length - 1)]
        else:
            pad_length = max(length - (self.window_size + 1), 0)
            slice_start_position = max((self.window_size + 1) - length, 0)

            return (F.pad(relative_embeddings, convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]])) if pad_length > 0 else relative_embeddings)[:, slice_start_position:(slice_start_position + 2 * length - 1)]  

    def _relative_position_to_absolute_position(self, x, onnx=False):
        batch, heads, length, _ = x.size()

        return (F.pad(F.pad(x, [0, 1, 0, 0, 0, 0, 0, 0]).view([batch, heads, length * 2 * length]), [0, length - 1, 0, 0, 0, 0]).view([batch, heads, length + 1, 2 * length - 1]) if onnx else F.pad(F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]])).view([batch, heads, length * 2 * length]), convert_pad_shape([[0, 0], [0, 0], [0, length - 1]])).view([batch, heads, length + 1, 2 * length - 1]))[:, :, :length, length - 1 :]

    def _absolute_position_to_relative_position(self, x, onnx=False):
        batch, heads, length, _ = x.size()

        return (F.pad(F.pad(x, [0, length - 1, 0, 0, 0, 0, 0, 0]).view([batch, heads, length*length + length * (length - 1)]), [length, 0, 0, 0, 0, 0]).view([batch, heads, length, 2 * length]) if onnx else F.pad(F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]])).view([batch, heads, length**2 + length * (length - 1)]), convert_pad_shape([[0, 0], [0, 0], [length, 0]])).view([batch, heads, length, 2 * length]))[:, :, :, 1:]

    def _attention_bias_proximal(self, length):
        r = torch.arange(length, dtype=torch.float32)

        return -(r.unsqueeze(0) - r.unsqueeze(1)).abs().log1p().unsqueeze(0).unsqueeze(0)

class FFN(nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0.0, activation=None, causal=False, onnx=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal
        self.onnx = onnx
        self.padding = self._causal_padding if causal else self._same_padding
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(self.padding(x * x_mask))

        return self.conv_2(self.padding(self.drop(((x * (1.702 * x).sigmoid()) if self.activation == "gelu" else x.relu())) * x_mask)) * x_mask

    def _causal_padding(self, x):
        if self.kernel_size == 1: return x

        return F.pad(x, [self.kernel_size - 1, 0, 0, 0, 0, 0]) if self.onnx else F.pad(x, convert_pad_shape([[0, 0], [0, 0], [(self.kernel_size - 1), 0]]))

    def _same_padding(self, x):
        if self.kernel_size == 1: return x
        
        return F.pad(x, [(self.kernel_size - 1) // 2, self.kernel_size // 2, 0, 0, 0, 0]) if self.onnx else F.pad(x, convert_pad_shape([[0, 0], [0, 0], [((self.kernel_size - 1) // 2), (self.kernel_size // 2)]]))
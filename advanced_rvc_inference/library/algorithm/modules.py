import os
import sys
import torch

import torch.nn.utils.parametrize as parametrize

sys.path.append(os.getcwd())

from .commons import fused_add_tanh_sigmoid_multiply

class WaveNet(torch.nn.Module):
    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
        super(WaveNet, self).__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = torch.nn.Dropout(p_dropout)
        if gin_channels != 0: self.cond_layer = torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1), name="weight")
        dilations = [dilation_rate ** i for i in range(n_layers)]
        paddings = [(kernel_size * d - d) // 2 for d in dilations]

        for i in range(n_layers):
            in_layer = torch.nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size, dilation=dilations[i], padding=paddings[i])
            in_layer = torch.nn.utils.parametrizations.weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)
            res_skip_channels = (hidden_channels if i == n_layers - 1 else 2 * hidden_channels)
            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.parametrizations.weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, g=None):
        output = x.clone().zero_()
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None: g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            g_l = (g[:, i * 2 * self.hidden_channels : (i + 1) * 2 * self.hidden_channels, :] if g is not None else 0)
            res_skip_acts = self.res_skip_layers[i](self.drop(fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)))

            if i < self.n_layers - 1:
                x = (x + (res_skip_acts[:, : self.hidden_channels, :])) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else: output = output + res_skip_acts

        return output * x_mask

    def remove_weight_norm(self):
        if self.gin_channels != 0: 
            if hasattr(self.cond_layer, "parametrizations") and "weight" in self.cond_layer.parametrizations: parametrize.remove_parametrizations(self.cond_layer, "weight", leave_parametrized=True)
            else: torch.nn.utils.remove_weight_norm(self.cond_layer)

        for l in self.in_layers:
            if hasattr(l, "parametrizations") and "weight" in l.parametrizations: parametrize.remove_parametrizations(l, "weight", leave_parametrized=True)
            else: torch.nn.utils.remove_weight_norm(l)

        for l in self.res_skip_layers:
            if hasattr(l, "parametrizations") and "weight" in l.parametrizations: parametrize.remove_parametrizations(l, "weight", leave_parametrized=True)
            else: torch.nn.utils.remove_weight_norm(l)
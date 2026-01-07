import os
import sys
import torch

import torch.nn.utils.parametrize as parametrize

from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm

sys.path.append(os.getcwd())

from .modules import WaveNet
from .commons import get_padding, init_weights

LRELU_SLOPE = 0.1

def create_conv1d_layer(channels, kernel_size, dilation):
    return weight_norm(torch.nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation, padding=get_padding(kernel_size, dilation)))

def apply_mask(tensor, mask):
    return tensor * mask if mask is not None else tensor

class ResBlockBase(torch.nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super(ResBlockBase, self).__init__()

        self.convs1 = torch.nn.ModuleList([create_conv1d_layer(channels, kernel_size, d) for d in dilations])
        self.convs1.apply(init_weights)

        self.convs2 = torch.nn.ModuleList([create_conv1d_layer(channels, kernel_size, 1) for _ in dilations])
        self.convs2.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            x = c2(apply_mask(torch.nn.functional.leaky_relu(c1(apply_mask(torch.nn.functional.leaky_relu(x, LRELU_SLOPE), x_mask)), LRELU_SLOPE), x_mask)) + x

        return apply_mask(x, x_mask)

    def remove_weight_norm(self):
        for conv in self.convs1 + self.convs2:
            if hasattr(conv, "parametrizations") and "weight" in conv.parametrizations: parametrize.remove_parametrizations(conv, "weight", leave_parametrized=True)
            else: remove_weight_norm(conv)

class ResBlock(ResBlockBase):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock, self).__init__(channels, kernel_size, dilation)

class Log(torch.nn.Module):
    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = x.clamp_min(1e-5).log() * x_mask
            return y, (-y).sum(dim=[1, 2])
        else: return x.exp() * x_mask

class Flip(torch.nn.Module):
    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])

        if not reverse: return x, torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
        else: return x

class ElementwiseAffine(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.m = torch.nn.Parameter(torch.zeros(channels, 1))
        self.logs = torch.nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse: return ((self.m + self.logs.exp() * x) * x_mask), (self.logs * x_mask).sum(dim=[1, 2])
        else: return (x - self.m) * (-self.logs).exp() * x_mask

class ResidualCouplingBlock(torch.nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, n_flows=4, gin_channels=0):
        super(ResidualCouplingBlock, self).__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels
        self.flows = torch.nn.ModuleList()

        for _ in range(n_flows):
            self.flows.append(ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
            self.flows.append(Flip())

    def forward(self, x, x_mask, g = None, reverse = False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow.forward(x, x_mask, g=g, reverse=reverse)

        return x

    def remove_weight_norm(self):
        for i in range(self.n_flows):
            self.flows[i * 2].remove_weight_norm()

class ResidualCouplingLayer(torch.nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=0, gin_channels=0, mean_only=False):
        assert channels % 2 == 0, "Channels/2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = torch.nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WaveNet(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout, gin_channels=gin_channels)
        self.post = torch.nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)

        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = x.split([self.half_channels] * 2, 1)
        stats = self.post(self.enc((self.pre(x0) * x_mask), x_mask, g=g)) * x_mask

        if not self.mean_only: m, logs = stats.split([self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse: return torch.cat([x0, (m + x1 * logs.exp() * x_mask)], 1), logs.sum(dim=[1, 2])
        else: return torch.cat([x0, ((x1 - m) * (-logs).exp() * x_mask)], 1)

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()
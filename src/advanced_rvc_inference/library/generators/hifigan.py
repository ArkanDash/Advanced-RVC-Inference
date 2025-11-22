import os
import sys
import torch

import torch.nn.utils.parametrize as parametrize

from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm

sys.path.append(os.getcwd())

from main.library.algorithm.commons import init_weights
from main.library.algorithm.residuals import ResBlock, LRELU_SLOPE

class HiFiGANGenerator(torch.nn.Module):
    def __init__(self, initial_channel, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(HiFiGANGenerator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = torch.nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        self.ups = torch.nn.ModuleList()
        self.resblocks = torch.nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(torch.nn.ConvTranspose1d(upsample_initial_channel // (2**i), upsample_initial_channel // (2 ** (i + 1)), k, u, padding=(k - u) // 2)))
            ch = upsample_initial_channel // (2 ** (i + 1))

            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d))

        self.conv_post = torch.nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)
        if gin_channels != 0: self.cond = torch.nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g = None):
        x = self.conv_pre(x)
        if g is not None: x += self.cond(g)

        for i in range(self.num_upsamples):
            x = self.ups[i](torch.nn.functional.leaky_relu(x, LRELU_SLOPE))
            xs = None

            for j in range(self.num_kernels):
                if xs is None: xs = self.resblocks[i * self.num_kernels + j](x)
                else: xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        return self.conv_post(torch.nn.functional.leaky_relu(x)).tanh()
    
    def remove_weight_norm(self):
        for l in self.ups:
            if hasattr(l, "parametrizations") and "weight" in l.parametrizations: parametrize.remove_parametrizations(l, "weight", leave_parametrized=True)
            else: remove_weight_norm(l)

        for l in self.resblocks:
            l.remove_weight_norm()
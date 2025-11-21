import os
import sys

import torch.nn as nn

sys.path.append(os.getcwd())

from main.library.predictors.DJCM.utils import ResConvBlock

class ResEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks, kernel_size):
        super(ResEncoderBlock, self).__init__()
        self.conv = nn.ModuleList([ResConvBlock(in_channels, out_channels)])
        for _ in range(n_blocks - 1):
            self.conv.append(ResConvBlock(out_channels, out_channels))

        self.pool = nn.MaxPool2d(kernel_size) if kernel_size is not None else None

    def forward(self, x):
        for each_layer in self.conv:
            x = each_layer(x)

        if self.pool is not None: return x, self.pool(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, n_blocks):
        super(Encoder, self).__init__()
        self.en_blocks = nn.ModuleList([
            ResEncoderBlock(in_channels, 32, n_blocks, (1, 2)), 
            ResEncoderBlock(32, 64, n_blocks, (1, 2)), 
            ResEncoderBlock(64, 128, n_blocks, (1, 2)), 
            ResEncoderBlock(128, 256, n_blocks, (1, 2)), 
            ResEncoderBlock(256, 384, n_blocks, (1, 2)), 
            ResEncoderBlock(384, 384, n_blocks, (1, 2))
        ])

    def forward(self, x):
        concat_tensors = []

        for layer in self.en_blocks:
            _, x = layer(x)
            concat_tensors.append(_)

        return x, concat_tensors
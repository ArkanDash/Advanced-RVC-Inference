import os
import sys
import torch

import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.getcwd())

from main.library.predictors.DJCM.encoder import ResEncoderBlock
from main.library.predictors.DJCM.utils import ResConvBlock, BiGRU, init_bn, init_layer, N_CLASS, WINDOW_LENGTH

class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks, stride):
        super(ResDecoderBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, stride, stride, (0, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=0.01)
        self.conv = nn.ModuleList([ResConvBlock(out_channels * 2, out_channels)])

        for _ in range(n_blocks - 1):
            self.conv.append(ResConvBlock(out_channels, out_channels))

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn1)
        init_layer(self.conv1)

    def forward(self, x, concat):
        x = self.conv1(F.relu_(self.bn1(x)))
        x = torch.cat((x, concat), dim=1)
    
        for each_layer in self.conv:
            x = each_layer(x)
    
        return x

class Decoder(nn.Module):
    def __init__(self, n_blocks):
        super(Decoder, self).__init__()
        self.de_blocks = nn.ModuleList([
            ResDecoderBlock(384, 384, n_blocks, (1, 2)), 
            ResDecoderBlock(384, 384, n_blocks, (1, 2)), 
            ResDecoderBlock(384, 256, n_blocks, (1, 2)), 
            ResDecoderBlock(256, 128, n_blocks, (1, 2)), 
            ResDecoderBlock(128, 64, n_blocks, (1, 2)), 
            ResDecoderBlock(64, 32, n_blocks, (1, 2))
        ])

    def forward(self, x, concat_tensors):
        for i, layer in enumerate(self.de_blocks):
            x = layer(x, concat_tensors[-1 - i])

        return x

class PE_Decoder(nn.Module):
    def __init__(self, n_blocks, seq_layers=1):
        super(PE_Decoder, self).__init__()
        self.de_blocks = Decoder(n_blocks)
        self.after_conv1 = ResEncoderBlock(32, 32, n_blocks, None)
        self.after_conv2 = nn.Conv2d(32, 1, (1, 1))
        self.fc = nn.Sequential(BiGRU((1, WINDOW_LENGTH // 2), 1, seq_layers), nn.Linear(WINDOW_LENGTH // 2, N_CLASS), nn.Sigmoid())
        init_layer(self.after_conv2)

    def forward(self, x, concat_tensors):
        return self.fc(self.after_conv2(self.after_conv1(self.de_blocks(x, concat_tensors)))).squeeze(1)
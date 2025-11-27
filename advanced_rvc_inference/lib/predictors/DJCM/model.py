import os
import sys

import torch.nn as nn

sys.path.append(os.getcwd())

from main.library.predictors.DJCM.decoder import PE_Decoder
from main.library.predictors.DJCM.utils import init_bn, WINDOW_LENGTH
from main.library.predictors.DJCM.encoder import ResEncoderBlock, Encoder

class LatentBlocks(nn.Module):
    def __init__(self, n_blocks, latent_layers):
        super(LatentBlocks, self).__init__()
        self.latent_blocks = nn.ModuleList([
            ResEncoderBlock(384, 384, n_blocks, None) 
            for _ in range(latent_layers)
        ])

    def forward(self, x):
        for layer in self.latent_blocks:
            x = layer(x)

        return x

class DJCMM(nn.Module):
    def __init__(self, in_channels, n_blocks, latent_layers):
        super(DJCMM, self).__init__()
        self.bn = nn.BatchNorm2d(WINDOW_LENGTH // 2 + 1, momentum=0.01)
        self.pe_encoder = Encoder(in_channels, n_blocks)
        self.pe_latent = LatentBlocks(n_blocks, latent_layers)
        self.pe_decoder = PE_Decoder(n_blocks)
        init_bn(self.bn)

    def forward(self, spec):
        x = self.bn(spec.transpose(1, 3)).transpose(1, 3)[..., :-1]
        x, concat_tensors = self.pe_encoder(x)
        pe_out = self.pe_decoder(self.pe_latent(x), concat_tensors)

        return pe_out
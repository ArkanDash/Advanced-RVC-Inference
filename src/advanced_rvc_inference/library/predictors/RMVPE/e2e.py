import os
import sys
import torch

import torch.nn as nn

sys.path.append(os.getcwd())

from main.library.predictors.RMVPE.deepunet import DeepUnet

N_MELS, N_CLASS = 128, 360

class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(input_features, hidden_features, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        try:
            return self.gru(x)[0]
        except:
            torch.backends.cudnn.enabled = False
            return self.gru(x)[0]
        
class E2E(nn.Module):
    def __init__(self, n_blocks, n_gru, kernel_size, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super(E2E, self).__init__()
        self.unet = DeepUnet(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        self.fc = nn.Sequential(BiGRU(3 * 128, 256, n_gru), nn.Linear(512, N_CLASS), nn.Dropout(0.25), nn.Sigmoid()) if n_gru else nn.Sequential(nn.Linear(3 * N_MELS, N_CLASS), nn.Dropout(0.25), nn.Sigmoid())

    def forward(self, mel):
        return self.fc(self.cnn(self.unet(mel.transpose(-1, -2).unsqueeze(1))).transpose(1, 2).flatten(-2))
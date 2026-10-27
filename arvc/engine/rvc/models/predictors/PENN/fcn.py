import torch

class FCN(torch.nn.Sequential):
    def __init__(self, channels = 256, pitch_bins = 1440, pooling = (2, 2)):
        super().__init__(*(Block(1, channels, 481, pooling), Block(channels, channels // 8, 225, pooling), Block(channels // 8, channels // 8, 97, pooling), Block(channels // 8, channels // 2, 66), Block(channels // 2, channels, 35), Block(channels, channels * 2, 4), torch.nn.Conv1d(channels * 2, pitch_bins, 4)))

    def forward(self, frames):
        return super().forward(frames[:, :, 16:-15])
    
class Block(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, length=1, pooling=None, kernel_size=32):
        layers = (torch.nn.Conv1d(in_channels, out_channels, kernel_size), torch.nn.ReLU())
        if pooling is not None: layers += (torch.nn.MaxPool1d(*pooling),)
        layers += (torch.nn.LayerNorm((out_channels, length)),)
        super().__init__(*layers)
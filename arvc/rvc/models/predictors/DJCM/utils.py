import torch

from torch import nn
from einops.layers.torch import Rearrange

SAMPLE_RATE, WINDOW_LENGTH, N_CLASS = 16000, 1024, 360

def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias") and layer.bias is not None: layer.bias.data.fill_(0.0)

def init_bn(bn):
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)
    bn.running_mean.data.fill_(0.0)
    bn.running_var.data.fill_(1.0)

class BiGRU(nn.Module):
    def __init__(self, patch_size, channels, depth):
        super(BiGRU, self).__init__()
        patch_width, patch_height = patch_size
        patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(Rearrange('b c (w p1) (h p2) -> b (w h) (p1 p2 c)', p1=patch_width, p2=patch_height))
        self.gru = nn.GRU(patch_dim, patch_dim // 2, num_layers=depth, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        try:
            return self.gru(x)[0]
        except:
            torch.backends.cudnn.enabled = False
            return self.gru(x)[0]

class ResConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ResConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.01)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.01)
        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()
        self.conv1 = nn.Conv2d(in_planes, out_planes, (3, 3), padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, (3, 3), padding=(1, 1), bias=False)
        self.is_shortcut = False

        if in_planes != out_planes:
            self.shortcut = nn.Conv2d(in_planes, out_planes, (1, 1))
            self.is_shortcut = True

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_layer(self.conv1)
        init_layer(self.conv2)
        if self.is_shortcut: init_layer(self.shortcut)

    def forward(self, x):
        out = self.conv2(self.act2(self.bn2(self.conv1(self.act1(self.bn1(x))))))

        if self.is_shortcut: return self.shortcut(x) + out
        else: return out + x
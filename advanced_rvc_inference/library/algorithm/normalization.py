import torch

import torch.nn.functional as F

class LayerNorm(torch.nn.Module):
    def __init__(self, channels, eps=1e-5, onnx=False):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.onnx = onnx
        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        return (F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps) if self.onnx else F.layer_norm(x, (x.size(-1),), self.gamma, self.beta, self.eps)).transpose(1, -1) 
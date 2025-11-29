"""
Utils file for backends module.
Contains custom implementations that may be used instead of PyTorch defaults.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence


class GRU(nn.Module):
    """
    Custom GRU implementation that may extend or modify PyTorch's default GRU.
    This is used to potentially add custom functionality or optimizations for specific backends.
    """
    def __init__(self, *args, **kwargs):
        super(GRU, self).__init__()
        # Use the original PyTorch GRU as the base implementation
        self.gru = nn.GRU(*args, **kwargs)
    
    def forward(self, input, hx=None):
        return self.gru(input, hx)
    
    def extra_repr(self):
        return self.gru.extra_repr()
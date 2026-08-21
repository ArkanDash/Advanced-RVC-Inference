from typing import Optional, Tuple
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.utils.parametrize as parametrize
from arvc.engine.models.weight_norm import weight_norm, remove_weight_norm


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

        return F.layer_norm(
            x, 
            (self.channels,) if self.onnx else (x.size(-1),), 
            self.gamma, 
            self.beta, 
            self.eps
        ).transpose(1, -1)


# ---------------------------------------------------------------------------
# SnakeBeta activation — general-purpose activation function
# ---------------------------------------------------------------------------

class SnakeBeta(nn.Module):
    def __init__(self, num_channels, init=1.0, beta_init=1.0, log_scale=True):
        super().__init__()
        self.num_channels = num_channels
        self.log_scale = log_scale
        self.log_scale_factor = nn.Parameter(torch.zeros(num_channels))
        if beta_init is not None:
            self.beta = nn.Parameter(torch.ones(num_channels) * beta_init)
        else:
            self.beta = None

    def forward(self, x):
        if self.beta is not None:
            x = x + (1.0 / (self.beta + 0.000000001)) * (torch.sin(x * self.beta) ** 2)
        return x


# ---------------------------------------------------------------------------
# Helpers for building conv layers inside residual blocks
# ---------------------------------------------------------------------------

_LRELU_SLOPE = 0.1


def _init_weights(m, mean=0.0, std=0.01):
    if m.__class__.__name__.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def _get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def _create_conv1d_layer(channels, kernel_size, dilation):
    return weight_norm(
        nn.Conv1d(channels, channels, kernel_size, 1,
                  dilation=dilation, padding=_get_padding(kernel_size, dilation))
    )


# ---------------------------------------------------------------------------
# ResBlock & ResBlock_SnakeBeta — general-purpose residual blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Multi-dilation residual block with LeakyReLU activation."""

    def __init__(self, channels: int, kernel_size: int = 3,
                 dilations: Tuple[int, ...] = (1, 3, 5)):
        super().__init__()
        self.convs1 = self._make_convs(channels, kernel_size, dilations)
        self.convs2 = self._make_convs(channels, kernel_size, [1] * len(dilations))

    @staticmethod
    def _make_convs(channels, kernel_size, dilations):
        layers = nn.ModuleList(
            [_create_conv1d_layer(channels, kernel_size, d) for d in dilations]
        )
        layers.apply(_init_weights)
        return layers

    def forward(self, x: torch.Tensor,
                x_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            x_residual = x
            xt = torch.nn.functional.leaky_relu(x, _LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = conv1(xt)
            xt = torch.nn.functional.leaky_relu(xt, _LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = conv2(xt)
            x = xt + x_residual
            if x_mask is not None:
                x = x * x_mask
        return x

    def remove_weight_norm(self):
        for conv in chain(self.convs1, self.convs2):
            if hasattr(conv, "parametrizations") and "weight" in conv.parametrizations: parametrize.remove_parametrizations(conv, "weight", leave_parametrized=True)
            else: remove_weight_norm(conv)


class ResBlock_SnakeBeta(nn.Module):
    """Multi-dilation residual block with SnakeBeta activation."""

    def __init__(self, channels: int, kernel_size: int = 3,
                 dilations: Tuple[int, ...] = (1, 3, 5)):
        super().__init__()
        self.convs1 = self._make_convs(channels, kernel_size, dilations)
        self.convs2 = self._make_convs(channels, kernel_size, [1] * len(dilations))
        self.acts1 = nn.ModuleList(
            [SnakeBeta(num_channels=channels, init=1.0, beta_init=1.0, log_scale=True)
             for _ in dilations]
        )
        self.acts2 = nn.ModuleList(
            [SnakeBeta(num_channels=channels, init=1.0, beta_init=1.0, log_scale=True)
             for _ in dilations]
        )

    @staticmethod
    def _make_convs(channels, kernel_size, dilations):
        layers = nn.ModuleList(
            [_create_conv1d_layer(channels, kernel_size, d) for d in dilations]
        )
        layers.apply(_init_weights)
        return layers

    def forward(self, x: torch.Tensor,
                x_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for i, (conv1, conv2) in enumerate(zip(self.convs1, self.convs2)):
            x_residual = x
            xt = self.acts1[i](x)
            if x_mask is not None:
                xt = xt * x_mask
            xt = conv1(xt)
            xt = self.acts2[i](xt)
            if x_mask is not None:
                xt = xt * x_mask
            xt = conv2(xt)
            x = xt + x_residual
            if x_mask is not None:
                x = x * x_mask
        return x

    def remove_weight_norm(self):
        for conv in chain(self.convs1, self.convs2):
            if hasattr(conv, "parametrizations") and "weight" in conv.parametrizations: parametrize.remove_parametrizations(conv, "weight", leave_parametrized=True)
            else: remove_weight_norm(conv)

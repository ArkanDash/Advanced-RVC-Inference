import os
import sys
import torch
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

sys.path.append(os.getcwd())

from main.library.algorithm.commons import get_padding
from main.library.algorithm.residuals import LRELU_SLOPE

class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, version, use_spectral_norm=False, checkpointing=False):
        super(MultiPeriodDiscriminator, self).__init__()
        self.checkpointing = checkpointing
        periods = ([2, 3, 5, 7, 11, 17] if version == "v1" else [2, 3, 5, 7, 11, 17, 23, 37])
        self.discriminators = torch.nn.ModuleList([DiscriminatorS(use_spectral_norm=use_spectral_norm, checkpointing=checkpointing)] + [DiscriminatorP(p, use_spectral_norm=use_spectral_norm, checkpointing=checkpointing) for p in periods])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []

        for d in self.discriminators:
            if self.training and self.checkpointing:
                def forward_discriminator(d, y, y_hat):
                    y_d_r, fmap_r = d(y)
                    y_d_g, fmap_g = d(y_hat)

                    return y_d_r, fmap_r, y_d_g, fmap_g
                y_d_r, fmap_r, y_d_g, fmap_g = checkpoint(forward_discriminator, d, y, y_hat, use_reentrant=False)
            else:
                y_d_r, fmap_r = d(y)
                y_d_g, fmap_g = d(y_hat)

            y_d_rs.append(y_d_r); fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g); fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False, checkpointing=False):
        super(DiscriminatorS, self).__init__()
        self.checkpointing = checkpointing
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = torch.nn.ModuleList([norm_f(torch.nn.Conv1d(1, 16, 15, 1, padding=7)), norm_f(torch.nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)), norm_f(torch.nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)), norm_f(torch.nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)), norm_f(torch.nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)), norm_f(torch.nn.Conv1d(1024, 1024, 5, 1, padding=2))])
        self.conv_post = norm_f(torch.nn.Conv1d(1024, 1, 3, 1, padding=1))
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x):
        fmap = []

        for conv in self.convs:
            x = checkpoint(self.lrelu, checkpoint(conv, x, use_reentrant = False), use_reentrant = False) if self.training and self.checkpointing else self.lrelu(conv(x))
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        return x.flatten(1, -1), fmap

class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, use_spectral_norm=False, checkpointing=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.checkpointing = checkpointing
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = torch.nn.ModuleList([norm_f(torch.nn.Conv2d(in_ch, out_ch, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))) for in_ch, out_ch, stride in zip([1, 32, 128, 512, 1024], [32, 128, 512, 1024, 1024], [3, 3, 3, 3, 1])])
        self.conv_post = norm_f(torch.nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0: x = F.pad(x, (0, (self.period - (t % self.period))), "reflect")
        x = x.view(b, c, -1, self.period)

        for conv in self.convs:
            x = checkpoint(self.lrelu, checkpoint(conv, x, use_reentrant = False), use_reentrant = False) if self.training and self.checkpointing else self.lrelu(conv(x))
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        return x.flatten(1, -1), fmap
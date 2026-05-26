import torch
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

from advanced_rvc_inference.engine.models.algorithms.commons import get_padding
from advanced_rvc_inference.engine.models.algorithms.residuals import LRELU_SLOPE

class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, version, use_spectral_norm=False, checkpointing=False):
        super(MultiPeriodDiscriminator, self).__init__()
        self.checkpointing = checkpointing

        if version == "v0":
            periods = [2, 3, 5, 7, 11]
            resolutions = []
        elif version == "v1":
            periods = [2, 3, 5, 7, 11, 17]
            resolutions = []
        elif version == "v2": 
            periods = [2, 3, 5, 7, 11, 17, 23, 37]
            resolutions = []
        elif version == "v3":
            periods = [2, 3, 5, 7, 11]
            resolutions = [[1024, 120, 600], [2048, 240, 1200], [512, 50, 240]]

        self.discriminators = torch.nn.ModuleList(
            [DiscriminatorS(use_spectral_norm=use_spectral_norm)] + 
            [DiscriminatorP(p, use_spectral_norm=use_spectral_norm) for p in periods] + 
            [DiscriminatorR(r, use_spectral_norm=use_spectral_norm) for r in resolutions]
        )

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []

        for d in self.discriminators:
            if self.training and self.checkpointing:
                y_d_r, fmap_r = checkpoint(d, y, use_reentrant=False)
                y_d_g, fmap_g = checkpoint(d, y_hat, use_reentrant=False)
            else:
                y_d_r, fmap_r = d(y)
                y_d_g, fmap_g = d(y_hat)

            y_d_rs.append(y_d_r); fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g); fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = torch.nn.ModuleList([
            norm_f(torch.nn.Conv1d(1, 16, 15, 1, padding=7)), 
            norm_f(torch.nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)), 
            norm_f(torch.nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)), 
            norm_f(torch.nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)), 
            norm_f(torch.nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)), 
            norm_f(torch.nn.Conv1d(1024, 1024, 5, 1, padding=2))
        ])
        self.conv_post = norm_f(torch.nn.Conv1d(1024, 1, 3, 1, padding=1))
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x):
        fmap = []
        for conv in self.convs:
            x = self.lrelu(conv(x))
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        return x.flatten(1, -1), fmap

class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = torch.nn.ModuleList([
            norm_f(
                torch.nn.Conv2d(
                    in_ch, 
                    out_ch, 
                    (kernel_size, 1), 
                    (stride, 1), 
                    padding=(get_padding(kernel_size, 1), 0)
                )
            ) 
            for in_ch, out_ch, stride in zip(
                [1, 32, 128, 512, 1024], 
                [32, 128, 512, 1024, 1024], 
                [3, 3, 3, 3, 1]
            )
        ])
        self.conv_post = norm_f(torch.nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0: x = F.pad(x, (0, (self.period - (t % self.period))), "reflect")
        x = x.view(b, c, -1, self.period)

        for conv in self.convs:
            x = self.lrelu(conv(x))
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        return x.flatten(1, -1), fmap

class DiscriminatorR(torch.nn.Module):
    def __init__(self, resolution, use_spectral_norm=False):
        super().__init__()
        self.resolution = resolution
        self.lrelu_slope = 0.1
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = torch.nn.ModuleList([
            norm_f(torch.nn.Conv2d( 1, 32, (3, 9), padding=(1, 4))), 
            norm_f(torch.nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))), 
            norm_f(torch.nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))), 
            norm_f(torch.nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))), 
            norm_f(torch.nn.Conv2d(32, 32, (3, 3), padding=(1, 1)))
        ])
        self.conv_post = norm_f(torch.nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        fmap = []
        x = self.spectrogram(x).unsqueeze(1)
        
        for layer in self.convs:
            x = F.leaky_relu(layer(x), self.lrelu_slope)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        return x.flatten(1, -1), fmap

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        pad = int((n_fft - hop_length) / 2)

        is_not_cuda = x.device.type in ["privateuseone", "ocl"]
        stft = torch.stft(
            F.pad(
                x.cpu() if is_not_cuda else x, 
                (pad, pad), 
                mode="reflect"
            ).squeeze(1), 
            n_fft=n_fft, 
            hop_length=hop_length, 
            win_length=win_length, 
            window=torch.ones(win_length, device="cpu" if is_not_cuda else x.device), 
            center=False, 
            return_complex=True
        )

        return torch.view_as_real(stft).norm(p=2, dim=-1).to(x.device)

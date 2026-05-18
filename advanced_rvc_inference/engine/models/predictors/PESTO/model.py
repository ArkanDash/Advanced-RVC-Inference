import math
import torch

from functools import partial

class PPESTO(torch.nn.Module):
    def __init__(self, encoder, preprocessor, crop_kwargs = None, reduction = "alwa"):
        super(PPESTO, self).__init__()
        self.encoder = encoder
        self.preprocessor = preprocessor
        self.confidence = ConfidenceClassifier()
        if crop_kwargs is None: crop_kwargs = {}
        self.crop_cqt = CropCQT(**crop_kwargs)
        self.reduction = reduction
        self.register_buffer('shift', torch.zeros((), dtype=torch.float), persistent=True)

    def forward(self, audio_waveforms, sr = 16000, convert_to_freq = True, return_activations = False):
        batch_size = audio_waveforms.size(0) if audio_waveforms.ndim == 2 else None
        x = self.preprocessor(audio_waveforms, sr=sr).flatten(0, 1)

        energy = x.mul_(math.log(10) / 10.).exp().squeeze_(1)
        vol = energy.sum(dim=-1)
        confidence = self.confidence(energy)

        x = self.crop_cqt(x)
        activations = self.encoder(x)

        if batch_size is None: confidence.squeeze_(0)
        else:
            activations = activations.view(batch_size, -1, activations.size(-1))
            confidence = confidence.view(batch_size, -1)
            vol = vol.view(batch_size, -1)

        activations = activations.roll(-(self.shift * self.bins_per_semitone).round().int().item(), -1)
        preds = self.reduce_activations(activations)
        if convert_to_freq: preds = 440 * 2 ** ((preds - 69) / 12)

        if return_activations: return preds, confidence, vol, activations
        return preds, confidence

    @property
    def bins_per_semitone(self):
        return self.preprocessor.hcqt_kwargs["bins_per_semitone"]

    @property
    def hop_size(self):
        return self.preprocessor.hop_size
    
    def reduce_activations(self, activations):
        device = activations.device
        num_bins = activations.size(-1)
        if torch.is_tensor(num_bins): num_bins = num_bins.item()

        bps, r = divmod(num_bins, 128)
        assert r == 0

        if self.reduction == "argmax": return activations.argmax(dim=-1).float() / bps

        all_pitches = torch.arange(num_bins, dtype=torch.float, device=device).div_(bps)
        if self.reduction == "mean": return activations.matmul(all_pitches)

        if self.reduction == "alwa":
            indices = (activations.argmax(dim=-1, keepdim=True) + (torch.arange(1, 2 * bps, device=device) - bps)).clip_(min=0, max=num_bins - 1)
            cropped_activations = activations.gather(-1, indices)
            return (cropped_activations * all_pitches.unsqueeze(0).expand_as(activations).gather(-1, indices)).sum(dim=-1) / cropped_activations.sum(dim=-1)

        raise ValueError
    
class ConfidenceClassifier(torch.nn.Module):
    def __init__(self):
        super(ConfidenceClassifier, self).__init__()
        self.conv = torch.nn.Conv1d(1, 1, 39, stride=3)
        self.linear = torch.nn.Linear(72, 1)

    def forward(self, x):
        return self.linear(torch.cat((torch.nn.functional.relu(self.conv(x.unsqueeze(1)).squeeze(1)), x.log().mean(dim=-1, keepdim=True).exp() / x.mean(dim=-1, keepdim=True).clip_(min=1e-8)), dim=-1)).sigmoid().squeeze(-1)

class CropCQT(torch.nn.Module):
    def __init__(self, min_steps, max_steps):
        super(CropCQT, self).__init__()
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.lower_bin = self.max_steps

    def forward(self, spectrograms):
        return spectrograms[..., self.max_steps: self.min_steps]
    
class Resnet1d(torch.nn.Module):
    def __init__(self, n_chan_input=1, n_chan_layers=(20, 20, 10, 1), n_prefilt_layers=1, prefilt_kernel_size=15, residual=False, n_bins_in=216, output_dim=128, activation_fn = "leaky", a_lrelu=0.3, p_dropout=0.2, **unused):
        super(Resnet1d, self).__init__()
        self.hparams = dict(n_chan_input=n_chan_input, n_chan_layers=n_chan_layers, n_prefilt_layers=n_prefilt_layers, prefilt_kernel_size=prefilt_kernel_size, residual=residual, n_bins_in=n_bins_in, output_dim=output_dim, activation_fn=activation_fn, a_lrelu=a_lrelu, p_dropout=p_dropout)

        if activation_fn == "relu":
            activation_layer = torch.nn.ReLU
        elif activation_fn == "silu":
            activation_layer = torch.nn.SiLU
        elif activation_fn == "leaky":
            activation_layer = partial(torch.nn.LeakyReLU, negative_slope=a_lrelu)
        else:
            raise ValueError

        n_in = n_chan_input
        n_ch = n_chan_layers
        if len(n_ch) < 5: n_ch.append(1)

        self.layernorm = torch.nn.LayerNorm(normalized_shape=[n_in, n_bins_in])
        prefilt_padding = prefilt_kernel_size // 2

        self.conv1 = torch.nn.Sequential(torch.nn.Conv1d(in_channels=n_in, out_channels=n_ch[0], kernel_size=prefilt_kernel_size, padding=prefilt_padding, stride=1), activation_layer(), torch.nn.Dropout(p=p_dropout))
        self.n_prefilt_layers = n_prefilt_layers
        self.prefilt_layers = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Conv1d(in_channels=n_ch[0], out_channels=n_ch[0], kernel_size=prefilt_kernel_size, padding=prefilt_padding, stride=1), activation_layer(), torch.nn.Dropout(p=p_dropout)) for _ in range(n_prefilt_layers-1)])
        self.residual = residual
        conv_layers = []

        for i in range(len(n_chan_layers)-1):
            conv_layers.extend([torch.nn.Conv1d(in_channels=n_ch[i], out_channels=n_ch[i + 1], kernel_size=1, padding=0, stride=1), activation_layer(), torch.nn.Dropout(p=p_dropout)])

        self.conv_layers = torch.nn.Sequential(*conv_layers)
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.fc = ToeplitzLinear(n_bins_in * n_ch[-1], output_dim)
        self.final_norm = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.conv1(self.layernorm(x))

        for p in range(0, self.n_prefilt_layers - 1):
            prefilt_layer = self.prefilt_layers[p]

            if self.residual:
                x = prefilt_layer(x) + x
            else:
                x = prefilt_layer(x)

        return self.final_norm(self.fc(self.flatten(self.conv_layers(x))))
    
class ToeplitzLinear(torch.nn.Conv1d):
    def __init__(self, in_features, out_features):
        super(ToeplitzLinear, self).__init__(in_channels=1, out_channels=1, kernel_size=in_features+out_features-1, padding=out_features-1, bias=False)

    def forward(self, input):
        return super(ToeplitzLinear, self).forward(input.unsqueeze(-2)).squeeze(-2)
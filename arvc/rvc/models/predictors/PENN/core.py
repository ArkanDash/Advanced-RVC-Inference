import math
import torch

import torch.nn.functional as F

PITCH_BINS, CENTS_PER_BIN, OCTAVE = 1440, 5, 1200

def frequency_to_bins(frequency, quantize_fn=torch.floor):
    return cents_to_bins(frequency_to_cents(frequency), quantize_fn)

def cents_to_bins(cents, quantize_fn=torch.floor):
    bins = quantize_fn(cents / CENTS_PER_BIN).long()
    bins[bins < 0] = 0
    bins[bins >= PITCH_BINS] = PITCH_BINS - 1
    return bins

def cents_to_frequency(cents):
    return 31 * 2 ** (cents / OCTAVE)

def bins_to_cents(bins):
    return CENTS_PER_BIN * bins

def frequency_to_cents(frequency):
    return OCTAVE * (frequency / 31).log2()

def seconds_to_samples(seconds, sample_rate=8000):
    return seconds * sample_rate

def interpolate(pitch, periodicity, value):
    voiced = periodicity > value
    if not voiced.any(): return pitch

    pitch = pitch.log2()
    pitch[..., 0] = pitch[voiced][..., 0]
    pitch[..., -1] = pitch[voiced][..., -1]
    voiced[..., 0] = True
    voiced[..., -1] = True
    pitch[~voiced] = _interpolate(torch.where(~voiced[0])[0][None], torch.where(voiced[0])[0][None], pitch[voiced][None])

    return 2 ** pitch

def _interpolate(x, xp, fp):
    if xp.shape[-1] == 0: return x
    if xp.shape[-1] == 1: return torch.full(x.shape, fp.squeeze(), device=fp.device, dtype=fp.dtype)

    m = (fp[:, 1:] - fp[:, :-1]) / (xp[:, 1:] - xp[:, :-1])
    b = fp[:, :-1] - (m.mul(xp[:, :-1]))

    indicies = x[:, :, None].ge(xp[:, None, :]).sum(-1) - 1
    indicies = indicies.clamp(0, m.shape[-1] - 1)
    line_idx = torch.linspace(0, indicies.shape[0], 1, device=indicies.device).to(torch.long).expand(indicies.shape)

    return m[line_idx, indicies].mul(x) + b[line_idx, indicies]

def entropy(logits):
    distribution = F.softmax(logits, dim=1)
    return (1 + 1 / math.log(PITCH_BINS) * (distribution * (distribution + 1e-7).log()).sum(dim=1))
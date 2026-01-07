import torch

def mean(signals, win_length=9):
    assert signals.dim() == 2

    signals = signals.unsqueeze(1)
    mask = ~torch.isnan(signals)
    padding = win_length // 2

    ones_kernel = torch.ones(signals.size(1), 1, win_length, device=signals.device)
    avg_pooled = torch.nn.functional.conv1d(torch.where(mask, signals, torch.zeros_like(signals)), ones_kernel, stride=1, padding=padding) / torch.nn.functional.conv1d(mask.float(), ones_kernel, stride=1, padding=padding).clamp(min=1) 
    avg_pooled[avg_pooled == 0] = float("nan")

    return avg_pooled.squeeze(1)

def median(signals, win_length):
    assert signals.dim() == 2

    signals = signals.unsqueeze(1)
    mask = ~torch.isnan(signals)
    padding = win_length // 2

    x = torch.nn.functional.pad(torch.where(mask, signals, torch.zeros_like(signals)), (padding, padding), mode="reflect")
    mask = torch.nn.functional.pad(mask.float(), (padding, padding), mode="constant", value=0)

    x = x.unfold(2, win_length, 1)
    mask = mask.unfold(2, win_length, 1)

    x = x.contiguous().view(x.size()[:3] + (-1,))
    mask = mask.contiguous().view(mask.size()[:3] + (-1,))

    x_sorted, _ = torch.where(mask.bool(), x.float(), float("inf")).to(x).sort(dim=-1)

    median_pooled = x_sorted.gather(-1, ((mask.sum(dim=-1) - 1) // 2).clamp(min=0).unsqueeze(-1).long()).squeeze(-1)
    median_pooled[torch.isinf(median_pooled)] = float("nan")

    return median_pooled.squeeze(1)
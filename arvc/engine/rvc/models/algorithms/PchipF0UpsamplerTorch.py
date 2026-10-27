import numpy as np
import torch
from typing import Tuple


def round_half_to_even_np32(x: torch.Tensor) -> torch.Tensor:
    """
    Emulate np.round(x.astype(np.float32)) on GPU for non-negative x.
    Operates in float32 domain and returns float32.
    """
    flo = torch.floor(x)
    frac = x - flo
    lt = frac < 0.5
    gt = frac > 0.5
    half_case = (~lt) & (~gt)
    flo_even = (torch.remainder(flo, 2) == 0)
    half_rounded = torch.where(flo_even, flo, flo + 1.0)
    return torch.where(lt, flo, torch.where(gt, flo + 1.0, half_rounded))


@torch.jit.script
def _pchip_end_slopes(dx: torch.Tensor, delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    dx: (B, N-1) or (N-1,)    positive step sizes
    delta: (B, N-1) or (N-1,) slopes between points
    returns endpoint derivatives (d0, dn) shape (B,) or scalars when input 1D.
    SciPy Fritsch-Butland endpoint rules.
    """
    h0 = dx[:, 0] if dx.dim() == 2 else dx[0]
    h1 = dx[:, 1] if dx.dim() == 2 else dx[1]
    hm2 = dx[:, -2] if dx.dim() == 2 else dx[-2]
    hm1 = dx[:, -1] if dx.dim() == 2 else dx[-1]

    d0 = ((2.0*h0 + h1) * delta[:, 0] - h0 * delta[:, 1]) / (h0 + h1)
    dn = ((2.0*hm1 + hm2) * delta[:, -1] - hm1 * delta[:, -2]) / (hm1 + hm2)

    mask0 = (d0 * delta[:, 0]) <= 0
    d0 = torch.where(mask0, torch.zeros_like(d0), d0)
    mask0b = ((delta[:, 0] * delta[:, 1]) < 0) & (~mask0)
    d0 = torch.where(mask0b & (d0.abs() > 3*delta[:, 0].abs()), 3*delta[:, 0], d0)

    maskn = (dn * delta[:, -1]) <= 0
    dn = torch.where(maskn, torch.zeros_like(dn), dn)
    masknb = ((delta[:, -1] * delta[:, -2]) < 0) & (~maskn)
    dn = torch.where(masknb & (dn.abs() > 3*delta[:, -1].abs()), 3*delta[:, -1], dn)

    return d0, dn


class PchipF0UpsamplerTorch(torch.nn.Module):
    """
    Auto-switching PCHIP upsampler:
      - If B == 1 -> uses the single-sample fast path (minimal overhead).
      - If B > 1  -> uses a batched/padded path to improve throughput.
    Matches SciPy PchipInterpolator semantics (voiced-only spline, Fritsch-Butland endpoints,
    NumPy round-half-to-even mapping for unvoiced masking). Uses float64 internally.
    """
    def __init__(self, scale_factor: int):
        super().__init__()
        if scale_factor < 1:
            raise ValueError("scale_factor must be >= 1")
        self.scale_factor = scale_factor

    def forward(self, f0: torch.Tensor) -> torch.Tensor:

        # print(f"PCHIP_UPSAMPLER:  Input dtype: {f0.dtype}")

        if f0.dim() != 3 or f0.size(1) != 1:
            raise ValueError(f"Expected (B,1,T), got {tuple(f0.shape)}")
        B, _, T = f0.shape
        if B == 1:
            return self._forward_single(f0)
        else:
            return self._forward_batched(f0)

    def _forward_single(self, f0: torch.Tensor) -> torch.Tensor:
        """
        Fast path for B==1. This mirrors the exact SciPy-matching single-sample version
        you already validated.
        """
        B, _, T = f0.shape
        device = f0.device
        up_len = T * self.scale_factor

        up_x64 = torch.linspace(0.0, float(T - 1), up_len, device=device, dtype=torch.float64)
        up_x32 = up_x64.to(torch.float32)

        y_orig64 = f0[0, 0].to(torch.float64)
        voiced_mask = y_orig64 > 0.0
        if voiced_mask.sum().item() < 2:
            return torch.zeros((1, 1, up_len), device=device, dtype=torch.float32)

        x_full64 = torch.arange(T, device=device, dtype=torch.float64)
        x_v = x_full64[voiced_mask]
        y_v = y_orig64[voiced_mask]
        Nv = x_v.numel()

        if Nv == 2:
            dx = x_v[1] - x_v[0]
            delta = (y_v[1] - y_v[0]) / dx
            d_v = torch.stack([delta, delta], dim=0)
        else:
            dx = x_v[1:] - x_v[:-1]
            dy = y_v[1:] - y_v[:-1]
            delta = dy / dx
            d_v = torch.zeros_like(y_v)
            hk = dx[1:]
            hkm1 = dx[:-1]
            w1 = 2.0*hk + hkm1
            w2 = hk + 2.0*hkm1
            del_km1, del_k = delta[:-1], delta[1:]
            mono = (del_km1 * del_k) > 0.0
            denom = (w1 / del_km1) + (w2 / del_k)
            hm = (w1 + w2) / denom
            d_v[1:-1] = torch.where(mono, hm, torch.zeros_like(hm))

            d0, dn = _pchip_end_slopes(dx.unsqueeze(0), delta.unsqueeze(0))
            d_v[0] = d0
            d_v[-1] = dn

        seg = torch.searchsorted(x_v, up_x64, right=True) - 1
        seg = torch.clamp(seg, 0, Nv - 2)

        x0 = x_v[seg]
        x1 = x_v[seg + 1]
        y0 = y_v[seg]
        y1 = y_v[seg + 1]
        d0 = d_v[seg]
        d1 = d_v[seg + 1]
        h = x1 - x0

        t = (up_x64 - x0) / h
        t2 = t * t
        t3 = t2 * t

        h00 = 2.0*t3 - 3.0*t2 + 1.0
        h10 = t3 - 2.0*t2 + t
        h01 = -2.0*t3 + 3.0*t2
        h11 = t3 - t2

        up_y64 = h00*y0 + h10*h*d0 + h01*y1 + h11*h*d1

        up_y32 = up_y64.to(torch.float32)
        up_y32 = torch.where(up_y32 < 0.0, torch.zeros_like(up_y32), up_y32)

        back_idx = round_half_to_even_np32(up_x32).to(torch.int64).clamp(0, T - 1)
        keep_mask = voiced_mask.to(torch.bool)[back_idx]
        up_y32 = torch.where(keep_mask, up_y32, torch.zeros_like(up_y32))

        return up_y32.unsqueeze(0).unsqueeze(0)

    def _forward_batched(self, f0: torch.Tensor) -> torch.Tensor:
        """
        Batched path for B > 1.
        Strategy:
         - pack voiced x,y per batch into padded tensors of shape (B, maxNv)
         - compute derivatives per batch (vectorized where possible)
         - evaluate spline per batch (loop over B for evaluation segments, but heavy ops are vectorized)
        This preserves SciPy semantics exactly.
        """
        B, _, T = f0.shape
        device = f0.device
        up_len = T * self.scale_factor

        up_x64 = torch.linspace(0.0, float(T - 1), up_len, device=device, dtype=torch.float64)
        up_x32 = up_x64.to(torch.float32)

        out = torch.zeros((B, 1, up_len), device=device, dtype=torch.float32)

        y_full64 = f0.squeeze(1).to(torch.float64)
        voiced = y_full64 > 0.0
        voiced_counts = voiced.sum(dim=1)

        maxNv = int(voiced_counts.max().item())
        if maxNv < 2:
            return out

        x_idx = torch.arange(T, device=device, dtype=torch.float64).unsqueeze(0).expand(B, -1)  # (B,T)
        x_packed = torch.zeros((B, maxNv), device=device, dtype=torch.float64)
        y_packed = torch.zeros((B, maxNv), device=device, dtype=torch.float64)
        valid_packed = torch.zeros((B, maxNv), device=device, dtype=torch.bool)

        for b in range(B):
            Nv = int(voiced_counts[b].item())
            if Nv == 0:
                continue
            maskb = voiced[b]
            xb = x_idx[b, maskb]
            yb = y_full64[b, maskb]
            x_packed[b, :Nv] = xb
            y_packed[b, :Nv] = yb
            valid_packed[b, :Nv] = True

        d_packed = torch.zeros_like(y_packed)
        for b in range(B):
            Nv = int(voiced_counts[b].item())
            if Nv < 2:
                continue
            x_v = x_packed[b, :Nv]
            y_v = y_packed[b, :Nv]
            if Nv == 2:
                dx = x_v[1] - x_v[0]
                delta = (y_v[1] - y_v[0]) / dx
                d_packed[b, 0] = delta
                d_packed[b, 1] = delta
            else:
                dx = x_v[1:] - x_v[:-1]
                dy = y_v[1:] - y_v[:-1]
                delta = dy / dx

                hk = dx[1:]
                hkm1 = dx[:-1]
                w1 = 2.0*hk + hkm1
                w2 = hk + 2.0*hkm1
                del_km1 = delta[:-1]
                del_k   = delta[1:]
                mono = (del_km1 * del_k) > 0.0
                denom = (w1 / del_km1) + (w2 / del_k)
                hm = (w1 + w2) / denom
                d_packed[b, 1:Nv-1] = torch.where(mono, hm, torch.zeros_like(hm))
                d0, dn = _pchip_end_slopes(dx.unsqueeze(0), delta.unsqueeze(0))

                d_packed[b, 0] = d0
                d_packed[b, Nv-1] = dn

        for b in range(B):
            Nv = int(voiced_counts[b].item())
            if Nv < 2:
                continue
            x_v = x_packed[b, :Nv]
            y_v = y_packed[b, :Nv]
            d_v = d_packed[b, :Nv]

            seg = torch.searchsorted(x_v, up_x64, right=True) - 1
            seg = torch.clamp(seg, 0, Nv - 2)

            x0 = x_v[seg]
            x1 = x_v[seg + 1]
            y0 = y_v[seg]
            y1 = y_v[seg + 1]
            d0 = d_v[seg]
            d1 = d_v[seg + 1]
            h = x1 - x0

            t = (up_x64 - x0) / h
            t2 = t * t
            t3 = t2 * t

            h00 = 2.0*t3 - 3.0*t2 + 1.0
            h10 = t3 - 2.0*t2 + t
            h01 = -2.0*t3 + 3.0*t2
            h11 = t3 - t2

            up_y64 = h00*y0 + h10*h*d0 + h01*y1 + h11*h*d1
            up_y32 = up_y64.to(torch.float32)
            up_y32 = torch.where(up_y32 < 0.0, torch.zeros_like(up_y32), up_y32)

            back_idx = round_half_to_even_np32(up_x32).long().clamp(0, T - 1)
            keep_mask = voiced[b, back_idx]
            up_y32 = torch.where(keep_mask, up_y32, torch.zeros_like(up_y32))

            out[b, 0] = up_y32

        return out

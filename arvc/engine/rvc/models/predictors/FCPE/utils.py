import os
import torch

from torch import nn
from io import BytesIO
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad

def decrypt_model(configs, input_path):
    with open(input_path, "rb") as f:
        data = f.read()

    with open(os.path.join(configs["binary_path"], "decrypt.bin"), "rb") as f:
        key = f.read()

    return BytesIO(unpad(AES.new(key, AES.MODE_CBC, data[:16]).decrypt(data[16:]), AES.block_size)).read()

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

def l2_regularization(model, l2_alpha):
    l2_loss = []
    for module in model.modules():
        if type(module) is nn.Conv2d: l2_loss.append((module.weight**2).sum() / 2.0)

    return l2_alpha * sum(l2_loss)

def torch_interp(x, xp, fp):
    sort_idx = xp.argsort()
    xp = xp[sort_idx]
    fp = fp[sort_idx]

    right_idxs = torch.searchsorted(xp, x).clamp(max=len(xp) - 1)
    left_idxs = (right_idxs - 1).clamp(min=0)
    x_left = xp[left_idxs]
    y_left = fp[left_idxs]

    interp_vals = y_left + ((x - x_left) * (fp[right_idxs] - y_left) / (xp[right_idxs] - x_left))
    interp_vals[x < xp[0]] = fp[0]
    interp_vals[x > xp[-1]] = fp[-1]

    return interp_vals

def batch_interp_with_replacement_detach(uv, f0):
    result = f0.clone()
    for i in range(uv.shape[0]):
        interp_vals = torch_interp(torch.where(uv[i])[-1], torch.where(~uv[i])[-1], f0[i][~uv[i]]).detach()
        result[i][uv[i]] = interp_vals
        
    return result

class DotDict(dict):
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, "dims == 2"
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)

class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()
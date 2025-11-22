import os
import sys
import torch
import platform
import subprocess

try:
    import pytorch_ocl
except:
    pytorch_ocl = None

sys.path.append(os.getcwd())

from main.library.backends.utils import GRU

torch_available = pytorch_ocl != None
if torch_available: adaptive_orig = torch.nn.AdaptiveAvgPool2d

def check_amd_gpu(gpu):
    for i in ["RX", "AMD", "Vega", "Radeon", "FirePro"]:
        return i in gpu

def get_amd_gpu_windows():
    gpus = ""

    try:
        gpus = subprocess.check_output("wmic path win32_VideoController get name", shell=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        gpus = subprocess.check_output('powershell "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"', shell=True, stderr=subprocess.DEVNULL)

    return [gpu.strip() for gpu in gpus.decode().split('\n')[1:] if check_amd_gpu(gpu)]

def get_amd_gpu_linux():
    try:
        return [gpu for gpu in subprocess.check_output("lspci | grep VGA", shell=True).decode().split('\n') if check_amd_gpu(gpu)]
    except:
        return []

def get_gpu_list():
    return (get_amd_gpu_windows() if platform.system() == "Windows" else get_amd_gpu_linux()) if torch_available else []

def device_count():
    return len(get_gpu_list()) if torch_available else 0

def device_name(device_id = 0):
    return (get_gpu_list()[device_id] if device_id >= 0 and device_id < device_count() else "") if torch_available else ""

def is_available():
    return (device_count() > 0) if torch_available else False

def group_norm(x, num_groups, weight=None, bias=None, eps=1e-5):
    N, C = x.shape[:2]
    assert C % num_groups == 0

    shape = (N, num_groups, C // num_groups) + x.shape[2:]
    x_reshaped = x.view(shape)

    dims = (2,) + tuple(range(3, x_reshaped.dim()))
    mean = x_reshaped.mean(dim=dims, keepdim=True)
    var = x_reshaped.var(dim=dims, keepdim=True, unbiased=False)

    x_norm = (x_reshaped - mean) / (var + eps).sqrt()
    x_norm = x_norm.view_as(x)

    if weight is not None:
        weight = weight.view(1, C, *([1] * (x.dim() - 2)))
        x_norm = x_norm * weight

    if bias is not None:
        bias = bias.view(1, C, *([1] * (x.dim() - 2)))
        x_norm = x_norm + bias

    return x_norm

def script(f, *_, **__):
    f.graph = pytorch_ocl.torch._C.Graph()
    return f

def AdaptiveAvgPool2d(input):
    input = input[0] if isinstance(input, tuple) else input
    return adaptive_orig(input)

if torch_available:
    torch.nn.GRU = GRU
    torch.nn.AdaptiveAvgPool2d = AdaptiveAvgPool2d
    torch.nn.functional.group_norm = group_norm
    torch.jit.script = script
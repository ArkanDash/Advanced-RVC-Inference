import os
import gc
import sys
import torch
import subprocess


from advanced_rvc_inference.engine.models.backends.utils import GRU

try:
    import torch_directml
except:
    torch_directml = None

torch_available = torch_directml != None

# Lazy import of fairseq - only needed when DirectML is actually used
_fairseq = None

def _get_fairseq():
    """Lazily import fairseq module when needed."""
    global _fairseq
    if _fairseq is None:
        from advanced_rvc_inference.engine.models.embedders import fairseq
        _fairseq = fairseq
    return _fairseq

def device_count():
    return torch_directml.device_count() if torch_available else 0

def device_name(device_id = 0):
    return torch_directml.device_name(device_id) if torch_available else ""

def is_available():
    return torch_directml.is_available() if torch_available else False

def empty_cache():
    # Resolve path relative to this file's package directory
    _backends_dir = os.path.dirname(os.path.abspath(__file__))
    empty_cache_path = os.path.join(_backends_dir, "dml_empty_cache", "empty_cache.exe")

    if torch_available and os.path.exists(empty_cache_path):
        subprocess.run([empty_cache_path], capture_output=True, text=True)
        gc.collect()

def forward_dml(ctx, x, scale):
    ctx.scale = scale
    res = x.clone().detach()
    return res

if torch_available: 
    torch.nn.GRU = GRU
    _get_fairseq().GradMultiply.forward = forward_dml
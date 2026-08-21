"""
ZLUDA Backend Module for Advanced RVC Inference.

ZLUDA (CUDA-on-AMD) is a CUDA compatibility layer that translates CUDA API
calls to HIP/ROCm, allowing CUDA applications to run on AMD GPUs.

When ZLUDA is active, torch.cuda.is_available() returns True but the actual
backend is AMD hardware, not NVIDIA. This means several CUDA-specific
optimizations must be disabled or adapted.

Detection methods:
- torch.version.cuda is None or empty (ZLUDA has no CUDA toolkit)
- torch.cuda.get_device_name() ends with "[ZLUDA]"
- DISABLE_ADDMM_CUDA_LT environment variable is set (common ZLUDA flag)
"""

import os
import torch


# ── ZLUDA Detection ──────────────────────────────────────────────────────────

def _detect_zluda() -> bool:
    """Detect whether the current environment is running under ZLUDA."""
    if not torch.cuda.is_available():
        return False

    # Method 1: Check CUDA version string (ZLUDA has no real CUDA version)
    if torch.version.cuda is None:
        return True

    # Method 2: Check for "zluda" in version string (some builds report it)
    if hasattr(torch.version, "cuda") and torch.version.cuda is not None:
        if "zluda" in str(torch.version.cuda).lower():
            return True

    # Method 3: Check GPU device name suffix
    try:
        if torch.cuda.device_count() > 0:
            if torch.cuda.get_device_name(0).endswith("[ZLUDA]"):
                return True
    except Exception:
        pass

    # Method 4: Check environment variable (commonly set in ZLUDA setups)
    if os.environ.get("DISABLE_ADDMM_CUDA_LT", "") == "1":
        return True

    return False


# Cached detection result (computed once at import time)
_zluda_detected = _detect_zluda()


def is_available() -> bool:
    """Check if ZLUDA is available and active.

    Returns:
        True if running under ZLUDA (AMD GPU via CUDA compatibility layer).
    """
    return _zluda_detected


def device_count() -> int:
    """Get the number of ZLUDA-visible GPUs.

    Returns:
        Number of AMD GPUs visible through ZLUDA, or 0 if not available.
    """
    if not _zluda_detected:
        return 0
    try:
        return torch.cuda.device_count()
    except Exception:
        return 0


def device_name(device_id: int = 0) -> str:
    """Get the name of a ZLUDA-visible AMD GPU.

    Args:
        device_id: GPU index (default: 0).

    Returns:
        GPU name string, or empty string if not available.
    """
    if not _zluda_detected:
        return ""
    try:
        if 0 <= device_id < torch.cuda.device_count():
            return torch.cuda.get_device_name(device_id)
    except Exception:
        pass
    return ""


# ── ZLUDA Compatibility Workarounds ──────────────────────────────────────────
# These monkeypatches are applied when ZLUDA is detected to fix known
# compatibility issues between CUDA-expecting PyTorch code and ZLUDA's
# HIP backend.

if _zluda_detected:
    # Override torch.stft with a ZLUDA-compatible implementation.
    # ZLUDA's HIP backend has issues with the native torch.stft on complex
    # tensors, so we provide a manual FFT-based STFT implementation.
    class _ZludaSTFT:
        def __init__(self):
            self.device = "cuda"
            self.fourier_bases = {}

        def _get_fourier_basis(self, n_fft):
            if n_fft in self.fourier_bases:
                return self.fourier_bases[n_fft]

            fourier_basis = torch.fft.fft(torch.eye(n_fft, device="cpu")).to(
                self.device
            )

            cutoff = n_fft // 2 + 1
            fourier_basis = torch.cat(
                [fourier_basis.real[:cutoff], fourier_basis.imag[:cutoff]], dim=0
            )

            self.fourier_bases[n_fft] = fourier_basis
            return fourier_basis

        def transform(self, input, n_fft, hop_length, window):
            fourier_basis = self._get_fourier_basis(n_fft)
            fourier_basis = fourier_basis * window

            pad_amount = n_fft // 2
            input = torch.nn.functional.pad(
                input, (pad_amount, pad_amount), mode="reflect"
            )

            input_frames = input.unfold(1, n_fft, hop_length).permute(0, 2, 1)
            fourier_transform = fourier_basis @ input_frames
            cutoff = n_fft // 2 + 1

            return torch.complex(
                fourier_transform[:, :cutoff, :], fourier_transform[:, cutoff:, :]
            )

    _stft_impl = _ZludaSTFT()
    _original_stft = torch.stft

    def _zluda_stft(input, window, *args, **kwargs):
        if (
            kwargs.get("win_length") is None
            and kwargs.get("center") is None
            and kwargs.get("return_complex") is True
        ):
            return _stft_impl.transform(
                input, kwargs.get("n_fft"), kwargs.get("hop_length"), window
            )
        else:
            return _original_stft(
                input=input.cpu(), window=window.cpu(), *args, **kwargs
            ).to(input.device)

    def _zluda_jit(f, *_, **__):
        """No-op jit.script replacement — ZLUDA does not support TorchScript."""
        f.graph = torch._C.Graph()
        return f

    torch.stft = _zluda_stft
    torch.jit.script = _zluda_jit

    # Disable cuDNN — ZLUDA uses HIP MIOpen, not NVIDIA cuDNN
    torch.backends.cudnn.enabled = False

    # Configure Scaled Dot Product Attention backends for ZLUDA/HIP
    # Flash SDP: Not supported on HIP
    # Memory-efficient SDP: Not supported on HIP
    # Math SDP: Fallback (slow but correct)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

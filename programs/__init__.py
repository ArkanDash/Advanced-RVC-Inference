"""
Advanced RVC Programs Module
Contains optimized implementations and utilities
"""

# Import KADVC kernels for easy access
try:
    from .kernels import (
        setup_kadvc_for_rvc,
        get_kadvc_optimizer,
        KADVCConfig,
        KADVCCUDAKernels
    )
    KADVC_AVAILABLE = True
except ImportError:
    KADVC_AVAILABLE = False
    setup_kadvc_for_rvc = None
    get_kadvc_optimizer = None
    KADVCConfig = None
    KADVCCUDAKernels = None

__all__ = [
    "KADVC_AVAILABLE",
    "setup_kadvc_for_rvc", 
    "get_kadvc_optimizer",
    "KADVCConfig",
    "KADVCCUDAKernels"
]
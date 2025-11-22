"""
KADVC - Kernel Advanced Voice Conversion
High-performance CUDA kernel optimizations for RVC training and inference

Features:
- 2x faster training and inference on Google Colab
- Custom CUDA kernels for F0 extraction
- Mixed precision training optimizations
- Memory-efficient algorithms
- GPU-specific optimizations

Usage:
    from programs.kernels import setup_kadvc_for_rvc
    
    # Initialize KADVC optimization
    kadvc = setup_kadvc_for_rvc()
    
    # Use optimized functions
    f0 = kadvc.fast_f0_extraction(audio_tensor)
    features = kadvc.fast_feature_extraction(audio_tensor)
"""

from .kadvc_kernels import (
    KADVCCUDAKernels,
    setup_kadvc_environment,
    get_kadvc_performance_stats
)

from .kadvc_config import (
    KADVCConfig,
    create_optimized_config,
    get_gpu_type,
    DEFAULT_KADVC_CONFIG,
    LOCAL_KADVC_CONFIG
)

from .kadvc_integration import (
    KADVCOptimizer,
    KADVCMonitor,
    get_kadvc_optimizer,
    setup_kadvc_for_rvc
)

__version__ = "1.0.0"
__author__ = "BF667"

__all__ = [
    # Core kernel functions
    "KADVCCUDAKernels",
    "setup_kadvc_environment", 
    "get_kadvc_performance_stats",
    
    # Configuration
    "KADVCConfig",
    "create_optimized_config",
    "get_gpu_type", 
    "DEFAULT_KADVC_CONFIG",
    "LOCAL_KADVC_CONFIG",
    
    # Integration
    "KADVCOptimizer",
    "KADVCMonitor",
    "get_kadvc_optimizer",
    "setup_kadvc_for_rvc"
]
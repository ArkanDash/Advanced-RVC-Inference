"""
Advanced RVC Inference - Main Package
Kernel Advanced RVC - 2x Faster Training & Inference
Version 3.5.2

This package provides advanced voice conversion capabilities using the RVC (Retrieval-based Voice Conversion) 
technology with KRVC kernel optimizations for enhanced performance.

Author: MiniMax Agent
"""

__version__ = "3.5.2"
__author__ = "MiniMax Agent"

# Core imports for easy access
from .core import (
    full_inference_program,
    import_voice_converter,
    get_config,
    download_file,
    add_audio_effects,
    merge_audios,
    check_fp16_support,
    real_time_voice_conversion,
    models_vocals,
    karaoke_models,
    denoise_models,
    dereverb_models,
    deecho_models,
    gpu_optimizer,
    gpu_settings,
    GPU_OPTIMIZATION_AVAILABLE
)

# KRVC kernel imports
try:
    from .krvc_kernel import (
        KRVCFeatureExtractor,
        krvc_speed_optimize,
        krvc_inference_mode,
        krvc_training_mode,
        krvc_mixed_precision_training,
        KRVCAdvancedOptimizer,
        KRVCInferenceOptimizer,
        KRVCPerformanceMonitor,
        KRVCRealTimeProcessor,
        cleanup_krvc_memory
    )
    KRVC_AVAILABLE = True
except ImportError:
    KRVC_AVAILABLE = False

# Import submodules with fallback for missing modules
from . import tabs, rvc, msep
try:
    from . import lib
except ImportError:
    # lib module may not exist in some installations
    pass

# Package metadata
__all__ = [
    # Core functions
    'full_inference_program',
    'import_voice_converter',
    'get_config',
    'download_file',
    'add_audio_effects',
    'merge_audios',
    'check_fp16_support',
    'real_time_voice_conversion',

    # Model lists
    'models_vocals',
    'karaoke_models',
    'denoise_models',
    'dereverb_models',
    'deecho_models',

    # Submodules
    'tabs',
    'rvc',
    'msep',
]

# Add lib to __all__ only if it exists
try:
    from . import lib
    __all__.append('lib')
except ImportError:
    pass

__all__.extend([
    # KRVC features (if available)
    'KRVCFeatureExtractor',
    'krvc_speed_optimize',
    'krvc_inference_mode',
    'krvc_training_mode',
    'krvc_mixed_precision_training',
    'KRVCAdvancedOptimizer',
    'KRVCInferenceOptimizer',
    'KRVCPerformanceMonitor',
    'KRVCRealTimeProcessor',
    'cleanup_krvc_memory',

    # GPU Optimization features
    'gpu_optimizer',
    'gpu_settings',
    'GPU_OPTIMIZATION_AVAILABLE',

    # Availability flags
    'KRVC_AVAILABLE'
])
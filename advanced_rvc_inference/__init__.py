"""
Advanced RVC Inference - Main Package
Kernel Advanced RVC - 2x Faster Training & Inference
Version 3.5.2

This package provides advanced voice conversion capabilities using the RVC (Retrieval-based Voice Conversion)
technology with KRVC kernel optimizations for enhanced performance.

Authors: ArkanDash & BF667
"""

__version__ = "3.5.2"
__author__ = "ArkanDash & BF667"

# Core imports for easy access (with conditional fallbacks)
try:
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
    
    # Import path manager for easy access
    from .lib.path_manager import get_path_manager, PathManager
    
    # Make path_manager easily accessible as a module-level function
    path_manager = get_path_manager()
    
except ImportError as e:
    # Fallback implementations when core functions are not available
    def full_inference_program(*args, **kwargs):
        """Fallback for full_inference_program"""
        print("full_inference_program not available due to missing dependencies")
        return None
    
    def import_voice_converter(*args, **kwargs):
        """Fallback for import_voice_converter"""
        print("import_voice_converter not available due to missing dependencies")
        return None
    
    def get_config(*args, **kwargs):
        """Fallback for get_config"""
        return {}
    
    def download_file(*args, **kwargs):
        """Fallback for download_file"""
        print("download_file not available due to missing dependencies")
        return None
    
    def add_audio_effects(*args, **kwargs):
        """Fallback for add_audio_effects"""
        print("add_audio_effects not available due to missing dependencies")
        return None
    
    def merge_audios(*args, **kwargs):
        """Fallback for merge_audios"""
        print("merge_audios not available due to missing dependencies")
        return None
    
    def check_fp16_support(*args, **kwargs):
        """Fallback for check_fp16_support"""
        return False
    
    def real_time_voice_conversion(*args, **kwargs):
        """Fallback for real_time_voice_conversion"""
        print("real_time_voice_conversion not available due to missing dependencies")
        return None
    
    def models_vocals(*args, **kwargs):
        """Fallback for models_vocals"""
        return {}
    
    def karaoke_models(*args, **kwargs):
        """Fallback for karaoke_models"""
        return {}
    
    def denoise_models(*args, **kwargs):
        """Fallback for denoise_models"""
        return {}
    
    def dereverb_models(*args, **kwargs):
        """Fallback for dereverb_models"""
        return {}
    
    def deecho_models(*args, **kwargs):
        """Fallback for deecho_models"""
        return {}
    
    def gpu_optimizer(*args, **kwargs):
        """Fallback for gpu_optimizer"""
        print("gpu_optimizer not available due to missing dependencies")
        return None
    
    def gpu_settings(*args, **kwargs):
        """Fallback for gpu_settings"""
        return {}
    
    GPU_OPTIMIZATION_AVAILABLE = False

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
# Import sub-modules with fallbacks
try:
    from . import tabs
except ImportError:
    # Fallback for tabs module
    class tabs:
        """Fallback tabs module"""
        pass

try:
    from . import rvc
except ImportError:
    # Fallback for rvc module
    class rvc:
        """Fallback rvc module"""
        pass

try:
    from . import uvr
except ImportError:
    # Fallback for uvr module
    class uvr:
        """Fallback uvr module"""
        pass

# Path manager fallback
try:
    from .lib.path_manager import get_path_manager, PathManager
    path_manager = get_path_manager()
except ImportError:
    # Fallback for path_manager when lib.path_manager is not available
    class PathManager:
        """Fallback PathManager class"""
        def __init__(self, now_dir=None):
            self.now_dir = now_dir or os.getcwd() if 'os' in globals() else '/tmp'
            self.config = {}
    
    def get_path_manager(now_dir=None):
        """Fallback get_path_manager function"""
        return PathManager(now_dir)
    
    path_manager = PathManager()

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

    # Path management
    'path_manager',
    'get_path_manager',
    'PathManager',

    # Submodules
    'tabs',
    'rvc',
    'uvr',
]



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

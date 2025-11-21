"""Advanced RVC Inference Package"""

__version__ = "3.4.0"
__author__ = "BF667"
__email__ = "bf667@example.com"
__description__ = "Advanced RVC Inference with Vietnamese-RVC Integration and Performance Optimizations"

# Core modules
from .core.f0_extractor import EnhancedF0Extractor
from .audio.separation import EnhancedAudioSeparator
from .audio.voice_changer import RealtimeVoiceChanger
from .models.manager import EnhancedModelManager
from .ui.components import EnhancedUIComponents

# KADVC kernels
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

# Applio compatibility
try:
    from .applio_code.rvc.infer.infer import VoiceConverter
    from .applio_code.rvc.infer.pipeline import RVC_Inference_Pipeline
    APPLIO_AVAILABLE = True
except ImportError:
    APPLIO_AVAILABLE = False
    VoiceConverter = None
    RVC_Inference_Pipeline = None

# Training modules
try:
    from .training.trainer import RVC_Trainer
    from .training.simple_trainer import SimpleTrainer
    from .training.data.dataset import RVC_Dataset
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False
    RVC_Trainer = None
    SimpleTrainer = None
    RVC_Dataset = None

# Music separation modules
try:
    from .music_separation_code.inference import (
        DemucsInference,
        MDXInference, 
        BSRoformerInference
    )
    SEPARATION_AVAILABLE = True
except ImportError:
    SEPARATION_AVAILABLE = False
    DemucsInference = None
    MDXInference = None
    BSRoformerInference = None

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__description__",
    
    # Core modules
    "EnhancedF0Extractor",
    "EnhancedAudioSeparator", 
    "RealtimeVoiceChanger",
    "EnhancedModelManager",
    "EnhancedUIComponents",
    
    # KADVC optimization
    "KADVC_AVAILABLE",
    "setup_kadvc_for_rvc",
    "get_kadvc_optimizer",
    "KADVCConfig",
    "KADVCCUDAKernels",
    
    # Applio compatibility
    "APPLIO_AVAILABLE",
    "VoiceConverter",
    "RVC_Inference_Pipeline",
    
    # Training modules
    "TRAINING_AVAILABLE",
    "RVC_Trainer",
    "SimpleTrainer", 
    "RVC_Dataset",
    
    # Separation modules
    "SEPARATION_AVAILABLE",
    "DemucsInference",
    "MDXInference",
    "BSRoformerInference",
]
"""Advanced RVC Inference Package"""

__version__ = "3.4.0"
__author__ = "BF667"
__email__ = "bf667@example.com"
__description__ = "Advanced RVC Inference with Vietnamese-RVC Integration and Performance Optimizations"

from .core.f0_extractor import EnhancedF0Extractor
from .audio.separation import EnhancedAudioSeparator
from .audio.voice_changer import RealtimeVoiceChanger
from .models.manager import EnhancedModelManager
from .ui.components import EnhancedUIComponents

__all__ = [
    "EnhancedF0Extractor",
    "EnhancedAudioSeparator",
    "RealtimeVoiceChanger",
    "EnhancedModelManager",
    "EnhancedUIComponents",
    "__version__",
    "__author__",
    "__email__",
    "__description__"
]
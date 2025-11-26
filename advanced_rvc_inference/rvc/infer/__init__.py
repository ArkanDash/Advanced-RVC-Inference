"""
RVC Inference Module

This module contains all inference-related functionality for the RVC system.
"""

# Conditional imports to avoid dependency failures
try:
    from .conversion.convert import VoiceConverter
except ImportError:
    VoiceConverter = None

try:
    from .audio_effects import add_audio_effects
except ImportError:
    add_audio_effects = None

try:
    from .create_dataset import create_dataset
except ImportError:
    create_dataset = None

try:
    from .create_index import create_index
except ImportError:
    create_index = None

try:
    from .create_reference import create_reference
except ImportError:
    create_reference = None

# Inference utilities (for backward compatibility)
class inference_utils:
    """Inference utilities container"""
    
    @staticmethod
    def load_model(model_path):
        """Load model for inference"""
        return None
    
    @staticmethod
    def preprocess_audio(audio_path):
        """Preprocess audio for inference"""
        return None
    
    @staticmethod
    def postprocess_output(output):
        """Postprocess inference output"""
        return output

__all__ = [
    'VoiceConverter',
    'add_audio_effects', 
    'create_dataset',
    'create_index',
    'create_reference',
    'inference_utils'
]
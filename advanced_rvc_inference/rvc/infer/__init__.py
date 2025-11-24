"""
RVC Inference Module

This module contains all inference-related functionality for the RVC system.
"""

from .conversion.convert import VoiceConverter
from .audio_effects import add_audio_effects
from .create_dataset import create_dataset
from .create_index import create_index
from .create_reference import create_reference

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
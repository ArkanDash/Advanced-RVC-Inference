"""
RVC (Retrieval-based Voice Conversion) Module

This module contains the core RVC functionality including:
- Voice conversion algorithms
- Model inference and training
- Audio processing utilities
- Configuration management

Author: MiniMax Agent
"""

# Core RVC imports
from .infer.conversion.convert import VoiceConverter
from .configs.config import Config

# Training imports
from .train import training_utils

# Inference imports  
from .infer import inference_utils

# Configuration imports
from .configs import (
    config_v1,
    config_v2
)

__all__ = [
    'VoiceConverter',
    'Config',
    'training_utils', 
    'inference_utils',
    'config_v1',
    'config_v2'
]
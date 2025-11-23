"""
MSEP (Music Source Separation) Module

This module provides music source separation capabilities including:
- Vocal isolation and removal
- Instrumental track extraction
- Karaoke effect generation
- Audio enhancement and filtering

Author: MiniMax Agent  
"""

# Core MSEP imports
from .inference import proc_file
from .utils import separation_utils

# Model imports
from .models import (
    demucs4ht,
    mdx23c_tfc_tdf_v3,
    segm_models,
    torchseg_models,
    upernet_swin_transformers
)

__all__ = [
    'proc_file',
    'separation_utils',
    'demucs4ht',
    'mdx23c_tfc_tdf_v3', 
    'segm_models',
    'torchseg_models',
    'upernet_swin_transformers'
]
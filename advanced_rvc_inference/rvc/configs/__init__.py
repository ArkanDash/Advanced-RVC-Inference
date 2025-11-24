# This file makes the directory a Python package

import warnings
import os

def config_v1():
    """
    Load RVC v1 configuration
    
    Returns:
        dict: RVC v1 configuration
    """
    warnings.warn("RVC v1 configuration not available, using defaults")
    return {
        'model_version': 'v1',
        'sample_rate': 40000,
        'hop_length': 512,
        'win_length': 1024,
        'n_fft': 1024,
        'f0_min': 50,
        'f0_max': 1100,
        'mel_fmin': 0,
        'mel_fmax': 8000,
        'n_mel_channels': 80,
        'filter_length': 1024,
        'embedder_name': 'hubert_base',
        'cmvn': True,
        'normalize_volume': False,
        'use_pitch_guidance': True,
        'pitch_guidance_alpha': 0.5
    }

def config_v2():
    """
    Load RVC v2 configuration
    
    Returns:
        dict: RVC v2 configuration
    """
    warnings.warn("RVC v2 configuration not available, using defaults")
    return {
        'model_version': 'v2',
        'sample_rate': 48000,
        'hop_length': 512,
        'win_length': 2048,
        'n_fft': 2048,
        'f0_min': 50,
        'f0_max': 1100,
        'mel_fmin': 0,
        'mel_fmax': 16000,
        'n_mel_channels': 80,
        'filter_length': 2048,
        'embedder_name': 'hubert_base',
        'cmvn': True,
        'normalize_volume': False,
        'use_pitch_guidance': True,
        'pitch_guidance_alpha': 0.5,
        'use_energy_guidance': True,
        'energy_alpha': 0.5
    }

__all__ = [
    'config_v1',
    'config_v2'
]

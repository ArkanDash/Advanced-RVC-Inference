"""
Separate Music Module - Wrapper for uvr.separate_music

This module provides backward compatibility for imports from:
    from advanced_rvc_inference.separate_music import ...

The actual implementation is located in uvr/separate_music.py
"""

from advanced_rvc_inference.uvr.separate_music import (
    main,
    separate,
    parse_arguments,
    vr_models,
)

# For backward compatibility with code that might import _separate
try:
    from advanced_rvc_inference.uvr.separate_music import _separate as _separate
except ImportError:
    _separate = None

__all__ = [
    'main',
    'separate',
    'parse_arguments',
    'vr_models',
    '_separate',
]

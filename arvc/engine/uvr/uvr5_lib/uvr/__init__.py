"""
UVR separator modules.

Provides MDX and VR audio separation implementations.
"""

from arvc.engine.uvr.uvr5_lib.uvr.mdx_separator import MDXSeparator
from arvc.engine.uvr.uvr5_lib.uvr.vr_separator import VRSeparator

__all__ = [
    "MDXSeparator",
    "VRSeparator",
]

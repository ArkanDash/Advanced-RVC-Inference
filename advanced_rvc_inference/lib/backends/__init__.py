"""
Backend utilities for Advanced RVC Inference.
"""

from . import utils

# Import backend modules for direct access
try:
    from .. import directml
except ImportError:
    directml = None

try:
    from .. import opencl
except ImportError:
    opencl = None

__all__ = ['utils', 'directml', 'opencl']
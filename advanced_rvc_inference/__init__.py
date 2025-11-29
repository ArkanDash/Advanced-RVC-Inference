"""
Advanced RVC Inference - A state-of-the-art web UI for rapid and effortless inference.
This package provides tools for voice conversion, model training, and audio processing.
"""

__version__ = "0.1.0"
__author__ = "ArkanDash"

# Import main components to make them easily accessible
from . import app
from . import core

# Define what gets imported with "from advanced_rvc_inference import *"
__all__ = ["app", "core"]
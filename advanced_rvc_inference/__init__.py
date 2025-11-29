"""
Advanced RVC Inference - A state-of-the-art web UI for rapid and effortless inference.
This package provides tools for voice conversion, model training, and audio processing.
"""

__version__ = "0.1.0"
__author__ = "ArkanDash"

# Import main components to make them easily accessible
# Delay imports to avoid circular dependencies
def __getattr__(name):
    if name == "app":
        import advanced_rvc_inference.app
        return advanced_rvc_inference.app
    elif name == "core":
        import advanced_rvc_inference.core
        return advanced_rvc_inference.core
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Define what gets imported with "from advanced_rvc_inference import *"
__all__ = ["app", "core"]
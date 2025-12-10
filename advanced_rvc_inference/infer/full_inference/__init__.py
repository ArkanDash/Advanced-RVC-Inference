"""
RVC X UVR - Full Inference Package
Complete AI cover generation combining RVC voice conversion with UVR vocal separation.
"""

from .full_inference import FullInferencePipeline, create_full_inference_interface

__version__ = "1.0.0"
__author__ = "MiniMax Agent"

__all__ = [
    "FullInferencePipeline",
    "create_full_inference_interface"
]
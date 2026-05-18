"""
Engine package for Advanced RVC Inference.

This package contains the core RVC engine components:
- models: ML model architectures, algorithms, predictors, generators, embedders
- inference: Audio conversion and voice cloning
- realtime: Real-time voice changing
- training: Model training pipeline
- uvr: Audio source separation (UVR5)
- speaker: Speaker diarization and Whisper speech recognition
"""

__all__ = [
    "models",
    "inference",
    "realtime",
    "training",
    "uvr",
    "speaker",
]

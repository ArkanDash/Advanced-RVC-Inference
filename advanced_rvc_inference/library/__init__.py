"""
Library modules for Advanced RVC Inference.

This package contains audio processing libraries, ML models,
and utility functions.
"""

__all__ = [
    "utils",
    "backends",
    "algorithms",
    "architectures",
    "generators",
    "embedders",
    "predictors",
    "onnx",
    "speaker_diarization",
    "uvr5_lib",
]

# Lazy imports for library modules
_LAZY_MODULES = {
    "utils": ".utils",
    "backends": ".backends",
    "algorithms": ".algorithm",
    "architectures": ".architectures",
    "generators": ".generators",
    "embedders": ".embedders",
    "predictors": ".predictors",
    "onnx": ".onnx",
    "speaker_diarization": ".speaker_diarization",
    "uvr5_lib": ".uvr5_lib",
}


def __getattr__(name: str):
    """Lazy import mechanism for library modules."""
    if name in _LAZY_MODULES:
        import importlib

        module = importlib.import_module(_LAZY_MODULES[name], __package__)
        return module
    raise AttributeError(f"Module '{__name__}' has no attribute '{name}'")

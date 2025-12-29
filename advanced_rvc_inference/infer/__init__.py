"""
Inference modules for Advanced RVC Inference.

This package contains specialized inference functionality
including RVC, training, and extraction modules.
"""

__all__ = [
    "rvc",
    "train",
    "realtime",
    "extracting",
    "separate_music",
    "create_reference",
]

# Lazy imports for inference modules
_LAZY_MODULES = {
    "rvc": ".rvc",
    "train": ".train",
    "realtime": ".realtime",
    "extracting": ".extracting",
    "separate_music": ".separate_music",
    "create_reference": ".create_reference",
}


def __getattr__(name: str):
    """Lazy import mechanism for inference modules."""
    if name in _LAZY_MODULES:
        import importlib

        module = importlib.import_module(_LAZY_MODULES[name], __package__)
        return module
    raise AttributeError(f"Module '{__name__}' has no attribute '{name}'")

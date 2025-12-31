"""
Core processing modules for Advanced RVC Inference.

This package contains the core inference, training, and processing
functionality for voice conversion.
"""

__all__ = [
    "inference",
    "process",
    "model_utils",
    "f0_extract",
    "downloads",
    "presets",
    "realtime",
    "training",
    "separate",
    "tts",
    "ui",
    "utils",
    "csrt",
    "restart",
    "realtime_client",
]

# Lazy imports for heavy modules
_LAZY_MODULES = {
    "inference": ".inference",
    "process": ".process",
    "model_utils": ".model_utils",
    "f0_extract": ".f0_extract",
    "downloads": ".downloads",
    "presets": ".presets",
    "realtime": ".realtime",
    "training": ".training",
    "separate": ".separate",
    "tts": ".tts",
    "ui": ".ui",
    "utils": ".utils",
    "csrt": ".csrt",
    "restart": ".restart",
    "realtime_client": ".realtime_client",
}


def __getattr__(name: str):
    """Lazy import mechanism for core modules."""
    if name in _LAZY_MODULES:
        import importlib

        module = importlib.import_module(_LAZY_MODULES[name], __package__)
        return module
    raise AttributeError(f"Module '{__name__}' has no attribute '{name}'")

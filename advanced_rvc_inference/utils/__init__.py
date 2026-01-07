"""
Utility tools for Advanced RVC Inference.

This package contains various utility tools for file downloads,
audio processing, and other helper functions.
"""

__all__ = [
    "gdown",
    "huggingface",
    "mediafire",
    "meganz",
    "noisereduce",
    "pixeldrain",
]

# Lazy imports for tool modules
_LAZY_MODULES = {
    "gdown": ".gdown",
    "huggingface": ".huggingface",
    "mediafire": ".mediafire",
    "meganz": ".meganz",
    "noisereduce": ".noisereduce",
    "pixeldrain": ".pixeldrain",
}


def __getattr__(name: str):
    """Lazy import mechanism for tool modules."""
    if name in _LAZY_MODULES:
        import importlib

        module = importlib.import_module(_LAZY_MODULES[name], __package__)
        return module
    raise AttributeError(f"Module '{__name__}' has no attribute '{name}'")

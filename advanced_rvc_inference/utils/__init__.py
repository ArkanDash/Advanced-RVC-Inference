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
    "strtobool",
]


def strtobool(val):
    """Convert a string representation of truth to boolean.
    
    Replacement for the deprecated distutils.util.strtobool which was
    removed in Python 3.12+.
    """
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    val_str = str(val).strip().lower()
    if val_str in ('y', 'yes', 'true', 't', '1', 'on'):
        return True
    elif val_str in ('n', 'no', 'false', 'f', '0', 'off'):
        return False
    else:
        raise ValueError(f"Invalid truth value: {val!r}")


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

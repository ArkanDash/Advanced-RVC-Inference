"""
Library package for Advanced RVC Inference.

This package contains the RVC model library: architectures, algorithms,
generators, ONNX export helpers, optimizers, and related utilities.

Pluggable neural modules (embedders, predictors) live in
``arvc.engine.rvc.modules`` instead.
"""

__all__ = [
    "utils",
    "backends",
    "algorithms",
    "generators",
    "onnx",
    "optimizers",
    "safe_load",
    "weight_norm",
]

# Lazy imports for lib modules
_LAZY_MODULES = {
    "utils": ".utils",
    "backends": ".backends",
    "algorithms": ".algorithms",
    "generators": ".generators",
    "onnx": ".onnx",
    "optimizers": ".optimizers",
    "safe_load": ".safe_load",
    "weight_norm": ".weight_norm",
}


def __getattr__(name: str):
    """Lazy import mechanism for lib modules."""
    if name in _LAZY_MODULES:
        import importlib

        module = importlib.import_module(_LAZY_MODULES[name], __package__)
        return module
    raise AttributeError(f"Module '{__name__}' has no attribute '{name}'")

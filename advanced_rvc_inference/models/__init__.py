"""
Models package for Advanced RVC Inference.

This package contains ML model architectures, algorithms,
predictors, generators, embedders, and related utilities.
"""

__all__ = [
    "utils",
    "backends",
    "algorithms",
    "generators",
    "embedders",
    "predictors",
    "onnx",
    "optimizers",
    "speaker",
]

# Lazy imports for models modules
_LAZY_MODULES = {
    "utils": ".utils",
    "backends": ".backends",
    "algorithms": ".algorithms",
    "generators": ".generators",
    "embedders": ".embedders",
    "predictors": ".predictors",
    "onnx": ".onnx",
    "optimizers": ".optimizers",
    "speaker": ".speaker",
}


def __getattr__(name: str):
    """Lazy import mechanism for models modules."""
    if name in _LAZY_MODULES:
        import importlib

        module = importlib.import_module(_LAZY_MODULES[name], __package__)
        return module
    raise AttributeError(f"Module '{__name__}' has no attribute '{name}'")

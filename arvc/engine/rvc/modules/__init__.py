"""
Pluggable neural modules for the RVC engine.

This package contains the swappable building blocks used by inference and
training:

- ``embedders`` — content encoders (HuBERT/fairseq, ONNX, PPG, transformers)
- ``predictors`` — F0/pitch predictors (RMVPE, FCPE, CREPE, PEStO, ...)

They are kept separate from ``arvc.engine.rvc.lib`` (the core model
library) so each family can be extended independently.
"""

__all__ = [
    "embedders",
    "predictors",
]

# Lazy imports for module families
_LAZY_MODULES = {
    "embedders": ".embedders",
    "predictors": ".predictors",
}


def __getattr__(name: str):
    """Lazy import mechanism for module families."""
    if name in _LAZY_MODULES:
        import importlib

        module = importlib.import_module(_LAZY_MODULES[name], __package__)
        return module
    raise AttributeError(f"Module '{__name__}' has no attribute '{name}'")

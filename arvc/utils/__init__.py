"""
Utility tools for Advanced RVC Inference.

This package contains lightweight, dependency-poor helpers that are safe to
import in any mode (CLI, GUI, Colab-no-UI, headless). Heavy or domain-specific
utilities live in dedicated subpackages:

- ``arvc.utils.downloaders`` — file-host downloaders (gdown, huggingface,
  mediafire, meganz, pixeldrain). Each downloader is a thin wrapper around
  a specific file-hosting service.
- ``arvc.utils.feedback`` — logging-only feedback functions (``gr_info``,
  ``gr_warning``, ``gr_error``) usable in headless mode. The GUI-aware
  versions live in ``arvc.ui.feedback``.
- ``arvc.utils.variables`` — global paths, the singleton ``Config`` instance,
  the package logger, and translation strings.

Audio processing helpers (e.g. ``noisereduce``) live under
``arvc.engine.inference`` because they are tightly coupled to the inference
pipeline.
"""

__all__ = [
    "downloaders",
    "feedback",
    "variables",
    "strtobool",
]


def strtobool(val):
    """Convert a string representation of truth to boolean.

    Replacement for the deprecated ``distutils.util.strtobool`` which was
    removed in Python 3.12+.

    Accepts: ``y/yes/true/t/1/on`` (case-insensitive) → ``True``
    Accepts: ``n/no/false/f/0/off`` (case-insensitive) → ``False``
    Anything else raises ``ValueError``.
    """
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    val_str = str(val).strip().lower()
    if val_str in ("y", "yes", "true", "t", "1", "on"):
        return True
    elif val_str in ("n", "no", "false", "f", "0", "off"):
        return False
    else:
        raise ValueError(f"Invalid truth value: {val!r}")


# Submodules are imported lazily on first access to keep import time low
# (some downloaders pull in heavy third-party deps like ``gdown``).
_LAZY_MODULES = {
    "downloaders": ".downloaders",
    "feedback": ".feedback",
    "variables": ".variables",
}


def __getattr__(name: str):
    """Lazy import mechanism for submodules."""
    if name in _LAZY_MODULES:
        import importlib

        module = importlib.import_module(_LAZY_MODULES[name], __package__)
        return module
    raise AttributeError(f"Module '{__name__}' has no attribute '{name}'")

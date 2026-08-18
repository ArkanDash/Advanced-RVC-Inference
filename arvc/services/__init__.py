"""
Services package for Advanced RVC Inference.

This package is the service layer that sits between the UI/CLI (``arvc.app``,
``arvc.api``) and the engine (``arvc.engine``). Each subpackage groups
related service modules by domain:

- ``arvc.services.inference``  — inference-related services
  (csrt, f0_extract, presets, separate, tts)
- ``arvc.services.training``   — training orchestration
  (training)
- ``arvc.services.realtime``   — realtime voice-conversion services
  (realtime, realtime_client)
- ``arvc.services.system``     — system/process/model management services
  (model_utils, process, restart, utils)
- ``arvc.services.downloads``  — download orchestration
  (downloads)

All submodules are imported lazily on first access to keep startup time low
and to allow headless/CLI mode to import only the bits it needs.
"""

__all__ = [
    "inference",
    "training",
    "realtime",
    "system",
    "downloads",
]


# Submodules are imported lazily on first access so that, e.g., importing
# ``arvc.services`` in CLI mode doesn't pull in ``realtime`` (which has
# heavy audio-device dependencies).
_LAZY_MODULES = {
    "inference": ".inference",
    "training": ".training",
    "realtime": ".realtime",
    "system": ".system",
    "downloads": ".downloads",
}


def __getattr__(name: str):
    """Lazy import mechanism for service subpackages."""
    if name in _LAZY_MODULES:
        import importlib

        return importlib.import_module(_LAZY_MODULES[name], __package__)
    raise AttributeError(f"Module '{__name__}' has no attribute '{name}'")

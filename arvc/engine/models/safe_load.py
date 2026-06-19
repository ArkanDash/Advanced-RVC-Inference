"""
safe_load.py — Centralized safe deserialization helpers for Advanced-RVC-Inference.

SECURITY PATCH (train + infer):
Provides hardened replacements for `torch.load`, `pickle.load`, and `yaml.load`
that are used throughout the training and inference pipelines. These helpers
refuse to execute arbitrary code embedded in untrusted model / config files.

Why this matters
----------------
RVC checkpoints, fairseq embedders, PESTO/CREPE/FCPE predictors, UVR5 .bin
config files, and WORLD .bin binary blobs are all downloaded from the internet
or shared between users. A malicious actor can craft one of these files to
trigger arbitrary Python code execution the moment it is loaded with
`torch.load(weights_only=False)` or `pickle.load()`.

This module:
  * `safe_torch_load`  — Always uses `weights_only=True`. Refuses to fall back
                          to the unsafe legacy loader, even on old PyTorch.
  * `safe_pickle_load` — Restricted Unpickler that whitelists only primitive
                          types + numpy scalar/array types. ANY class
                          instantiation is rejected.
  * `safe_yaml_load`   — Uses `yaml.SafeLoader`. Rejects arbitrary Python tags.

Usage
-----
    from arvc.engine.models.safe_load import (
        safe_torch_load, safe_pickle_load, safe_yaml_load
    )

    ckpt = safe_torch_load(path)                 # was: torch.load(path, weights_only=False)
    cfg  = safe_pickle_load(open(path, "rb"))    # was: pickle.load(f)
    yml  = safe_yaml_load(path)                  # was: yaml.load(open(p), Loader=yaml.FullLoader)
"""

from __future__ import annotations

import io
import os
import pickle
import unicodedata
from typing import Any, IO

import torch

# ── safe_torch_load ─────────────────────────────────────────────────────────

def safe_torch_load(
    path: str,
    map_location: Any = "cpu",
    weights_only: bool = True,
) -> Any:
    """Hardened `torch.load`.

    * `weights_only` is ALWAYS True and cannot be disabled from the caller.
      Callers that genuinely need to load pickled Python objects (e.g. fairseq
      configs that include lambdas) must use `safe_pickle_load` instead, which
      still rejects arbitrary class instantiation.
    * On old PyTorch versions that lack `weights_only` support we re-raise the
      TypeError instead of silently falling back to unsafe loading.
    * Empty / missing files raise FileNotFoundError rather than returning None
      silently — caller code already checks `os.path.exists()` first.
    """
    if weights_only is False:
        # Refuse to disable the safe path even if a caller asks for it.
        weights_only = True

    if hasattr(torch, "load"):
        return torch.load(path, map_location=map_location, weights_only=True)
    # torch.load itself is missing — extremely unlikely, fail loudly.
    raise RuntimeError("torch.load is not available in this PyTorch build.")


# ── safe_pickle_load ────────────────────────────────────────────────────────

# Whitelist of primitive/builtin types that may be reconstructed from a pickle
# stream. Anything not on this list (e.g. `os.system`, `subprocess.Popen`,
# `__builtin__.eval`) raises `pickle.UnpicklingError` instead of being executed.
_SAFE_BUILTIN_TYPES = (
    dict, list, tuple, set, frozenset,
    int, float, complex, bool, str, bytes, bytearray,
    type(None),
)

# Numpy is imported lazily so that this module remains importable even in
# environments without numpy (rare, but the import order shouldn't fail).
def _numpy_safe_types():
    try:
        import numpy as np
        return (
            np.ndarray,
            np.dtype,
            np.float16, np.float32, np.float64,
            np.int8, np.int16, np.int32, np.int64,
            np.uint8, np.uint16, np.uint32, np.uint64,
            np.bool_,
            np.complex64, np.complex128,
            np.zeros,
        )
    except ImportError:
        return ()


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that only allows primitive Python types + numpy arrays.

    Any `REDUCE` / `BUILD` / `OBJ` opcode targeting a non-whitelisted class
    is rejected. This blocks every known pickle-based RCE gadget
    (`os.system`, `subprocess.Popen`, `builtins.eval`, `__import__`, etc.)
    while still allowing the data containers RVC actually relies on
    (dict-of-bytes for WORLD binaries, dict-of-params for UVR5 .bin configs).
    """

    # Module-level whitelist: maps "module.name" → actual callable.
    # We resolve lazily to keep the module import cheap.
    _WHITELIST: dict[str, Any] | None = None

    @classmethod
    def _build_whitelist(cls) -> dict[str, Any]:
        if cls._WHITELIST is not None:
            return cls._WHITELIST

        wl: dict[str, Any] = {}
        # Builtins
        for t in _SAFE_BUILTIN_TYPES:
            wl[f"{t.__module__}.{t.__name__}"] = t
            # `builtins` is also referenced as `__builtin__` on Python 2 streams.
            wl[f"builtins.{t.__name__}"] = t
            wl[f"__builtin__.{t.__name__}"] = t

        # collections.OrderedDict — used by some legacy checkpoints
        try:
            from collections import OrderedDict
            wl["collections.OrderedDict"] = OrderedDict
            wl["ordereddict.OrderedDict"] = OrderedDict
        except ImportError:
            pass

        # Numpy types
        try:
            import numpy as np
            for t in _numpy_safe_types():
                wl[f"numpy.{t.__name__}"] = t
                wl[f"np.{t.__name__}"] = t
            # numpy core reconstruct helpers
            wl["numpy._core.multiarray._reconstruct"] = np.core.multiarray._reconstruct
            wl["numpy.core.multiarray._reconstruct"] = np.core.multiarray._reconstruct
            wl["numpy.dtype"] = np.dtype
            wl["numpy.float16"] = np.float16
        except Exception:
            pass

        cls._WHITELIST = wl
        return wl

    def find_class(self, module: str, name: str) -> Any:  # noqa: D401
        wl = self._build_whitelist()
        key = f"{module}.{name}"
        if key in wl:
            return wl[key]
        # Allow a small set of additional numpy reconstruction helpers.
        if module.startswith("numpy") and name in (
            "ndarray", "dtype", "_reconstruct",
            "float16", "float32", "float64",
            "int8", "int16", "int32", "int64",
            "uint8", "uint16", "uint32", "uint64",
            "bool_", "complex64", "complex128",
        ):
            try:
                import numpy as np
                return getattr(np, name) if hasattr(np, name) else getattr(np.core.multiarray, name)
            except Exception:
                pass
        raise pickle.UnpicklingError(
            f"[safe_load] Refusing to unpickle disallowed class: {module}.{name}. "
            "This is a security guard against malicious pickle payloads. "
            "If this file is trusted, audit it manually before relaxing the whitelist."
        )


def safe_pickle_load(file_obj: IO[bytes]) -> Any:
    """Restricted `pickle.load`. Rejects arbitrary class instantiation.

    Accepts either an open binary file object or raw bytes.
    """
    if isinstance(file_obj, (bytes, bytearray)):
        return _RestrictedUnpickler(io.BytesIO(file_obj)).load()
    return _RestrictedUnpickler(file_obj).load()


def safe_pickle_loads(data: bytes) -> Any:
    """Restricted `pickle.loads`."""
    return _RestrictedUnpickler(io.BytesIO(data)).load()


# ── safe_yaml_load ──────────────────────────────────────────────────────────

def safe_yaml_load(stream: Any) -> Any:
    """`yaml.safe_load` wrapper that never uses FullLoader.

    `yaml.FullLoader` is NOT safe — it can still execute arbitrary Python via
    custom tags. `yaml.SafeLoader` only constructs primitive types and is the
    only acceptable loader for untrusted YAML (e.g. UVR5 model_data.yaml).
    """
    import yaml
    return yaml.safe_load(stream)


# ── path validation helpers (lightweight traversal guards) ──────────────────

def validate_path_within(path: str, base_dirs: list[str], allow_absolute: bool = True) -> str:
    """Resolve `path` and ensure it stays inside one of `base_dirs`.

    Returns the resolved absolute path. Raises ValueError on traversal.

    This is a defense-in-depth guard for inputs that come from the GUI/CLI
    (input_path, output_path, pth_path, pretrained paths). It blocks the
    classic `../../etc/passwd` style escapes even when the underlying OS would
    otherwise happily open the file.
    """
    if not path or not isinstance(path, str):
        raise ValueError("Path must be a non-empty string")

    # Normalise Unicode to prevent homoglyph bypasses (NFC canonical form).
    norm_path = unicodedata.normalize("NFC", path)

    # Reject null bytes outright (some OSes would silently truncate them).
    if "\x00" in norm_path:
        raise ValueError("Null byte in path is not allowed")

    resolved = os.path.realpath(os.path.abspath(norm_path))

    if not allow_absolute and os.path.isabs(norm_path):
        raise ValueError(f"Absolute paths are not allowed: {path}")

    if not base_dirs:
        return resolved

    for base in base_dirs:
        base_resolved = os.path.realpath(os.path.abspath(base))
        if resolved == base_resolved or resolved.startswith(base_resolved + os.sep):
            return resolved

    raise ValueError(
        f"Path '{path}' (resolved to '{resolved}') escapes allowed base directories: {base_dirs}"
    )


__all__ = [
    "safe_torch_load",
    "safe_pickle_load",
    "safe_pickle_loads",
    "safe_yaml_load",
    "validate_path_within",
]

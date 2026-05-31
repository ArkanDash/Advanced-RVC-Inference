"""
Conditional weight_norm import for RVC fork compatibility.

By default, uses the old-style ``torch.nn.utils.weight_norm`` which stores
weights as ``.weight_g`` / ``.weight_v`` — the format used by virtually every
RVC fork.  This ensures trained models are interchangeable across projects.

Set the config option ``new_pytorch_weight_norm: true`` to switch to
``torch.nn.utils.parametrizations.weight_norm`` (PyTorch 2.0+), which stores
weights as ``.parametrizations.weight.original0`` / ``original1``.
"""

import logging

logger = logging.getLogger(__name__)

# ── Global flag ──────────────────────────────────────────────────────────
# False = old-style weight_norm (default, compatible with most RVC forks)
# True  = new PyTorch 2.0+ parametrizations.weight_norm
_new_pytorch_mode = False


def configure_weight_norm(new_pytorch: bool = False) -> None:
    """Set the weight_norm mode.  Call once at startup before model creation."""
    global _new_pytorch_mode
    _new_pytorch_mode = bool(new_pytorch)
    mode = "new PyTorch 2.0+ parametrizations" if _new_pytorch_mode else "old-style (RVC fork compatible)"
    logger.info(f"weight_norm mode: {mode}")


def use_new_pytorch() -> bool:
    """Return whether new PyTorch weight_norm mode is active."""
    return _new_pytorch_mode


# ── Public API ───────────────────────────────────────────────────────────

def weight_norm(module, name: str = "weight", dim: int = 0):
    """Apply weight normalization to a module.

    Dispatches to the old or new PyTorch implementation depending on the
    current mode (see :func:`configure_weight_norm`).

    Args:
        module: The module to apply weight normalization to.
        name: Name of the weight parameter.
        dim: Dimension along which to apply weight normalization.
    """
    if _new_pytorch_mode:
        from torch.nn.utils.parametrizations import weight_norm as _wn
    else:
        from torch.nn.utils.weight_norm import weight_norm as _wn
    return _wn(module, name=name, dim=dim)


def remove_weight_norm(module, name: str = "weight"):
    """Remove weight normalization from a module.

    Works with both old-style and new parametrizations weight_norm.
    """
    if _new_pytorch_mode:
        # PyTorch 2.0+: remove_weight_norm handles both styles
        try:
            from torch.nn.utils.parametrizations import remove_weight_norm as _rwn
            return _rwn(module, name=name)
        except (ImportError, RuntimeError):
            from torch.nn.utils import remove_weight_norm as _rwn
            return _rwn(module, name=name)
    else:
        from torch.nn.utils import remove_weight_norm as _rwn
        return _rwn(module, name=name)


def weight_norm_v(module):
    """Return the *v* (direction) parameter of a weight-normed layer.

    Old-style: ``module.weight_v``
    New-style: ``module.parametrizations.weight.original1``
    """
    if _new_pytorch_mode:
        return getattr(getattr(module, "parametrizations", None), "weight", None).original1 if hasattr(module, "parametrizations") else getattr(module, "weight_v", None)
    return getattr(module, "weight_v", None)


def weight_norm_g(module):
    """Return the *g* (magnitude) parameter of a weight-normed layer.

    Old-style: ``module.weight_g``
    New-style: ``module.parametrizations.weight.original0``
    """
    if _new_pytorch_mode:
        return getattr(getattr(module, "parametrizations", None), "weight", None).original0 if hasattr(module, "parametrizations") else getattr(module, "weight_g", None)
    return getattr(module, "weight_g", None)


# ── Key conversion helpers ───────────────────────────────────────────────
# These are only needed when new_pytorch mode is active.

def needs_key_conversion() -> bool:
    """Return True if checkpoint key conversion between old/new format is needed.

    Conversion is needed only when the model uses new PyTorch parametrizations
    but checkpoints are stored in old ``.weight_g``/``.weight_v`` format.
    """
    return _new_pytorch_mode


def convert_old_to_new(state_dict):
    """Convert old-style weight_norm keys to new parametrizations keys.

    Only applies when ``new_pytorch_weight_norm`` is enabled.
    No-op otherwise (model and checkpoint both use old format).
    """
    if not _new_pytorch_mode:
        return state_dict

    from collections import OrderedDict

    def _replace(d, old_part, new_part):
        updated = OrderedDict() if isinstance(d, OrderedDict) else {}
        for key, value in d.items():
            updated[key.replace(old_part, new_part) if isinstance(key, str) else key] = (
                _replace(value, old_part, new_part) if isinstance(value, dict) else value
            )
        return updated

    result = _replace(state_dict, ".weight_v", ".parametrizations.weight.original1")
    result = _replace(result, ".weight_g", ".parametrizations.weight.original0")
    return result


def convert_new_to_old(state_dict):
    """Convert new parametrizations keys to old-style weight_norm keys.

    Only applies when ``new_pytorch_weight_norm`` is enabled.
    No-op otherwise (model and checkpoint both use old format).
    """
    if not _new_pytorch_mode:
        return state_dict

    from collections import OrderedDict

    def _replace(d, old_part, new_part):
        updated = OrderedDict() if isinstance(d, OrderedDict) else {}
        for key, value in d.items():
            updated[key.replace(old_part, new_part) if isinstance(key, str) else key] = (
                _replace(value, old_part, new_part) if isinstance(value, dict) else value
            )
        return updated

    result = _replace(state_dict, ".parametrizations.weight.original1", ".weight_v")
    result = _replace(result, ".parametrizations.weight.original0", ".weight_g")
    return result

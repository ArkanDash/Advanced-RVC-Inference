"""
Conditional weight_norm import for RVC fork compatibility.

Uses ``torch.nn.utils.parametrizations.weight_norm`` (PyTorch 2.0+) by default,
which stores weights as ``.parametrizations.weight.original0`` / ``original1``.
This matches the approach used by Applio and Vietnamese-RVC.

All checkpoints saved to disk use the old ``.weight_g`` / ``.weight_v`` format
for maximum backward compatibility across RVC forks.  Key conversion happens
automatically at save/load boundaries.
"""

import logging

logger = logging.getLogger(__name__)

# ── Global flag ──────────────────────────────────────────────────────────
# True  = new PyTorch 2.0+ parametrizations.weight_norm (default, matches Applio/VRVC)
# False = old-style weight_norm (legacy compatibility)
_new_pytorch_mode = True


def configure_weight_norm(new_pytorch: bool = True) -> None:
    """Set the weight_norm mode.  Call once at startup before model creation."""
    global _new_pytorch_mode
    _new_pytorch_mode = bool(new_pytorch)
    mode = "new PyTorch 2.0+ parametrizations" if _new_pytorch_mode else "old-style (legacy)"
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
# Always active regardless of mode.  This ensures seamless interoperability
# between old-format checkpoints and new-format models (and vice versa).

def _replace_keys_in_dict(d, old_part, new_part):
    """Recursively replace key substrings in a (possibly nested) dict."""
    from collections import OrderedDict
    updated = OrderedDict() if isinstance(d, OrderedDict) else {}
    for key, value in d.items():
        updated[(
            key.replace(old_part, new_part) if isinstance(key, str) else key
        )] = (
            _replace_keys_in_dict(value, old_part, new_part) if isinstance(value, dict) else value
        )
    return updated


def convert_old_to_new(state_dict):
    """Convert old-style weight_norm keys to new parametrizations keys.

    Converts ``.weight_v`` → ``.parametrizations.weight.original1``
    and       ``.weight_g`` → ``.parametrizations.weight.original0``

    Always active — ensures old-format checkpoints load correctly into
    models that use the new PyTorch parametrization format.
    """
    result = _replace_keys_in_dict(state_dict, ".weight_v", ".parametrizations.weight.original1")
    result = _replace_keys_in_dict(result, ".weight_g", ".parametrizations.weight.original0")
    return result


def convert_new_to_old(state_dict):
    """Convert new parametrizations keys to old-style weight_norm keys.

    Converts ``.parametrizations.weight.original1`` → ``.weight_v``
    and       ``.parametrizations.weight.original0`` → ``.weight_g``

    Always active — ensures saved checkpoints use the old format for
    maximum backward compatibility with other RVC forks.
    """
    result = _replace_keys_in_dict(state_dict, ".parametrizations.weight.original1", ".weight_v")
    result = _replace_keys_in_dict(result, ".parametrizations.weight.original0", ".weight_g")
    return result

"""Theme loader for ARVC.

This is a tiny re-implementation of Applio's ``assets/themes/loadThemes.py``
that supports the same external theme names (e.g. ``NeoPy/Soft``,
``freddyaboulton/dracula_revamped``) plus the built-in ``Applio`` theme
shipped under ``arvc/assets/themes/Applio.py``.

External theme names are resolved via ``gr.Theme.from_name`` (Gradio 4+)
or the older ``gr.themes.``` registry. Built-in themes are matched against
the modules registered in this file's ``BUILTIN_THEMES`` mapping.
"""

from __future__ import annotations

import importlib
from typing import Optional

import gradio as gr

# Map of "local" theme names → "arvc.assets.themes.<module>" that exposes the
# class. Each module must export a class with the same capitalized name.
BUILTIN_THEMES = {
    "Applio": "arvc.assets.themes.Applio",
}


def list_builtin_themes() -> list[str]:
    return sorted(BUILTIN_THEMES.keys())


def load_builtin_theme(name: str):
    """Return a Gradio theme instance for a built-in name, or ``None``.

    Returns ``None`` if *name* is not a built-in ARVC theme (so the caller
    can fall back to ``gr.Theme.from_name`` for external names like
    ``NeoPy/Soft``).
    """
    if name not in BUILTIN_THEMES:
        return None
    module_path = BUILTIN_THEMES[name]
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        return None
    cls = getattr(module, name, None)
    if cls is None:
        return None
    try:
        return cls()
    except Exception:  # noqa: BLE001
        return None


def load_theme(name: Optional[str]):
    """Resolve *name* into a Gradio theme.

    Resolution order:
      1. Built-in ARVC themes (``Applio``).
      2. External themes registered with ``gr.themes`` (e.g. ``NeoPy/Soft``).
      3. ``None`` if both fail (caller should fall back to default).
    """
    if not name:
        return None
    builtin = load_builtin_theme(name)
    if builtin is not None:
        return builtin
    try:
        return gr.Theme.from_name(name)
    except Exception:  # noqa: BLE001
        return None


__all__ = [
    "load_theme",
    "load_builtin_theme",
    "list_builtin_themes",
    "BUILTIN_THEMES",
]

"""Settings tab for the RMVPE high-register correction feature.

Ported from Applio (IAHispano/Applio) ``tabs/settings/sections/rmvpe_high_register.py``,
adapted to ARVC's existing ``arvc.utils.variables.configs`` config store and
``arvc.utils.variables.translations`` i18n dict.

The actual correction logic lives next to the RMVPE predictor — see
``arvc/engine/rvc/models/predictors/high_register.py``. This module only
handles the UI + persistence of the three user-facing knobs (``enabled``,
``mode``, ``f0_ceil``).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import gradio as gr

from arvc.utils.variables import (
    configs,
    translations,
    logger,
    CONFIGS_PATH,
)

# Keep in sync with HIGH_REGISTER_DEFAULTS in
# arvc/engine/rvc/models/predictors/high_register.py
DEFAULTS = {"enabled": False, "mode": "true_pitch", "f0_ceil": 1250}
MODES = ["true_pitch", "fold"]
F0_CEIL_RANGE = (1000, 2000)


def _config_file() -> Path:
    """Path to the ARVC config.json this module reads/writes."""
    # CONFIGS_PATH is a Path from arvc.utils.variables; honour it.
    if isinstance(CONFIGS_PATH, Path):
        return CONFIGS_PATH / "config.json"
    return Path(str(CONFIGS_PATH)) / "config.json"


def load_config_high_register() -> dict:
    """Read the ``rmvpe_high_register`` section from ARVC's config.json.

    Always returns a fully-populated dict (defaults fill missing keys); a
    corrupt or absent section disables the corrector, so RMVPE behaves
    exactly as before until the user opts in.
    """
    settings = dict(DEFAULTS)
    try:
        section = configs.get("rmvpe_high_register")
        if isinstance(section, dict):
            settings.update(section)
    except Exception:  # noqa: BLE001 — defensive: configs may be a mock
        pass

    # Sanitize hand-edited values so the UI always gets something valid.
    if settings["mode"] not in MODES:
        settings["mode"] = DEFAULTS["mode"]
    try:
        settings["f0_ceil"] = float(settings["f0_ceil"])
        settings["f0_ceil"] = min(
            max(settings["f0_ceil"], F0_CEIL_RANGE[0]), F0_CEIL_RANGE[1]
        )
    except (TypeError, ValueError):
        settings["f0_ceil"] = DEFAULTS["f0_ceil"]
    settings["enabled"] = bool(settings["enabled"])
    return settings


def save_config_high_register(enabled, mode, f0_ceil):
    """Persist the high-register settings into ARVC's config.json.

    Writes through the in-memory ``configs`` dict and re-dumps the file so
    other modules (and the next process launch) see the change immediately.
    """
    try:
        configs["rmvpe_high_register"] = {
            "enabled": bool(enabled),
            "mode": mode if mode in MODES else DEFAULTS["mode"],
            "f0_ceil": float(f0_ceil),
        }
        path = _config_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf8") as f:
            json.dump(configs, f, indent=4)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to persist rmvpe_high_register settings: {exc}")


def rmvpe_high_register_tab():
    """Build the Gradio Accordion shown under the Extra → Settings tab."""
    settings = load_config_high_register()
    with gr.Accordion(
        translations.get(
            "rmvpe_high_register",
            "RMVPE high-register correction",
        ),
        open=False,
    ):
        enabled = gr.Checkbox(
            label=translations.get(
                "rmvpe_high_register_enable",
                "Enable RMVPE high-register correction",
            ),
            info=translations.get(
                "rmvpe_high_register_enable_info",
                "Fixes the octave errors rmvpe.pt makes above ~1040 Hz (flutes, "
                "high strings, whistle register) with a second pass on "
                "octave-down-resampled audio. Off by default; when disabled RMVPE "
                "behaves exactly as before.",
            ),
            value=settings["enabled"],
            interactive=True,
        )
        mode = gr.Dropdown(
            filterable=False,
            label=translations.get(
                "rmvpe_high_register_mode",
                "Correction mode",
            ),
            info=translations.get(
                "rmvpe_high_register_mode_info",
                "true_pitch writes the real pitch, capped at the F0 ceiling. "
                "fold only repairs wrong-octave frames using half-pitch drives; "
                "use it with models trained on uncorrected labels to keep their "
                "timbre.",
            ),
            choices=MODES,
            value=settings["mode"],
            interactive=True,
        )
        f0_ceil = gr.Slider(
            label=translations.get(
                "rmvpe_high_register_f0_ceil",
                "F0 ceiling (Hz)",
            ),
            info=translations.get(
                "rmvpe_high_register_f0_ceil_info",
                "Highest pitch written in true_pitch mode. Models trained on "
                "stock RMVPE labels cannot synthesize above ~1250 Hz; raise it "
                "only for models trained with the correction enabled.",
            ),
            minimum=F0_CEIL_RANGE[0],
            maximum=F0_CEIL_RANGE[1],
            step=10,
            value=settings["f0_ceil"],
            interactive=True,
        )
        for component in (enabled, mode, f0_ceil):
            component.change(
                fn=save_config_high_register,
                inputs=[enabled, mode, f0_ceil],
                outputs=[],
            )

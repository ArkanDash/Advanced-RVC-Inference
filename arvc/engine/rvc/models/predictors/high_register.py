"""RMVPE high-register corrector — ported from Applio's ``rvc/lib/predictors/f0.py``.

The stock ``rmvpe.pt`` model was trained on a vocal pitch corpus that tops
out near C6 (~1040 Hz). Above that, the salience net confidently reports
f/2 or f/3 instead of the true pitch. The corrector runs a *second* RMVPE
pass on audio resampled to half speed (i.e. an octave down) and uses the
result only to repair the octave class of the normal pass — the normal
10 ms contour is preserved wherever it already agrees with the guide.

Two modes are supported (selected from the Extra → Settings tab):

``true_pitch`` (default)
    Writes the true pitch, capped at ``f0_ceil`` (default 1250 Hz). RVC
    models trained on stock RMVPE labels cannot synthesize above that
    ceiling; above it the stock octave-down drive is kept so the vocoder
    folds its harmonics back up.

``fold``
    Compatibility mode for models trained on uncorrected labels: only
    wrong-pitch-class frames are fixed, using *half* the true pitch so
    every drive stays in the model's trained register.

Settings are read from ``arvc/utils/variables.configs`` (i.e. from
``arvc/configs/config.json``) under the ``rmvpe_high_register`` key.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np

# Keep in sync with DEFAULTS in arvc/app/tabs/extra/child/rmvpe_high_register.py
HIGH_REGISTER_DEFAULTS = {"enabled": False, "mode": "true_pitch", "f0_ceil": 1250.0}


def load_high_register_settings() -> dict:
    """Read the high-register settings from ARVC's config.json.

    Returns a fully-populated dict (defaults fill missing keys); a corrupt
    or absent section disables the corrector so RMVPE keeps stock behaviour
    outside the UI.
    """
    settings = dict(HIGH_REGISTER_DEFAULTS)
    try:
        from arvc.utils.variables import configs

        section = configs.get("rmvpe_high_register")
        if isinstance(section, dict):
            settings.update(section)
    except (ImportError, RuntimeError, ValueError):
        pass

    # Sanitize — a hand-edited config might have wrong types.
    try:
        settings["f0_ceil"] = float(settings["f0_ceil"])
    except (TypeError, ValueError, KeyError):
        settings["f0_ceil"] = HIGH_REGISTER_DEFAULTS["f0_ceil"]
    if settings.get("mode") not in ("true_pitch", "fold"):
        settings["mode"] = HIGH_REGISTER_DEFAULTS["mode"]
    settings["enabled"] = bool(settings.get("enabled", False))
    return settings


def _half_speed_guide(model, x, n_target, sample_rate, hop_size, thred):
    """Assemble a bounded octave-down guide on the normal pitch frame clock.

    Resamples the audio to 2× the original sample rate (so a 10 ms hop
    becomes a 20 ms hop at the original rate, i.e. an octave down), runs the
    same RMVPE predictor on it, and interpolates the resulting f0 contour
    back onto the original frame clock — multiplying by 2 to undo the
    octave shift. The guide is bounded to ``n_target`` frames so callers
    can rely on its length matching the normal pass.
    """
    import librosa

    n_target = int(n_target)
    y = np.asarray(x, dtype=np.float32)
    # Half-speed resample: target_sr = 2 * sample_rate ⇒ same audio duration
    # but twice as many samples ⇒ original f0 mapped to f0/2.
    y2 = librosa.resample(
        y,
        orig_sr=sample_rate,
        target_sr=2 * sample_rate,
        res_type="soxr_vhq",
    )
    f0h = model.infer_from_audio(y2, thred=thred)
    pos = np.arange(n_target) * 2.0
    j0 = np.minimum(np.floor(pos).astype(int), len(f0h) - 1)
    j1 = np.minimum(j0 + 1, len(f0h) - 1)
    w = pos - np.floor(pos)
    v0, v1 = f0h[j0] > 0, f0h[j1] > 0
    voiced = np.where(w == 0, v0, v0 & v1)
    lg0 = np.log2(np.where(f0h[j0] > 0, f0h[j0], 1.0))
    lg1 = np.log2(np.where(f0h[j1] > 0, f0h[j1], 1.0))
    lg = (1 - w) * lg0 + w * lg1
    # ×2 to undo the half-speed resample; +1 in log space becomes ×2 in Hz.
    return np.where(voiced, 2.0 ** (lg + 1.0), 0.0)


def fix_high_register(
    model,
    x: np.ndarray,
    normal: np.ndarray,
    thred: float,
    *,
    settings: Optional[dict] = None,
    sample_rate: int = 16000,
    hop_size: int = 160,
    gate: float = 150.0,
) -> np.ndarray:
    """Apply the high-register corrector to ``normal``.

    Parameters
    ----------
    model
        Anything exposing ``infer_from_audio(audio, thred=...)`` — typically
        an ``RMVPE0Predictor`` instance.
    x
        Input audio samples (mono, float32).
    normal
        The f0 contour produced by the *normal* RMVPE pass (one value per
        hop-sized frame).
    thred
        Filter radius passed to ``model.infer_from_audio``.
    settings
        Optional dict overriding the on-disk config (``enabled``, ``mode``,
        ``f0_ceil``). Realtime callers pass ``{"enabled": False}`` to skip
        the second pass; the extraction CLI passes the pinned mode.
    sample_rate, hop_size
        Audio sample rate and hop size used by ``model``. Defaults match
        RMVPE's standard 16 kHz / 160 samples.
    gate
        Max tolerated cent deviation between normal and guide before the
        frame is treated as "wrong octave". 150 cents ≈ a semitone and a
        half — generous enough to absorb vibrato.
    """
    settings = {**HIGH_REGISTER_DEFAULTS, **(settings or {})}
    if not settings.get("enabled"):
        return normal

    ceil = float(settings.get("f0_ceil", HIGH_REGISTER_DEFAULTS["f0_ceil"]))
    mode = settings.get("mode", "true_pitch")
    guide = _half_speed_guide(model, x, len(normal), sample_rate, hop_size, thred)

    out = normal.copy()
    nv, gv = normal > 0, guide > 0
    both = nv & gv
    c_keep = np.full(len(normal), 1e9)
    c_double = np.full(len(normal), 1e9)
    c_keep[both] = np.abs(1200.0 * np.log2(normal[both] / guide[both]))
    c_double[both] = np.abs(1200.0 * np.log2(2.0 * normal[both] / guide[both]))
    agree = both & (c_keep < gate)
    cand_dbl = both & ~agree & (c_double < gate)

    if mode == "fold":
        fold_other = (both & ~agree & ~cand_dbl) & (guide >= 990.0)
        fold_fill = (~nv) & gv & (guide >= 900.0)
        out[fold_other] = guide[fold_other] / 2.0
        out[fold_fill] = guide[fold_fill] / 2.0
        return out

    # true_pitch
    dbl = cand_dbl & (guide >= 460.0) & (2.0 * normal <= ceil)
    other = (both & ~agree & ~cand_dbl) & (guide >= 990.0) & (guide <= ceil)
    filled = (~nv) & gv & (guide >= 460.0) & (guide <= ceil)
    out[dbl] = 2.0 * normal[dbl]
    out[other] = guide[other]
    out[filled] = guide[filled]
    return out

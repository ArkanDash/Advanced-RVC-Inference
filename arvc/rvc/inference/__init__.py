"""
Inference-related services for Advanced RVC Inference.

This subpackage contains service-layer modules that orchestrate end-to-end
inference tasks (audio conversion, source separation, TTS, F0 extraction,
SRT generation, preset management).

Modules:
- ``csrt``       — SRT subtitle generation from audio
- ``f0_extract`` — F0 (pitch) extraction as a standalone service
- ``presets``    — load/save inference effect presets
- ``separate``   — music/vocal separation orchestration
- ``tts``        — text-to-speech synthesis (Edge TTS, Google Translate TTS)

For backward compatibility, every public symbol from each module is also
re-exported at the subpackage level, so both of these work:

    from arvc.rvc.inference.csrt import create_srt
    from arvc.rvc.inference import create_srt
"""

from . import csrt, f0_extract, presets, separate, tts
from .csrt import *          # noqa: F401, F403
from .f0_extract import *    # noqa: F401, F403
from .presets import *       # noqa: F401, F403
from .separate import *      # noqa: F401, F403
from .tts import *           # noqa: F401, F403

__all__ = ["csrt", "f0_extract", "presets", "separate", "tts"]

"""
Asset files for Advanced RVC Inference.

This package contains resource files including:
- Audio samples
- Binary files
- Dataset templates
- F0 models
- Language packs
- Logs
- Model weights
- Presets
- UVR5 models
"""

__all__ = [
    "audios",
    "binary",
    "dataset",
    "f0",
    "languages",
    "logs",
    "models",
    "presets",
    "weights",
    "zluda",
]

import os
from pathlib import Path

# Asset directory path
ASSETS_PATH = Path(__file__).parent


def get_asset_path(asset_name: str) -> Path:
    """Get the path to an asset directory.

    Args:
        asset_name: Name of the asset directory

    Returns:
        Path to the asset directory
    """
    return ASSETS_PATH / asset_name


def get_audio_path() -> Path:
    """Get the path to audio files."""
    return get_asset_path("audios")


def get_weights_path() -> Path:
    """Get the path to model weights."""
    return get_asset_path("weights")


def get_logs_path() -> Path:
    """Get the path to log files."""
    return get_asset_path("logs")


def get_presets_path() -> Path:
    """Get the path to preset files."""
    return get_asset_path("presets")


def get_languages_path() -> Path:
    """Get the path to language files."""
    return get_asset_path("languages")

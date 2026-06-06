"""
Asset files for Advanced RVC Inference.

This package contains resource files including:
- Audio samples
- Binary files
- F0 models
- Language packs
- Logs (training data + trained model output)
- Model weights (embedders, predictors, pretrained, UVR5)
- Presets
- ZLUDA scripts

Datasets are stored separately at arvc/datasets/.
Training logs and trained model .pth files are both in arvc/assets/logs/.
"""

__all__ = [
    "audios",
    "binary",
    "f0",
    "languages",
    "logs",
    "models",
    "presets",
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


def get_logs_path() -> Path:
    """Get the path to log files and trained models."""
    return get_asset_path("logs")


def get_presets_path() -> Path:
    """Get the path to preset files."""
    return get_asset_path("presets")


def get_languages_path() -> Path:
    """Get the path to language files."""
    return get_asset_path("languages")

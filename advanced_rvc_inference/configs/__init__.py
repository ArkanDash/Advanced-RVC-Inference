"""
Configuration modules for Advanced RVC Inference.

This package contains configuration files and settings.
"""

__all__ = ["config", "config_settings"]

from .config import Config
from . import config as config_module

__all__.extend(config_module.__all__ if hasattr(config_module, "__all__") else [])

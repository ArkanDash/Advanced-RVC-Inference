"""
UI Tab Modules

This package contains all the Gradio interface tabs for the Advanced RVC Inference application.
"""

from .full_inference import full_inference_tab
from .download_model import download_model_tab
from .download_music import download_music_tab
from .settings import select_themes_tab
from .training import training_tab
from .model_manager import model_manager_tab
from .enhancement import enhancement_tab
from .real_time import real_time_inference_tab
from .config_options import extra_options_tab

__all__ = [
    'full_inference_tab',
    'download_model_tab',
    'download_music_tab', 
    'select_themes_tab',
    'training_tab',
    'model_manager_tab',
    'enhancement_tab',
    'real_time_inference_tab',
    'extra_options_tab'
]
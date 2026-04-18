# App Module
# Main application entry points (GUI, CLI, TensorBoard)
#
# This module contains the Gradio web interface and supporting tools.
# Import heavy GUI dependencies lazily so headless / CLI-only usage
# (e.g. Colab no-UI notebook, rvc-cli commands) is not blocked.

from .gui import launch as launch_gui
from .run_tensorboard import launch_tensorboard


__all__ = ['launch_gui', 'launch_tensorboard']

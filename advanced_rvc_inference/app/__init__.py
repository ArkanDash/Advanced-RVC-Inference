# App Module
# Main application entry points (GUI, CLI, TensorBoard)
#
# This module contains the Gradio web interface and supporting tools.
# Import heavy GUI dependencies lazily so headless / CLI-only usage
# (e.g. Colab no-UI notebook, rvc-cli commands) is not blocked.


def launch_gui(*args, **kwargs):
    """Launch the Gradio web interface (lazy import)."""
    from .gui import launch
    return launch(*args, **kwargs)


def launch_tensorboard(*args, **kwargs):
    """Launch TensorBoard (lazy import)."""
    from .run_tensorboard import launch_tensorboard as _launch
    return _launch(*args, **kwargs)


__all__ = ['launch_gui', 'launch_tensorboard']

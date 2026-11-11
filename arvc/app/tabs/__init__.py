"""
UI tabs for Advanced RVC Inference Gradio interface.

This package contains the tab modules for the web interface.
"""

__all__ = [
    "inference_tab",
    "realtime_tab",
    "training_tab",
    "downloads_tab",
    "extra_tab",
]


def inference_tab():
    """Load and return the inference tab."""
    from .inference.inference import inference_tab as tab

    return tab()


def realtime_tab():
    """Load and return the realtime tab."""
    from .realtime.realtime import realtime_tab as tab

    return tab()


def training_tab():
    """Load and return the training tab."""
    from .training.training import training_tab as tab

    return tab()


def downloads_tab():
    """Load and return the downloads tab."""
    from .downloads.downloads import download_tab as tab

    return tab()


def extra_tab(app=None):
    """Load and return the extra tab."""
    from .extra.extra import extra_tab as tab

    return tab(app)

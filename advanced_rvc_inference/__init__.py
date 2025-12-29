"""
Advanced RVC Inference
======================

A modular Retrieval-based Voice Conversion (RVC) framework providing
a programmatic API, Command Line Interface (CLI), and Gradio Web UI
for voice training and inference.

This package provides tools for:
- Voice conversion with multiple pitch extraction methods
- Model training capabilities
- Text-to-speech integration
- Audio separation tools
- Real-time voice processing

Basic Usage:
    >>> import advanced_rvc_inference as arvc
    >>> rvc = arvc.RVCInference(device="cuda:0")
    >>> rvc.load_model("path/to/model.pth")
    >>> audio = rvc.infer("input.wav", pitch_change=0)

CLI Usage:
    $ rvc-cli infer --model model.pth --input audio.wav
    $ rvc-gui  # Launch web interface
"""

import sys
from pathlib import Path

# Package version - single source of truth
try:
    from ._version import __version__
except ImportError:
    __version__ = "2.0.0"

# Define public API
__all__ = [
    "__version__",
    "RVCInference",
    "RVCConfig",
    "RVCModel",
    "RVCTrainer",
    "RVCRealtime",
    "cli",
    "gui",
    "launch",
]

# Lazy import for heavy dependencies to speed up initial import
_LAZY_IMPORTS = {
    "torch": ("torch", "torch"),
    "numpy": ("np", "numpy"),
    "librosa": ("librosa", "librosa"),
    "gradio": ("gr", "gradio"),
}


def __getattr__(name: str):
    """Lazy import mechanism for heavy dependencies."""
    if name in _LAZY_IMPORTS:
        module_name, alias = _LAZY_IMPORTS[name]
        try:
            module = __import__(module_name, fromlist=[alias])
            globals()[alias] = module
            return module
        except ImportError:
            raise AttributeError(
                f"Module '{__name__}' has no attribute '{name}'. "
                f"Required dependency '{module_name}' may not be installed."
            )
    raise AttributeError(f"Module '{__name__}' has no attribute '{name}'")


def _setup_paths():
    """Setup package paths for asset resolution."""
    # Get the package directory
    package_dir = Path(__file__).parent.resolve()

    # Add package directory to path if not already there
    if str(package_dir) not in sys.path:
        sys.path.insert(0, str(package_dir))

    # Set environment variables for asset paths
    import os

    # Default asset paths relative to package
    os.environ.setdefault("ARVC_ASSETS_PATH", str(package_dir / "assets"))
    os.environ.setdefault("ARVC_CONFIGS_PATH", str(package_dir / "configs"))
    os.environ.setdefault("ARVC_WEIGHTS_PATH", str(package_dir / "assets" / "weights"))
    os.environ.setdefault("ARVC_LOGS_PATH", str(package_dir / "assets" / "logs"))


# Setup paths on import
_setup_paths()


# Import main classes for convenience (these use lazy loading internally)
def _import_main_classes():
    """Import main classes with lazy loading to avoid circular imports."""
    try:
        from .api import RVCInference, RVCConfig, RVCModel, RVCTrainer, RVCRealtime

        return {
            "RVCInference": RVCInference,
            "RVCConfig": RVCConfig,
            "RVCModel": RVCModel,
            "RVCTrainer": RVCTrainer,
            "RVCRealtime": RVCRealtime,
        }
    except ImportError:
        # API module not available yet - return None
        return None


# Expose main classes
_main_classes = None


def __getattr__(name: str):
    """Get main classes with lazy loading."""
    global _main_classes

    if name in ("RVCInference", "RVCConfig", "RVCModel", "RVCTrainer", "RVCRealtime"):
        if _main_classes is None:
            _main_classes = _import_main_classes()
        if _main_classes and name in _main_classes:
            return _main_classes[name]

    # Handle lazy imports for heavy dependencies
    if name in _LAZY_IMPORTS:
        module_name, alias = _LAZY_IMPORTS[name]
        try:
            module = __import__(module_name, fromlist=[alias])
            globals()[alias] = module
            return module
        except ImportError:
            raise AttributeError(
                f"Module '{__name__}' has no attribute '{name}'. "
                f"Required dependency '{module_name}' may not be installed."
            )

    raise AttributeError(f"Module '{__name__}' has no attribute '{name}'")


# CLI and GUI entry points
def launch_cli():
    """Launch the command-line interface."""
    # Import cli module directly to avoid recursion
    from advanced_rvc_inference.cli import main as cli_main

    cli_main()


def launch_gui():
    """Launch the Gradio web interface."""
    from advanced_rvc_inference.gui import launch as gui_launch

    gui_launch()


def gui():
    """Launch the Gradio web interface (alias for launch_gui())."""
    launch_gui()


def launch():
    """Launch the Gradio web interface (alias for launch_gui())."""
    launch_gui()


class _CLIModule:
    """Module-like class that provides CLI functionality."""

    @staticmethod
    def main():
        """Run the CLI main function."""
        launch_cli()


class _GUIModule:
    """Module-like class that provides GUI functionality."""

    @staticmethod
    def launch():
        """Run the GUI launch function."""
        launch_gui()


# Add cli and gui as module-like objects for backward compatibility
cli = _CLIModule()
gui = _GUIModule()


def _check_dependencies():
    """Check if all required dependencies are installed."""
    missing = []
    for package in ["torch", "gradio", "numpy"]:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        import warnings

        warnings.warn(
            f"Missing optional dependencies: {', '.join(missing)}. "
            f"Install them with: pip install advanced-rvc-inference",
            ImportWarning,
        )


# Check dependencies on import
_check_dependencies()

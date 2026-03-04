

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
    "launch_cli",
    "launch_gui",
]

_LAZY_IMPORTS = {
    "torch": ("torch", "torch"),
    "numpy": ("np", "numpy"),
    "librosa": ("librosa", "librosa"),
    "gradio": ("gr", "gradio"),
}

_main_classes = None



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


# CLI and GUI entry points
def launch_cli():
    """Launch the command-line interface."""
    from advanced_rvc_inference.api.cli import main as cli_main

    cli_main()


def launch_gui():
    """Launch the Gradio web interface."""
    from advanced_rvc_inference.app.gui import launch as gui_launch

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
    for package in ["gradio"]:
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

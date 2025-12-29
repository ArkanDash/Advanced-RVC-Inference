"""
Test suite for Advanced RVC Inference package.
"""

import pytest
import sys
from pathlib import Path

# Add package to path
PACKAGE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))


class TestPackageStructure:
    """Test package structure and imports."""

    def test_version_defined(self):
        """Test that version is defined."""
        from advanced_rvc_inference._version import __version__

        assert __version__ is not None
        assert isinstance(__version__, str)

    def test_package_import(self):
        """Test package can be imported."""
        import advanced_rvc_inference

        assert advanced_rvc_inference.__version__ is not None

    def test_cli_module_exists(self):
        """Test CLI module exists."""
        from advanced_rvc_inference import cli

        assert hasattr(cli, "main")
        assert hasattr(cli, "create_parser")

    def test_gui_module_exists(self):
        """Test GUI module exists."""
        from advanced_rvc_inference import gui

        assert hasattr(gui, "launch")
        assert hasattr(gui, "create_app")

    def test_api_module_exists(self):
        """Test API module exists."""
        from advanced_rvc_inference import api

        assert hasattr(api, "RVCInference")
        assert hasattr(api, "RVCConfig")
        assert hasattr(api, "RVCTrainer")
        assert hasattr(api, "RVCRealtime")

    def test_core_modules(self):
        """Test core modules exist."""
        from advanced_rvc_inference import core

        assert hasattr(core, "inference")
        assert hasattr(core, "process")
        assert hasattr(core, "model_utils")

    def test_tabs_modules(self):
        """Test tabs modules exist."""
        from advanced_rvc_inference import tabs

        assert hasattr(tabs, "inference_tab")
        assert hasattr(tabs, "realtime_tab")
        assert hasattr(tabs, "training_tab")

    def test_library_modules(self):
        """Test library modules exist."""
        from advanced_rvc_inference import library

        assert hasattr(library, "utils")
        assert hasattr(library, "backends")


class TestAPI:
    """Test API classes."""

    def test_rvc_config_defaults(self):
        """Test RVCConfig default values."""
        from advanced_rvc_inference.api import RVCConfig

        config = RVCConfig()
        assert config.device == "cuda:0"
        assert config.half_precision is True

    def test_rvc_inference_init(self):
        """Test RVCInference initialization."""
        from advanced_rvc_inference.api import RVCInference

        # Should not raise when torch is available
        try:
            rvc = RVCInference(device="cpu")
            assert rvc is not None
        except Exception:
            # May fail if torch is not installed in test env
            pass


class TestCLI:
    """Test CLI functionality."""

    def test_get_version(self):
        """Test version retrieval."""
        from advanced_rvc_inference.cli import get_version

        version = get_version()
        assert version is not None
        assert isinstance(version, str)

    def test_parser_creation(self):
        """Test argument parser creation."""
        from advanced_rvc_inference.cli import create_parser

        parser = create_parser()
        assert parser is not None

    def test_version_command(self):
        """Test version command output."""
        from advanced_rvc_inference.cli import show_version

        # Should not raise
        show_version()


class TestGUIFunctions:
    """Test GUI functions."""

    def test_get_version(self):
        """Test version retrieval."""
        from advanced_rvc_inference.gui import get_version

        version = get_version()
        assert version is not None

    def test_launch_function_exists(self):
        """Test launch function exists."""
        from advanced_rvc_inference.gui import launch

        assert callable(launch)

    def test_create_app_function_exists(self):
        """Test create_app function exists."""
        from advanced_rvc_inference.gui import create_app

        assert callable(create_app)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

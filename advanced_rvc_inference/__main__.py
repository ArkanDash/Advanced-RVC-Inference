#!/usr/bin/env python3
"""
Main entry point for the advanced_rvc_inference package.

This module allows the package to be run as a module:

    python -m advanced_rvc_inference
    python -m advanced_rvc_inference --help
"""

import sys
from pathlib import Path


def main():
    """Main entry point when running as a module."""
    # Add the parent directory to path for imports
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    # Import and run CLI
    from advanced_rvc_inference import cli

    sys.exit(cli.main())


if __name__ == "__main__":
    main()

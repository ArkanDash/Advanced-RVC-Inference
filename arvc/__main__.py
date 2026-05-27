#!/usr/bin/env python3
"""
Main entry point for the arvc package.

This module allows the package to be run as a module:

    python -m arvc
    python -m arvc --help
"""

import sys
import os
from pathlib import Path


def main():
    """Main entry point when running as a module."""
    # Add the parent directory to path for imports
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    # Also add current working directory for imports
    sys.path.append(os.getcwd())

    # Import and run CLI
    from arvc import cli

    sys.exit(cli.main())


if __name__ == "__main__":
    main()

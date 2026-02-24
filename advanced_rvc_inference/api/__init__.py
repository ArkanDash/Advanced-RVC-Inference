# API Module
# REST API endpoints and CLI interface

from .cli_complete import main as cli_main
from .cli import main as cli_main_basic

__all__ = [
    'cli_main',
    'cli_main_basic'
]

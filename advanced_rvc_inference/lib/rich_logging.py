"""
Rich-based Logging System for Advanced RVC Inference
Enhanced logging with beautiful formatting and structured output

Author: MiniMax Agent
Date: 2025-11-27
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

try:
    from rich.logging import RichHandler
    from rich.console import Console
    from rich.theme import Theme
    from rich.panel import Panel
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TaskProgressColumn
    from rich.text import Text
    from rich.table import Table
    from rich.live import Live
    from rich.status import Status
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Rich console with custom theme
if RICH_AVAILABLE:
    custom_theme = Theme({
        "info": "cyan",
        "warning": "yellow", 
        "error": "red",
        "success": "green",
        "debug": "dim",
        "critical": "red bold",
        "header": "bold cyan",
        "accent": "magenta",
        "value": "white"
    })
    
    console = Console(theme=custom_theme, force_terminal=True)
    
    class RichLogger:
        """Enhanced logger with Rich formatting"""
        
        def __init__(self, name: str = "AdvancedRVC", level: int = logging.INFO):
            self.name = name
            self.console = console
            self.level = level
            self.setup_logger()
        
        def setup_logger(self):
            """Setup the logger with Rich handler"""
            self.logger = logging.getLogger(self.name)
            self.logger.setLevel(self.level)
            
            # Remove existing handlers
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)
            
            # Add Rich handler
            if RICH_AVAILABLE:
                rich_handler = RichHandler(
                    console=self.console,
                    show_path=True,
                    show_time=True,
                    markup=True
                )
                self.logger.addHandler(rich_handler)
            
            # Prevent propagation to root logger
            self.logger.propagate = False
        
        def info(self, message: str, **kwargs):
            """Log info message with Rich formatting"""
            if RICH_AVAILABLE:
                self.console.print(f"[info]ℹ {message}[/info]", **kwargs)
            self.logger.info(message)
        
        def warning(self, message: str, **kwargs):
            """Log warning message with Rich formatting"""
            if RICH_AVAILABLE:
                self.console.print(f"[warning]⚠ {message}[/warning]", **kwargs)
            self.logger.warning(message)
        
        def error(self, message: str, **kwargs):
            """Log error message with Rich formatting"""
            if RICH_AVAILABLE:
                self.console.print(f"[error]❌ {message}[/error]", **kwargs)
            self.logger.error(message)
        
        def success(self, message: str, **kwargs):
            """Log success message with Rich formatting"""
            if RICH_AVAILABLE:
                self.console.print(f"[success]✅ {message}[/success]", **kwargs)
            # Use info level for success messages
            self.logger.info(f"SUCCESS: {message}")
        
        def debug(self, message: str, **kwargs):
            """Log debug message with Rich formatting"""
            if RICH_AVAILABLE:
                self.console.print(f"[debug]🐛 {message}[/debug]", **kwargs)
            self.logger.debug(message)
        
        def critical(self, message: str, **kwargs):
            """Log critical message with Rich formatting"""
            if RICH_AVAILABLE:
                self.console.print(f"[critical]🚨 {message}[/critical]", **kwargs)
            self.logger.critical(message)
        
        def header(self, message: str, **kwargs):
            """Log header message"""
            if RICH_AVAILABLE:
                self.console.print(f"[header]{message}[/header]", **kwargs)
            self.logger.info(f"HEADER: {message}")
        
        def panel(self, title: str, content: str, style: str = "info", **kwargs):
            """Create a panel with Rich formatting"""
            if RICH_AVAILABLE:
                panel = Panel(
                    Text(content, style=style),
                    title=f"[header]{title}[/header]",
                    expand=False
                )
                self.console.print(panel, **kwargs)
            else:
                self.logger.info(f"{title}: {content}")
        
        def table(self, title: str, data: list, columns: list):
            """Create a table with Rich formatting"""
            if RICH_AVAILABLE:
                table = Table(title=title)
                for col in columns:
                    table.add_column(col, style="cyan")
                
                for row in data:
                    table.add_row(*row)
                
                self.console.print(table)
            else:
                self.logger.info(f"{title}: {data}")
        
        def progress(self, tasks: list):
            """Create progress tracking"""
            if RICH_AVAILABLE:
                progress = Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeRemainingColumn(),
                )
                return progress
            else:
                return None
        
        def status(self, message: str, spinner: str = "dots"):
            """Create status message with spinner"""
            if RICH_AVAILABLE:
                return Status(f"[cyan]{message}[/cyan]", spinner=spinner, console=self.console)
            else:
                return self.logger.info(message)

    # Create global logger instance
    logger = RichLogger("AdvancedRVC", logging.INFO)
    
else:
    # Fallback to standard logging if Rich is not available
    logger = logging.getLogger(__name__)
    console = None
    
    class RichLogger:
        """Fallback logger when Rich is not available"""
        
        def __init__(self, name: str = "AdvancedRVC", level: int = logging.INFO):
            self.name = name
            self.level = level
            self.setup_logger()
        
        def setup_logger(self):
            self.logger = logging.getLogger(self.name)
            self.logger.setLevel(self.level)
            
            # Add standard handler
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.propagate = False
        
        def info(self, message: str, **kwargs):
            print(f"INFO: {message}")
            self.logger.info(message)
        
        def warning(self, message: str, **kwargs):
            print(f"WARNING: {message}")
            self.logger.warning(message)
        
        def error(self, message: str, **kwargs):
            print(f"ERROR: {message}")
            self.logger.error(message)
        
        def success(self, message: str, **kwargs):
            print(f"SUCCESS: {message}")
            self.logger.info(message)
        
        def debug(self, message: str, **kwargs):
            print(f"DEBUG: {message}")
            self.logger.debug(message)
        
        def critical(self, message: str, **kwargs):
            print(f"CRITICAL: {message}")
            self.logger.critical(message)
        
        def header(self, message: str, **kwargs):
            print(f"=== {message} ===")
            self.logger.info(f"HEADER: {message}")
        
        def panel(self, title: str, content: str, style: str = "info", **kwargs):
            print(f"=== {title} ===\n{content}")
            self.logger.info(f"{title}: {content}")
        
        def table(self, title: str, data: list, columns: list):
            print(f"=== {title} ===")
            print(columns)
            for row in data:
                print(row)
            self.logger.info(f"{title}: {data}")
        
        def progress(self, tasks: list):
            return None
        
        def status(self, message: str, spinner: str = "dots"):
            print(f"STATUS: {message}")
            return self.logger.info(message)

# Initialize global logger
global_logger = RichLogger("AdvancedRVC")

# Export the logger for use throughout the application
__all__ = ['logger', 'console', 'RichLogger', 'RICH_AVAILABLE']

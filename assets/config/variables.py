# Config variables fallback
import os
from pathlib import Path

class Config:
    def __init__(self):
        self.config = {}
        self.logger = None
        self.translations = {}
        self.configs = {}

config = Config()

# Export translations and configs directly for imports
translations = config.translations
configs = config.configs

# Mock logger
class MockLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")

logger = MockLogger()

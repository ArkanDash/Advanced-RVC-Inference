"""
Path Manager for Advanced RVC Inference
Handles all path-related operations consistently across the application
"""
import os
import json
from pathlib import Path


class PathManager:
    """
    A comprehensive path manager class to handle all path-related operations
    in the Advanced RVC Inference application.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the PathManager with configuration
        
        Args:
            config_path (str, optional): Path to the config file. 
                                       Defaults to assets/config.json
        """
        if config_path is None:
            # Use the project root directory
            self.project_root = Path(__file__).parent.parent.parent
            self.config_path = self.project_root / "assets" / "config.json"
        else:
            self.config_path = Path(config_path)
            self.project_root = self.config_path.parent.parent
        
        self.config = self._load_config()
    
    def _load_config(self):
        """
        Load configuration from the config file.
        
        Returns:
            dict: Configuration dictionary
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            # Create a default configuration if the file doesn't exist
            print(f"Config file not found at {self.config_path}, using defaults")
            return self._get_default_config()
        except json.JSONDecodeError:
            print(f"Error decoding config file at {self.config_path}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self):
        """
        Get default configuration.
        
        Returns:
            dict: Default configuration dictionary
        """
        return {
            "paths": {
                "audio_files": "audio_files",
                "models": "assets/weights",
                "outputs": "assets/audios/output", 
                "temp": "temp",
                "datasets": "datasets"
            }
        }

    def _resolve_path(self, key):
        """
        Resolve a path key to an absolute path.

        Args:
            key (str): The path key (e.g., 'weights_dir', 'audios_dir', etc.)

        Returns:
            Path: The resolved path
        """
        # Map the keys to the actual path names in config
        key_mapping = {
            'weights_dir': 'models',
            'audios_dir': 'audio_files',
            'logs_dir': 'logs',  # Special case: logs might not be in the config, so map separately
            'outputs_dir': 'outputs',
            'inputs_dir': 'inputs',
            'temp_dir': 'temp',
            'datasets_dir': 'datasets',
            'pretrained_custom': 'assets/weights/pretrained'
        }

        path_key = key_mapping.get(key, key)

        # Map keys to config entries
        config_key_mapping = {
            'weights_dir': 'models',
            'audios_dir': 'audio_files',
            'logs_dir': 'logs',
            'outputs_dir': 'outputs',
            'inputs_dir': 'inputs',
            'temp_dir': 'temp',
            'datasets_dir': 'datasets'
        }

        # If it's logs_dir, handle it as a special case since it might not be in config
        if key == 'logs_dir':
            if 'logs' in self.config.get('paths', {}):
                path_str = self.config['paths']['logs']
            else:
                path_str = 'logs'  # Fallback
        elif key in config_key_mapping and config_key_mapping[key] in self.config.get('paths', {}):
            path_str = self.config['paths'][config_key_mapping[key]]
        elif path_key in self.config.get('paths', {}):
            path_str = self.config['paths'][path_key]
        elif key in self.config.get('paths', {}):
            path_str = self.config['paths'][key]
        else:
            # If the key is not found in config, use it as-is relative to project root
            path_str = key

        # Create the full path
        full_path = self.project_root / path_str

        # Create directory if it doesn't exist
        full_path.mkdir(parents=True, exist_ok=True)

        return full_path

    def __call__(self, key):
        """
        Make the PathManager callable to return paths.
        
        Args:
            key (str): Path key to resolve
            
        Returns:
            Path: The resolved path
        """
        return self._resolve_path(key)

    def get_path(self, key):
        """
        Get a path by key.
        
        Args:
            key (str): Path key to resolve
            
        Returns:
            Path: The resolved path
        """
        return self._resolve_path(key)

    def validate_path(self, path):
        """
        Validate if a path is safe and within project boundaries.
        
        Args:
            path (Path or str): Path to validate
            
        Returns:
            bool: True if path is valid and safe, False otherwise
        """
        path = Path(path)
        try:
            # Resolve the path to handle .. and . components
            resolved_path = path.resolve()
            # Check if the path is within the project root
            resolved_path.relative_to(self.project_root.resolve())
            return True
        except ValueError:
            # Path is outside the project root
            return False


def get_path_manager(config_path=None):
    """
    Get a singleton instance of the PathManager.
    
    Args:
        config_path (str, optional): Path to the config file
        
    Returns:
        PathManager: A PathManager instance
    """
    return PathManager(config_path)


# Create a global instance
_path_manager_instance = None


def path(key):
    """
    Global function to get paths using the PathManager.
    
    Args:
        key (str): Path key to resolve
        
    Returns:
        Path: The resolved path
    """
    global _path_manager_instance
    if _path_manager_instance is None:
        _path_manager_instance = get_path_manager()
    return _path_manager_instance(key)


# Backward compatibility - make the path function the main export
__all__ = ['PathManager', 'get_path_manager', 'path']
"""
Advanced RVC Inference - Path Management System
Enhanced path management with caching, validation, and optimization
Version: 1.0.0
Authors: ArkanDash & BF667
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from functools import lru_cache
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PathConfig:
    """Configuration for path management"""
    cache_enabled: bool = True
    auto_create: bool = True
    validate_paths: bool = True
    cache_ttl: int = 3600  # 1 hour

class PathManager:
    """
    Advanced Path Manager for Advanced RVC Inference
    
    Features:
    - Centralized path management with validation
    - Path caching with TTL
    - Automatic directory creation
    - Multi-platform compatibility
    - Model and asset discovery
    - Temporary file management
    - Project structure validation
    """
    
    def __init__(self, config: Optional[PathConfig] = None):
        """Initialize PathManager with configuration"""
        self.config = config or PathConfig()
        self._path_cache = {}
        self._cache_times = {}
        self._project_root = Path.cwd()
        
        # Define critical project paths
        self.paths = {
            # Core project structure
            'project_root': self._project_root,
            'models_dir': self._project_root / 'assets' / 'models',
            'weights_dir': self._project_root / 'assets' / 'weights',
            'configs_dir': self._project_root / 'advanced_rvc_inference' / 'rvc' / 'configs',
            'logs_dir': self._project_root / 'logs',
            'temp_dir': self._project_root / 'temp',
            'cache_dir': self._project_root / '.cache',
            
            # Model-specific directories
            'pretrained_v1': self._project_root / 'assets' / 'models' / 'pretrained_v1',
            'pretrained_v2': self._project_root / 'assets' / 'models' / 'pretrained_v2',
            'pretrained_custom': self._project_root / 'assets' / 'models' / 'pretrained_custom',
            'embedders': self._project_root / 'assets' / 'models' / 'embedders',
            'predictors': self._project_root / 'assets' / 'models' / 'predictors',
            'uvr5': self._project_root / 'assets' / 'models' / 'uvr5',
            'speaker_diarization': self._project_root / 'assets' / 'models' / 'speaker_diarization',
            
            # Audio directories
            'audios_dir': self._project_root / 'assets' / 'audios',
            'dataset_dir': self._project_root / 'assets' / 'dataset',
            'f0_dir': self._project_root / 'assets' / 'f0',
            
            # Configuration
            'config_file': self._project_root / 'config.json',
            'config_zluda': self._project_root / 'config_zluda.json',
            'requirements': self._project_root / 'requirements.txt',
            
            # Documentation and assets
            'docs_dir': self._project_root / 'docs',
            'notebooks_dir': self._project_root / 'notebooks',
            'themes_dir': self._project_root / 'assets' / 'themes',
            'i18n_dir': self._project_root / 'assets' / 'i18n',
            'binary_dir': self._project_root / 'assets' / 'binary',
            'zluda_dir': self._project_root / 'assets' / 'zluda',
            
            # Training specific
            'training_logs': self._project_root / 'logs' / 'training',
            'checkpoints': self._project_root / 'logs' / 'checkpoints',
            'tensorboard_logs': self._project_root / 'logs' / 'tensorboard',
        }
        
        # Ensure critical directories exist
        if self.config.auto_create:
            self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Create directories if they don't exist"""
        critical_dirs = [
            'models_dir', 'weights_dir', 'logs_dir', 'temp_dir', 'cache_dir',
            'audios_dir', 'dataset_dir', 'f0_dir', 'training_logs', 'checkpoints'
        ]
        
        for path_key in critical_dirs:
            self.get_path(path_key, create_if_missing=True)
    
    @lru_cache(maxsize=128)
    def get_path(self, key: str, create_if_missing: bool = False) -> Path:
        """
        Get a path by key with validation and caching
        
        Args:
            key: Path key identifier
            create_if_missing: Whether to create directory if missing
            
        Returns:
            Path object
            
        Raises:
            ValueError: If path key is invalid
            FileNotFoundError: If path doesn't exist and create_if_missing is False
        """
        if key not in self.paths:
            raise ValueError(f"Unknown path key: {key}")
        
        path = self.paths[key]
        
        if self.config.validate_paths and not path.exists():
            if create_if_missing:
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created directory: {path}")
                except Exception as e:
                    logger.error(f"Failed to create directory {path}: {e}")
            else:
                raise FileNotFoundError(f"Path does not exist: {path}")
        
        return path
    
    def add_custom_path(self, key: str, path: Union[str, Path]) -> None:
        """
        Add a custom path to the manager
        
        Args:
            key: Unique identifier for the path
            path: Path string or Path object
        """
        self.paths[key] = Path(path)
        if self.config.cache_enabled:
            # Clear cache if adding new paths
            self.get_path.cache_clear()
    
    def resolve_relative_path(self, base_key: str, relative_path: Union[str, Path]) -> Path:
        """
        Resolve a path relative to a base path
        
        Args:
            base_key: Key of the base path
            relative_path: Relative path to resolve
            
        Returns:
            Resolved Path object
        """
        base_path = self.get_path(base_key)
        return (base_path / relative_path).resolve()
    
    def find_models(self, 
                   model_type: Optional[str] = None,
                   extensions: List[str] = None) -> Dict[str, Path]:
        """
        Discover models in the models directory
        
        Args:
            model_type: Filter by model type ('pretrained', 'embedders', etc.)
            extensions: List of file extensions to include
            
        Returns:
            Dictionary mapping model names to their paths
        """
        if extensions is None:
            extensions = ['.pth', '.pt', '.onnx', '.safetensors', '.ckpt']
        
        models = {}
        models_dir = self.get_path('models_dir')
        
        # Search in all subdirectories
        for ext in extensions:
            for model_file in models_dir.rglob(f"*{ext}"):
                # Determine model type from directory structure
                relative_path = model_file.relative_to(models_dir)
                parts = relative_path.parts
                
                if model_type and len(parts) > 1:
                    if parts[0] != model_type:
                        continue
                
                model_name = model_file.stem
                models[model_name] = model_file
        
        return models
    
    def get_temp_file(self, prefix: str = "rvc_", suffix: str = ".tmp") -> Path:
        """
        Create and return a temporary file path
        
        Args:
            prefix: File prefix
            suffix: File suffix
            
        Returns:
            Path to temporary file
        """
        import tempfile
        temp_dir = self.get_path('temp_dir', create_if_missing=True)
        
        # Create temporary file in our temp directory
        temp_file = tempfile.NamedTemporaryFile(
            dir=temp_dir,
            prefix=prefix,
            suffix=suffix,
            delete=False
        )
        temp_file.close()
        
        return Path(temp_file.name)
    
    def cleanup_temp_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up temporary files older than specified age
        
        Args:
            max_age_hours: Maximum age in hours
            
        Returns:
            Number of files cleaned up
        """
        import time
        temp_dir = self.get_path('temp_dir')
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        cleaned = 0
        if temp_dir.exists():
            for temp_file in temp_dir.iterdir():
                if temp_file.is_file():
                    file_age = current_time - temp_file.stat().st_mtime
                    if file_age > max_age_seconds:
                        try:
                            temp_file.unlink()
                            cleaned += 1
                        except Exception as e:
                            logger.warning(f"Failed to delete {temp_file}: {e}")
        
        return cleaned
    
    def validate_project_structure(self) -> Dict[str, bool]:
        """
        Validate that all critical project directories exist
        
        Returns:
            Dictionary mapping directory keys to validation status
        """
        required_dirs = [
            'models_dir', 'weights_dir', 'logs_dir', 'audios_dir', 
            'dataset_dir', 'docs_dir', 'notebooks_dir'
        ]
        
        validation_results = {}
        for dir_key in required_dirs:
            try:
                path = self.get_path(dir_key)
                validation_results[dir_key] = path.exists()
            except Exception:
                validation_results[dir_key] = False
        
        return validation_results
    
    def export_config(self, output_file: Optional[Path] = None) -> Dict[str, Any]:
        """
        Export current path configuration
        
        Args:
            output_file: Optional file to save configuration
            
        Returns:
            Configuration dictionary
        """
        config_data = {
            'version': '1.0.0',
            'project_root': str(self._project_root),
            'paths': {k: str(v) for k, v in self.paths.items()},
            'validation': self.validate_project_structure(),
            'config': {
                'cache_enabled': self.config.cache_enabled,
                'auto_create': self.config.auto_create,
                'validate_paths': self.config.validate_paths,
                'cache_ttl': self.config.cache_ttl
            }
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(config_data, f, indent=2)
        
        return config_data
    
    def import_config(self, config_file: Union[str, Path]) -> None:
        """
        Import path configuration from file
        
        Args:
            config_file: Configuration file path
        """
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        # Update paths
        for key, path_str in config_data.get('paths', {}).items():
            self.paths[key] = Path(path_str)
        
        # Update config
        if 'config' in config_data:
            cfg = config_data['config']
            self.config.cache_enabled = cfg.get('cache_enabled', True)
            self.config.auto_create = cfg.get('auto_create', True)
            self.config.validate_paths = cfg.get('validate_paths', True)
            self.config.cache_ttl = cfg.get('cache_ttl', 3600)
        
        # Clear cache after import
        self.get_path.cache_clear()
    
    def get_path_hash(self, key: str) -> str:
        """
        Get hash of a path for caching purposes
        
        Args:
            key: Path key identifier
            
        Returns:
            MD5 hash of the path
        """
        path_str = str(self.get_path(key))
        return hashlib.md5(path_str.encode()).hexdigest()
    
    def list_files(self, 
                   path_key: str, 
                   pattern: str = "*",
                   recursive: bool = False) -> List[Path]:
        """
        List files in a directory matching a pattern
        
        Args:
            path_key: Directory path key
            pattern: File pattern to match
            recursive: Whether to search recursively
            
        Returns:
            List of matching file paths
        """
        base_path = self.get_path(path_key)
        
        if not base_path.exists():
            return []
        
        if recursive:
            return list(base_path.rglob(pattern))
        else:
            return list(base_path.glob(pattern))
    
    def backup_path(self, 
                   path_key: str, 
                   backup_dir: Optional[str] = None) -> Optional[Path]:
        """
        Create backup of a file or directory
        
        Args:
            path_key: Path key to backup
            backup_dir: Optional backup directory key
            
        Returns:
            Path to backup file/directory
        """
        import shutil
        import datetime
        
        source_path = self.get_path(path_key)
        
        if not source_path.exists():
            return None
        
        if backup_dir is None:
            backup_dir_key = 'cache_dir'  # Use cache as default backup location
        else:
            backup_dir_key = backup_dir
        
        backup_parent = self.get_path(backup_dir_key, create_if_missing=True)
        
        # Create timestamped backup name
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{source_path.name}_{timestamp}"
        
        if source_path.is_file():
            backup_name += source_path.suffix
            backup_path = backup_parent / backup_name
            shutil.copy2(source_path, backup_path)
        else:
            backup_path = backup_parent / backup_name
            shutil.copytree(source_path, backup_path)
        
        return backup_path

# Global path manager instance
_global_path_manager = None

def get_path_manager() -> PathManager:
    """
    Get global path manager instance
    
    Returns:
        Global PathManager instance
    """
    global _global_path_manager
    if _global_path_manager is None:
        _global_path_manager = PathManager()
    return _global_path_manager

def path(key: str, create_if_missing: bool = False) -> Path:
    """
    Convenience function to get a path
    
    Args:
        key: Path key identifier
        create_if_missing: Whether to create directory if missing
        
    Returns:
        Path object
    """
    return get_path_manager().get_path(key, create_if_missing)

# Path resolution functions for convenience
def model_path(*args) -> Path:
    """Get path relative to models directory"""
    return path('models_dir') / Path(*args)

def log_path(*args) -> Path:
    """Get path relative to logs directory"""
    return path('logs_dir') / Path(*args)

def config_path(*args) -> Path:
    """Get path relative to configs directory"""
    return path('configs_dir') / Path(*args)

def asset_path(*args) -> Path:
    """Get path relative to assets directory"""
    return path('project_root') / 'assets' / Path(*args)

if __name__ == "__main__":
    # Demo usage
    pm = PathManager()
    
    # Validate project structure
    validation = pm.validate_project_structure()
    print("Project Structure Validation:")
    for key, status in validation.items():
        print(f"  {key}: {'✓' if status else '✗'}")
    
    # Find available models
    models = pm.find_models()
    print(f"\nFound {len(models)} models")
    
    # Export configuration
    config = pm.export_config()
    print("\nConfiguration exported successfully")

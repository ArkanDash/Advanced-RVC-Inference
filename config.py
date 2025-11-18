"""
Enhanced Configuration Manager for Advanced RVC Inference
Handles environment variables, config files, and validation
"""

import os
import configparser
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging
from decouple import config, Config, RepositoryEnv

logger = logging.getLogger(__name__)

class ConfigurationManager:
    """Enhanced configuration management with validation and environment support."""
    
    def __init__(self, config_file: Optional[str] = None, env_file: Optional[str] = None):
        self.config_file = Path(config_file) if config_file else Path("config.ini")
        self.env_file = Path(env_file) if env_file else Path(".env")
        
        # Initialize configuration
        self._config = configparser.ConfigParser()
        self._load_configuration()
        
        # Validate configuration
        self._validate_configuration()
        
        logger.info(f"Configuration loaded from: {self.config_file}")
        logger.info(f"Environment file: {self.env_file}")
    
    def _load_configuration(self):
        """Load configuration from files."""
        # Load .env file first (takes priority)
        if self.env_file.exists():
            try:
                from decouple import config
                self._config_dict = config
                logger.debug(f"Loaded environment variables from: {self.env_file}")
            except Exception as e:
                logger.warning(f"Could not load .env file: {e}")
                self._config_dict = config
        else:
            self._config_dict = config
        
        # Load config.ini file
        if self.config_file.exists():
            try:
                self._config.read(self.config_file)
                logger.debug(f"Loaded configuration from: {self.config_file}")
            except Exception as e:
                logger.warning(f"Could not load config.ini: {e}")
        else:
            logger.info(f"Config file not found, using defaults: {self.config_file}")
    
    def _validate_configuration(self):
        """Validate configuration values and set defaults."""
        validation_rules = {
            'MAX_FILE_SIZE_MB': (1, 10000, 500),
            'MAX_AUDIO_DURATION_MINUTES': (1, 180, 30),
            'MAX_THREADS': (1, 16, 4),
            'MEMORY_LIMIT_GB': (1, 64, 8),
            'CACHE_SIZE_MB': (10, 10000, 1000),
            'SERVER_PORT': (1024, 65535, 7860),
            'DEFAULT_CHUNK_SIZE_MS': (10, 1000, 100),
            'DEFAULT_VAD_SENSITIVITY': (0, 5, 3),
            'MAX_PITCH_SHIFT': (0, 24, 12),
            'DEFAULT_PROTECT_RATE': (0.0, 1.0, 0.33),
            'LOG_FILE_MAX_SIZE_MB': (1, 100, 10),
            'LOG_FILE_BACKUP_COUNT': (1, 20, 5),
            'BACKUP_INTERVAL_HOURS': (1, 8760, 168),
            'MAX_BACKUP_FILES': (1, 100, 10),
            'MAX_TEXT_LENGTH': (100, 50000, 5000),
            'MAX_SEPARATION_MODELS': (1, 50, 10),
            'BATCH_SIZE_LIMIT': (1, 20, 5)
        }
        
        for key, (min_val, max_val, default_val) in validation_rules.items():
            try:
                current_val = self.get(key, default=default_val)
                if isinstance(current_val, str):
                    # Try to convert string to appropriate type
                    try:
                        if '.' in current_val:
                            current_val = float(current_val)
                        else:
                            current_val = int(current_val)
                    except ValueError:
                        current_val = default_val
                
                if not (min_val <= current_val <= max_val):
                    logger.warning(f"Configuration value {key}={current_val} is out of range [{min_val}, {max_val}]. Using default: {default_val}")
                    os.environ[key] = str(default_val)
            except Exception as e:
                logger.warning(f"Could not validate {key}: {e}. Using default: {default_val}")
                os.environ[key] = str(default_val)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with environment variable override."""
        try:
            # Try environment variable first
            if key in os.environ:
                env_val = os.environ[key]
                # Type conversion based on environment variable name
                if key.startswith('ENABLE_') or key.endswith('_ENABLED') or key.endswith('_DEBUG'):
                    return env_val.lower() in ('true', '1', 'yes', 'on')
                elif any(x in key.upper() for x in ['SIZE', 'MB', 'GB', 'MINUTES', 'HOURS', 'PORT', 'THREADS']):
                    return int(env_val)
                elif any(x in key.upper() for x in ['RATE', 'LEVEL', 'SENSITIVITY']):
                    return float(env_val)
                else:
                    return env_val
            
            # Fallback to config file
            for section in self._config.sections():
                if self._config.has_option(section, key):
                    return self._config.get(section, key)
            
            # Fallback to decouple config
            try:
                return self._config_dict(key, default=default)
            except:
                return default
                
        except Exception as e:
            logger.debug(f"Could not get configuration for {key}: {e}")
            return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        result = {}
        if self._config.has_section(section):
            for key, value in self._config.items(section):
                result[key] = value
        return result
    
    def set(self, key: str, value: Any, section: str = 'application'):
        """Set configuration value."""
        if section not in self._config.sections():
            self._config.add_section(section)
        self._config.set(section, key, str(value))
    
    def save(self, file_path: Optional[str] = None):
        """Save configuration to file."""
        save_path = Path(file_path) if file_path else self.config_file
        try:
            with open(save_path, 'w') as f:
                self._config.write(f)
            logger.info(f"Configuration saved to: {save_path}")
        except Exception as e:
            logger.error(f"Could not save configuration: {e}")
    
    def get_paths(self) -> Dict[str, Path]:
        """Get all configured paths."""
        paths = {}
        path_keys = [
            'CACHE_DIR', 'LOGS_DIR', 'MODELS_DIR', 'TEMP_DIR', 
            'AUDIO_FILES_DIR', 'OUTPUT_DIR'
        ]
        
        for key in path_keys:
            path_str = self.get(key, default=key.lower().replace('_', '_'))
            paths[key] = Path(path_str)
        
        return paths
    
    def is_development_mode(self) -> bool:
        """Check if running in development mode."""
        return self.get('APP_DEBUG', default=False) or self.get('DEV_MODE', default=False)
    
    def is_gpu_enabled(self) -> bool:
        """Check if GPU acceleration is enabled."""
        return self.get('ENABLE_GPU_ACCELERATION', default=True)
    
    def get_security_settings(self) -> Dict[str, Any]:
        """Get security-related configuration."""
        return {
            'max_file_size_mb': self.get('MAX_FILE_SIZE_MB', 500),
            'max_audio_duration_minutes': self.get('MAX_AUDIO_DURATION_MINUTES', 30),
            'allowed_extensions': self.get('ALLOWED_EXTENSIONS', 'wav,mp3,flac,ogg,m4a,aac,alac,wma').split(','),
            'enable_validation': self.get('ENABLE_FILE_VALIDATION', True),
            'enable_audit_logging': self.get('ENABLE_AUDIT_LOGGING', False)
        }
    
    def get_performance_settings(self) -> Dict[str, Any]:
        """Get performance-related configuration."""
        return {
            'max_threads': self.get('MAX_THREADS', 4),
            'memory_limit_gb': self.get('MEMORY_LIMIT_GB', 8),
            'cache_size_mb': self.get('CACHE_SIZE_MB', 1000),
            'cache_ttl_hours': self.get('CACHE_TTL_HOURS', 24),
            'chunk_size_ms': self.get('DEFAULT_CHUNK_SIZE_MS', 100)
        }
    
    def get_audio_settings(self) -> Dict[str, Any]:
        """Get audio processing configuration."""
        return {
            'default_sample_rate': self.get('DEFAULT_SAMPLE_RATE', 44100),
            'default_bit_depth': self.get('DEFAULT_BIT_DEPTH', 16),
            'normalize_output': self.get('NORMALIZE_OUTPUT', True),
            'enable_post_processing': self.get('ENABLE_POST_PROCESSING', True),
            'enable_noise_reduction': self.get('ENABLE_NOISE_REDUCTION', True),
            'enable_reverb_removal': self.get('ENABLE_REVERB_REMOVAL', True)
        }
    
    def get_ui_settings(self) -> Dict[str, Any]:
        """Get user interface configuration."""
        return {
            'theme': self.get('UI_THEME', 'default'),
            'language': self.get('UI_LANGUAGE', 'en_US'),
            'show_progress_bars': self.get('SHOW_PROGRESS_BARS', True),
            'show_detailed_logs': self.get('SHOW_DETAILED_LOGS', False),
            'auto_open_browser': self.get('AUTO_OPEN_BROWSER', True)
        }
    
    def validate_file_size(self, file_size_bytes: int) -> bool:
        """Validate file size against configuration."""
        max_size_bytes = self.get('MAX_FILE_SIZE_MB', 500) * 1024 * 1024
        return file_size_bytes <= max_size_bytes
    
    def validate_file_extension(self, file_path: Union[str, Path]) -> bool:
        """Validate file extension against allowed list."""
        file_path = Path(file_path)
        allowed_extensions = self.get('ALLOWED_EXTENSIONS', 'wav,mp3,flac,ogg,m4a,aac,alac,wma').split(',')
        return file_path.suffix.lower() in [ext.lower() for ext in allowed_extensions]

# Global configuration instance
config_manager = ConfigurationManager()

# Export configuration values as module-level constants
MAX_FILE_SIZE_MB = config_manager.get('MAX_FILE_SIZE_MB', 500)
MAX_AUDIO_DURATION_MINUTES = config_manager.get('MAX_AUDIO_DURATION_MINUTES', 30)
ENABLE_GPU_ACCELERATION = config_manager.get('ENABLE_GPU_ACCELERATION', True)
LOG_LEVEL = config_manager.get('APP_LOG_LEVEL', 'INFO')

logger.debug("Configuration manager initialized successfully")
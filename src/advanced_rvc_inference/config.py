"""
Centralized Configuration Management for Advanced RVC Inference
Professional Singleton Pattern with Type Hints and Validation
"""

import os
import json
import logging
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from functools import lru_cache


class DeviceType(Enum):
    """Supported device types for computation."""
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    MPS = "mps"


class AudioFormat(Enum):
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"
    M4A = "m4a"
    AAC = "aac"


@dataclass
class GPUConfig:
    """GPU-specific configuration."""
    device: str = "auto"
    memory_fraction: float = 0.8
    allow_growth: bool = True
    mixed_precision: bool = True
    benchmark: bool = True


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 44100
    bit_depth: int = 16
    channels: int = 1
    chunk_size: int = 1024
    normalization: bool = True
    noise_reduction: bool = True
    format: AudioFormat = AudioFormat.WAV


@dataclass
class ModelConfig:
    """Model-specific configuration."""
    f0_method: str = "rmvpe"  # rmvpe, crepe, parselmouth
    pitch_extractor: str = "rmvpe"
    model_name: str = "hubert_base"
    chunk_length: int = 32
    pad_seconds: float = 0.04
    pred_seconds: float = 0.6
    filter_radius: int = 3
    rms_mix_rate: float = 0.25
    protect: float = 0.33
    hop_length: int = 160
    stereo: bool = False


@dataclass
class TrainingConfig:
    """Training-specific configuration."""
    batch_size: int = 4
    learning_rate: float = 1e-4
    epochs: int = 1000
    early_stopping_patience: int = 50
    validation_split: float = 0.1
    shuffle_buffer: int = 1000
    preprocess_threads: int = 8


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    max_threads: int = 4
    memory_limit_gb: float = 8.0
    cache_size_mb: int = 1000
    enable_mixed_precision: bool = True
    enable_compile: bool = True
    pin_memory: bool = True
    non_blocking: bool = True


class Config:
    """
    Professional Singleton Configuration Manager
    Thread-safe with comprehensive validation and type hints
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, config_file: Optional[str] = None):
        """Implement singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration with validation."""
        if self._initialized:
            return
            
        self._config_file = Path(config_file) if config_file else Path("config.json")
        self._logger = logging.getLogger(__name__)
        
        # Load default configurations
        self.gpu_config = GPUConfig()
        self.audio_config = AudioConfig()
        self.model_config = ModelConfig()
        self.training_config = TrainingConfig()
        self.performance_config = PerformanceConfig()
        
        # Application settings
        self.app_config = {
            "name": "Advanced RVC Inference",
            "version": "3.4.0",
            "debug": False,
            "log_level": "INFO",
            "theme": "gradio/default",
            "language": "en_US"
        }
        
        # Server settings
        self.server_config = {
            "host": "0.0.0.0",
            "port": 7860,
            "share": False,
            "show_error": True,
            "inbrowser": True
        }
        
        # Load configuration from file
        self._load_config()
        
        # Validate and setup
        self._validate_config()
        self._setup_directories()
        self._setup_gpu()
        
        self._initialized = True
        self._logger.info("Configuration initialized successfully")
    
    def _load_config(self) -> None:
        """Load configuration from JSON file."""
        if not self._config_file.exists():
            self._logger.info(f"Config file not found, using defaults: {self._config_file}")
            return
        
        try:
            with open(self._config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Update configurations from file
            if 'gpu' in config_data:
                self.gpu_config = GPUConfig(**config_data['gpu'])
            if 'audio' in config_data:
                self.audio_config = AudioConfig(**config_data['audio'])
            if 'model' in config_data:
                self.model_config = ModelConfig(**config_data['model'])
            if 'training' in config_data:
                self.training_config = TrainingConfig(**config_data['training'])
            if 'performance' in config_data:
                self.performance_config = PerformanceConfig(**config_data['performance'])
            if 'app' in config_data:
                self.app_config.update(config_data['app'])
            if 'server' in config_data:
                self.server_config.update(config_data['server'])
            
            self._logger.info(f"Configuration loaded from: {self._config_file}")
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self._logger.warning(f"Failed to load configuration: {e}. Using defaults.")
    
    def _validate_config(self) -> None:
        """Validate configuration values."""
        # Validate GPU configuration
        if not 0.1 <= self.gpu_config.memory_fraction <= 1.0:
            self._logger.warning("GPU memory fraction out of range, using default")
            self.gpu_config.memory_fraction = 0.8
        
        # Validate audio configuration
        if not 8000 <= self.audio_config.sample_rate <= 192000:
            self._logger.warning("Sample rate out of range, using default")
            self.audio_config.sample_rate = 44100
        
        # Validate model configuration
        valid_f0_methods = ["rmvpe", "crepe", "parselmouth"]
        if self.model_config.f0_method not in valid_f0_methods:
            self._logger.warning(f"Invalid F0 method: {self.model_config.f0_method}")
            self.model_config.f0_method = "rmvpe"
        
        # Validate training configuration
        if self.training_config.batch_size <= 0:
            self._logger.warning("Invalid batch size, using default")
            self.training_config.batch_size = 4
        
        if self.training_config.learning_rate <= 0:
            self._logger.warning("Invalid learning rate, using default")
            self.training_config.learning_rate = 1e-4
    
    def _setup_directories(self) -> None:
        """Setup required directories."""
        directories = [
            "weights",
            "indexes", 
            "logs",
            "cache",
            "temp",
            "audio_files",
            "outputs"
        ]
        
        for dir_name in directories:
            dir_path = Path(dir_name)
            dir_path.mkdir(exist_ok=True)
    
    def _setup_gpu(self) -> None:
        """Setup GPU configuration."""
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            self._logger.info(f"CUDA available with {device_count} GPUs")
            
            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory
                self._logger.info(f"GPU {i}: {gpu_name} ({memory_total / 1024**3:.1f} GB)")
            
            # Auto-detect optimal batch size based on GPU memory
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory_gb >= 24:  # A100
                self.training_config.batch_size = 8
            elif gpu_memory_gb >= 16:  # V100
                self.training_config.batch_size = 6
            elif gpu_memory_gb >= 8:   # T4
                self.training_config.batch_size = 4
            else:  # Smaller GPUs
                self.training_config.batch_size = 2
            
            self._logger.info(f"Auto-configured batch size: {self.training_config.batch_size}")
        else:
            self._logger.info("CUDA not available, using CPU")
    
    def get_device(self) -> DeviceType:
        """Get the optimal device for computation."""
        if torch.cuda.is_available():
            return DeviceType.CUDA
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return DeviceType.MPS
        elif torch.backends.rocm.is_available():
            return DeviceType.ROCM
        else:
            return DeviceType.CPU
    
    def get_device_string(self) -> str:
        """Get device string for PyTorch."""
        device = self.get_device()
        if device == DeviceType.CUDA:
            return "cuda"
        elif device == DeviceType.MPS:
            return "mps"
        elif device == DeviceType.ROCM:
            return "rocm"
        else:
            return "cpu"
    
    @lru_cache(maxsize=128)
    def get_model_path(self, model_name: str) -> Path:
        """Get model path with caching."""
        return Path("weights") / f"{model_name}.pth"
    
    @lru_cache(maxsize=128)
    def get_index_path(self, model_name: str) -> Path:
        """Get index path with caching."""
        return Path("indexes") / f"{model_name}.index"
    
    def get_cache_path(self, cache_name: str) -> Path:
        """Get cache path."""
        return Path("cache") / f"{cache_name}.cache"
    
    def get_log_path(self, log_name: str) -> Path:
        """Get log path."""
        return Path("logs") / f"{log_name}.log"
    
    def save_config(self, config_file: Optional[str] = None) -> None:
        """Save current configuration to file."""
        save_path = Path(config_file) if config_file else self._config_file
        
        config_data = {
            'gpu': asdict(self.gpu_config),
            'audio': asdict(self.audio_config),
            'model': asdict(self.model_config),
            'training': asdict(self.training_config),
            'performance': asdict(self.performance_config),
            'app': self.app_config,
            'server': self.server_config
        }
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            self._logger.info(f"Configuration saved to: {save_path}")
        except Exception as e:
            self._logger.error(f"Failed to save configuration: {e}")
    
    def update_config(self, section: str, key: str, value: Any) -> None:
        """Update configuration value."""
        if hasattr(self, f"{section}_config"):
            config_obj = getattr(self, f"{section}_config")
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
                self._logger.debug(f"Updated {section}.{key} = {value}")
            else:
                self._logger.warning(f"Unknown config key: {section}.{key}")
        else:
            self._logger.warning(f"Unknown config section: {section}")
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults."""
        self.__init__(str(self._config_file))
        self._logger.info("Configuration reset to defaults")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        device = self.get_device()
        gpu_memory_gb = 0
        
        if device == DeviceType.CUDA:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_utilization = torch.cuda.utilization(0) if hasattr(torch.cuda, 'utilization') else "N/A"
        else:
            gpu_utilization = "N/A"
        
        return {
            'device': device.value,
            'device_string': self.get_device_string(),
            'gpu_memory_gb': gpu_memory_gb,
            'gpu_utilization': gpu_utilization,
            'batch_size': self.training_config.batch_size,
            'sample_rate': self.audio_config.sample_rate,
            'mixed_precision': self.performance_config.enable_mixed_precision,
            'compile_enabled': self.performance_config.enable_compile
        }
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"Config(device={self.get_device().value}, batch_size={self.training_config.batch_size}, sample_rate={self.audio_config.sample_rate})"


# Global configuration instance
config = Config()


# Convenience functions for common configurations
def get_config() -> Config:
    """Get global configuration instance."""
    return config


def get_device() -> str:
    """Get optimal device string."""
    return config.get_device_string()


def get_batch_size() -> int:
    """Get optimal batch size."""
    return config.training_config.batch_size


def get_sample_rate() -> int:
    """Get sample rate."""
    return config.audio_config.sample_rate


def is_gpu_available() -> bool:
    """Check if GPU is available."""
    return config.get_device() != DeviceType.CPU


def enable_memory_efficient_attention() -> bool:
    """Check if memory efficient attention should be enabled."""
    return is_gpu_available() and config.performance_config.enable_mixed_precision


# Export type hints for other modules
__all__ = [
    'Config', 'GPUConfig', 'AudioConfig', 'ModelConfig', 
    'TrainingConfig', 'PerformanceConfig', 'DeviceType', 'AudioFormat',
    'config', 'get_config', 'get_device', 'get_batch_size', 
    'get_sample_rate', 'is_gpu_available', 'enable_memory_efficient_attention'
]
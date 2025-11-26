# Advanced RVC Inference - Configuration Guide

**Version:** 4.0.0  
**Authors:** ArkanDash & BF667

This guide provides comprehensive information about configuring Advanced RVC Inference for optimal performance.

## Configuration Files

### Main Configuration (`config.json`)

The primary configuration file located in the project root:

```json
{
    "project": {
        "name": "Advanced RVC Inference",
        "version": "4.0.0",
        "authors": ["ArkanDash", "BF667"]
    },
    "paths": {
        "model_dir": "assets/models/",
        "weight_dir": "assets/weights/",
        "log_dir": "logs/",
        "temp_dir": "temp/",
        "cache_dir": ".cache/",
        "config_dir": "advanced_rvc_inference/rvc/configs/",
        "theme_dir": "assets/themes/",
        "i18n_dir": "assets/i18n/"
    },
    "audio": {
        "default_sample_rate": 44100,
        "default_bit_depth": 16,
        "supported_formats": ["wav", "mp3", "flac", "ogg", "m4a", "aac"],
        "max_duration": 600,
        "buffer_size": 1024
    },
    "gpu": {
        "enabled": true,
        "device_id": 0,
        "precision": "float16",
        "memory_fraction": 0.8,
        "allow_memory_growth": true,
        "cudnn_benchmark": true
    },
    "krvc": {
        "enabled": true,
        "type": "v2",
        "optimization_level": "balanced",
        "enable_mixed_precision": true,
        "enable_memory_optimization": true,
        "compile_optimization": "max-autotune"
    },
    "models": {
        "auto_discover": true,
        "validation_required": true,
        "cache_models": true,
        "max_models_in_memory": 3,
        "fallback_model": "contentvec"
    },
    "performance": {
        "batch_size": 1,
        "num_workers": 4,
        "pin_memory": true,
        "persistent_workers": true,
        "torch_compile": true,
        "enable_profiling": false
    },
    "audio_processing": {
        "enhancer": {
            "enabled": false,
            "type": "basic",
            "strength": 0.5
        },
        "autotune": {
            "enabled": false,
            "strength": 0.5,
            "speed": 0.1,
            "threshold": 0.1
        },
        "effects": {
            "reverb": {
                "enabled": false,
                "room_size": 0.3,
                "damping": 0.5,
                "wet_level": 0.1
            },
            "compression": {
                "enabled": false,
                "ratio": 4.0,
                "threshold": 0.5,
                "attack": 0.003,
                "release": 0.1
            }
        }
    },
    "logging": {
        "level": "INFO",
        "file": "logs/application.log",
        "max_size": "10MB",
        "backup_count": 5,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    },
    "ui": {
        "theme": "default",
        "language": "en",
        "show_advanced_options": false,
        "auto_save_settings": true,
        "notification_timeout": 3000
    },
    "development": {
        "debug": false,
        "profiling": false,
        "benchmarking": false,
        "save_intermediate": false
    }
}
```

### ZLUDA Configuration (`config_zluda.json`)

Configuration for AMD GPU support via ZLUDA:

```json
{
    "zluda": {
        "enabled": false,
        "device_type": "gpu",
        "platform": "opencl"
    },
    "amd_gpu": {
        "rocblas_enabled": true,
        "hipfft_enabled": true,
        "rocprims_enabled": true,
        "miopen_enabled": true,
        "memory_pool_size": 4096
    },
    "opencl": {
        "device_type": "gpu",
        "context_properties": ["CL_CONTEXT_PLATFORM"],
        "work_group_size": 256,
        "max_work_items": [1024, 1024, 1024]
    },
    "performance": {
        "use_host_pointer": false,
        "use_unified_memory": true,
        "cache_kernels": true,
        "parallel_copy": true
    },
    "compatibility": {
        "cuda_api_stubs": true,
        "cuda_runtime_stubs": true,
        "cublas_stubs": true,
        "cusparse_stubs": true
    }
}
```

## Environment Variables

### GPU Configuration

```bash
# CUDA device selection
export CUDA_VISIBLE_DEVICES=0

# PyTorch CUDA settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# Memory optimization
export PYTORCH_CUDA_MEMORY_FRACTION=0.8
export PYTORCH_MPS_PREFER_METAL=1  # For Apple Silicon
```

### PyTorch Optimization

```bash
# JIT compilation
export PYTORCH_JIT=1
export TORCH_COMPILE_DEBUG=1

# Performance tuning
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# Benchmark settings
export TORCH_BENCHMARK=1
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
```

### Path Configuration

```bash
# Custom path overrides
export RVC_MODEL_DIR=/custom/models
export RVC_WEIGHT_DIR=/custom/weights
export RVC_LOG_DIR=/custom/logs
export RVC_TEMP_DIR=/custom/temp
export RVC_CACHE_DIR=/custom/cache

# Audio directory
export RVC_AUDIO_DIR=/custom/audio
export RVC_DATASET_DIR=/custom/dataset
```

### Development Settings

```bash
# Debug and profiling
export RVC_DEBUG=0
export RVC_PROFILE=0
export RVC_SAVE_INTERMEDIATE=0

# Logging
export RVC_LOG_LEVEL=INFO
export RVC_LOG_FILE=/custom/logs/app.log

# Performance monitoring
export RVC_ENABLE_METRICS=1
export RVC_BENCHMARK_MODE=0
```

## Runtime Configuration

### Using Configuration Manager

```python
from advanced_rvc_inference.core import get_config, save_config
from pathlib import Path

# Get current configuration
config = get_config()

# Modify configuration
config['gpu']['precision'] = 'float32'
config['krvc']['optimization_level'] = 'maximum'
config['performance']['batch_size'] = 2

# Save configuration
save_config(config)

# Load from file
config_path = Path('custom_config.json')
custom_config = load_config_from_file(config_path)
```

### Dynamic Configuration

```python
# Configure GPU settings dynamically
from advanced_rvc_inference import gpu_settings, gpu_optimizer

# Get optimal GPU settings
gpu_settings = gpu_settings(
    batch_size=4,
    precision="float16",
    memory_fraction=0.9
)

# Apply GPU optimization
optimizer = gpu_optimizer()
optimizer.configure(**gpu_settings)
```

### KRVC Configuration

```python
# Configure KRVC optimizations
from advanced_rvc_inference import (
    krvc_speed_optimize,
    krvc_inference_mode,
    krvc_mixed_precision_training
)

# Apply optimizations
krvc_speed_optimize()
krvc_inference_mode()
krvc_mixed_precision_training(enable=True, opt_level="O2")
```

## Performance Tuning

### GPU Optimization

#### Memory Management

```python
# Configure memory optimization
config = {
    'gpu': {
        'enabled': True,
        'device_id': 0,
        'precision': 'float16',
        'memory_fraction': 0.8,
        'allow_memory_growth': True,
        'max_memory_growth': '2GB'
    }
}

# Clear GPU memory
import torch
torch.cuda.empty_cache()

# Monitor memory usage
def monitor_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return allocated, reserved
    return 0, 0
```

#### Precision Settings

```python
# Precision configuration
precision_configs = {
    'float32': {
        'precision': 'float32',
        'memory_usage': 'high',
        'quality': 'maximum',
        'performance': 'standard'
    },
    'float16': {
        'precision': 'float16',
        'memory_usage': 'medium',
        'quality': 'high',
        'performance': 'fast'
    },
    'bfloat16': {
        'precision': 'bfloat16',
        'memory_usage': 'medium',
        'quality': 'high',
        'performance': 'fast'
    },
    'int8': {
        'precision': 'int8',
        'memory_usage': 'low',
        'quality': 'medium',
        'performance': 'maximum'
    }
}

# Choose precision based on use case
def select_precision(use_case='balanced'):
    if use_case == 'quality':
        return 'float32'
    elif use_case == 'performance':
        return 'float16'
    elif use_case == 'memory':
        return 'int8'
    else:  # balanced
        return 'float16'
```

### Batch Processing Configuration

```python
# Batch size optimization
def optimize_batch_size(
    available_memory_gb: float,
    model_size_mb: float,
    audio_duration: float = 10.0
) -> int:
    """
    Calculate optimal batch size based on available resources.
    
    Args:
        available_memory_gb: Available GPU memory in GB
        model_size_mb: Model size in MB
        audio_duration: Average audio duration in seconds
        
    Returns:
        Optimal batch size
    """
    # Reserve 20% for system overhead
    usable_memory = available_memory_gb * 0.8
    
    # Estimate memory per batch item (rough approximation)
    memory_per_item = model_size_mb * 2  # Model + activations
    memory_per_item += audio_duration * 0.1  # Audio data
    
    # Calculate batch size
    max_batch_size = int((usable_memory * 1024) / memory_per_item)
    
    # Bound batch size
    return max(1, min(max_batch_size, 8))
```

### Model Caching Configuration

```python
# Model caching settings
cache_config = {
    'enabled': True,
    'max_models': 3,
    'eviction_policy': 'LRU',  # Least Recently Used
    'cache_size_gb': 2.0,
    'preload_models': [],
    'validate_models': True
}

# Configure model caching
from advanced_rvc_inference.lib.utils import configure_model_cache
configure_model_cache(**cache_config)
```

## Advanced Configuration

### Plugin Configuration

```python
# Plugin system configuration
plugin_config = {
    'enabled': True,
    'auto_load': True,
    'safe_mode': True,
    'timeout': 30,
    'allowed_operations': [
        'audio_processing',
        'model_loading',
        'feature_extraction'
    ]
}

# Load plugin configuration
from advanced_rvc_inference.tabs.plugins import configure_plugins
configure_plugins(plugin_config)
```

### Real-time Configuration

```python
# Real-time processing settings
realtime_config = {
    'buffer_size': 1024,
    'target_latency': 0.05,  # 50ms
    'max_latency': 0.1,      # 100ms
    'dropout_threshold': 0.9,
    'buffer_overflow_policy': 'drop_oldest',
    'threading': True,
    'num_threads': 4
}

# Configure real-time processing
from advanced_rvc_inference import KRVCRealTimeProcessor
processor = KRVCRealTimeProcessor(**realtime_config)
```

### Audio Processing Configuration

```python
# Audio pipeline configuration
audio_config = {
    'sample_rate': 44100,
    'channels': 1,
    'bit_depth': 16,
    'buffer_size': 1024,
    'preprocessing': {
        'normalize': True,
        'denoise': False,
        'trim_silence': True,
        'format_convert': 'auto'
    },
    'postprocessing': {
        'normalize': True,
        'compressor': False,
        'limiter': False,
        'fade_in': 0.01,
        'fade_out': 0.01
    }
}
```

## Configuration Validation

### Configuration Schema

```python
from typing import Dict, Any
from pydantic import BaseModel, validator

class RVCConfig(BaseModel):
    """Configuration schema for RVC settings."""
    
    gpu: Dict[str, Any]
    krvc: Dict[str, Any]
    performance: Dict[str, Any]
    
    @validator('gpu')
    def validate_gpu(cls, v):
        if v.get('enabled', False):
            assert v.get('device_id', 0) >= 0, "Device ID must be non-negative"
            assert 0 < v.get('memory_fraction', 0.8) <= 1, "Memory fraction must be in (0, 1]"
        return v
    
    @validator('krvc')
    def validate_krvc(cls, v):
        assert v.get('type') in ['v1', 'v2', 'custom'], "Invalid KRVC type"
        assert v.get('optimization_level') in ['speed', 'memory', 'balanced'], "Invalid optimization level"
        return v

# Validate configuration
try:
    config = RVCConfig(**config_dict)
    print("Configuration is valid")
except Exception as e:
    print(f"Configuration error: {e}")
```

### Configuration Testing

```python
# Test configuration settings
def test_config(config: Dict[str, Any]) -> Dict[str, bool]:
    """Test configuration settings."""
    results = {}
    
    # Test GPU settings
    try:
        import torch
        results['gpu'] = torch.cuda.is_available() if config.get('gpu', {}).get('enabled') else True
    except:
        results['gpu'] = False
    
    # Test memory settings
    try:
        import psutil
        available_memory = psutil.virtual_memory().available / (1024**3)
        required_memory = config.get('performance', {}).get('batch_size', 1) * 2
        results['memory'] = available_memory > required_memory
    except:
        results['memory'] = False
    
    # Test disk space
    try:
        import shutil
        free_space = shutil.disk_usage('.').free / (1024**3)
        results['disk_space'] = free_space > 5  # 5GB minimum
    except:
        results['disk_space'] = False
    
    return results
```

## Configuration Examples

### High Performance Setup

```json
{
    "gpu": {
        "enabled": true,
        "device_id": 0,
        "precision": "float16",
        "memory_fraction": 0.9,
        "allow_memory_growth": true
    },
    "krvc": {
        "enabled": true,
        "type": "v2",
        "optimization_level": "speed",
        "compile_optimization": "max-autotune"
    },
    "performance": {
        "batch_size": 4,
        "num_workers": 8,
        "torch_compile": true,
        "enable_profiling": false
    }
}
```

### Memory-Constrained Setup

```json
{
    "gpu": {
        "enabled": true,
        "device_id": 0,
        "precision": "float16",
        "memory_fraction": 0.6
    },
    "krvc": {
        "enabled": true,
        "type": "v1",
        "optimization_level": "memory",
        "enable_mixed_precision": true,
        "enable_memory_optimization": true
    },
    "performance": {
        "batch_size": 1,
        "num_workers": 2,
        "memory_efficient": true
    }
}
```

### Quality-Focused Setup

```json
{
    "gpu": {
        "enabled": true,
        "precision": "float32"
    },
    "krvc": {
        "enabled": true,
        "type": "v2",
        "optimization_level": "balanced"
    },
    "audio_processing": {
        "enhancer": {
            "enabled": true,
            "type": "advanced",
            "strength": 0.7
        },
        "autotune": {
            "enabled": true,
            "strength": 0.3
        }
    }
}
```

## Troubleshooting Configuration

### Common Configuration Issues

1. **GPU Memory Issues**
   - Reduce `memory_fraction` to 0.6-0.7
   - Set `batch_size` to 1
   - Enable `memory_optimization`

2. **Performance Issues**
   - Set `precision` to `float16`
   - Enable `torch_compile`
   - Increase `batch_size` if memory allows

3. **Quality Issues**
   - Use `float32` precision
   - Disable aggressive optimizations
   - Enable audio enhancement

### Configuration Recovery

```python
# Reset to default configuration
def reset_to_defaults():
    from pathlib import Path
    import shutil
    
    default_config = Path('config_default.json')
    current_config = Path('config.json')
    
    if current_config.exists():
        backup_path = current_config.with_suffix('.json.backup')
        shutil.copy2(current_config, backup_path)
        print(f"Backup created: {backup_path}")
    
    if default_config.exists():
        shutil.copy2(default_config, current_config)
        print("Configuration reset to defaults")

# Validate and fix configuration
def fix_configuration():
    try:
        config = get_config()
        validated_config = validate_and_fix_config(config)
        save_config(validated_config)
        print("Configuration fixed")
    except Exception as e:
        print(f"Configuration fix failed: {e}")
```

## Best Practices

1. **Start with defaults** and adjust incrementally
2. **Monitor performance** when changing settings
3. **Test configurations** before applying to production
4. **Document custom settings** for reproducibility
5. **Use environment variables** for deployment-specific settings
6. **Validate configurations** regularly
7. **Keep backups** of working configurations

---

*Configuration guide maintained by ArkanDash & BF667 for Advanced RVC Inference V4.0.0*
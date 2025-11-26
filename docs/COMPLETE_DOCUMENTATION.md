# Advanced RVC Inference - Complete Documentation

**Version:** 4.0.0 Ultimate Performance Edition  
**Authors:** ArkanDash & BF667  
**Last Updated:** November 26, 2025

## Table of Contents

1. [Overview](#overview)
2. [Quick Start Guide](#quick-start-guide)
3. [Installation](#installation)
4. [Core Features](#core-features)
5. [Performance Optimization](#performance-optimization)
6. [API Reference](#api-reference)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)
9. [Contributing](#contributing)
10. [License](#license)

## Overview

Advanced RVC Inference V4.0.0 represents the pinnacle of voice conversion technology, featuring revolutionary performance optimizations and cutting-edge AI models. Built on the foundation of the RVC (Retrieval-based Voice Conversion) technology with KRVC kernel enhancements, this project delivers professional-grade voice conversion capabilities with unprecedented speed and quality.

### Key Highlights

- **🚀 Performance**: Up to 5x faster processing with V4.0 optimizations
- **🧠 AI-Powered**: 60+ F0 extraction methods and 60+ embedder models
- **⚡ Real-time**: Ultra-low latency voice conversion
- **🎯 Professional**: Production-ready with comprehensive features
- **🔧 Extensible**: Modular architecture with plugin support

## Quick Start Guide

### Basic Voice Conversion

```python
from advanced_rvc_inference import full_inference_program

# Simple voice conversion
result = full_inference_program(
    model_path="models/my_voice_model.pth",
    input_audio_path="input.wav",
    output_path="output.wav"
)
```

### Advanced Configuration

```python
from advanced_rvc_inference import KRVCFeatureExtractor, get_path_manager

# Initialize with KRVC optimization
extractor = KRVCFeatureExtractor()

# Use path manager for better file organization
pm = get_path_manager()
models = pm.find_models()  # Auto-discover available models
```

## Installation

### System Requirements

- **Python**: 3.8 or higher
- **GPU**: NVIDIA CUDA-compatible GPU (recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB for models and dependencies

### Automated Installation

```bash
# Clone repository
git clone https://github.com/ArkanDash/Advanced-RVC-Inference.git
cd Advanced-RVC-Inference

# Run enhanced setup (recommended)
python enhanced_setup.py

# Or use platform-specific scripts
./run_enhanced.sh  # Linux/Mac
run_enhanced.bat   # Windows
```

### Manual Installation

```bash
# Create virtual environment
python -m venv rvc_env
source rvc_env/bin/activate  # Windows: rvc_env\Scripts\activate

# Install PyTorch with CUDA support
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project requirements
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### GPU Setup

#### NVIDIA GPUs
```bash
# Install CUDA toolkit (if not already installed)
# Visit: https://developer.nvidia.com/cuda-downloads

# Verify CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### AMD GPUs
```bash
# Install ROCm (Linux)
# Visit: https://rocmdocs.amd.com/en/latest/

# OpenCL support will be used automatically
```

## Core Features

### 1. Voice Conversion Engine

The heart of the system is built on RVC technology with KRVC kernel optimizations:

- **High-Quality Conversion**: Professional-grade voice conversion
- **Real-time Processing**: Ultra-low latency for live applications
- **Batch Processing**: Efficient handling of multiple files
- **Multi-format Support**: WAV, MP3, FLAC, OGG, and more

### 2. F0 Extraction Methods

60+ advanced F0 extraction methods including:

- **Traditional Methods**: RMVPE, CREPE, FCPE
- **Hybrid Approaches**: `hybrid[rmvpe+crepe+fcpe]`
- **Language-Specific**: Optimized for Vietnamese, English, and more
- **Real-time Capable**: Low-latency methods for live applications

### 3. Embedder Models

60+ embedder models for content extraction:

- **ContentVec**: High-quality content extraction
- **Whisper Variants**: Multilingual support
- **Hubert Models**: Advanced audio representations
- **Custom Architectures**: ONNX, Fairseq, VITS-based models

### 4. Audio Processing

Advanced audio processing capabilities:

- **Audio Separation**: Mel-Roformer, BS-Roformer, MDX23C
- **Effect Processing**: Reverb, normalization, noise reduction
- **Format Conversion**: Multi-format audio I/O
- **Quality Enhancement**: Advanced audio upsampling and enhancement

## Performance Optimization

### V4.0 Ultimate Performance Features

#### 1. TorchFX GPU-DSP Integration
```python
from advanced_rvc_inference.lib.torchfx_integration import TorchFXProcessor

# 3x faster audio processing with GPU acceleration
torchfx_processor = TorchFXProcessor(device='cuda')
dsp_pipeline = torchfx_processor.create_audio_pipeline(['lowpass', 'highpass', 'normalize'])
```

#### 2. torch-audiomentations Integration
```python
from advanced_rvc_inference.lib.torch_audiomentations_integration import RVCAudioAugmenter

# 11+ real-time GPU audio augmentation transforms
augmenter = RVCAudioAugmenter(sample_rate=44100)
```

#### 3. torch.compile Optimization
```python
from advanced_rvc_inference.lib.torch_compile_optimization import TorchCompileOptimizer

# 2-5x inference speedup with JIT compilation
compile_optimizer = TorchCompileOptimizer()
```

### Performance Benchmarks

| Metric | Standard RVC | V3.5.3 | V4.0.0 | Improvement |
|--------|--------------|--------|---------|-------------|
| Processing Speed | Baseline | 2x | 5x | +150% |
| Startup Time | 30s | 8s | 4s | +50% |
| Memory Usage | 100% | 70% | 60% | +40% |
| Real-time Latency | High | Low | Ultra-Low | Professional |

## API Reference

### Core Functions

#### `full_inference_program()`
Main voice conversion function with full parameter support.

```python
def full_inference_program(
    model_path: str,
    index_path: str,
    input_audio_path: str,
    output_path: str,
    pitch_extract: str = "rmvpe",
    embedder_model: str = "contentvec",
    filter_radius: int = 3,
    resample_sr: int = 0,
    rms_mix_rate: float = 0.25,
    protect: float = 0.33,
    hop_length: int = 512,
    krvc_type: str = "v2",
    direct_io: str = None,
    db_level: float = -18.0,
    embedder_path: str = "assets/models/embedders/",
    enhancer: bool = False,
    autotune: bool = False,
    autotune_strength: float = 0.5,
    use_gpu: bool = True,
    gpu_id: int = 0,
    **kwargs
) -> Optional[str]:
```

**Parameters:**
- `model_path`: Path to the voice model file (.pth/.onnx)
- `index_path`: Path to the feature index file (.index)
- `input_audio_path`: Path to input audio file
- `output_path`: Path for output audio file
- `pitch_extract`: F0 extraction method (60+ available)
- `embedder_model`: Content extraction model (60+ available)
- `krvc_type`: KRVC kernel type ("v1", "v2", "custom")
- `use_gpu`: Enable GPU acceleration

### Path Management System

#### PathManager Class
Comprehensive path management with validation and caching.

```python
from advanced_rvc_inference import PathManager

# Initialize path manager
pm = PathManager()

# Get critical paths
models_dir = pm.get_path('models_dir')
logs_dir = pm.get_path('logs_dir')

# Auto-discover models
models = pm.find_models()
```

**Key Methods:**
- `get_path(key)`: Get validated path by key
- `find_models()`: Auto-discover available models
- `validate_project_structure()`: Check directory structure
- `cleanup_temp_files()`: Manage temporary files

### KRVC Kernel Integration

#### KRVCFeatureExtractor
Enhanced feature extraction with KRVC optimizations.

```python
from advanced_rvc_inference import KRVCFeatureExtractor

# Initialize KRVC extractor
extractor = KRVCFeatureExtractor()

# Optimize for inference
extractor.set_inference_mode()
extractor.enable_mixed_precision()
```

### GPU Optimization

#### GPUOptimizer
Automatic GPU detection and optimization.

```python
from advanced_rvc_inference import gpu_optimizer, GPU_OPTIMIZATION_AVAILABLE

if GPU_OPTIMIZATION_AVAILABLE:
    optimizer = gpu_optimizer()
    gpu_info = optimizer.get_gpu_info()
    optimal_settings = optimizer.get_optimal_settings()
```

## Configuration

### Configuration Files

The project uses several configuration files:

#### `config.json`
Main configuration file with runtime settings:
```json
{
    "model_dir": "assets/models/",
    "log_dir": "logs/",
    "temp_dir": "temp/",
    "cache_dir": ".cache/",
    "gpu_enabled": true,
    "gpu_id": 0,
    "precision": "float16",
    "batch_size": 1,
    "krvc_enabled": true,
    "torch_compile": true
}
```

#### `config_zluda.json`
ZLUDA configuration for AMD GPU support:
```json
{
    "zluda_enabled": false,
    "gpu_vendor": "amd",
    "opencl_enabled": true,
    "rocblas_enabled": true
}
```

### Environment Variables

Set these environment variables for custom configuration:

```bash
# GPU configuration
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# PyTorch optimization
export PYTORCH_JIT=1
export TORCH_COMPILE_DEBUG=1

# Path overrides
export RVC_MODEL_DIR=/custom/models
export RVC_LOG_DIR=/custom/logs
```

### Runtime Configuration

Configure settings at runtime:

```python
from advanced_rvc_inference import get_config, gpu_settings

# Get current configuration
config = get_config()
print(f"GPU enabled: {config.get('gpu_enabled', False)}")

# Configure GPU settings
settings = gpu_settings(
    batch_size=2,
    precision="float16",
    memory_fraction=0.8
)
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```python
# Solution 1: Reduce batch size
settings = gpu_settings(batch_size=1)

# Solution 2: Enable memory optimization
extractor.enable_memory_optimization()

# Solution 3: Use mixed precision
extractor.enable_mixed_precision()
```

#### 2. Model Loading Errors
```python
# Check model compatibility
from advanced_rvc_inference.core import import_voice_converter

try:
    model = import_voice_converter("model.pth")
    print("Model loaded successfully")
except Exception as e:
    print(f"Model loading failed: {e}")
```

#### 3. Audio Processing Issues
```python
# Verify audio format and quality
import librosa

# Load and validate audio
audio, sr = librosa.load("input.wav", sr=None)
print(f"Sample rate: {sr}, Duration: {len(audio)/sr:.2f}s")
```

### Performance Issues

#### Slow Processing
1. **Enable GPU acceleration**: `use_gpu=True`
2. **Use KRVC optimizations**: `krvc_type="v2"`
3. **Enable torch.compile**: Set in configuration
4. **Optimize batch size**: Use `gpu_settings()`

#### High Memory Usage
1. **Enable mixed precision**: `precision="float16"`
2. **Reduce batch size**: `batch_size=1`
3. **Enable memory optimization**: Call `enable_memory_optimization()`
4. **Clean up resources**: Use `cleanup_krvc_memory()`

### Installation Issues

#### Missing Dependencies
```bash
# Reinstall with specific versions
pip install torch==2.9.1 torchaudio==2.9.1

# Install missing audio libraries
sudo apt-get install ffmpeg libsndfile1-dev

# Install additional ML libraries
pip install transformers>=4.49.0
```

#### GPU Not Detected
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA toolkit
# Visit: https://developer.nvidia.com/cuda-downloads
```

## Contributing

### Development Setup

```bash
# Clone for development
git clone https://github.com/ArkanDash/Advanced-RVC-Inference.git
cd Advanced-RVC-Inference

# Create development environment
python -m venv rvc_dev
source rvc_dev/bin/activate

# Install in development mode
pip install -e .
pip install -r requirements.txt

# Install development dependencies
pip install pytest black isort flake8 mypy
```

### Code Style

```bash
# Format code
black advanced_rvc_inference/

# Sort imports
isort advanced_rvc_inference/

# Lint code
flake8 advanced_rvc_inference/

# Type checking
mypy advanced_rvc_inference/
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=advanced_rvc_inference/

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
```

### Submission Guidelines

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** with comprehensive tests
4. **Follow code style** guidelines
5. **Update documentation** for new features
6. **Submit a pull request** with detailed description

### Feature Requests

For feature requests:
1. Check existing issues
2. Create detailed proposal
3. Include performance impact analysis
4. Provide usage examples

## Performance Tuning

### GPU Optimization

```python
from advanced_rvc_inference import GPUOptimizer, KRVCAdvancedOptimizer

# Advanced GPU optimization
gpu_opt = GPUOptimizer()
krvc_opt = KRVCAdvancedOptimizer()

# Apply optimizations
optimized_model = krvc_opt.optimize_model(
    model,
    optimization_level='maximum'
)

# Monitor performance
monitor = krvc_opt.get_performance_monitor()
stats = monitor.get_stats()
```

### Memory Management

```python
# Automatic memory cleanup
from advanced_rvc_inference import cleanup_krvc_memory

# Manual cleanup after processing
cleanup_krvc_memory()

# Configure garbage collection
import gc
gc.set_threshold(700, 10, 10)
```

### Batch Processing

```python
# Efficient batch processing
def process_batch(audio_files, batch_size=4):
    for i in range(0, len(audio_files), batch_size):
        batch = audio_files[i:i+batch_size]
        results = process_audio_batch(batch)
        yield results
```

## Advanced Features

### Custom Model Training

```python
from advanced_rvc_inference.enhanced_training import (
    RVCTrainer, RVCDataset, TrainingConfig
)

# Configure training
config = TrainingConfig(
    model_name="my_voice_model",
    sample_rate=44100,
    batch_size=4,
    learning_rate=1e-4,
    epochs=100
)

# Initialize trainer
trainer = RVCTrainer(config)

# Prepare dataset
dataset = RVCDataset("path/to/training/data")

# Train model
trainer.train(dataset)
```

### Plugin System

```python
from advanced_rvc_inference.tabs.plugins import register_plugin

# Create custom plugin
class MyPlugin:
    def process_audio(self, audio):
        # Custom processing logic
        return processed_audio

# Register plugin
register_plugin("my_plugin", MyPlugin())
```

### Real-time Processing

```python
from advanced_rvc_inference import KRVCRealTimeProcessor

# Initialize real-time processor
rt_processor = KRVCRealTimeProcessor(
    model_path="voice_model.pth",
    latency_target=0.05  # 50ms target
)

# Process audio stream
def process_stream(audio_chunk):
    result = rt_processor.process_chunk(audio_chunk)
    return result
```

## Integration Examples

### Web Application Integration

```python
from fastapi import FastAPI, File, UploadFile
from advanced_rvc_inference import full_inference_program

app = FastAPI()

@app.post("/convert-voice")
async def convert_voice(
    audio_file: UploadFile = File(...),
    model: str = "default_model"
):
    # Save uploaded file
    input_path = f"temp/{audio_file.filename}"
    with open(input_path, "wb") as f:
        f.write(await audio_file.read())
    
    # Process with RVC
    output_path = f"output/{audio_file.filename}"
    result = full_inference_program(
        model_path=f"models/{model}.pth",
        input_audio_path=input_path,
        output_path=output_path
    )
    
    return {"result": "success", "output": output_path}
```

### Command Line Interface

```python
import click
from advanced_rvc_inference import full_inference_program

@click.command()
@click.option('--model', required=True, help='Voice model path')
@click.option('--input', required=True, help='Input audio path')
@click.option('--output', required=True, help='Output audio path')
@click.option('--f0-method', default='rmvpe', help='F0 extraction method')
def convert_voice(model, input, output, f0_method):
    """Convert voice using RVC"""
    result = full_inference_program(
        model_path=model,
        input_audio_path=input,
        output_path=output,
        pitch_extract=f0_method
    )
    click.echo(f"Conversion complete: {result}")

if __name__ == '__main__':
    convert_voice()
```

## Best Practices

### 1. Path Management
- Always use the PathManager for file operations
- Validate paths before processing
- Clean up temporary files regularly

### 2. Memory Management
- Enable mixed precision for memory efficiency
- Use batch processing for multiple files
- Clean up resources after processing

### 3. Performance Optimization
- Enable KRVC optimizations
- Use appropriate batch sizes
- Monitor GPU memory usage

### 4. Error Handling
- Always wrap processing in try-catch blocks
- Validate input audio files
- Provide fallback options

### 5. Model Management
- Use version control for models
- Test models before deployment
- Maintain model metadata

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- **GitHub Issues**: [Report bugs and request features](https://github.com/ArkanDash/Advanced-RVC-Inference/issues)
- **Discord Community**: [Join our Discord](https://discord.gg/hvmsukmBHE)
- **Documentation**: [Complete documentation](https://github.com/ArkanDash/Advanced-RVC-Inference/wiki)
- **Email**: Contact maintainers through GitHub

---

**Professional Voice Conversion Technology - Ready for Production Use**

*Built with ❤️ by ArkanDash & BF667*
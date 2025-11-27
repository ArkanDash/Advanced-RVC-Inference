# Advanced RVC Inference

<div align="center">

**Last Updated:** November 27, 2025

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArkanDash/Advanced-RVC-Inference/blob/master/notebooks/Advanced_RVC_Inference.ipynb)

</div>

**Web-based Graphical User Interface for RVC (Retrieval-based Voice Conversion)**

Advanced RVC Inference provides a user-friendly Gradio interface for voice conversion using RVC technology, based on the Applio project foundation and enhanced with KRVC kernel optimizations.

## Features

### Voice Conversion
- **Full Inference**: Convert audio files using trained RVC models
- **Real-time Voice Changing**: Live voice conversion with low latency
- **Model Training**: Interface for training RVC models
- **Vietnamese RVC Support**: Enhanced support for Vietnamese voice conversion

### Audio Tools
- **Audio Enhancement**: Post-processing tools for audio quality improvement
- **Model Management**: Download, organize, and manage RVC models
- **Music Downloader**: Built-in music downloading capabilities
- **Multiple Format Support**: WAV, MP3, FLAC, OGG, and more

### Performance Optimizations
- **KRVC Kernel**: Custom kernel optimizations for improved performance
- **GPU Acceleration**: CUDA support for faster processing
- **Memory Management**: Efficient GPU memory usage

## Quick Start

### Standard Installation

```bash
# Clone repository
git clone https://github.com/ArkanDash/Advanced-RVC-Inference.git
cd Advanced-RVC-Inference

# Create virtual environment
python -m venv rvc_env
source rvc_env/bin/activate  # Windows: rvc_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python -m advanced_rvc_inference.main
```

### Using Scripts

#### Windows
- `run.bat` - Launch the application
- `update.bat` - Update the application

#### Linux/Mac
- `run.sh` - Launch the application
- `update.sh` - Update the application

### Google Colab

Click the "Open in Colab" badge above to run in your browser without local installation.

### Docker Installation

```bash
# CPU version
docker pull advanced-rvc-inference:latest

# GPU version
docker pull advanced-rvc-inference:gpu

# Run container
docker run -p 7860:7860 -v $(pwd)/models:/app/models advanced-rvc-inference:latest
```

## Project Structure

The application provides several main features through tabs:

- **🎵 Inference**: Full inference and real-time voice conversion
- **📥 Downloader**: Download music and models
- **🎓 Training**: Train RVC models
- **🔧 Audio Tools**: Audio enhancement and processing tools
- **⚙️ Settings**: Configuration and theme selection

## Core Features

### RVC Voice Conversion
- Standard RVC inference using PyTorch models
- Support for multiple model formats (.pth, .onnx)
- Index file support for improved quality
- Batch processing capabilities

### KRVC Optimizations
- Custom kernel optimizations for better performance
- GPU memory management
- Mixed precision support
- Optimized for various NVIDIA GPUs

### Vietnamese RVC Integration
- Enhanced F0 extraction methods
- Vietnamese-specific voice conversion optimizations
- Support for Vietnamese language characteristics

### Model Management
- Integration with public model repositories
- Automatic model categorization
- One-click model downloads
- Model validation and integrity checking

## API Usage

### Basic Voice Conversion

```python
from advanced_rvc_inference.core import full_inference_program

# Basic inference example
result = full_inference_program(
    model_path="path/to/model.pth",
    index_path="path/to/index.index",
    input_audio_path="input.wav",
    output_path="output.wav",
    pitch_extract="rmvpe",  # F0 extraction method
    embedder_model="contentvec",   # Content embedding model
)
```

### KRVC Optimizations

```python
from advanced_rvc_inference.krvc_kernel import (
    krvc_inference_mode,
    krvc_training_mode,
    krvc_speed_optimize
)

# Enable KRVC optimizations
krvc_speed_optimize()
krvc_inference_mode()  # For inference
# or
krvc_training_mode()   # For training
```

### GPU Optimization

```python
from advanced_rvc_inference.gpu_optimization import get_gpu_optimizer

# Get GPU optimizer with automatic hardware detection
gpu_optimizer = get_gpu_optimizer()
gpu_info = gpu_optimizer.gpu_info
print(f"Detected GPU: {gpu_info['type']}")

# Get optimal settings for your hardware
optimal_settings = gpu_optimizer.get_optimal_settings()
print(f"Optimal batch size: {optimal_settings['batch_size']}")
```

## Requirements

- Python 3.10 or higher
- PyTorch 2.0 or higher
- CUDA-compatible GPU (recommended for optimal performance)
- 8GB+ RAM (16GB recommended for training)
- FFmpeg for audio processing

See `requirements.txt` for full dependency list.

## Model Setup

1. **Place Models**: Put .pth/.onnx files in the models directory
2. **Index Files**: Add corresponding .index files for better quality
3. **Automatic Detection**: Application will list all available models
4. **Model Metadata**: Models are automatically categorized and described

## Recent Updates

### Version Updates (November 27, 2025)

#### 🔧 Bug Fixes
- **Fixed ModuleNotFoundError**: Resolved import issues with theme loading
- **Corrected Relative Imports**: Fixed import path issues throughout the codebase
- **Path Resolution**: Improved asset import handling
- **Import Compatibility**: Enhanced module execution compatibility

#### 🚀 Code Improvements
- **Refactored Import System**: Better module organization
- **Enhanced Error Handling**: Improved error messages
- **Code Organization**: Better separation of concerns
- **Documentation Updates**: Updated README with accurate information

## Development

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/ArkanDash/Advanced-RVC-Inference.git
cd Advanced-RVC-Inference
python -m venv rvc_dev_env
source rvc_dev_env/bin/activate
pip install -e .
pip install -r requirements.txt

# Code quality tools (optional)
pip install black isort flake8 mypy pytest
black advanced_rvc_inference/
isort advanced_rvc_inference/
```

### Architecture Overview

- **Modular Design**: Clear separation of concerns
- **Gradio Interface**: Web-based GUI for easy access
- **KRVC Kernel**: Custom performance optimizations
- **Memory Management**: GPU memory optimization utilities
- **Type Hints**: Python type annotations where applicable
- **Error Handling**: Comprehensive exception management

## Documentation

### Available Guides

- **[Directory Structure](docs/directory_structure.md)**: Detailed folder organization
- **[API Usage](docs/api_usage.md)**: Python API documentation
- **[Troubleshooting](docs/troubleshooting.md)**: Common issues and solutions
- **[Configuration Guide](docs/configuration.md)**: Advanced setup options

### Quick Links

- **Local Installation**: `pip install -e .` then `python -m advanced_rvc_inference.main`
- **Colab Demo**: Click the Colab badge above
- **GitHub Issues**: [Bug reports and feature requests](https://github.com/ArkanDash/Advanced-RVC-Inference/issues)

## Performance Notes

### KRVC Kernel Benefits
- Custom optimized kernels for RVC operations
- Improved GPU memory efficiency
- Better performance on supported hardware
- Memory management utilities

### Expected Performance
- **Training**: Standard RVC training performance with KRVC optimizations
- **Inference**: Improved inference speed with GPU acceleration
- **Memory Usage**: Better GPU memory management compared to baseline RVC

## Citation & Credits

### Project Foundation

- **[Applio](https://github.com/IAHispano/Applio)**: Original project foundation and inspiration
- **[RVC Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)**: Core RVC technology
- **[Vietnamese-RVC](https://github.com/PhamHuynhAnh16/Vietnamese-RVC)**: Vietnamese RVC enhancements

### Maintainers & Authors

- **ArkanDash**: Project maintainer and developer
- **Contributors**: Various contributors to the RVC ecosystem

### License

MIT License - See [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/ArkanDash/Advanced-RVC-Inference/issues)
- **Documentation**: [Complete Guide](docs/)
- **Email**: Contact maintainers through GitHub

## Limitations and Disclaimers

- This project provides a GUI interface for RVC technology but is not the core RVC implementation
- Performance improvements are dependent on hardware and specific use cases
- Some features may require specific GPU hardware for optimal performance
- Training requires significant computational resources and time

## Contributing

Contributions are welcome! Please read the contributing guidelines and feel free to submit pull requests or open issues for bugs and feature requests.

---

*Advanced RVC Inference - Making RVC technology more accessible through an intuitive web interface*

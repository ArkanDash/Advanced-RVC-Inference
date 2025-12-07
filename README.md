# Advanced RVC Inference

Advanced RVC Inference presents itself as a state-of-the-art web UI crafted to streamline rapid and effortless inference. This comprehensive toolset encompasses a model downloader, a voice splitter, and the added efficiency of batch inference.

## Features

- Voice conversion with multiple pitch extraction methods
- Model training capabilities
- Batch inference support
- Text-to-speech integration
- Audio separation tools
- Web UI interface with Gradio

## Table of Contents

- [Installation](#installation)
- [Quick Start Guide](#quick-start-guide)
- [Using the Web UI](#using-the-web-ui)
- [Command Line Usage](#command-line-usage)
- [Development Setup](#development-setup)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Support](#support)

## Installation

```bash
# Clone the repository
git clone https://github.com/ArkanDash/Advanced-RVC-Inference.git
cd Advanced-RVC-Inference

# Install in development mode
pip install -e .
```

## Quick Start Guide

1. **Install the package** using one of the methods above.

2. **Download prerequisites**:
   ```bash
   python -c "from advanced_rvc_inference.core import run_prerequisites_script; run_prerequisites_script(pretraineds_hifigan=True, models=True, exe=True)"
   ```

3. **Launch the web interface**:
   ```bash
   python -m advanced_rvc_inference.app
   ```

4. Access the UI in your browser at the displayed URL (typically http://127.0.0.1:6969)

## Using the Web UI

The web interface provides an intuitive way to use all features:

1. **Voice Conversion**: Upload your source audio and target model
2. **Model Training**: Upload datasets and configure training parameters
3. **Batch Processing**: Process multiple files simultaneously
4. **Audio Analysis**: Analyze audio characteristics and quality

### Web UI Features

- **Real-time Preview**: Listen to results before saving
- **Parameter Adjustment**: Fine-tune pitch, tone, and other parameters
- **Progress Monitoring**: Track training and inference progress
- **Model Management**: Organize and manage your voice models



## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- [uv](https://github.com/astral-sh/uv) (optional but recommended)

### Development Workflow

1. Clone the repository:
   ```bash
   git clone https://github.com/ArkanDash/Advanced-RVC-Inference.git
   cd Advanced-RVC-Inference
   ```

2. Install in development mode:
   ```bash
   pip install -e .
   # or with uv:
   uv pip install -e .
   ```

3. Run the application:
   ```bash
   python -m advanced_rvc_inference.app
   ```




### Adding New Features

1. **Core Functions**: Add to appropriate modules in `core/`, `infer/`, or `library/`
2. **Web UI**: Create new tabs in `tabs/` directory
3. **CLI Commands**: Extend `parser.py` with new command definitions
4. **Configuration**: Update `variables.py` for new settings

## Troubleshooting

### Common Issues

**CUDA Out of Memory Errors:**
- Reduce batch size in training
- Use `cache_dataset_in_gpu=False`
- Use `checkpointing=True` for memory-efficient training

**Model Loading Issues:**
- Ensure model files (.pth, .index) are in the correct locations
- Check file permissions and paths
- Verify model compatibility with your installation

**Audio Format Issues:**
- Convert audio to supported formats (WAV, MP3, FLAC, etc.)
- Ensure correct sample rate (32000, 40000, 48000 Hz)
- Use preprocessing tools to normalize audio

**Web Interface Not Starting:**
- Check if the port is already in use
- Try using `--port` option with a different port
- Ensure all dependencies are installed

**Import Errors:**
- Ensure you're using the correct import structure: `from advanced_rvc_inference.xxx import ...`
- Check that the package is installed correctly
- Verify Python path and virtual environment activation

### Logging Issues
To get more detailed logs, you can modify logging levels by setting environment variables:
```bash
PYTHONPATH=. python -m advanced_rvc_inference.app
```

### Performance Optimization

- **GPU Memory**: Monitor GPU usage and adjust batch sizes accordingly
- **CPU Usage**: Use multiple CPU cores for preprocessing and feature extraction
- **Disk Space**: Ensure sufficient space for models and temporary files
- **Network**: Stable internet connection for model downloads

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests to ensure everything works
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Standards
- Use 4 spaces for indentation (not tabs)
- Follow PEP 8 style guide
- Write docstrings for public functions
- Include type hints where appropriate
- Add tests for new functionality


```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

If you encounter any issues, please open an issue on [GitHub](https://github.com/ArkanDash/Advanced-RVC-Inference/issues).

For questions and discussions, join our community:
- [GitHub Discussions](https://github.com/ArkanDash/Advanced-RVC-Inference/discussions)




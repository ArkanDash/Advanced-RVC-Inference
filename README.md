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

### Using pip

```bash
pip install advanced-rvc-inference
```

### Using uv (recommended)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
uv pip install -e .
```

### For development

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



## Command Line Usage

### Basic Inference
```bash
python -m advanced_rvc_inference.core infer \
  --input_path path/to/input.wav \
  --output_path path/to/output.wav \
  --pth_path path/to/model.pth \
  --index_path path/to/index.index \
  --pitch 12 \
  --f0_method rmvpe \
  --index_rate 0.5
```

### Batch Inference
```bash
python -m advanced_rvc_inference.core batch_infer \
  --input_folder path/to/input_folder \
  --output_folder path/to/output_folder \
  --pth_path path/to/model.pth \
  --index_path path/to/index.index \
  --pitch 12 \
  --f0_method rmvpe \
  --index_rate 0.5
```

### Training
```bash
# Preprocess
python -m advanced_rvc_inference.core preprocess \
  --model_name my_model \
  --dataset_path path/to/dataset \
  --sample_rate 40000 \
  --cpu_cores 4

# Extract features
python -m advanced_rvc_inference.core extract \
  --model_name my_model \
  --f0_method rmvpe \
  --sample_rate 40000 \
  --embedder_model contentvec

# Train
python -m advanced_rvc_inference.core train \
  --model_name my_model \
  --save_every_epoch 10 \
  --total_epoch 200 \
  --sample_rate 40000 \
  --batch_size 8 \
  --pretrained True
```

### Available Commands
- `infer`: Single audio inference
- `batch_infer`: Batch audio inference
- `tts`: Text-to-speech inference
- `preprocess`: Dataset preprocessing
- `extract`: Feature extraction
- `train`: Model training
- `index`: Index file generation
- `model_information`: Model details
- `model_blender`: Model blending
- `tensorboard`: Launch tensorboard
- `download`: Model download
- `prerequisites`: Install prerequisites
- `audio_analyzer`: Audio analysis

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

### Logging Issues
To get more detailed logs, you can modify logging levels by setting environment variables:
```bash
PYTHONPATH=. python -m advanced_rvc_inference.app
```

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

### Testing
Run the test suite to ensure your changes don't break existing functionality:
```bash
python test_package.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

If you encounter any issues, please open an issue on [GitHub](https://github.com/ArkanDash/Advanced-RVC-Inference/issues).

For questions and discussions, join our community:
- [GitHub Discussions](https://github.com/ArkanDash/Advanced-RVC-Inference/discussions)


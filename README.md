<div align="center">

# Advanced RVC Inference

**A modular Retrieval-based Voice Conversion framework with Gradio UI, training capabilities, and audio processing tools**



</div>

## Features

- **Voice Conversion**: High-quality voice conversion with multiple pitch extraction methods
- **Model Training**: Complete training pipeline for creating custom RVC models
- **Real-time Processing**: Low-latency real-time voice conversion support
- **Web UI**: Intuitive Gradio-based web interface
- **CLI Support**: Command-line interface for scripting and automation
- **API Access**: Python API for programmatic access
- **Audio Separation**: Built-in tools for vocal/instrument separation
- **Text-to-Speech**: Integration with edge-tts for TTS-based voice conversion

## Installation

### From PyPI (Recommended)

```bash
pip install advanced-rvc-inference
```

### From GitHub (Latest Version)

```bash
pip install git+https://github.com/ArkanDash/Advanced-RVC-Inference.git
```

### With GPU Support

For CUDA-enabled GPUs:

```bash
pip install advanced-rvc-inference[gpu]
# or
pip install git+https://github.com/ArkanDash/Advanced-RVC-Inference.git#egg=advanced-rvc-inference[gpu]
```

### From Source

```bash
git clone https://github.com/ArkanDash/Advanced-RVC-Inference.git
cd Advanced-RVC-Inference
pip install -e .
```

### Development Installation

```bash
pip install advanced-rvc-inference[dev]
# or
pip install git+https://github.com/ArkanDash/Advanced-RVC-Inference.git#egg=advanced-rvc-inference[dev]
```

## Quick Start

### Web Interface

Launch the Gradio web UI:

```bash
rvc-gui
# or
python -m advanced_rvc_inference.gui
```

The web interface will be available at `http://localhost:7860`

### Command Line Interface

Run voice conversion from the command line:

```bash
rvc-cli infer --model path/to/model.pth --input audio.wav --output converted.wav --pitch 0
```

View help:

```bash
rvc-cli --help
rvc-cli infer --help
```

### Python API

```python
from advanced_rvc_inference import RVCInference

# Initialize the inference engine
rvc = RVCInference(device="cuda:0")

# Load a model
rvc.load_model("path/to/model.pth")

# Run inference
audio = rvc.infer("input.wav", pitch_change=0, output_path="output.wav")

# Or use batch processing
audio_files = rvc.infer_batch(
    input_dir="input_folder",
    output_dir="output_folder",
    pitch_change=2,
    format="wav"
)

# Cleanup
rvc.unload_model()
```

## Command Reference

### CLI Commands

| Command | Description |
|---------|-------------|
| `rvc-cli infer` | Run voice conversion inference |
| `rvc-cli train` | Train RVC models (use web UI) |
| `rvc-cli serve` | Launch the web interface |
| `rvc-cli version` | Show version information |
| `rvc-cli info` | Show system information |

### Inference Options

```bash
rvc-cli infer \
    --model MODEL.pth \
    --input input.wav \
    --output output.wav \
    --pitch 0 \
    --format wav \
    --index INDEX.index
```

## Project Structure

```
advanced-rvc-inference/
├── pyproject.toml          # Package configuration
├── README.md               # This file
├── LICENSE                 # MIT License
├── MANIFEST.in             # Package manifest
├── requirements.txt        # Legacy requirements file
├── src/
│   └── advanced_rvc_inference/
│       ├── __init__.py     # Package entry point
│       ├── __main__.py     # Module execution
│       ├── _version.py     # Version information
│       ├── api.py          # High-level Python API
│       ├── cli.py          # Command-line interface
│       ├── gui.py          # Gradio web UI
│       ├── variables.py    # Global configuration
│       ├── core/           # Core processing modules
│       ├── tabs/           # UI tab components
│       ├── library/        # ML libraries and utilities
│       ├── infer/          # Inference engines
│       ├── tools/          # Utility tools
│       ├── configs/        # Configuration files
│       └── assets/         # Resource files
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ARVC_ASSETS_PATH` | Path to asset directory | Package assets folder |
| `ARVC_CONFIGS_PATH` | Path to configs directory | Package configs folder |
| `ARVC_WEIGHTS_PATH` | Path to model weights | assets/weights |
| `ARVC_LOGS_PATH` | Path to logs directory | assets/logs |

### Configuration File

Configuration is managed through `advanced_rvc_inference/configs/config.json`:

```json
{
    "device": "cuda:0",
    "fp16": true,
    "app_port": 7860,
    "language": "vi-VN",
    "theme": "NoCrypt/miku"
}
```

## Dependencies

### Core Dependencies

- Python 3.10+
- PyTorch 2.3.1+
- torchaudio 2.3.1+
- NumPy, SciPy
- librosa (audio processing)
- Gradio (web UI)

### Optional Dependencies

- onnxruntime-gpu (GPU inference acceleration)
- faiss-gpu (vector similarity search)
- tensorboard (training visualization)

See `pyproject.toml` for the complete dependency list.

## Documentation

- [API Reference](https://github.com/ArkanDash/Advanced-RVC-Inference#api-reference)
- [Usage Guide](https://github.com/ArkanDash/Advanced-RVC-Inference#usage)
- [Contributing](CONTRIBUTING.md)

## Troubleshooting

### GPU Not Detected

Ensure you have CUDA installed and PyTorch with CUDA support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Memory Issues

Reduce batch size or use CPU mode:

```python
rvc = RVCInference(device="cpu")
```

### Import Errors

Reinstall the package:

```bash
pip install --upgrade --force-reinstall advanced-rvc-inference
```

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Terms of Use

The use of the converted voice for the following purposes is prohibited:

- Criticizing or attacking individuals
- Advocating for or opposing specific political positions, religions, or ideologies
- Publicly displaying strongly stimulating expressions without proper zoning
- Selling of voice models and generated voice clips
- Impersonation of the original owner of the voice with malicious intentions
- Fraudulent purposes that lead to identity theft or fraudulent phone calls

## Credits

| Repository | Owner |
|------------|-------|
| [Vietnamese-RVC](https://github.com/PhamHuynhAnh16/Vietnamese-RVC) | [PhamHuynhAnh16](https://github.com/PhamHuynhAnh16/) |
| [Applio](https://github.com/IAHispano/Applio) | [IAHispano](https://github.com/IAHispano) |

## Support

For issues and feature requests, please use the [GitHub Issues](https://github.com/ArkanDash/Advanced-RVC-Inference/issues) page.

---

<div align="center">

**Made with by ArkanDash**

</div>

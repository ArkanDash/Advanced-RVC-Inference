<div align="center">

# Advanced RVC Inference

**A modular Retrieval-based Voice Conversion framework with Gradio UI, training capabilities, and audio processing tools**

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArkanDash/Advanced-RVC-Inference/blob/master/Advanced-RVC.ipynb)

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

```bash
pip install git+https://github.com/ArkanDash/Advanced-RVC-Inference.git
```

### With GPU Support

For CUDA-enabled GPUs:

```bash
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

## UVR Vocal Separation

Advanced RVC Inference includes built-in vocal separation using UVR (Ultimate Vocal Remover) technology. The system automatically detects separation settings and options, making it easy to process multi-track audio.

### Separation Process

1. Select your audio file in the UVR tab
2. Choose your separation model and options
3. Click "Start Separation"
4. Separated files will be saved to the assets/audios directory

### Auto-Detection Feature

When using separated audio files for voice conversion, the system automatically detects:

- **Separation Model**: MDX, VR, or Demucs-based separation
- **Backing Vocals**: Whether backing track separation was enabled
- **Reverb Options**: Whether reverb removal was applied
- **Output Format**: Audio format used for separation

This information is stored in `uvr_options.json` within each separation folder, allowing the system to correctly process your audio without manual configuration.

### Audio Folder Structure

After separation, your audio files will be organized as follows:

```
assets/
└── audios/
    └── [Song Name]/
        ├── Original_Vocals.wav      # Main vocal track
        ├── Instruments.wav          # Instrumental track
        ├── Main_Vocals.wav          # Lead vocals (karaoke mode)
        ├── Backing_Vocals.wav       # Backing vocals (if enabled)
        ├── Original_Vocals_No_Reverb.wav  # De-reverbbed vocals
        └── uvr_options.json         # Separation settings (auto-generated)
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
    "theme": "NoCrypt/miku",
    "uvr_path": "advanced_rvc_inference/assets/audios"
}
```

### UVR Path Configuration

The `uvr_path` setting specifies where separated audio files are stored. By default, files are saved to:

- **Project Directory**: `advanced_rvc_inference/assets/audios/` (when running from source)
- **Package Directory**: `advanced_rvc_inference/assets/audios/` (when installed via pip)

You can customize this path in the configuration file if needed.

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

### Separated Audio Not Found

If the system cannot find your separated audio files:

1. Ensure files are in the correct directory structure: `assets/audios/[Song Name]/`
2. Check that audio files have proper extensions (.wav, .mp3, .flac, etc.)
3. Verify the `uvr_path` setting in your configuration file points to the correct location
4. Ensure the folder name matches exactly (case-sensitive)

### UVR Separation Errors

If you encounter errors during vocal separation:

1. Ensure ffmpeg is installed (required for audio processing)
2. Check that input audio is not corrupted or in an unsupported format
3. Reduce batch size or segment size if experiencing memory issues
4. For Demucs models, ensure the demucs package is installed:
   ```bash
   pip install demucs
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

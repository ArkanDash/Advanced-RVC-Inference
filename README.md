<div align="center">

# Advanced RVC Inference

Advanced RVC Inference presents itself as a state-of-the-art web UI crafted to streamline rapid and effortless inference. This comprehensive toolset encompasses a model downloader, a voice splitter, training and more.

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

see guides mkre at [Wiki](https://github.com/ArkanDash/Advanced-RVC-Inference/wiki/Cli-Guide)


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
| `rvc-cli infer` | Run voice conversion inference on a single audio file |
| `rvc-cli infer-batch` | Run batch voice conversion on multiple files |
| `rvc-cli train` | Train RVC models (use web UI for full features) |
| `rvc-cli dataset` | Create and manage training datasets |
| `rvc-cli preprocess` | Preprocess training data |
| `rvc-cli extract` | Extract features for training |
| `rvc-cli index` | Create index for feature retrieval |
| `rvc-cli separate` | Separate music into vocals and instruments |
| `rvc-cli reference` | Create reference audio for training |
| `rvc-cli tts` | Text-to-speech voice conversion |
| `rvc-cli serve` | Launch the web interface |
| `rvc-cli info` | Show system information |

### Quick Inference

Run voice conversion on a single audio file:

```bash
rvc-cli infer --model path/to/model.pth --input audio.wav --output converted.wav
```

With pitch shift (one octave up):

```bash
rvc-cli infer --model vocals.pth --input audio.wav --pitch 12 --output output.wav
```

### Batch Processing

Process multiple audio files at once:

```bash
rvc-cli infer-batch --model model.pth --input_dir ./songs --output_dir ./converted
```

### Music Separation

Separate vocals from instrumental tracks:

```bash
rvc-cli separate --input song.mp3 --output_dir ./separated
```

### Web Interface

Launch the Gradio web UI:

```bash
rvc-cli serve --port 7860
```

View help for any command:

```bash
rvc-cli --help
rvc-cli infer --help
rvc-cli separate --help


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
| [Vietnamese-RVC](https://github.com/PhamHuynhAnh16/Vietnamese-RVC) | Phạm Huỳnh Anh |
| [Applio](https://github.com/IAHispano/Applio) | IAHispano |
| [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) | Nomad Karaoke |
| [whisper](https://github.com/openai/whisper) | OpenAI |

## Support

For issues and feature requests, please use the [GitHub Issues](https://github.com/ArkanDash/Advanced-RVC-Inference/issues) page.

## NOTES
if you want use older verion use v1 branch (py3.12+ support comming soon), dont't lazy
---

<div align="center">

**Made with by ArkanDash**

</div>

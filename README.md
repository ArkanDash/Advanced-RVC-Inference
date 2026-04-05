<div align="center">

# Advanced RVC Inference

**Version 2.1.0** — Stability-focused update with improved error handling, security hardening, and modernized dependencies.

Advanced RVC Inference is a state-of-the-art web UI crafted to streamline rapid and effortless RVC (Retrieval-based Voice Conversion) inference. This comprehensive toolset encompasses a model downloader, a voice splitter, training, real-time voice conversion, speaker diarization, and much more — all powered by a Gradio-based interface.

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArkanDash/Advanced-RVC-Inference/blob/master/Advanced-RVC.ipynb)

</div>

---

## What's New in v2.1.0

This release focuses on **stability and robustness improvements** across the entire codebase:

- **Security**: Removed global SSL certificate verification bypass — HTTPS connections are now properly validated
- **Error Handling**: Replaced all bare `except:` clauses with specific exception types and proper logging
- **Deprecated APIs**: Replaced deprecated `distutils.util.strtobool` with a portable, type-aware implementation
- **Process Safety**: Added timeout handling and return-code checking to all `subprocess.run()` calls
- **Signal Handling**: Fixed signal handlers to allow graceful shutdown instead of blocking termination
- **File Safety**: All file I/O now uses `with` statements and proper encoding; removed dangerous monkey-patching of PyTorch internals
- **Missing Module**: Created the previously missing Downloads tab module that caused import errors
- **Logic Bugs**: Fixed UI logic error in backing track choices and improved `gr.Error` compatibility across Gradio versions
- **Dependencies**: Updated Gradio to use the [BF667-IDLE/gradio](https://github.com/BF667-IDLE/gradio) fork for enhanced stability

---

## Requirements

- **Python**: 3.10, 3.11, or 3.12
- **OS**: Windows, macOS, or Linux
- **GPU**: Optional — CUDA, DirectML, OpenCL, ROCm, or Apple Silicon (MPS) supported
- **RAM**: Minimum 8 GB recommended (16+ GB for training)

---

# Getting Started

## Installation

### From PyPI (recommended)

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

### Using the Custom Gradio Fork

This version uses a custom [Gradio fork](https://github.com/BF667-IDLE/gradio) for improved stability. If you prefer the standard Gradio, you can override the dependency:

```bash
pip install gradio>=6.5.1
```

> **Note**: The custom fork is installed automatically via the dependency specification. No manual action is needed unless you want to switch back to the standard Gradio package.


## Quick Start

### Web Interface

Launch the Gradio web UI:

```bash
rvc-gui
```

Or equivalently:

```bash
python -m advanced_rvc_inference.app.gui
```

The web interface will be available at `http://localhost:7860` by default.

**Command-line options for the GUI:**

```bash
# Custom host and port
python -m advanced_rvc_inference.app.gui --host 0.0.0.0 --port 7860

# Create a public share link
rvc-gui --share

# Force local access only
rvc-gui --no-share

# Open in browser automatically
rvc-gui --open
```

### Command Line Interface

See the [Wiki](https://github.com/ArkanDash/Advanced-RVC-Inference/wiki/Cli-Guide) for detailed CLI guides.

#### Quick Inference

Run voice conversion on a single audio file:

```bash
rvc-cli infer --model path/to/model.pth --input audio.wav --output converted.wav
```

With pitch shift (one octave up):

```bash
rvc-cli infer --model vocals.pth --input audio.wav --pitch 12 --output output.wav
```

#### Batch Processing

Process multiple audio files at once:

```bash
rvc-cli infer-batch --model model.pth --input_dir ./songs --output_dir ./converted
```

#### Music Separation

Separate vocals from instrumental tracks:

```bash
rvc-cli separate --input song.mp3 --output_dir ./separated
```

#### Web Interface via CLI

```bash
rvc-cli serve --port 7860
```

#### View Help

```bash
rvc-cli --help
rvc-cli infer --help
rvc-cli separate --help
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Voice Conversion** | Convert voices using RVC v1/v2 models with support for pitch shifting, formant shifting, and audio processing |
| **Music Separation** | Separate vocals, instruments, and backing tracks using UVR/MDX models |
| **Real-Time Conversion** | Live voice conversion with low latency using microphone input |
| **Training** | Train custom RVC models from datasets with support for multiple architectures |
| **Speaker Diarization** | Multi-speaker detection and conversion using Whisper |
| **TTS Integration** | Text-to-speech with Edge TTS and Google TTS, with optional voice conversion |
| **Model Management** | Download, search, and manage RVC models from HuggingFace and other sources |
| **F0 Methods** | 30+ pitch extraction methods including RMVPE, CREPE, FCPE, Harvest, SWIPE, PENN, and hybrid modes |
| **Embedders** | Support for HuBERT, ContentVec, Whisper, SPIN embedders in fairseq/ONNX/transformers modes |
| **Multi-GPU** | Support for CUDA, DirectML, OpenCL, ROCm, and Apple Silicon |
| **Multi-Language** | Interface available in English, Vietnamese, and Indonesian |
| **Themes** | 50+ built-in Gradio themes |
| **ONNX Export** | Convert models to ONNX format for optimized inference |

---

## Supported F0 Methods

### Standard Methods
`rmvpe`, `crepe-full`, `fcpe`, `harvest`, `pyin`, `hybrid`

### Extended Methods (Full Mode)
`pm-ac`, `pm-cc`, `pm-shs`, `dio`, `crepe-tiny/small/medium/large/full`, `fcpe-legacy/previous`, `rmvpe-clipping/medfilt`, `hpa-rmvpe`, `harvest`, `yin`, `swipe`, `piptrack`, `penn`, `djcm`, `swift`, `pesto`, and all hybrid combinations

---

## Troubleshooting

### GPU Not Detected

Ensure you have CUDA installed and PyTorch with CUDA support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Port Already in Use

If port 7860 is occupied, use a different port:

```bash
rvc-gui --port 7861
```

### Gradio Share Link Fails

The app will automatically fall back to local-only mode if share link creation fails. You can also explicitly disable it:

```bash
rvc-gui --no-share
```

### FP16 Not Supported on CPU/MPS

The application will automatically detect and disable FP16 on devices that do not support it. No manual intervention needed.

---

## Documentation

- [API Reference](https://github.com/ArkanDash/Advanced-RVC-Inference#api-reference)
- [Usage Guide](https://github.com/ArkanDash/Advanced-RVC-Inference#usage)
- [CLI Guide](https://github.com/ArkanDash/Advanced-RVC-Inference/wiki/Cli-Guide)
- [Contributing](CONTRIBUTING.md)

---

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Terms of Use

The use of the converted voice for the following purposes is prohibited:

- Criticizing or attacking individuals
- Advocating for or opposing specific political positions, religions, or ideologies
- Publicly displaying strongly stimulating expressions without proper zoning
- Selling of voice models and generated voice clips
- Impersonation of the original owner of the voice with malicious intentions
- Fraudulent purposes that lead to identity theft or fraudulent phone calls

---

## Credits

| Repository | Owner |
|------------|-------|
| [Vietnamese-RVC](https://github.com/PhamHuynhAnh16/Vietnamese-RVC) | Phạm Huỳnh Anh |
| [Applio](https://github.com/IAHispano/Applio) | IAHispano |
| [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) | Nomad Karaoke |
| [whisper](https://github.com/openai/whisper) | OpenAI |
| [BigVGAN](https://github.com/NVIDIA/BigVGAN.git) | Nvidia |
| [Gradio (custom fork)](https://github.com/BF667-IDLE/gradio) | BF667-IDLE |

---

## Support

For issues and feature requests, please use the [GitHub Issues](https://github.com/ArkanDash/Advanced-RVC-Inference/issues) page.

[Discord Community](https://discord.gg/hvmsukmBHE)

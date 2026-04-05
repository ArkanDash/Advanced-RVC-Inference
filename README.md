<div align="center">

# Advanced RVC Inference

**A state-of-the-art web UI for Retrieval-based Voice Conversion (RVC) — featuring fast inference, model downloading, voice splitting, training, real-time conversion, and a full command-line interface.**

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArkanDash/Advanced-RVC-Inference/blob/master/Advanced-RVC.ipynb)
[![Discord](https://img.shields.io/badge/Chat-Discord-5865F2?style=flat-square&logo=discord&logoColor=white)](https://discord.gg/hvmsukmBHE)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

> [!NOTE]
> Advanced RVC Inference will no longer receive frequent updates. Going forward, development will focus mainly on security patches, dependency updates, and occasional feature improvements. This is because the project is already stable and mature with limited room for further improvements. Pull requests are still welcome and will be reviewed.

---

## Features

| Category | Details |
|----------|---------|
| **Voice Inference** | Single & batch audio conversion, TTS synthesis, pitch shifting, F0 autotune, formant shifting, audio cleaning, and Whisper-based transcription |
| **Audio Separation** | Vocal/instrumental isolation using UVR5 models (MDX-Net, Roformer, BS-Roformer), karaoke separation, reverb removal, and denoising |
| **Real-Time Conversion** | Live microphone voice conversion with VAD (Voice Activity Detection) and low-latency processing |
| **Training Pipeline** | End-to-end training from dataset creation (YouTube/local), preprocessing, feature extraction, and model training with overtraining detection |
| **Model Management** | Download models from URLs (HuggingFace, direct links), create .index files, model format conversion, and reference set creation |
| **Extra Tools** | F0 extraction, voice fusion, SRT subtitle generation, model info reader, and configurable settings |
| **CLI** | Full command-line interface for all operations — `rvc-cli` with subcommands for inference, separation, training, and more |
| **Downloads Tab** | Built-in model and asset downloader accessible directly from the web UI |

## Supported F0 Methods

Advanced RVC Inference supports an extensive range of pitch extraction algorithms:

**Standard Methods:**
`rmvpe` · `crepe-full` · `fcpe` · `harvest` · `pyin` · `hybrid`

**Extended Methods (30+):**
`mangio-crepe-tiny/small/medium/large/full` · `crepe-tiny/small/medium/large/full` · `fcpe-legacy` · `fcpe-previous` · `rmvpe-clipping` · `rmvpe-medfilt` · `hpa-rmvpe` · `hpa-rmvpe-medfilt` · `dio` · `yin` · `swipe` · `piptrack` · `penn` · `mangio-penn` · `djcm` · `swift` · `pesto` · and more

**Hybrid Methods (combine two algorithms):**
`hybrid[pm+dio]` · `hybrid[pm+crepe-tiny]` · `hybrid[pm+crepe]` · `hybrid[pm+fcpe]` · `hybrid[pm+rmvpe]` · `hybrid[crepe-tiny+crepe]` · `hybrid[dio+crepe]` · and more combinations

> `rmvpe` is the recommended default for most use cases, offering the best balance of speed and accuracy.

---

## Installation

### Prerequisites

- **Python** 3.10, 3.11, or 3.12
- **PyTorch** ≥ 2.3.1 (with CUDA support recommended for GPU acceleration)
- **FFmpeg** installed and available in your system PATH

### Install from PyPI

```bash
pip install git+https://github.com/ArkanDash/Advanced-RVC-Inference.git
```

### With GPU Support (CUDA)

```bash
pip install git+https://github.com/ArkanDash/Advanced-RVC-Inference.git
pip install onnxruntime-gpu
```

### Install from Source

```bash
git clone https://github.com/ArkanDash/Advanced-RVC-Inference.git
cd Advanced-RVC-Inference
pip install -r requirements.txt
```

### Google Colab

Click the badge below to open the notebook directly in Colab — everything installs and runs with a single click:

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArkanDash/Advanced-RVC-Inference/blob/master/Advanced-RVC.ipynb)

---

## Quick Start

### Web Interface

Launch the Gradio web UI — this is the easiest way to get started:

```bash
# Using the GUI entry point
rvc-gui

# Or via Python module
python -m advanced_rvc_inference.app.gui

# With a public share link
python -m advanced_rvc_inference.app.gui --share
```

The web interface will be available at `http://localhost:7860` by default.

### Command Line Interface

The `rvc-cli` tool provides full access to all features directly from the terminal. For the complete command reference, see the [CLI Guide](https://github.com/ArkanDash/Advanced-RVC-Inference/wiki/Cli-Guide).

```bash
# Show all available commands
rvc-cli --help
```

#### Voice Conversion

```bash
# Basic conversion
rvc-cli infer -m model.pth -i input.wav -o output.wav

# With pitch shift (one octave up = +12 semitones)
rvc-cli infer -m model.pth -i input.wav -p 12 -o output.wav

# With a specific F0 method and format
rvc-cli infer -m model.pth -i input.wav --f0_method crepe-full -f flac
```

#### Audio Separation

```bash
# Separate vocals from instrumental
rvc-cli uvr -i song.mp3

# Use a specific UVR model
rvc-cli uvr -i song.mp3 --model BS-Roformer
```

#### Model Download

```bash
# Download from HuggingFace or direct URL
rvc-cli download -l "https://huggingface.co/user/model/resolve/main/model.pth"
```

#### System Information

```bash
# Show system info, GPU status, and installed models
rvc-cli info
rvc-cli list-models
rvc-cli list-f0-methods
```

---

## Web UI Tabs Overview

The Gradio web interface is organized into several tabs, each dedicated to a specific workflow:

### Inference Tab
The main workspace for voice conversion. Supports single file conversion, batch processing on folders, audio separation (UVR5), Whisper-based transcription, and TTS synthesis. Fine-tune parameters like pitch shift, filter radius, index rate, F0 method, formant shifting, audio cleaning, and more.

### Real-Time Tab
Perform live voice conversion using your microphone. Configure input/output devices, pitch, and conversion parameters for real-time processing with minimal latency.

### Training Tab
Complete training pipeline accessible from the web UI:
- **Create Dataset** — Build training data from YouTube URLs or local audio files, with optional vocal separation and cleaning
- **Create Reference** — Generate reference audio sets for improved inference quality
- **Train** — Train RVC models with configurable epochs, batch size, optimizer, overtraining detection, and more

### Downloads Tab
Built-in model and asset downloader. Paste URLs from HuggingFace or other sources to download models directly into the correct directory.

### Extra Tab
Additional utilities:
- **Model Reader** — Inspect model metadata and configuration
- **Model Converter** — Convert between model formats (v1/v2, PyTorch/ONNX)
- **F0 Extract** — Extract pitch contours from audio files
- **Fusion** — Blend two voice models together
- **SRT Generator** — Create subtitle files from audio
- **Settings** — Configure application preferences

---

## Project Structure

```
Advanced-RVC-Inference/
├── advanced_rvc_inference/
│   ├── app/
│   │   ├── gui.py              # Main entry point & Gradio app
│   │   └── tabs/
│   │       ├── inference/       # Inference, separation, TTS, Whisper
│   │       ├── realtime/        # Real-time mic conversion
│   │       ├── training/        # Dataset creation, extraction, training
│   │       ├── downloads/       # Model downloader tab
│   │       └── extra/           # Extra tools (fusion, SRT, settings, etc.)
│   ├── api/
│   │   └── cli.py              # Full CLI interface (rvc-cli)
│   ├── configs/                 # Model configs (v1, v2, ringformer, etc.)
│   ├── core/                    # Core utilities (UI, process, restart)
│   ├── library/                 # ML backends (predictors, embedders, ONNX)
│   ├── rvc/
│   │   ├── infer/               # Inference engine & audio conversion
│   │   ├── realtime/            # Real-time voice conversion
│   │   └── train/               # Preprocessing, extraction, training
│   ├── uvr/                     # UVR5 audio separation library
│   └── utils/                   # Shared variables & utilities
├── Advanced-RVC.ipynb           # Google Colab notebook
├── rvc-cli.sh                   # CLI wrapper script
├── requirements.txt             # Python dependencies
└── pyproject.toml               # Package configuration
```

---

## Troubleshooting

### GPU Not Detected

Make sure you have the CUDA toolkit installed and PyTorch built with CUDA support:

```bash
# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify your GPU is detected:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### FFmpeg Not Found

FFmpeg is required for audio processing. Install it via your package manager:

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows — download from https://ffmpeg.org/download.html and add to PATH
```

### CUDA Out of Memory

If you encounter OOM errors during inference or training, try enabling memory checkpointing:

- **CLI:** Add `--checkpointing` to your command
- **Web UI:** Enable the "Checkpointing" toggle in the inference tab
- Reduce batch size during training

### Common Dependency Issues

```bash
# If FAISS fails on Python 3.12+
pip install faiss-cpu --upgrade

# If ONNX Runtime causes issues on macOS
pip install onnxruntime --upgrade

# For NVIDIA GPUs, ensure the GPU variant of ONNX Runtime
pip install onnxruntime-gpu
```

---

## Contributing

Contributions are welcome! Whether it's bug fixes, new features, or documentation improvements, feel free to open a pull request. Please ensure your changes pass any existing tests and follow the project's coding conventions.

---

## Terms of Use

The use of the converted voice for the following purposes is **strictly prohibited**:

- Criticizing or attacking individuals
- Advocating for or opposing specific political positions, religions, or ideologies
- Publicly displaying strongly stimulating expressions without proper zoning
- Selling of voice models and generated voice clips
- Impersonation of the original owner of the voice with malicious intentions
- Fraudulent purposes that lead to identity theft or fraudulent phone calls

---

## Credits

This project builds upon the work of several open-source repositories and their contributors:

| Repository | Owner | Purpose |
|------------|-------|---------|
| [Vietnamese-RVC](https://github.com/PhamHuynhAnh16/Vietnamese-RVC) | Phạm Huỳnh Anh | Core RVC implementation |
| [Applio](https://github.com/IAHispano/Applio) | IAHispano | UI/UX inspiration & components |
| [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) | Nomad Karaoke | UVR5 audio separation |
| [whisper](https://github.com/openai/whisper) | OpenAI | Speech-to-text transcription |
| [BigVGAN](https://github.com/NVIDIA/BigVGAN.git) | Nvidia | Vocoder implementation |

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

## Links

- **GitHub:** [ArkanDash/Advanced-RVC-Inference](https://github.com/ArkanDash/Advanced-RVC-Inference)
- **Discord:** [Join the community](https://discord.gg/hvmsukmBHE)
- **CLI Guide:** [Wiki - CLI Guide](https://github.com/ArkanDash/Advanced-RVC-Inference/wiki/Cli-Guide)
- **Issues:** [Report a bug](https://github.com/ArkanDash/Advanced-RVC-Inference/issues)

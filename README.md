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
| **Easy GUI** | Simplified one-click interface for quick conversion and training — inspired by EasierGUI |
| **Auto Pretrained Download** | Automatically downloads default pretrained models from HuggingFace when no custom paths are specified |
| **ZLUDA Support** | Full AMD GPU support via ZLUDA (CUDA compatibility layer) for training and inference |
| **T4 / Low-VRAM Optimizations** | Auto-detected GPU-class optimizations for Colab T4 and low-VRAM GPUs (FP16, gradient accumulation, memory-efficient attention) |
| **Training Optimizations** | Gradient accumulation, torch.compile(), 8-bit Adam, set_to_none gradients, DDP tuning, CUDA cache cleanup |
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

### ZLUDA (AMD GPU) Setup

ZLUDA allows CUDA applications to run on AMD GPUs. No additional installation steps are needed — just install PyTorch with ZLUDA support and Advanced RVC will automatically detect and configure itself for AMD hardware.

```bash
# Follow the ZLUDA installation guide for your AMD GPU
# Then install Advanced RVC normally — ZLUDA is auto-detected
pip install git+https://github.com/ArkanDash/Advanced-RVC-Inference.git
```

### Google Colab

**With Web UI** — full Gradio interface with a public share link. T4 GPU optimizations (FP16, gradient accumulation, memory-efficient attention) are automatically enabled when a Tesla T4 is detected:

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArkanDash/Advanced-RVC-Inference/blob/master/Advanced-RVC.ipynb)

**CLI Only (No UI)** — lightweight headless mode using `pip install`. No web UI, no repo clone — just `rvc-cli` commands. Paths are resolved automatically from the installed package:

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArkanDash/Advanced-RVC-Inference/blob/master/colab-noui.ipynb)

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

### Easy GUI (Simplified Mode)

Launch a streamlined interface designed for quick workflows — just convert or train with minimal configuration:

```bash
# Launch Easy GUI
rvc-cli serve --easy true

# Alternative shorthand
rvc-cli serve -ez true
```

The Easy GUI includes three tabs:
- **Quick Convert** — Simple voice conversion with model selection, pitch, and F0 settings
- **One-Click Train** — Full training pipeline in a single button (Preprocess → Extract Features → Train → Create Index)
- **Download** — Quick model download from URLs

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

### Easy GUI Tabs
A simplified interface accessible via `rvc-cli serve --easy true`:
- **Quick Convert** — Simplified voice conversion with essential settings
- **One-Click Train** — Train a new model in one click with automatic pretrained model download
- **Download** — Quick model download from HuggingFace and other sources

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

## Training Optimizations

Advanced RVC Inference includes several training optimizations to improve speed and reduce memory usage:

### Gradient Accumulation
Simulate larger batch sizes without extra VRAM by accumulating gradients over multiple steps:

```bash
# Effective batch size = batch_size * grad_accum_steps
rvc-cli train mymodel --batch_size 4 --grad_accum_steps 4
```

### torch.compile() (PyTorch 2.x+)
Compile the generator model for faster training iterations (not compatible with ZLUDA):

```bash
rvc-cli train mymodel --compile_model True
```

### 8-bit Adam Optimizer
Reduce optimizer memory by ~50% using 8-bit quantized Adam (requires `bitsandbytes`):

```bash
rvc-cli train mymodel --use_8bit_adam True
```

### Auto Pretrained Model Download
When no custom pretrained paths are specified, the training script automatically downloads the appropriate default pretrained G and D models from HuggingFace (Vietnamese-RVC-Project). Models are selected based on RVC version, sample rate, and pitch guidance setting, then cached locally for reuse.

### GPU-Auto-Detection
The training system automatically detects your GPU hardware and applies appropriate optimizations:

| GPU | Detection | Optimizations Applied |
|-----|-----------|----------------------|
| **Tesla T4** (Colab) | `tesla t4` in device name | FP16 AMP, gradient accumulation (auto), memory-efficient attention, reduced workers/prefetch |
| **Low VRAM** (≤16 GB) | Memory size check | Reduced DataLoader workers/prefetch, memory-efficient attention |
| **ZLUDA** (AMD) | CUDA version check, device name suffix, env vars | gloo DDP backend (no NCCL), FP16 AMP (no BF16), skip TF32/cuDNN/torch.compile, custom STFT |
| **High VRAM CUDA** | Default | Full CUDA optimizations: TF32, cuDNN benchmark, torch.compile, CUDA streams |

Run `rvc-cli info` to see your detected GPU class and active optimizations.

---

## ZLUDA (AMD GPU) Support

ZLUDA is a CUDA compatibility layer that translates CUDA API calls to HIP/ROCm, allowing PyTorch CUDA applications to run on AMD GPUs without code changes. Advanced RVC Inference automatically detects ZLUDA and adjusts its behavior:

### What Works
- Model inference (single & batch)
- Full training pipeline (single GPU)
- Audio preprocessing and feature extraction
- Easy GUI and web interface

### ZLUDA-Specific Adaptations
- **DDP Backend**: Uses `gloo` instead of `nccl` (no multi-GPU support)
- **AMP Precision**: Forces FP16 instead of BF16 (not reliably supported on HIP)
- **CUDA Features Disabled**: TF32, cuDNN benchmark/deterministic, torch.compile(), CUDA allocator settings, and CUDA streams are all automatically disabled
- **STFT**: Custom ZLUDA-compatible STFT implementation for complex tensor operations
- **ONNX Runtime**: Prefers ROCMExecutionProvider or CPUExecutionProvider over CUDAExecutionProvider
- **DataLoader**: Disables pin_memory for more reliable HIP memory transfers

### Detection
ZLUDA is detected through multiple methods for reliability: CUDA version string check, GPU device name suffix `[ZLUDA]`, and the `DISABLE_ADDMM_CUDA_LT` environment variable. Check your setup:

```bash
rvc-cli info
# Look for: "ZLUDA: Detected (AMD GPU via CUDA compatibility layer)"
```

---

## Project Structure

```
Advanced-RVC-Inference/
├── advanced_rvc_inference/
│   ├── app/
│   │   ├── gui.py              # Main entry point & Gradio app
│   │   ├── easy_gui.py         # Simplified Easy GUI interface
│   │   └── tabs/
│   │       ├── inference/       # Inference, separation, TTS, Whisper
│   │       ├── realtime/        # Real-time mic conversion
│   │       ├── training/        # Dataset creation, extraction, training
│   │       ├── downloads/       # Model downloader tab
│   │       └── extra/           # Extra tools (fusion, SRT, settings, etc.)
│   ├── api/
│   │   └── cli.py              # Full CLI interface (rvc-cli)
│   ├── configs/                 # Model configs (v1, v2, ringformer, etc.)
│   ├── core/                    # Core utilities (UI, process, training, restart)
│   ├── library/
│   │   ├── backends/            # GPU backends (CUDA, ZLUDA, DirectML, OpenCL)
│   │   ├── algorithm/           # Model architectures and algorithms
│   │   ├── generators/          # Vocoder implementations
│   │   ├── optimizers/          # Training optimizers
│   │   ├── predictors/          # F0 extraction algorithms
│   │   ├── embedders/           # Speaker embedding models
│   │   └── onnx/                # ONNX export utilities
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

If you encounter OOM errors during inference or training, try these solutions:

- **Gradient accumulation**: Use `--grad_accum_steps 2` (or higher) to reduce VRAM usage
- **8-bit Adam**: Use `--use_8bit_adam True` to reduce optimizer memory by ~50%
- **Smaller batch size**: Reduce training batch size
- **Checkpointing**: Add `--checkpointing` to your command
- **Web UI**: Enable the "Checkpointing" toggle in the inference tab

For T4/Colab users, most of these optimizations are applied automatically when the Tesla T4 GPU is detected.

### ZLUDA Issues

If you encounter problems running on AMD GPUs with ZLUDA:

```bash
# Verify ZLUDA is detected
rvc-cli info

# Check that torch.cuda reports your AMD GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"

# If STFT issues occur, check that ZLUDA STFT override is loaded
python -c "from advanced_rvc_inference.library.backends import zluda; print(f'ZLUDA: {zluda.is_available()}')"
```

### Common Dependency Issues

```bash
# If FAISS fails on Python 3.12+
pip install faiss-cpu --upgrade

# If ONNX Runtime causes issues on macOS
pip install onnxruntime --upgrade

# For NVIDIA GPUs, ensure the GPU variant of ONNX Runtime
pip install onnxruntime-gpu

# For 8-bit Adam optimizer support
pip install bitsandbytes
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
| [Vietnamese-RVC](https://github.com/PhamHuynhAnh16/Vietnamese-RVC) | Phạm Huỳnh Anh | Core RVC implementation & pretrained models |
| [Applio](https://github.com/IAHispano/Applio) | IAHispano | UI/UX inspiration & components |
| [Mangio-Kalo-Tweaks](https://github.com/kalomaze/Mangio-Kalo-Tweaks) | kalomaze | EasyGUI inspiration |
| [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) | Nomad Karaoke | UVR5 audio separation |
| [whisper](https://github.com/openai/whisper) | OpenAI | Speech-to-text transcription |
| [BigVGAN](https://github.com/NVIDIA/BigVGAN.git) | Nvidia | Vocoder implementation |
| [ZLUDA](https://github.com/vlsid/ZLUDA) | vlsid | AMD GPU CUDA compatibility layer |

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

## Links

- **GitHub:** [ArkanDash/Advanced-RVC-Inference](https://github.com/ArkanDash/Advanced-RVC-Inference)
- **Discord:** [Join the community](https://discord.gg/hvmsukmBHE)
- **CLI Guide:** [Wiki - CLI Guide](https://github.com/ArkanDash/Advanced-RVC-Inference/wiki/Cli-Guide)
- **Issues:** [Report a bug](https://github.com/ArkanDash/Advanced-RVC-Inference/issues)

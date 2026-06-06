<div align="center">

# Advanced RVC Inference

**A state-of-the-art web UI crafted to streamline rapid and effortless RVC inference — featuring a model downloader, voice splitter, batch inference, training pipeline, real-time conversion, and a full CLI.**

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArkanDash/Advanced-RVC-Inference/blob/master/Advanced-RVC.ipynb)
[![Discord](https://img.shields.io/badge/Chat-Discord-5865F2?style=flat-square&logo=discord&logoColor=white)](https://discord.gg/hvmsukmBHE)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

> [!NOTE]
> Advanced RVC Inference will no longer receive frequent updates. Going forward, development will focus mainly on security patches, dependency updates, and occasional feature improvements. This is because the project is already stable and mature with limited room for further improvements. Pull requests are still welcome and will be reviewed.

> [!NOTE]
> If you want to use old version switch to v1 branch.

---

## Features

### Inference
- **Voice Inference** — Single & batch conversion, TTS, pitch shifting, formant shifting, audio cleaning, Whisper transcription
- **Real-Time Conversion** — Live mic voice conversion with VAD and low-latency processing
- **30+ F0 Methods** — rmvpe, crepe, fcpe, harvest, hybrid, and many more
- **F0 Autotune** — Automatic pitch correction with configurable strength
- **Audio Cleaning** — Built-in denoising for cleaner output

### Audio Processing
- **Audio Separation** — Vocal/instrumental isolation (MDX-Net, Roformer, BS-Roformer), karaoke, reverb removal, denoising
- **Auto Pretrained Download** — Automatically downloads pretrained models from HuggingFace

### Training Pipeline
- **End-to-End Training** — Dataset creation → preprocessing → feature extraction → training → model export
- **4 Vocoders** — HiFi-GAN NSF (Default), BigVGAN, MRF-HiFi-GAN, RefineGAN
- **5 Optimizers** — AdamW, RAdam, AnyPrecisionAdamW, AdaBelief, AdaBeliefV2
- **Training Quality Improvements** — Multi-scale mel spectrogram loss (8 scales), scaled v3 discriminator loss, proper feature loss gradient flow, cuDNN benchmark
- **Advanced Options** — Gradient accumulation, torch.compile(), 8-bit Adam, cosine annealing LR, overtraining detection
- **Architecture Support** — RVC and SVC (from Vietnamese-RVC)
- **Embedder Mix** — Layer-wise embedding mixing with configurable ratios (from Vietnamese-RVC)

### Platform & Integration
- **CLI** — Full command-line interface via `rvc-cli`
- **ZLUDA Support** — Full AMD GPU support via ZLUDA
- **XPU Support** — Intel GPU support via XPU backend
- **Push to Hub** — Upload trained models directly to HuggingFace Hub
- **44 Languages** — Full UI translation support

---

## Supported Vocoders

Advanced RVC Inference supports the same vocoders as [Vietnamese-RVC](https://github.com/PhamHuynhAnh16/Vietnamese-RVC):

| Vocoder | Description | Pitch Required |
|---------|-------------|----------------|
| **Default** (HiFi-GAN NSF) | HiFi-GAN with Neural Sine Filter. Adds harmonic sine wave injection for improved pitch accuracy. **Recommended for best compatibility.** | Yes |
| **BigVGAN** | Snake activations with Anti-Aliasing (SnakeBeta + AMP blocks). State-of-the-art audio quality. | Yes |
| **MRF-HiFi-GAN** | HiFi-GAN with Multi-Receptive Field fusion. Richer feature extraction with MRF blocks. | Yes |
| **RefineGAN** | U-Net based vocoder with parallel residual blocks and anti-aliased resampling. High-fidelity spectral detail. | Yes |

When training without pitch guidance (`pitch_guidance=False`), a plain HiFi-GAN generator (no NSF) is used automatically regardless of the selected vocoder.

---

## Supported Optimizers

Advanced RVC Inference provides **5 carefully selected optimizers** for model training, covering the most effective choices for RVC/audio model training:

| Optimizer | Category | Rating | Best For |
|-----------|----------|--------|----------|
| **AdamW** | PyTorch Built-in | ⭐⭐⭐⭐⭐ | General-purpose, most reliable (default) |
| **RAdam** | PyTorch Built-in | ⭐⭐⭐⭐ | Warmup-free training, short training runs |
| **AnyPrecisionAdamW** | Mixed-Precision | ⭐⭐⭐⭐ | Bfloat16 training, long runs with Kahan summation |
| **AdaBelief** | Belief-Based | ⭐⭐⭐ | Better conditioned adaptive learning rates |
| **AdaBeliefV2** | Belief-Based | ⭐⭐⭐ | Stable deep training with AMSGrad + InverseSqrt scheduler |

See the [Optimizer Reference Guide](docs/optimizer.md) for detailed descriptions, hyperparameters, and recommendations.

---

## Getting Started

### 1. Install

```bash
git clone https://github.com/ArkanDash/Advanced-RVC-Inference.git
cd Advanced-RVC-Inference
pip install -r requirements.txt
```

Or install from PyPI:

```bash
pip install git+https://github.com/ArkanDash/Advanced-RVC-Inference.git
```

<details>
<summary>GPU Support (CUDA)</summary>

```bash
pip install git+https://github.com/ArkanDash/Advanced-RVC-Inference.git
pip install onnxruntime-gpu
```
</details>

<details>
<summary>ZLUDA (AMD GPU)</summary>

ZLUDA allows CUDA applications to run on AMD GPUs. Just install PyTorch with ZLUDA support — Advanced RVC will auto-detect and configure itself.

```bash
# Follow the ZLUDA installation guide for your AMD GPU
# Then install Advanced RVC normally — ZLUDA is auto-detected
pip install git+https://github.com/ArkanDash/Advanced-RVC-Inference.git
```
</details>

### 2. Run

```bash
# Launch the web UI
rvc-gui

# Or via Python module
python -m arvc.app.gui

# With a public share link
python -m arvc.app.gui --share
```

The interface will be available at `http://localhost:7860`.

### 3. CLI Usage

```bash
# Voice conversion
rvc-cli infer -m model.pth -i input.wav -o output.wav

# Audio separation
rvc-cli uvr -i song.mp3

# Show all commands
rvc-cli --help
```

### 4. Google Colab

| Notebook | Description |
|----------|-------------|
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArkanDash/Advanced-RVC-Inference/blob/master/Advanced-RVC.ipynb) | Full Web UI |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArkanDash/Advanced-RVC-Inference/blob/master/colab-noui.ipynb) | CLI only — lightweight headless mode |

---

## Project Structure

```
arvc/
├── app/                    # Gradio web UI (tabs, pages, layouts)
│   ├── tabs/               #   inference, training, downloads, realtime, extra
├── engine/                 # Core logic (no UI dependency)
│   ├── inference/          #   Voice conversion pipeline, TTS
│   ├── training/           #   preprocess, extract, train, export
│   │   ├── preprocess/     #   Audio slicing & normalization
│   │   ├── extract/        #   Embedding & F0 extraction
│   │   └── runner/         #   Training loop, losses, data loading
│   ├── uvr/                #   Audio separation (UVR5)
│   ├── realtime/           #   Live mic conversion
│   └── models/             #   Model loading, generators, optimizers, embedders
│       ├── generators/     #   HiFi-GAN NSF, BigVGAN, MRF-HiFi-GAN, RefineGAN
│       ├── optimizers/     #   AdamW, RAdam, AnyPrecisionAdamW, AdaBelief, AdaBeliefV2
│       ├── embedders/      #   Hubert, ContentVec embedders
│       ├── predictors/     #   F0 predictors (RMVPE, Crepe, FCPE, etc.)
│       └── backends/       #   CUDA, DirectML, OpenCL, XPU, ZLUDA
├── services/               # Business logic layer (bridges UI <-> engine)
├── ui/                     # UI helpers (feedback, dropdown updates, formatting)
├── utils/                  # Shared utilities (variables, download helpers)
├── configs/                # Configuration files (training configs, model templates)
│   ├── v1/                 #   V1 model configs (32k, 40k, 48k)
│   ├── v2/                 #   V2 model configs (24k, 32k, 40k, 48k)
│   ├── ringformer_v2/      #   RingFormer V2 configs
│   └── pcph_gan/           #   PCPH-GAN configs
├── datasets/               # Training datasets (organized per model)
├── assets/                 # Runtime assets
│   ├── models/             #   Pretrained models, embedders, predictors, UVR5
│   │   ├── pretrained_v1/  #     V1 pretrained G/D weights
│   │   ├── pretrained_v2/  #     V2 pretrained G/D weights
│   │   ├── pretrained_custom/ #  Custom pretrained weights
│   │   ├── embedders/      #     Hubert/ContentVec models
│   │   ├── predictors/     #     F0 predictor models
│   │   └── uvr5/           #     UVR5 separation models
│   ├── logs/               #   Training logs, checkpoints, extracted features, indexes
│   ├── audios/             #   Audio files (input, output, TTS, UVR results)
│   ├── f0/                 #   F0 cache files
│   ├── binary/             #   Binary resources
│   ├── languages/          #   44 translation JSON files
│   └── presets/            #   Inference presets
└── _version.py             # Version management
```

**Key rule**: `engine/` should never import from `app/` or `services/`. Keep the core independent.

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

## Changelog

### v2.1.0
- **VRVC Training Integration** — Cloned Vietnamese-RVC training pipeline including architecture selector (RVC/SVC), embedder mix, include mutes, nprobe, alpha, and F0 autotune with configurable strength
- **Training Quality Fixes** — Multi-scale mel spectrogram loss (8 scales with dynamic windows from PolTrain), proper feature loss gradient flow (removed `.detach()` from Applio), scaled v3 discriminator loss for BigVGAN/RefineGAN, cuDNN benchmark enabled by default
- **Optimizer Cleanup** — Reduced from 43 optimizers to 5 proven choices (AdamW, RAdam, AnyPrecisionAdamW, AdaBelief, AdaBeliefV2)
- **Directory Structure** — Cleaned up assets: datasets moved to `arvc/datasets/`, weights merged into `arvc/assets/logs/`
- **EasyGUI Removed** — Deleted `easy_gui.py` and all references; Web UI is the only interface
- **Bug Fixes** — Fixed robotic chirping (#69), `get_gpu_info()` unpack error, faiss AVX512/AVX2 import crash, missing `--predictor_onnx` argument, synced all training params across UI → service → subprocess
- **Colab Updates** — Removed EasyGUI toggle and CLI Usage section from main notebook

---

## Credits

This project builds upon the work of many open-source projects and contributors. We gratefully acknowledge the following:

### Core RVC Foundation
| Project | Author |
|---------|--------|
| [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) | RVC Project |
| [Vietnamese-RVC](https://github.com/PhamHuynhAnh16/Vietnamese-RVC) | Phạm Huỳnh Anh |
| [Mangio-Kalo-Tweaks](https://github.com/kalomaze/Mangio-Kalo-Tweaks) | kalomaze |

### Training Improvements
| Project | Author |
|---------|--------|
| [PolTrain](https://github.com/Politrees/PolTrain) | Politrees |
| [Applio](https://github.com/IAHispano/Applio) | IAHispano |

### Audio & Models
| Project | Author |
|---------|--------|
| [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) | Nomad Karaoke |
| [whisper](https://github.com/openai/whisper) | OpenAI |
| [BigVGAN](https://github.com/NVIDIA/BigVGAN) | Nvidia |
| [Ultimate-RVC-Models](https://huggingface.co/R-Kentaren/Ultimate-RVC-Models) | R-Kentaren |

### Hardware & Platform Support
| Project | Author |
|---------|--------|
| [ZLUDA](https://github.com/vlsid/ZLUDA) | vlsid |
| [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) | Tim Dettmers |

### Collaborators
| Collaborator | Role |
|-------------|------|
| [ArkanDash](https://github.com/ArkanDash) | Creator & Maintainer |
| [BF667](https://github.com/BF667) | Collaborator |

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

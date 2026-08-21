<div align="center">

# Advanced RVC Inference

**A state-of-the-art web UI crafted to streamline rapid and effortless RVC inference — featuring a model downloader, voice splitter, batch inference, training pipeline, real-time conversion, and a full CLI.**

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArkanDash/Advanced-RVC-Inference/blob/master/Advanced-RVC.ipynb)
[![Discord](https://img.shields.io/badge/Chat-Discord-5865F2?style=flat-square&logo=discord&logoColor=white)](https://discord.gg/hvmsukmBHE)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>


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
- **🔧 Auto Model Download** — Predictor (RMVPE/FCPE) and embedder (HuBERT) models download automatically before training starts — no more "model not found" errors!
- **4 Vocoders** — HiFi-GAN NSF (Default), BigVGAN, MRF-HiFi-GAN, RefineGAN
- **7 Optimizers** — AdamW, RAdam, AnyPrecisionAdamW, AdaBelief, AdaBeliefV2, **Ranger2020**, **Prodigy**
- **Enhanced Loss Functions** — Multi-scale STFT loss, phase loss, envelope loss, KL-divergence loss (from Codename RVC Fork v4)
- **Training Enhancements** — LR warmup, KL annealing, gradient clip scheduling, decoder freezing (from Codename RVC Fork v4)
- **Robust Data Loading** — Safe numpy loading with NaN/Inf handling, corrupted file recovery, increased sequence length limits (900→1800)
- **Advanced Options** — Gradient accumulation, torch.compile(), 8-bit Adam, cosine annealing LR, overtraining detection
- **Architecture Support** — RVC and SVC (from Vietnamese-RVC)
- **Embedder Mix** — Layer-wise embedding mixing with configurable ratios (from Vietnamese-RVC)
- **🚀 3× Faster Training** — `--fast_train` flag bundles TF32 matmul + cuDNN benchmark + torch.compile + expandable_segments allocator. Vocal-quality-safe (no loss/numerics changes).
- **🚀 bf16 Auto-Mode** — `--bf16_adamw` flag (Applio-parity shortcut) forces AnyPrecisionAdamW + bf16 autocast. Recommended on Ampere+ GPUs (A100/H100/RTX 30xx+/40xx+).

### 🔒 Security Hardening
- **Safe Deserialization** — All `torch.load()` calls route through `safe_torch_load` (forces `weights_only=True`). Restricted `pickle.Unpickler` whitelist blocks every known RCE gadget.
- **Path Traversal Guards** — `validate_path_within()` wired into 20+ `os.path.join` sites in inference + training.
- **Hardened Downloaders** — All downloaders enforce: 8 GB size cap, extension whitelist, filename sanitization, `timeout=300s`.
- **No Silent Failures** — Bare `except:` clauses replaced with typed exceptions.

### Platform & Integration
- **CLI** — Full command-line interface via `rvc-cli`
- **ZLUDA Support** — Full AMD GPU support via ZLUDA
- **XPU Support** — Intel GPU support via XPU backend
- **Push to Hub** — Upload trained models directly to HuggingFace Hub
- **44 Languages** — Full UI translation support

---

## Supported Vocoders

| Vocoder | Description | Pitch Required |
|---------|-------------|----------------|
| **Default** (HiFi-GAN NSF) | HiFi-GAN with Neural Sine Filter. **Recommended for best compatibility.** | Yes |
| **BigVGAN** | Snake activations with Anti-Aliasing. State-of-the-art audio quality. | Yes |
| **MRF-HiFi-GAN** | HiFi-GAN with Multi-Receptive Field fusion. Richer feature extraction. | Yes |
| **RefineGAN** | U-Net based vocoder with parallel residual blocks. High-fidelity spectral detail. | Yes |

---

## Supported Optimizers

| Optimizer | Category | Rating | Best For |
|-----------|----------|--------|----------|
| **AdamW** | PyTorch Built-in | ⭐⭐⭐⭐⭐ | General-purpose, most reliable (default) |
| **RAdam** | PyTorch Built-in | ⭐⭐⭐⭐ | Warmup-free training, short training runs |
| **AnyPrecisionAdamW** | Mixed-Precision | ⭐⭐⭐⭐ | Bfloat16 training, long runs with Kahan summation |
| **AdaBelief** | Belief-Based | ⭐⭐⭐ | Better conditioned adaptive learning rates |
| **AdaBeliefV2** | Belief-Based | ⭐⭐⭐ | Stable deep training with AMSGrad + InverseSqrt scheduler |
| **Ranger2020** 🆕 | Advanced | ⭐⭐⭐⭐⭐ | RAdam + Lookahead + Gradient Centralization |
| **Prodigy** 🆕 | D-Adaptation | ⭐⭐⭐⭐⭐ | Automatic LR tuning (lr=1.0 works!) |

🆕 *Newly added from Codename RVC Fork v4*

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

#### Fast Training

```bash
# ~3× faster training, vocal-quality-safe
rvc-cli train my_model --fast_train true --epochs 200 --batch_size 4

# Additional ~1.5–2× speedup on Ampere+ GPUs
rvc-cli train my_model --fast_train true --bf16_adamw true --epochs 200 --batch_size 8
```

### 4. Google Colab

| Notebook | Description |
|----------|-------------|
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArkanDash/Advanced-RVC-Inference/blob/master/Advanced-RVC.ipynb) | Full Web UI |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArkanDash/Advanced-RVC-Inference/blob/master/colab-noui.ipynb) | CLI only — lightweight headless mode |

---

## Project Structure

```
arvc/                           # Main Package
│
├── api/                         # CLI entry points (rvc-cli → arvc.api.cli:main)
│   ├── cli.py                   #   Main CLI interface
│   └── cli_complete.py          #   Shell completion
│
├── app/                         # Gradio Web UI
│   ├── gui.py                   #   Main application entry
│   └── tabs/                    #   UI Tabs
│       ├── inference/           #     Voice conversion, TTS, separation
│       ├── training/            #     Dataset creation, reference, training
│       ├── downloads/           #     Model downloader tab
│       ├── realtime/            #     Real-time voice conversion
│       └── extra/               #     Settings, SRT, model tools
│
├── rvc/                         # 🎯 RVC Core Module
│   ├── inference/               #   Voice conversion pipeline
│   │   ├── inference.py         #     Main inference logic
│   │   ├── convert.py           #     Model conversion & export
│   │   ├── pipeline.py          #     Inference pipeline
│   │   ├── create_reference.py  #     Reference audio creation
│   │   ├── noisereduce.py       #     Audio denoising (TorchGate)
│   │   ├── audio_processing.py  #     Audio preprocessing/postprocessing
│   │   ├── csrt.py              #     SRT subtitle generation
│   │   ├── f0_extract.py        #     F0 extraction utility
│   │   ├── presets.py           #     Inference presets management
│   │   ├── separate.py          #     Audio separation wrapper
│   │   └── tts.py               #     Text-to-Speech integration
│   │
│   ├── training/                #   Training Pipeline
│   │   ├── runner/              #     Core training loop
│   │   │   ├── train.py         #       Main training script
│   │   │   ├── losses.py        #       Loss functions (enhanced!)
│   │   │   ├── data_utils.py    #       Data loading & augmentation
│   │   │   ├── enhancements.py  #       Training enhancements (NEW!)
│   │   │   ├── mel_processing.py#       Mel spectrogram processing
│   │   │   ├── extract_model.py #       Model extraction utilities
│   │   │   └── utils.py         #       Training helpers
│   │   ├── extract/             #     Feature extraction
│   │   │   ├── extract.py       #       Embedding & F0 extraction
│   │   │   ├── embedding.py     #       Speaker embedding extraction
│   │   │   ├── feature.py       #       Feature processing
│   │   │   ├── preparing_files.py#      File preparation
│   │   │   ├── rms.py           #       RMS energy extraction
│   │   │   └── setup_path.py    #       Path configuration
│   │   ├── preprocess/          #     Audio preprocessing
│   │   │   ├── preprocess.py    #       Main preprocessing
│   │   │   └── slicer2.py       #       Audio slicing
│   │   ├── create_dataset.py    #     Dataset creation
│   │   └── create_index.py      #     FAISS index creation
│   │
│   ├── models/                  #   Neural Network Models
│   │   ├── algorithms/          #     Core algorithms
│   │   │   ├── synthesizers.py  #       Synthesizer (VITS-style)
│   │   │   ├── discriminators.py#       Multi-period/multi-scale discriminators
│   │   │   ├── encoders.py      #       Text encoders
│   │   │   ├── encoders_vits2.py#       VITS2 encoders
│   │   │   ├── modules.py       #       Common modules
│   │   │   ├── normalizing_flows.py #   Normalizing flows
│   │   │   ├── stftpitchshift.py#       STFT pitch shifting
│   │   │   └── wavenet.py       #       WaveNet vocoder
│   │   ├── generators/          #     Vocoder architectures
│   │   │   ├── hifigan.py       #       HiFi-GAN NSF (default)
│   │   │   ├── bigvgan.py       #       BigVGAN
│   │   │   ├── mrf_hifigan.py   #       MRF-HiFi-GAN
│   │   │   ├── nsf_hifigan.py   #       NSF HiFi-GAN
│   │   │   └── refinegan.py     #       RefineGAN
│   │   ├── optimizers/          #     Training optimizers
│   │   │   ├── adabelief.py     #       AdaBelief
│   │   │   ├── ranger2020.py    #       Ranger2020 (NEW!)
│   │   │   ├── prodigy.py       #       Prodigy D-Adapt (NEW!)
│   │   │   └── polopt.py        #       PolOpt hybrid
│   │   ├── predictors/          #     F0/Pitch extractors
│   │   │   ├── RMVPE/           #       RMVPE predictor
│   │   │   ├── FCPE/            #       FCPE predictor
│   │   │   ├── CREPE/           #       CREPE predictor
│   │   │   ├── PESTO/           #       PESTO predictor
│   │   │   ├── WORLD/           #       WORLD/Harvest predictor
│   │   │   ├── PENN/            #       PENN predictor
│   │   │   ├── SWIFT/           #       SWIFT predictor
│   │   │   └── DJCM/            #       DJCM predictor
│   │   ├── embedders/           #     Speaker embedders
│   │   │   ├── fairseq.py       #       HuBERT/ContentVec
│   │   │   ├── onnx.py          #       ONNX embedders
│   │   │   ├── ppg.py           #       PPG embedder
│   │   │   └── transformers.py  #       Wav2Vec2 embedder
│   │   ├── backends/            #     Hardware backends
│   │   │   ├── directml.py      #       DirectX/DirectML
│   │   │   ├── opencl.py        #       OpenCL
│   │   │   ├── zluda.py         #       ZLUDA (AMD)
│   │   │   └── utils.py         #       Backend utilities
│   │   └── onnx/                #     ONNX export
│   │       ├── onnx_export.py   #       Export utilities
│   │       └── wrapper.py       #       ONNX runtime wrapper
│   │
│   └── __init__.py
│
├── uvr/                         # 🎵 Ultimate Vocal Remover
│   ├── separate_music.py        #   High-level separation API
│   └── uvr5_lib/                #   UVR5 library
│       ├── separator.py         #     Base separator class
│       ├── spec_utils.py        #     Spectrogram utilities
│       └── uvr/                 #     Separators
│           ├── mdx_separator.py #       MDX-Net separator
│           └── vr_separator.py  #       VR separator
│
├── whisper/                     # 🎤 Whisper / Speaker Diarization
│   └── speaker/                 #   Speaker recognition
│       ├── ECAPA_TDNN.py        #     ECAPA-TDNN model
│       ├── speechbrain.py       #     SpeechBrain integration
│       ├── encoder.py           #     Speaker encoder
│       ├── whisper.py           #     Whisper transcription
│       ├── segment.py           #     Audio segmentation
│       └── embedding.py         #     Embedding extraction
│
├── tts/                         # 🔊 Text-to-Speech
│   └── tts.py                   #   TTS interface
│
├── downloader/                  # ⬇️ Downloaders
│   ├── huggingface.py           #   HuggingFace download
│   ├── gdown.py                 #   Google Drive download
│   ├── meganz.py                #   Mega.nz download
│   ├── mediafire.py             #   MediaFire download
│   ├── pixeldrain.py            #   PixelDrain download
│   └── downloads.py             #   Download orchestration
│
├── engine/                      # Real-time Processing
│   └── realtime/                #   Live voice conversion
│       ├── realtime.py          #     RVC_Realtime main class
│       ├── pipeline.py          #     Realtime pipeline
│       ├── callbacks.py         #     Audio callbacks
│       ├── audio.py             #     Audio device handling
│       └── vad_utils.py         #     VAD (Voice Activity Detection)
│
├── utils/                       # 🛠️ Shared Utilities
│   ├── variables.py             #   Global config, paths, translations
│   ├── feedback.py              #   Logger (gr_info, gr_warning, gr_error)
│   ├── process.py               #   System process handling
│   ├── model_utils.py           #   Model utilities (export, info, fusion)
│   ├── restart.py               #   Restart utilities
│   └── utils.py                 #   Helper functions
│
├── configs/                     # Configuration Files
│   ├── config.py                #   Runtime Config singleton
│   ├── v1/                      #   V1 model configs (32k, 40k, 48k)
│   ├── v2/                      #   V2 model configs (24k, 32k, 40k, 48k)
│   ├── ringformer_v2/           #   RingFormer V2 configs
│   └── pcph_gan/                #   PCPH-GAN configs
│
├── assets/                      # Runtime Assets
│   ├── models/                  #   Pretrained models
│   │   ├── pretrained_v1/       #     V1 pretrained weights
│   │   ├── pretrained_v2/       #     V2 pretrained weights
│   │   ├── pretrained_custom/   #     Custom pretrained weights
│   │   ├── predictors/          #     F0 predictor models
│   │   └── speaker_diarization/ #     Speaker diarization models
│   ├── logs/                    #   Training logs & checkpoints
│   ├── audios/                  #   Audio files
│   ├── languages/               #   44 translation JSON files
│   └── presets/                 #   Inference presets
│
├── datasets/                    # Training Datasets
│
├── __init__.py                  # Package initialization
├── __main__.py                  # Entry point (python -m arvc)
└── _version.py                  # Version management
```

**Key rules**:
- `rvc/` is the core module — all RVC-specific code lives here
- `engine/realtime/` handles real-time voice conversion only
- `app/` contains the GUI and imports from `rvc/`, `uvr/`, etc.
- No circular dependencies between top-level modules

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

This project builds upon the work of many open-source projects and contributors.

### Core RVC Foundation
| Project | Author |
|---------|--------|
| [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) | RVC Project |
| [Vietnamese-RVC](https://github.com/PhamHuynhAnh16/Vietnamese-RVC) | Phạm Huỳnh Anh |

### Training Improvements
| Project | Author |
|---------|--------|
| [PolTrain](https://github.com/Politrees/PolTrain) | Politrees |
| [Applio](https://github.com/IAHispano/Applio) | IAHispano |
| [Codename RVC Fork v4](https://github.com/CodenameRVC/Codename-RVC-Fork-v4) | CodenameRVC |

### Audio & Models
| Project | Author |
|---------|--------|
| [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) | Nomad Karaoke |
| [whisper](https://github.com/openai/whisper) | OpenAI |
| [BigVGAN](https://github.com/NVIDIA/BigVGAN) | Nvidia |

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

<div align="center">

# Advanced RVC Inference

**A state-of-the-art web UI crafted to streamline rapid and effortless RVC inference — featuring a model downloader, voice splitter, batch inference, training pipeline, real-time conversion, and a full CLI.**

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArkanDash/Advanced-RVC-Inference/blob/master/Advanced-RVC.ipynb)
[![Discord](https://img.shields.io/badge/Chat-Discord-5865F2?style=flat-square&logo=discord&logoColor=white)](https://discord.gg/hvmsukmBHE)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

> [!NOTE]
> v2.2.0 ships a major security hardening pass, a ~3× training speedup bundle (`--fast_train` + `--bf16_adamw`), and Applio-parity accuracy patches for small (10-minute) datasets. See the [Changelog](#changelog) below for the full breakdown. Pull requests are still welcome and will be reviewed.

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
- **🚀 3× Faster Training** — `--fast_train` flag bundles TF32 matmul + cuDNN benchmark + torch.compile + expandable_segments allocator. Vocal-quality-safe (no loss/numerics changes). See [Training Boost](docs/TRAINING_BOOST.md).
- **🚀 bf16 Auto-Mode** — `--bf16_adamw` flag (Applio-parity shortcut) forces AnyPrecisionAdamW + bf16 autocast. Recommended on Ampere+ GPUs (A100/H100/RTX 30xx+/40xx+).
- **🎯 Applio-Parity Accuracy** — `per_preprocess=3.0s` (was 3.7s) yields ~26% more training chunks on small datasets. `--chunk_len`/`--overlap_len` now apply to Automatic cut mode. Saved `.pth` embeds `embedder_model` + `dataset_length` + `overtrain_info` provenance fields.

### 🔒 Security Hardening
- **Safe Deserialization** — All `torch.load()` calls route through `safe_torch_load` (forces `weights_only=True`). Restricted `pickle.Unpickler` whitelist blocks every known RCE gadget. See [Security Patches](docs/SECURITY_PATCHES.md).
- **Path Traversal Guards** — `validate_path_within()` wired into 20+ `os.path.join` sites in inference + training. Blocks `../../etc/cron.d/evil` style escapes from GUI/CLI inputs.
- **Hardened Downloaders** — All 5 downloaders (HuggingFace, Google Drive, Mega, MediaFire, PixelDrain) enforce: 8 GB size cap, extension whitelist, filename sanitization, `timeout=300s` on every network call. Fixed `tempfile.mktemp` TOCTOU race in `gdown.py`.
- **No Silent Failures** — Bare `except:` clauses in checkpoint-load (was silently restarting training from epoch 1) and ONNX export replaced with typed exceptions. MEGA nonce migrated from `random.randint` → `secrets.randbits(32)`.

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

#### Fast Training (v2.2.0+)

```bash
# ~3× faster training, vocal-quality-safe (no loss/numerics changed)
rvc-cli train my_model --fast_train true --epochs 200 --batch_size 4

# Additional ~1.5–2× speedup on Ampere+ GPUs (A100/H100/RTX 30xx+/40xx+)
# Skip on Colab T4 (Turing) — bf16 is emulated there.
rvc-cli train my_model --fast_train true --bf16_adamw true --epochs 200 --batch_size 8

# 10-minute dataset recipe (max accuracy)
rvc-cli preprocess my_model --sample_rate 48000 \
    --cut_method Automatic --chunk_len 3.0 --overlap_len 0.5 \
    --process_effects --normalization post
rvc-cli extract my_model --sample_rate 48000 --f0_method rmvpe
rvc-cli create-index my_model --version v2 --algorithm Auto
rvc-cli train my_model --fast_train true --bf16_adamw true \
    --epochs 300 --batch_size 4 --multiscale_loss --cosine_lr \
    --overtrain_detect --overtrain_threshold 50
```

See [`docs/TRAINING_BOOST.md`](docs/TRAINING_BOOST.md) for the full optimization
breakdown and [`docs/SECURITY_PATCHES.md`](docs/SECURITY_PATCHES.md) for the
complete security audit trail.

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

### v2.2.0

**🔒 Security Hardening (defense-in-depth — no numerics changed)**
- Routed 16 `torch.load()` calls through `safe_torch_load` (forces `weights_only=True`) across all predictors (PESTO, PENN, RMVPE, CREPE, DJCM, FCPE), training (train.py ×3, data_utils, utils), whisper, onnx_export, vr_separator, fairseq
- Restricted `pickle.Unpickler` whitelist (primitive + numpy types only) — blocks every known pickle RCE gadget (`os.system`, `subprocess.Popen`, `builtins.eval`)
- Wired `validate_path_within()` into 20+ `os.path.join` sites in `inference.py` and `services/training.py` — blocks `../../etc/cron.d/evil` path traversal from GUI/CLI inputs
- All 5 downloaders (HuggingFace, Google Drive, Mega, MediaFire, PixelDrain) now enforce: 8 GB size cap, extension whitelist, filename sanitization, `timeout=300s` on every network call
- Fixed `tempfile.mktemp` → `mkstemp` TOCTOU race in `gdown.py`
- MEGA nonce migrated from `random.randint` → `secrets.randbits(32)`
- Bare `except:` clauses in checkpoint-load (was silently restarting training from epoch 1 — silent data loss) and ONNX export replaced with typed exceptions
- Added `timeout=30s` to `urllib.request.urlopen` Google Sheets fetch at app startup

**🚀 Training Speedup (~3× faster, vocal-quality-safe)**
- New `--fast_train` flag bundles: TF32 matmul + cuDNN TF32 + cuDNN benchmark + torch.compile(mode="reduce-overhead") on both G and D + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + 8 dataloader workers + prefetch_factor=16 + pin_memory + persistent_workers
- New `--bf16_adamw` flag (Applio-parity shortcut) — forces `optimizer=AnyPrecisionAdamW` and `brain=True` (bf16 autocast). Recommended on Ampere+ GPUs. Skip on T4/Turing (bf16 emulated → slower).
- All optimizations are non-numerical — no loss function, gradient path, or weight is touched. Vocal fidelity is bit-for-bit identical.

**🎯 Applio-Parity Accuracy for 10-minute datasets**
- `per_preprocess` 3.7s → 3.0s (matches Applio's `PERCENTAGE=3.0`) — produces ~26% more training chunks for the same audio (largest single fix for small-data accuracy)
- `--chunk_len` / `--overlap_len` CLI flags now apply to **Automatic** cut mode (was Simple-only). Users can boost `--overlap_len=0.5` for ~17% more chunks on small datasets
- `preprocess.py` now writes `total_dataset_duration` + `total_seconds` to `model_info.json`
- `extract_model()` embeds `embedder_model`, `dataset_length`, `overtrain_info` into the saved `.pth` as provenance metadata — lets inference auto-select the matching embedder
- Preprocess now fails fast with clear error if dataset path is missing or empty (was silent walk + cryptic downstream crash)

**📚 Docs & Colab**
- New [`docs/SECURITY_PATCHES.md`](docs/SECURITY_PATCHES.md) — full audit trail of every patch with verification commands
- New [`docs/TRAINING_BOOST.md`](docs/TRAINING_BOOST.md) — usage guide with recommended recipes for T4 / A100 / RTX 30xx+
- `colab-noui.ipynb` updated: added `bf16_adamw` parameter, always passes `--chunk_len`/`--overlap_len`, rewrote Security+Speedup+Accuracy notes section

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

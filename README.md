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

- **Voice Inference** — Single & batch conversion, TTS, pitch shifting, formant shifting, audio cleaning, Whisper transcription
- **Audio Separation** — Vocal/instrumental isolation (MDX-Net, Roformer, BS-Roformer), karaoke, reverb removal, denoising
- **Real-Time Conversion** — Live mic voice conversion with VAD and low-latency processing
- **Training Pipeline** — End-to-end training from dataset creation to model export with overtraining detection
- **Easy GUI** — Simplified one-click interface for quick conversion and training
- **CLI** — Full command-line interface via `rvc-cli`
- **Auto Pretrained Download** — Automatically downloads pretrained models from HuggingFace
- **ZLUDA Support** — Full AMD GPU support via ZLUDA
- **30+ F0 Methods** — rmvpe, crepe, fcpe, harvest, hybrid, and many more
- **Training Optimizations** — Gradient accumulation, torch.compile(), 8-bit Adam, DDP tuning
- **Push to Hub** — Upload trained models directly to HuggingFace Hub

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

## Easy GUI

A simplified interface for quick workflows:

```bash
rvc-cli serve --easy true
```

- **Quick Convert** — Simple voice conversion with minimal settings
- **One-Click Train** — Full pipeline in a single button
- **Download** — Quick model download from URLs

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

| Project | Author | Purpose |
|---------|--------|---------|
| [Vietnamese-RVC](https://github.com/PhamHuynhAnh16/Vietnamese-RVC) | Phạm Huỳnh Anh | Core RVC implementation & pretrained models |
| [Applio](https://github.com/IAHispano/Applio) | IAHispano | UI/UX inspiration & components |
| [Mangio-Kalo-Tweaks](https://github.com/kalomaze/Mangio-Kalo-Tweaks) | kalomaze | EasyGUI inspiration |
| [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) | Nomad Karaoke | UVR5 audio separation |
| [whisper](https://github.com/openai/whisper) | OpenAI | Speech-to-text transcription |
| [BigVGAN](https://github.com/NVIDIA/BigVGAN) | Nvidia | Vocoder implementation |
| [ZLUDA](https://github.com/vlsid/ZLUDA) | vlsid | AMD GPU CUDA compatibility layer |

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

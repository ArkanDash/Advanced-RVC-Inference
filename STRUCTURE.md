# Advanced-RVC-Inference - Folder Structure

## 📁 Organized Structure (v2.0)

```
Advanced-RVC-Inference/
│
├── arvc/                    # Main Application Package
│   ├── api/                 # CLI & API interfaces
│   ├── app/                 # GUI Application (Gradio)
│   │   └── tabs/            # UI Tabs (training, inference, etc.)
│   ├── assets/              # Static assets (models, configs, audio)
│   │   ├── audios/          # Sample audio files
│   │   ├── models/          # Pretrained models storage
│   │   └── logs/            # Training logs
│   ├── configs/             # Model configurations (v1/v2/custom)
│   ├── datasets/            # Dataset handling
│   ├── engine/              # Engine wrappers (legacy - use rvc/)
│   ├── services/            # High-level services
│   └── utils/               # Utilities (legacy - use Utils/)
│
├── rvc/                     # 🎯 RVC Core Module
│   ├── models/              # Neural Network Models
│   │   ├── algorithms/      # Core algorithms (synthesizers, discriminators)
│   │   │   ├── conformer/   # Conformer attention modules
│   │   │   ├── generators/  # Generator architectures (HiFi-GAN, etc.)
│   │   │   └── discriminators/
│   │   ├── predictors/      # F0/Pitch extractors
│   │   │   ├── RMVPE/       # RMVPE predictor
│   │   │   ├── FCPE/        # FCPE predictor
│   │   │   ├── CREPE/       # CREPE predictor
│   │   │   ├── PESTO/       # PESTO predictor
│   │   │   ├── WORLD/       # WORLD (Harvest) predictor
│   │   │   ├── PENN/        # PENN predictor
│   │   │   ├── SWIFT/       # SWIFT predictor
│   │   │   └── DJCM/        # DJCM predictor
│   │   ├── optimizers/       # Training optimizers
│   │   │   ├── adamw.py     # AdamW variants
│   │   │   ├── ranger2020.py # Ranger2020 (NEW!)
│   │   │   ├── prodigy.py   # Prodigy D-Adaptation (NEW!)
│   │   │   └── polopt.py    # PolOpt hybrid
│   │   ├── embedders/       # Speaker embedders
│   │   ├── backends/        # Hardware backends (CUDA, ZLUDA, etc.)
│   │   └── onnx/            # ONNX export utilities
│   │
│   ├── training/            # Training Pipeline
│   │   ├── runner/          # Main training loop
│   │   │   ├── train.py     # Training script
│   │   │   ├── losses.py    # Loss functions (ENHANCED!)
│   │   │   ├── data_utils.py # Data loading
│   │   │   └── enhancements.py # Training enhancements (NEW!)
│   │   ├── extract/          # Feature extraction
│   │   └── preprocess/       # Audio preprocessing
│   │
│   └── inference/           # Inference pipeline
│       ├── inference.py     # Main inference
│       ├── pipeline.py      # Inference pipeline
│       └── convert.py       # Model conversion
│
├── uvr/                     # 🎵 UVR (Ultimate Vocal Remover)
│   └── uvr5_lib/            # UVR5 library
│       ├── uvr/             # UVR separators (MDX, VR)
│       └── vr_network/      # VR network architecture
│
├── whisper/                 # 🎤 Whisper / Speaker Diarization
│   └── speaker/             # Speaker recognition
│       ├── ECAPA_TDNN.py    # ECAPA-TDNN model
│       ├── speechbrain.py   # SpeechBrain integration
│       └── encoder.py       # Speaker encoder
│
├── tts/                     # 🔊 Text-to-Speech
│   └── tts.py               # TTS implementation
│
├── downloader/              # ⬇️ Downloaders & Tools
│   ├── huggingface.py       # HuggingFace download
│   ├── meganz.py            # Mega.nz download
│   ├── pixeldrain.py        # PixelDrain download
│   ├── mediafire.py         # MediaFire download
│   ├── gdown.py             # Google Drive download
│   └── model_download.py    # Model download manager
│
├── Utils/                   # 🛠️ General Utilities
│   ├── variables.py         # Global variables & config
│   ├── feedback.py          # User feedback system
│   └── downloaders/         # (moved to downloader/)
│
├── configs/                 # Configuration files
├── docs/                    # Documentation
├── notebook/                # Jupyter notebooks
└── assets/                  # Binary assets
```

---

## 🚀 Quick Import Guide

### RVC Models & Training
```python
# Optimizers (8 available!)
from rvc.models.optimizers import get_optimizer_class, OPTIMIZER_REGISTRY
from rvc.models.optimizers.ranger2020 import Ranger2020
from rvc.models.optimizers.prodigy import Prodigy

# Loss Functions (enhanced)
from rvc.training.runner.losses import (
    phase_loss, envelope_loss, kl_loss_fb,
    MultiScaleSTFTLoss, EnhancedLossCalculator
)

# Training Enhancements (NEW!)
from rvc.training.runner.enhancements import (
    LRWarmupScheduler, KLAnnealer, GradientClipScheduler,
    DecoderFreezer, RollingLossTracker, TrainingEnhancements
)

# Predictors
from rvc.models.predictors.RMVPE import RMVE
from rvc.models.predictors.FCPE import FCPE

# Algorithms
from rvc.models.algorithms.synthesizers import Synthesizer
from rvc.models.algorithms.discriminators import MultiPeriodDiscriminator
```

### UVR (Vocal Separation)
```python
from uvr.uvr5_lib.separator import Separator
from uvr.uvr5_lib.vr.vr_separator import VRSeparator
```

### Whisper/Speaker
```python
from whisper.speaker.ECAPA_TDNN import ECAPA_TDNN
```

### TTS
```python
from tts.tts import TTSInterface
```

### Downloaders
```python
from downloader.huggingface import HF_download_file
from downloader.gdown import gdown_download
```

---

## 📦 What's New in v2.0

### New Optimizers (from Codename RVC Fork v4)
- **Ranger2020**: RAdam + Lookahead + Gradient Centralization
- **Prodigy**: D-Adaptation with automatic LR tuning (lr=1.0!)

### New Loss Functions
- `phase_loss`: Phase coherence for RingFormer vocoders
- `envelope_loss`: Waveform envelope matching
- `kl_loss_fb`: Free-bits KL divergence (prevents collapse)
- `MultiScaleSTFTLoss`: Multi-resolution spectral loss

### Training Enhancements
- LR Warmup scheduler
- KL Annealing (cyclical cosine)
- Gradient Clipping Scheduler (two-phase)
- Decoder Layer Freezing/Slowing
- Rolling Loss Tracking
- Best Step per Epoch tracking

---

## 🔄 Migration Notes

**Old imports still work!** The `arvc/engine/*` paths are kept for backward compatibility.

To use the new structure:
```python
# Old way (still works)
from arvc.engine.models.optimizers import AdamW

# New way (recommended)
from rvc.models.optimizers import AdamW
```

---

*Last updated: 2026-08-21*
*Version: 2.0 - Restructured*

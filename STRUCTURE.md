# Advanced-RVC-Inference - Folder Structure

## 📁 arvc/ Package Structure (v2.1)

```
Advanced-RVC-Inference/
│
├── arvc/                    # 📦 MAIN PACKAGE (Everything is here!)
│   │
│   ├── __init__.py          # Package init
│   ├── __main__.py          # Entry point
│   ├── _version.py          # Version info
│   │
│   ├── api/                 # CLI & API interfaces
│   ├── app/                 # GUI Application (Gradio)
│   │   └── tabs/            # UI Tabs
│   ├── assets/              # Static assets
│   │   ├── audios/          # Sample audio files
│   │   ├── models/          # Pretrained models storage
│   │   └── logs/            # Training logs
│   ├── configs/             # Model configurations (v1/v2/custom)
│   ├── datasets/            # Dataset handling
│   ├── engine/              # Engine wrappers (legacy)
│   │   ├── models/          # Model definitions
│   │   │   └── optimizers/  # Optimizers (Ranger2020, Prodigy)
│   │   └── training/
│   │       └── runner/      # Training loop + enhancements
│   ├── services/            # High-level services
│   ├── ui/                  # UI components
│   ├── utils/               # Internal utilities
│   │
│   ├── rvc/                 # 🎯 RVC Core Module
│   │   ├── models/          # Neural Network Models
│   │   │   ├── algorithms/  # Core algorithms
│   │   │   │   ├── conformer/
│   │   │   │   ├── generators/
│   │   │   │   └── discriminators/
│   │   │   ├── predictors/  # F0/Pitch extractors
│   │   │   │   ├── RMVPE/
│   │   │   │   ├── FCPE/
│   │   │   │   ├── CREPE/
│   │   │   │   ├── PESTO/
│   │   │   │   ├── WORLD/
│   │   │   │   ├── PENN/
│   │   │   │   ├── SWIFT/
│   │   │   │   └── DJCM/
│   │   │   ├── optimizers/  # Training optimizers
│   │   │   ├── embedders/  # Speaker embedders
│   │   │   ├── backends/    # Hardware backends
│   │   │   └── onnx/        # ONNX export
│   │   ├── training/        # Training Pipeline
│   │   │   ├── runner/      # Main training loop
│   │   │   ├── extract/     # Feature extraction
│   │   │   └── preprocess/  # Audio preprocessing
│   │   └── inference/       # Inference pipeline
│   │
│   ├── uvr/                 # 🎵 UVR (Ultimate Vocal Remover)
│   │   └── uvr5_lib/        # UVR5 library
│   │       ├── uvr/         # UVR separators
│   │       └── vr_network/  # VR network
│   │
│   ├── whisper/             # 🎤 Whisper / Speaker Diarization
│   │   └── speaker/         # Speaker recognition
│   │       ├── ECAPA_TDNN.py
│   │       ├── speechbrain.py
│   │       └── encoder.py
│   │
│   ├── tts/                 # 🔊 Text-to-Speech
│   │   └── tts.py
│   │
│   ├── downloader/          # ⬇️ Downloaders & Tools
│   │   ├── huggingface.py
│   │   ├── meganz.py
│   │   ├── pixeldrain.py
│   │   ├── mediafire.py
│   │   └── gdown.py
│   │
│   └── Utils/               # 🛠️ General Utilities
│       ├── variables.py
│       └── feedback.py
│
├── configs/                 # Root config files
├── docs/                    # Documentation
├── notebook/                # Jupyter notebooks
├── README.md                # Main readme
├── pyproject.toml           # Package config
├── requirements.txt         # Dependencies
└── installer.bat            # Windows installer
```

---

## 🚀 Import Guide (from arvc package)

### RVC Models & Training
```python
from arvc.rvc.models.optimizers import Ranger2020, Prodigy
from arvc.rvc.training.runner.losses import EnhancedLossCalculator
from arvc.rvc.training.runner.enhancements import TrainingEnhancements
from arvc.rvc.models.predictors.RMVPE import RMVPE
```

### UVR (Vocal Separation)
```python
from arvc.uvr.uvr5_lib.separator import Separator
```

### Whisper/Speaker
```python
from arvc.whisper.speaker.ECAPA_TDNN import ECAPA_TDNN
```

### TTS
```python
from arvc.tts.tts import TTSInterface
```

### Downloaders
```python
from arvc.downloader.huggingface import HF_download_file
```

---

## ✅ v2.1 Fixes
- All modules now properly inside `arvc/` package
- Clean hierarchical structure
- Single package import: `import arvc`

---
*Version: 2.1 - Fixed Package Structure*

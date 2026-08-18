# Package Structure Reference

This document is the canonical reference for the layout of the `arvc/` Python
package. It is intended for contributors who need to know where to add new
code, where to find existing code, and what the dependency rules are between
subpackages.

If you only need a high-level overview, see the "Project Structure" section of
the top-level [README.md](../README.md). This document is the deep-dive
companion.

---

## 1. Top-level layout

```
arvc/
├── __init__.py             # Public API surface + lazy entry points
├── __main__.py             # `python -m arvc` dispatcher
├── _version.py             # Single source of truth for __version__
│
├── api/                    # CLI entry points (no business logic)
├── app/                    # Gradio web UI (tabs, pages, layouts)
├── assets/                 # Static runtime assets (models, audios, languages, …)
├── configs/                # Model + runtime config (v1/v2 JSON, Config class)
├── datasets/               # Training dataset root (path placeholder)
├── engine/                 # Core logic (no UI dependency)
├── services/               # Service layer (bridges UI/CLI ↔ engine)
├── ui/                     # UI helpers (feedback, dropdown updates, formatting)
└── utils/                  # Shared utilities (variables, feedback, downloaders)
```

### Dependency direction (MANDATORY)

```
        app/  ─────►  services/  ─────►  engine/
         │                                ▲
         ▼                                │
        ui/  ─────►  utils/  ─────────────┘
        api/  ─────►  services/  ─────►  engine/
```

**Rules:**

- `engine/` **MUST NEVER** import from `app/`, `services/`, `ui/`, or `api/`.
  It is the dependency-free core.
- `services/` may import from `engine/` and `utils/`, but NOT from `app/` or
  `ui/` (it is the bridge to those, not the other way around).
- `app/` and `api/` may import from anywhere except `app/` siblings (each tab
  is independent).
- `utils/` may import from `engine/` (for path resolution), but NOT from
  `services/`, `app/`, or `ui/`.

---

## 2. Subpackage reference

### 2.1 `arvc/api/` — CLI entry points

Holds the command-line interface. Each module here is a thin argparse wrapper
that dispatches into `arvc.services.*` or `arvc.engine.*`.

```
api/
├── __init__.py
├── cli.py            # Main `rvc-cli` entry point
└── cli_complete.py   # Full-featured CLI variant with all subcommands
```

**Registered entry points** (in `pyproject.toml`):

- `rvc-cli` → `arvc.api.cli:main`
- `rvc-gui` → `arvc.app.gui:launch`

### 2.2 `arvc/app/` — Gradio web UI

Holds all Gradio UI code. Each tab is a self-contained module that imports
its service-layer counterparts.

```
app/
├── __init__.py
├── gui.py                    # Top-level Gradio Blocks launcher
├── mainjs.py                 # JavaScript injected into the Gradio page
├── run_tensorboard.py        # Launches TensorBoard in a subprocess
└── tabs/
    ├── __init__.py
    ├── downloads/downloads.py
    ├── extra/extra.py        # Container for "extra" sub-tabs
    ├── extra/child/
    │   ├── convert_model.py
    │   ├── create_srt.py
    │   ├── f0_extract.py
    │   ├── fushion.py
    │   ├── read_model.py
    │   └── settings.py
    ├── inference/inference.py
    ├── inference/child/
    │   ├── convert.py
    │   ├── convert_tts.py
    │   ├── convert_with_whisper.py
    │   └── separate.py
    ├── realtime/
    │   ├── realtime.py
    │   └── realtime_client.py
    └── training/
        ├── training.py
        └── child/
            ├── create_dataset.py
            ├── create_reference.py
            └── training.py
```

### 2.3 `arvc/assets/` — Static runtime assets

Static files used at runtime. Mostly gitkeep-tracked empty directories that
fill up with user data, plus bundled defaults (languages, mute npy files,
binary helpers).

```
assets/
├── audios/{others,rvc,tts,uvr}/   # User audio workspace (input/output/TTS/UVR)
├── binary/                          # Bundled binaries (e.g., ZLUDA helpers on Windows)
├── f0/                              # F0 extraction cache
├── languages/                       # 44 translation JSON files (<locale>.json)
├── logs/                            # Training output (checkpoints, weights, indexes, feature caches)
│   └── mute/                        # Bundled "silent" training augmentation data
│       ├── energy/
│       ├── f0/, f0_voiced/
│       ├── sliced_audios/, sliced_audios_16k/
│       └── v1_extracted/, v2_extracted/
├── models/
│   ├── embedders/                   # ContentVec / HuBERT embedder models
│   ├── predictors/                  # F0 predictor models (RMVPE, FCPE, CREPE, …)
│   ├── pretrained_v1/               # V1 pretrained G/D weights
│   ├── pretrained_v2/               # V2 pretrained G/D weights
│   ├── pretrained_custom/           # User-supplied custom pretrained weights
│   ├── speaker_diarization/         # Speaker-diarization model + assets
│   └── uvr5/                        # UVR5 separation models
├── presets/                         # Inference effect presets (.json)
└── zluda/                           # ZLUDA launcher for AMD GPUs (Windows)
```

### 2.4 `arvc/configs/` — Model & runtime configuration

Two distinct kinds of config live here:

1. **Per-version model JSONs** (`v1/`, `v2/`, `ringformer_v2/`, `pcph_gan/`)
   define the architecture hyperparameters for each supported model version
   and sample rate (e.g., `v2/48000.json`).
2. **Runtime `Config` class** in `config.py` — a singleton that detects
   available devices, providers, paths, and translations. Used everywhere
   as `from arvc.configs.config import Config`.

```
configs/
├── __init__.py
├── config.py             # Config singleton (device, providers, paths, translations)
├── v1/                   # V1 model JSONs (32k, 40k, 48k)
├── v2/                   # V2 model JSONs (24k, 32k, 40k, 48k)
├── ringformer_v2/        # RingFormer V2 JSONs
└── pcph_gan/             # PCPH-GAN JSONs
```

> **Note:** The user-facing `config.json` (with paths, language, theme) lives
> at the project root, NOT inside this package. The `Config` class in
> `arvc/configs/config.py` loads it at runtime.

### 2.5 `arvc/datasets/` — Training dataset root

A path-placeholder package. Users place per-model training datasets under
`arvc/datasets/<model_name>/`. The `__init__.py` exposes `DATASETS_PATH` so
other modules can resolve dataset paths without hardcoding.

### 2.6 `arvc/engine/` — Core logic (NO UI dependency)

The heart of the project. All actual ML work happens here.

```
engine/
├── __init__.py
│
├── inference/                # Voice-conversion inference
│   ├── __init__.py
│   ├── audio_processing.py   # Audio I/O helpers (load, resample, format convert)
│   ├── convert.py            # Top-level convert_audio / convert_selection
│   ├── create_reference.py   # Reference set creation
│   ├── inference.py          # Core inference loop (single + batch + TTS)
│   ├── noisereduce.py        # TorchGate noise-reduction utility
│   └── pipeline.py           # Pipeline orchestration (model load → infer → postprocess)
│
├── models/                   # All model definitions + loaders
│   ├── __init__.py
│   ├── safe_load.py          # safe_torch_load + pickle.Unpickler whitelist
│   ├── utils.py              # Model load/download/cache helpers
│   ├── weight_norm.py        # Weight-norm configure/strip helpers
│   ├── algorithms/           # VITS-style algorithm primitives
│   │   ├── attentions.py, commons.py, modules.py, normalization.py
│   │   ├── residuals.py, synthesizers.py, encoders.py, encoders_vits2.py
│   │   ├── discriminators.py, normalizing_flows.py, wavenet.py
│   │   ├── stftpitchshift.py, PchipF0UpsamplerTorch.py
│   │   ├── conformer/        # Conformer block
│   │   └── generators/       # Generator-internal blocks
│   ├── backends/             # Device backends
│   │   ├── directml.py, opencl.py, zluda.py, utils.py
│   ├── embedders/            # Content embedders
│   │   ├── fairseq.py, onnx.py, ppg.py, transformers.py
│   ├── generators/           # Vocoders
│   │   ├── hifigan.py, bigvgan.py, mrf_hifigan.py, refinegan.py, nsf_hifigan.py
│   ├── onnx/                 # ONNX export + wrapper
│   │   ├── onnx_export.py, wrapper.py
│   ├── optimizers/           # Custom optimizers
│   │   ├── adabelief.py, adabeliefv2.py, anyprecision_optimizer.py, polopt.py
│   └── predictors/           # F0 predictors (each in its own subpackage)
│       ├── Generator.py      # Top-level F0 generator facade
│       ├── CREPE/, DJCM/, FCPE/, PENN/, PESTO/, RMVPE/, SWIFT/, WORLD/
│
├── realtime/                 # Realtime voice-conversion server
│   ├── __init__.py
│   ├── audio.py              # Audio capture/playback
│   ├── callbacks.py          # Stream callbacks
│   ├── pipeline.py           # Realtime pipeline
│   ├── realtime.py           # Server entry point
│   └── vad_utils.py          # Voice Activity Detection
│
├── speaker/                  # Speaker diarization & embedding
│   ├── ECAPA_TDNN.py, encoder.py, embedding.py
│   ├── audio.py, features.py, segment.py
│   ├── parameter_transfer.py, speechbrain.py, whisper.py
│
├── training/                 # Training pipeline
│   ├── __init__.py
│   ├── create_dataset.py     # Dataset prep orchestration
│   ├── create_index.py       # FAISS index creation
│   ├── extract/              # Feature extraction
│   │   ├── embedding.py, extract.py, feature.py
│   │   ├── preparing_files.py, rms.py, setup_path.py
│   ├── preprocess/           # Audio slicing & normalization
│   │   ├── preprocess.py, slicer2.py
│   └── runner/               # Training loop
│       ├── train.py, data_utils.py, mel_processing.py
│       ├── losses.py, utils.py, extract_model.py
│       └── anyprecision_optimizer.py
│
└── uvr/                      # UVR5 music/vocal separation
    ├── __init__.py
    ├── separate_music.py
    └── uvr5_lib/
        ├── separator.py, common_separator.py, spec_utils.py
        ├── uvr/{mdx_separator.py, vr_separator.py}
        └── vr_network/{layers.py, layers_new.py, nets.py, nets_new.py, model_param_init.py}
```

### 2.7 `arvc/services/` — Service layer

The service layer bridges the UI/CLI (`app/`, `api/`) and the engine
(`engine/`). It is organized by domain into five subpackages. Each
subpackage's `__init__.py` re-exports all public names from its modules, so
you can import either way:

```python
# Either works:
from arvc.services.training import create_dataset
from arvc.services.training.training import create_dataset
```

```
services/
├── __init__.py                # Lazy re-export hub
│
├── inference/                 # Inference-time services
│   ├── csrt.py                #   SRT subtitle generation
│   ├── f0_extract.py          #   Standalone F0 extraction
│   ├── presets.py             #   Load/save inference presets
│   ├── separate.py            #   Music/vocal separation orchestration
│   └── tts.py                 #   Text-to-speech (Edge TTS, Google Translate)
│
├── training/                  # Training orchestration
│   └── training.py            #   Dataset/ref/training launch + status
│
├── realtime/                  # Realtime services
│   ├── realtime.py            #   Realtime server
│   └── realtime_client.py     #   Client for remote realtime server
│
├── system/                    # System-level services
│   ├── model_utils.py         #   Model info, fusion, ONNX export
│   ├── process.py             #   Process management, archives, file movement
│   ├── restart.py             #   App restart, language/theme switching
│   └── utils.py               #   Generic helpers (stop_pid, google_translate)
│
└── downloads/                 # Download orchestration
    └── downloads.py           #   Model lookup, URL resolution, extraction
```

### 2.8 `arvc/ui/` — UI helpers

Gradio-aware helpers that need the `gradio` import. Split from `arvc.utils.feedback`
so that headless/CLI mode can use the lightweight logger without paying the
Gradio import cost.

```
ui/
├── __init__.py
└── feedback.py    # Gradio toast messages, dropdown updaters, file pickers
```

### 2.9 `arvc/utils/` — Shared utilities

Lightweight, dependency-poor helpers. Anything that needs heavy imports
(`torch`, `gradio`, `librosa`) usually lives elsewhere.

```
utils/
├── __init__.py            # strtobool + lazy submodule access
├── feedback.py            # Headless-safe logger (gr_info, gr_warning, gr_error)
├── variables.py           # Global paths, Config singleton, package logger, translations
└── downloaders/           # File-host download backends
    ├── __init__.py
    ├── gdown.py           # Google Drive
    ├── huggingface.py     # Hugging Face Hub (HF_download_file)
    ├── mediafire.py       # MediaFire
    ├── meganz.py          # MEGA.nz (encrypted)
    └── pixeldrain.py      # Pixeldrain
```

> **Note:** `noisereduce.py` (the `TorchGate` noise-reduction utility) used
> to live in `arvc/utils/` but has been relocated to
> `arvc/engine/inference/noisereduce.py` because it is only consumed by the
> inference/training/preprocess pipelines and depends on `torch`.

---

## 3. Public API surface

The top-level `arvc/__init__.py` exposes a small, stable public API:

```python
import arvc

arvc.__version__              # str — package version
arvc.launch_cli()             # Run the CLI
arvc.launch_gui()             # Launch the Gradio UI
arvc.gui                      # Module-like alias for the GUI
arvc.cli                      # Module-like alias for the CLI
```

Heavy classes (`RVCInference`, `RVCModel`, `RVCTrainer`, etc.) are listed in
`__all__` but are imported lazily to keep `import arvc` fast.

---

## 4. Adding new code — where does it go?

| You are adding… | Put it in… |
|------------------|------------|
| A new Gradio tab | `arvc/app/tabs/<domain>/<tab_name>.py` |
| A new CLI subcommand | `arvc/api/cli.py` (or `cli_complete.py`) |
| A new service (orchestration of multiple engine calls) | `arvc/services/<domain>/<service_name>.py` |
| A new model architecture | `arvc/engine/models/algorithms/` (primitives) or `arvc/engine/models/generators/` (vocoders) |
| A new F0 predictor | `arvc/engine/models/predictors/<NAME>/` (own subpackage) |
| A new optimizer | `arvc/engine/models/optimizers/<name>.py` + register in `__init__.py` |
| A new vocoder | `arvc/engine/models/generators/<name>.py` + register in `__init__.py` |
| A new embedder | `arvc/engine/models/embedders/<name>.py` |
| A new device backend | `arvc/engine/models/backends/<name>.py` |
| A new file-host downloader | `arvc/utils/downloaders/<name>.py` + register in `__init__.py` |
| A new translation | Add `<locale>.json` to `arvc/assets/languages/` (and update `support_language` in `config.json`) |
| A new model config (v1/v2/etc.) | `arvc/configs/<version>/<sample_rate>.json` |
| A new runtime config flag | `arvc/utils/variables.py` (Config class) + `config.json` schema |
| A new audio-processing utility (torch-dependent) | `arvc/engine/inference/audio_processing.py` (or a new module under `engine/inference/`) |
| A new headless-safe logging helper | `arvc/utils/feedback.py` |
| A new Gradio-aware UI helper | `arvc/ui/feedback.py` |

---

## 5. Backward compatibility

As of v2.2.2, the package was reorganized. To avoid breaking existing user
scripts and downstream forks, the following shims are in place:

- **Service re-exports.** Each `arvc/services/<domain>/__init__.py` does
  `from .<module> import *`, so `from arvc.services.training import X`
  still works (it now resolves to the subpackage, which re-exports `X`
  from `training.py`).
- **Top-level `cli`/`gui` aliases.** `arvc.cli` and `arvc.gui` are
  module-like objects that forward to `arvc.api.cli:main` and
  `arvc.app.gui:launch` respectively.
- **Lazy submodule access.** `arvc.utils` and `arvc.services` use
  `__getattr__`-based lazy imports so that `import arvc` does not pull in
  every downloader / service module.

If you maintain a fork that imported e.g. `arvc.utils.noisereduce` directly,
update to `arvc.engine.inference.noisereduce`. The other moves are
backward-compatible thanks to the re-export shims.

---

## 6. Changelog (structure-only)

### v2.2.2 — Package tidy-up

- **`arvc/utils/`** slimmed down:
  - 5 file-host downloaders (`gdown`, `huggingface`, `mediafire`, `meganz`,
    `pixeldrain`) moved into a new `arvc/utils/downloaders/` subpackage.
  - `noisereduce.py` moved to `arvc/engine/inference/noisereduce.py`
    (it is a torch-dependent audio utility, belongs with the inference
    pipeline).
  - `utils/__init__.py` rewritten with a lazy-import `__getattr__` so
    `import arvc` does not eagerly pull in every downloader.
- **`arvc/services/`** grouped into domain subpackages:
  - `services/inference/` — `csrt`, `f0_extract`, `presets`, `separate`, `tts`
  - `services/training/` — `training`
  - `services/realtime/` — `realtime`, `realtime_client`
  - `services/system/` — `model_utils`, `process`, `restart`, `utils`
  - `services/downloads/` — `downloads`
  - Each subpackage `__init__.py` re-exports its modules' public names for
    backward compatibility with `from arvc.services.X import Y` style imports.
- **Docs.** New `docs/PACKAGE_STRUCTURE.md` (this file). README.md and
  CONTRIBUTING.md project-structure sections refreshed.

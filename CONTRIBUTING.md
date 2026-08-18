# Contributing to Advanced-RVC-Inference

Thanks for checking this out! Whether you're fixing a typo, adding a feature, or reporting a bug — every contribution matters. This guide will help you get started without a ton of overhead.

## Quick Start

1. **Fork** the repo on GitHub
2. **Clone** your fork:
   ```bash
   git clone https://github.com/YOUR-USERNAME/Advanced-RVC-Inference.git
   cd Advanced-RVC-Inference
   ```
3. **Set up upstream**:
   ```bash
   git remote add upstream https://github.com/ArkanDash/Advanced-RVC-Inference.git
   ```
4. **Install** dependencies:
   ```bash
   pip install -e .
   ```
5. **Create a branch**, make changes, push, and open a PR!

## Project Structure

Understanding the codebase helps you find where to contribute. See
[`docs/PACKAGE_STRUCTURE.md`](docs/PACKAGE_STRUCTURE.md) for the canonical
deep-dive; the summary below is the quick reference.

```
arvc/
├── api/                    # CLI entry points (rvc-cli → arvc.api.cli:main)
├── app/                    # Gradio web UI (tabs, pages, layouts)
│   └── tabs/               #   inference, training, downloads, realtime, extra
├── engine/                 # Core logic (no UI dependency)
│   ├── inference/          #   voice conversion pipeline, TTS, noisereduce
│   ├── training/           #   preprocess, extract, train, export
│   │   ├── preprocess/     #     audio slicing & normalization
│   │   ├── extract/        #     embedding & F0 extraction
│   │   └── runner/         #     training loop, losses, data loading
│   ├── uvr/                #   audio separation (UVR5)
│   ├── realtime/           #   live mic conversion
│   ├── speaker/            #   speaker diarization & embedding
│   └── models/             #   model loading, generators, optimizers, backends
│       ├── algorithms/     #     VITS-style primitives
│       ├── generators/     #     HiFi-GAN NSF, BigVGAN, MRF-HiFi-GAN, RefineGAN
│       ├── optimizers/     #     AdamW, RAdam, AnyPrecisionAdamW, AdaBelief, AdaBeliefV2, PolOpt
│       ├── embedders/      #     Hubert, ContentVec, PPG, transformers
│       ├── predictors/     #     F0 predictors (RMVPE, Crepe, FCPE, PESTO, PENN, DJCM, SWIFT, WORLD)
│       ├── backends/       #     CUDA, DirectML, OpenCL, XPU, ZLUDA
│       └── onnx/           #     ONNX export & wrapper
├── services/               # Business logic layer (bridges UI ↔ engine)
│   ├── inference/          #   csrt, f0_extract, presets, separate, tts
│   ├── training/           #   training orchestration
│   ├── realtime/           #   realtime server + client
│   ├── system/             #   process, restart, model_utils, utils
│   └── downloads/          #   download orchestration
├── ui/                     # UI helpers (feedback, dropdown updates, formatting)
├── utils/                  # Shared utilities
│   ├── variables.py        #   Global paths, Config singleton, logger, translations
│   ├── feedback.py         #   Headless-safe logger (gr_info, gr_warning, gr_error)
│   └── downloaders/        #   File-host backends (gdown, huggingface, mediafire, meganz, pixeldrain)
├── configs/                # Configuration files (training configs, model templates)
│   ├── config.py           #   Runtime Config singleton
│   ├── v1/, v2/            #   V1/V2 model JSONs
│   ├── ringformer_v2/      #   RingFormer V2 configs
│   └── pcph_gan/           #   PCPH-GAN configs
├── datasets/               # Training datasets (organized per model)
├── assets/                 # Runtime assets (models, audios, languages, presets, …)
└── _version.py             # Version management
```

**Key rules** (enforced — violations will break headless mode):
- `engine/` **MUST NEVER** import from `app/`, `services/`, `ui/`, or `api/`. It is the dependency-free core.
- `services/` may import from `engine/` and `utils/`, but NOT from `app/` or `ui/`.
- `app/` and `api/` may import from anywhere except `app/` siblings (each tab is independent).
- `utils/` may import from `engine/` (for path resolution), but NOT from `services/`, `app/`, or `ui/`.

### Where to put new code — quick lookup

| You are adding… | Put it in… |
|------------------|------------|
| A new Gradio tab | `arvc/app/tabs/<domain>/<tab_name>.py` |
| A new CLI subcommand | `arvc/api/cli.py` |
| A new service (orchestration) | `arvc/services/<domain>/<service_name>.py` |
| A new vocoder / optimizer / embedder / predictor / backend | `arvc/engine/models/<type>/<name>.py` (+ register in `__init__.py`) |
| A new file-host downloader | `arvc/utils/downloaders/<name>.py` (+ register in `__init__.py`) |
| A new translation | `arvc/assets/languages/<locale>.json` (update `support_language` in `config.json`) |
| A new model config | `arvc/configs/<version>/<sample_rate>.json` |
| A new runtime config flag | `arvc/utils/variables.py` (Config class) + `config.json` schema |
| A new audio-processing utility (torch-dependent) | `arvc/engine/inference/audio_processing.py` or a new module under `engine/inference/` |
| A new headless-safe logging helper | `arvc/utils/feedback.py` |
| A new Gradio-aware UI helper | `arvc/ui/feedback.py` |

## Ways to Contribute

### Reporting Bugs

Found something broken? Open an [issue](https://github.com/ArkanDash/Advanced-RVC-Inference/issues) with:

- What you expected to happen vs. what actually happened
- Steps to reproduce
- Error messages or logs (paste them, don't screenshot)
- Your environment: OS, Python version, GPU, how you launched the app

Try to search existing issues first — someone might have already reported it.

### Suggesting Features

We're open to ideas! When suggesting something:

- Describe the problem you're trying to solve
- Explain how your feature would help
- Any alternatives you've considered

### Writing Code

Areas where help is always welcome:

| Area | What |
|------|------|
| **UI/UX** | Gradio interface improvements, new tabs, better layout |
| **Translations** | Fix or improve any of the 44 language files in `arvc/assets/languages/` |
| **Core Engine** | Inference optimizations, new F0 methods, training pipeline |
| **Bug Fixes** | Pick an open issue and go for it |
| **Documentation** | Tutorials, code comments, README improvements |
| **Testing** | Unit tests, integration tests — currently very limited |

### Improving Translations

Each language file is a JSON dict at `arvc/assets/languages/<locale>.json`. When adding new UI keys:

1. Add the key to **all 44 language files** with at least an English fallback
2. Provide proper translations for languages you know
3. Use `translations.get("key", "English fallback")` in code — never bare `translations["key"]`

## Coding Style

We're not picky, but follow these basics:

- **PEP 8** — standard Python style, 4 spaces, no tabs
- **Line length** — try to stay under 120 characters
- **Type hints** — appreciated for public functions, not required everywhere
- **Docstrings** — add them for new public functions and classes
- **Import order** — stdlib → third-party → local

```python
# Good
import os
from typing import Optional

import gradio as gr
import torch

from arvc.utils.variables import configs, translations
```

### A Few Project Conventions

- Use `translations.get("key", "fallback")` instead of `translations["key"]` — this prevents crashes when a translation key is missing
- Keep `engine/` free of UI imports — it should work headless
- Log errors with `logger.error()` and show user-facing messages with `gr_warning()` / `gr_error()` / `gr_info()`
- If you add a new Gradio component, make sure event handler outputs match the number of return values from the function

## Submitting Changes

### Branch Naming

Use whatever makes sense — just keep it descriptive:

- `fix/audio-output-format`
- `feature/batch-download`
- `translate/ja-JP-updates`

### Commit Messages

We're relaxed about format, but try to be descriptive:

```
Fix: pretrained model list showing 0 when switching to List Model
Add: Download Audio tab separated from Download Model
Update: Japanese translations for new download keys
```

If you want to use conventional commits (`feat:`, `fix:`, `docs:`, etc.), that's cool too.

### Pull Requests

When you're ready:

1. **Sync with upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/master
   ```
2. **Push** to your fork
3. **Open a PR** against the `master` branch

In your PR description, include:

- What the PR does (brief)
- Why it's needed (context)
- How you tested it
- Any related issues (e.g., "Fixes #69")

### PR Checklist

Before submitting, quickly check:

- [ ] Does the code run without errors?
- [ ] Did you test the feature/fix manually?
- [ ] Are translation keys added to all language files (if you added new UI text)?
- [ ] No hardcoded strings that should be translatable?
- [ ] Event handler outputs match function return values?

Don't stress if it's not perfect — we'll work through it in the review.

## Development Tips

### Running the App

```bash
# GUI mode
python -m arvc

# CLI mode
python -m arvc --cli

# Or use the shell script
./rvc-cli.sh
```

### Running Tests

```bash
pytest
```

Tests are limited right now, so adding new ones is a great contribution.

### Debugging

- Set `debug_mode: true` in `arvc/configs/config.json` for verbose logging
- Check `logger.debug()` calls — they're silenced by default but visible in debug mode

## Community

- **Discord**: [https://discord.gg/hvmsukmBHE](https://discord.gg/hvmsukmBHE)
- **GitHub Issues**: [https://github.com/ArkanDash/Advanced-RVC-Inference/issues](https://github.com/ArkanDash/Advanced-RVC-Inference/issues)
- **GitHub Discussions**: [https://github.com/ArkanDash/Advanced-RVC-Inference/discussions](https://github.com/ArkanDash/Advanced-RVC-Inference/discussions)

## Recognition

Contributors and collaborators are recognized in:

- The [README credits & collaborators](README.md#credits)
- The [HTML documentation](docs/Advanced-RVC-Documentation.html#credits-license)
- Release notes
- Community channels

---

Thanks for contributing! Every fix, feature, and translation makes this project better for everyone.

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
├── app/                     # User-facing layer (Gradio UI + CLI entry points)
│   ├── api/                 #   CLI entry points (rvc-cli → arvc.app.api.cli:main)
│   ├── tabs/                #   inference, training, downloads, realtime, extra, terminal
│   └── gui.py               #   Gradio launcher + theme/CSS wiring
├── engine/                  # Core logic (no UI dependency)
│   ├── rvc/                 #   voice conversion: inference, training, models
│   │   ├── inference/       #     voice conversion pipeline, TTS, noisereduce
│   │   ├── training/        #     preprocess, extract, train, export
│   │   │   ├── preprocess/  #       audio slicing & normalization
│   │   │   ├── extract/     #       embedding & F0 extraction
│   │   │   └── runner/      #       training loop, losses, data loading
│   │   └── models/          #     model loading, generators, optimizers, backends
│   │       ├── algorithms/  #       VITS-style primitives
│   │       ├── generators/  #       HiFi-GAN NSF, BigVGAN, MRF-HiFi-GAN, RefineGAN
│   │       ├── optimizers/  #       AdamW, RAdam, AnyPrecisionAdamW, AdaBelief, AdaBeliefV2, PolOpt
│   │       ├── embedders/   #       Hubert, ContentVec, PPG, transformers
│   │       ├── predictors/  #       F0 predictors (RMVPE, Crepe, FCPE, PESTO, PENN, DJCM, SWIFT, WORLD) + high_register.py
│   │       ├── backends/    #       CUDA, DirectML, OpenCL, XPU, ZLUDA
│   │       └── onnx/        #       ONNX export & wrapper
│   ├── uvr/                 #   audio separation (UVR5)
│   ├── whisper/             #   speech-to-text + speaker diarization & embedding
│   └── realtime/            #   live mic conversion
├── utils/                   # Shared utilities
│   ├── variables.py         #   Global paths, Config singleton, logger, translations
│   ├── feedback.py          #   Headless-safe logger (gr_info, gr_warning, gr_error)
│   └── terminal.py          #   Terminal-in-GUI capture (ported from Applio)
├── downloader/              # File-host backends (gdown, huggingface, mediafire, meganz, pixeldrain)
├── configs/                 # Configuration files (training configs, model templates)
│   ├── config.py            #   Runtime Config singleton
│   ├── config.json          #   User-facing settings (theme, language, fp16, rmvpe_high_register, …)
│   ├── v1/, v2/             #   V1/V2 model JSONs
│   ├── ringformer_v2/       #   RingFormer V2 configs
│   └── pcph_gan/            #   PCPH-GAN configs
├── assets/                  # Runtime assets
│   ├── themes/              #   Built-in Gradio themes (Applio.py + loadThemes.py)
│   ├── ui/                  #   shell.css — dark shell styling (ported from Applio)
│   ├── languages/           #   i18n JSON files (en-US, id-ID, vi-VN, …)
│   ├── models/              #   Pretrained, predictor, embedder, UVR5 weights
│   ├── audios/              #   Sample / default audio inputs
│   ├── presets/             #   Inference presets
│   └── logs/                #   Training run logs (per-model)
├── datasets/                # Training datasets (organized per model)
└── _version.py              # Version management
```

**Key rules** (enforced — violations will break headless mode):
- `engine/` **MUST NEVER** import from `app/`, `utils/feedback.py`, or any UI module. It is the dependency-free core.
- `app/` (both `app/gui` and `app/api`) may import from anywhere except `app/` siblings (each tab is independent).
- `utils/` may import from `engine/` (for path resolution) and `configs/`, but NOT from `app/`.

### Where to put new code — quick lookup

| You are adding… | Put it in… |
|------------------|------------|
| A new Gradio tab | `arvc/app/tabs/<domain>/<tab_name>.py` |
| A new CLI subcommand | `arvc/app/api/cli.py` (or `arvc/app/api/cli_complete.py` for the legacy dispatcher) |
| A new vocoder / optimizer / embedder / predictor / backend | `arvc/engine/rvc/models/<type>/<name>.py` (+ register in `__init__.py`) |
| A new F0 predictor helper (e.g. high-register correction) | `arvc/engine/rvc/models/predictors/<name>.py` |
| A new UVR5 separator variant | `arvc/engine/uvr/uvr5_lib/uvr/<variant>_separator.py` (+ register in `__init__.py`) |
| A new file-host downloader | `arvc/downloader/<name>.py` (+ register in `__init__.py`) |
| A new translation | `arvc/assets/languages/<locale>.json` (update `support_language` in `config.json`) |
| A new model config | `arvc/configs/<version>/<sample_rate>.json` |
| A new runtime config flag | `arvc/utils/variables.py` (Config class) + `config.json` schema |
| A new audio-processing utility (torch-dependent) | `arvc/engine/rvc/inference/audio_processing.py` or a new module under `engine/rvc/inference/` |
| A new headless-safe logging helper | `arvc/utils/feedback.py` |
| A new built-in Gradio theme | `arvc/assets/themes/<Name>.py` (+ register in `arvc/assets/themes/loadThemes.py:BUILTIN_THEMES`) |
| A new dark shell CSS rule | `arvc/assets/ui/shell.css` (gated by `configs["use_shell_css"]`) |

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

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

Understanding the codebase helps you find where to contribute:

```
arvc/
├── app/               # Gradio web UI (tabs, pages, layouts)
│   ├── tabs/          #   inference, training, downloads, realtime, extra
├── engine/            # Core logic (no UI dependency)
│   ├── inference/     #   voice conversion pipeline, TTS
│   ├── training/      #   preprocess, extract, train, export
│   ├── uvr/           #   audio separation (UVR5)
│   ├── realtime/      #   live mic conversion
│   └── models/        #   model loading, backends (CUDA, DirectML, OpenCL)
├── services/          # Business logic layer (bridges UI ↔ engine)
├── ui/                # UI helpers (feedback, dropdown updates, formatting)
├── utils/             # Shared utilities (variables, download helpers)
├── configs/           # Configuration files (config.json, training configs)
└── assets/            # Runtime assets (models, languages, presets, weights)
    └── languages/     #   44 translation JSON files
```

**Key rule**: `engine/` should never import from `app/` or `services/`. Keep the core independent.

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

Contributors are recognized in:

- The [README credits](README.md#credits)
- Release notes
- Community channels

---

Thanks for contributing! Every fix, feature, and translation makes this project better for everyone.

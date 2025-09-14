<div align="center">

# Advanced RVC Inference V3.1

[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-yellow?style=for-the-badge&logo=google-colab&logoColor=white)](https://colab.research.google.com/github/ArkanDash/Advanced-RVC-Inference/blob/master/Advanced-RVC.ipynb)

</div>

---

## 📺 Quick Start Guide

Watch the full setup and usage guide here:  
[![YouTube Guide](https://img.shields.io/badge/YouTube-Setup%20Guide-red?style=for-the-badge&logo=youtube)](https://youtu.be/8CzEFMmyRag?si=M8SYyal4RWtD07VM)

---

## 📖 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Running the WebUI](#running-the-webui)
- [Language Support](#language-support)
- [Terms of Use](#terms-of-use)
- [Disclaimer](#disclaimer)
- [Credits](#credits)

---

## 📝 Overview

<div align="center">

**Advanced RVC Inference V3.1** is a state-of-the-art WebUI designed for fast and effortless inference.  
Packed with powerful tools including a model downloader, voice splitter, and more — all accessible from your browser.

<br>

> Special thanks to [Applio](https://github.com/IAHispano/Applio). This project wouldn't exist without their pioneering work.

[![Original Applio](https://img.shields.io/badge/Github-Original%20Applio%20Repository-blue?style=for-the-badge&logo=github)](https://github.com/IAHispano/Applio)

</div>

---

## ✨ Features

- ✅ **V1 & V2 Model Support**
- ✅ **YouTube Audio Downloader**
- ✅ **Text-to-Speech (TTS)**
- ✅ **Audio Separator (Voice Splitter)**  
  *Requires internet to download model*
- ✅ **Model Downloader**
- ✅ **Gradio WebUI**
- ✅ **Multi-language Support (16+ Languages)**

---

## ⚡ Installation

1. **Install Python Dependencies**

   ```bash
   python -m pip install -r requirements.txt
   pip install torch torchvision torchaudio numpy==1.23.5
   ```

2. **Install [ffmpeg](https://ffmpeg.org/)**  
   Download and follow instructions for your OS.

---

## 🚀 Running the WebUI

```bash
python app.py
```

---

## 🌍 Language Support

Advanced RVC Inference now supports 16+ languages, making it accessible to users around the world. The application automatically detects your system language and uses the appropriate translation if available.

### Supported Languages:
- 🇺🇸 English (US)
- 🇩🇪 German (Deutsch)
- 🇪🇸 Spanish (Español)
- 🇫🇷 French (Français)
- 🇮🇩 Indonesian (Bahasa Indonesia)
- 🇯🇵 Japanese (日本語)
- 🇧🇷 Portuguese (Português)
- 🇨🇳 Chinese (中文)
- 🇸🇦 Arabic (العربية)
- 🇮🇳 Hindi (हिन्दी)
- 🇮🇹 Italian (Italiano)
- 🇰🇷 Korean (한국어)
- 🇳🇱 Dutch (Nederlands)
- 🇵🇱 Polish (Polski)
- 🇷🇺 Russian (Русский)
- 🇹🇷 Turkish (Türkçe)

### Changing Language:
1. Open the application
2. Go to the "Settings" tab
3. Select your preferred language from the dropdown
4. Restart the application for changes to take effect

### Contributing Translations:
We welcome translations from the community! If you'd like to add support for your language or improve existing translations, please see our [Translation Guide](TRANSLATION.md).

---

## 📜 Terms of Use

The converted voices **must not** be used for:

- Criticizing or attacking individuals
- Advocating or opposing political positions, religions, or ideologies
- Public display of strongly stimulating expressions without proper zoning
- Selling voice models or generated voice clips
- Malicious impersonation of original voice owners
- Fraudulent activities (identity theft, deceptive calls, etc.)

---

## ⚠️ Disclaimer

The author is **not liable** for any direct, indirect, consequential, incidental, or special damages arising from the use, misuse, or inability to use this software.

---

## 🙏 Credits

- [ArkanDash](https://github.com/ArkanDash) — Original project owner
- [Shirou's RVC AI Cover Maker UI](https://github.com/Eddycrack864/RVC-AI-Cover-Maker-UI.git) — Project base

---
<div align="center">

# Advanced RVC Inference

[![Colab](https://img.shields.io/badge/Colab-Advanced%20RVC%20Inference-blue?style=for-the-badge&logo=googlecolab)](https://colab.research.google.com/github/ArkanDash/Advanced-RVC-Inference/blob/main/Advanced-RVC.ipynb)
</div>

### Information
Advanced RVC Inference presents itself as a state-of-the-art web UI crafted to streamline rapid and effortless inference. This comprehensive toolset encompasses a model downloader, a voice splitter, and the added efficiency of batch inference.

Please support the original RVC. This inference won't be possible to make without it.<br />
[![Original RVC Repository](https://img.shields.io/badge/Github-Original%20RVC%20Repository-blue?style=for-the-badge&logo=github)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

#### Features
- Support V1 & V2 Model ✅
- Youtube Audio Downloader ✅
- Demucs (Voice Splitter) [Internet required for downloading model] ✅
- Microphone Support ✅
- TTS Support ✅
- Model Downloader ✅

#### Currently Working
- Batch Inference 🛠
- Settings 🛠

### Installation

1. Install Dependencies <br />
```bash
pip install torch torchvision torchaudio

pip install -r requirements.txt
```
2. Install [ffmpeg](https://ffmpeg.org/)

3. Download [Hubert Model](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/hubert_base.pt)

4. [OPTIONAL] To use rmvpe pitch extraction, download this [rvmpe.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt)

### Run WebUI <br />

For Windows:
```bash
Open run.bat
```
For Other:
```bash
python infer.py
```

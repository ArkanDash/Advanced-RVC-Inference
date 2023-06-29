<div align="center">

# Advanced RVC Inference


[![GitHub](https://img.shields.io/github/license/arkandash/Multi-Model-RVC-Inference)](https://github.com/ArkanDash/Multi-Model-RVC-Inference/blob/master/LICENSE)
</div>

### Information
Support V1 Model and V2 Model

Original Repository: [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

#### Features
- Youtube Audio Downloader ✅
- Demucs (Voice Splitter) [Internet required for downloading model] ✅
- TTS Support ✅
- Model Downloader ✅

#### Currently Working
- Batch Inference 🛠

#### Plans
- UVR

### Installation

1. Install Requirement <br />
```bash
pip install torch torchvision torchaudio

pip install -r requirements.txt
```

2. Download [Hubert Model](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/hubert_base.pt)

3. Run WebUI <br />
```bash
python infer.py
```
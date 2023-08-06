<div align="center">

# Advanced RVC Inference

</div>

### Information
Please support the original RVC. without it, this inference wont be possible to make.<br />
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

### Installation

1. Install Dependencies <br />
```bash
pip install torch torchvision torchaudio

pip install -r requirements.txt
```
2. Install [ffmpeg](https://ffmpeg.org/)

3. Download [Hubert Model](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/hubert_base.pt)

4. [OPTIONAL] To use rmvpe pitch extraction, download this [rvmpe.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt)

5. Run WebUI <br />
```bash
python infer.py
```
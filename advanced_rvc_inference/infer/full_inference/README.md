# RVC X UVR - Full Inference Pipeline

A comprehensive AI cover generation system that combines **RVC (Retrieval-based Voice Conversion)** with **UVR (Ultimate Vocal Remover)** to create complete AI covers from YouTube videos or local audio files.

## 🎯 Features

### Core Pipeline
- **YouTube Audio Download**: Automatically download audio from YouTube URLs
- **Vocal Separation**: Use UVR-MDX-NET models to separate vocals from instrumentals
- **Voice Conversion**: Apply RVC voice models to convert vocals to target voice
- **Audio Effects**: Apply reverb, compression, and other audio effects
- **Final Mixing**: Combine processed vocals with instrumentals

### Advanced Controls
- **Pitch Adjustment**: Control pitch for vocals and full mix
- **Voice Quality**: Adjust index rate, filter radius, RMS mix rate
- **Pitch Detection**: Choose from rmvpe, harvest, or mangio-crepe algorithms
- **Audio Effects**: Fine-tune reverb settings (size, wet/dry, damping)
- **Volume Control**: Independent gain control for vocals and instrumentals
- **Output Formats**: Support for WAV and MP3 output

## 🚀 Usage

### Method 1: Launch Standalone Interface
```bash
python launch_full_inference.py
```

### Method 2: Access via Main Application
1. Launch the main Advanced RVC Inference application
2. Navigate to the **"RVC X UVR Full Inference"** tab
3. Configure your settings and generate covers

### Method 3: Command Line Interface
```bash
python advanced_rvc_inference/infer/full_inference/full_inference.py \
    --input "https://www.youtube.com/watch?v=..." \
    --model "your_voice_model" \
    --pitch-change 0 \
    --output-format wav
```

## 📋 Parameters

### Basic Parameters
- `--input`: YouTube URL or local audio file path
- `--model`: RVC voice model name
- `--pitch-change`: Pitch change for vocals (octaves, -12 to 12)
- `--pitch-change-all`: Pitch change for all tracks (semitones, -12 to 12)

### Voice Conversion Quality
- `--index-rate`: Control AI accent (0.0 to 1.0, default: 0.5)
- `--filter-radius`: Median filtering radius (0 to 7, default: 3)
- `--rms-mix-rate`: Original vs fixed loudness balance (0.0 to 1.0, default: 0.25)
- `--f0-method`: Pitch detection algorithm (rmvpe/harvest/mangio-crepe)

### Audio Effects
- `--reverb-size`: Reverb room size (0.0 to 1.0, default: 0.15)
- `--reverb-wet`: Reverb wet level (0.0 to 1.0, default: 0.2)
- `--reverb-dry`: Reverb dry level (0.0 to 1.0, default: 0.8)
- `--reverb-damping`: High frequency absorption (0.0 to 1.0, default: 0.7)

### Audio Mixing
- `--vocals-gain`: Vocals volume gain in dB (-20 to 20, default: 0)
- `--instrumentals-gain`: Instrumentals volume gain in dB (-20 to 20, default: -7)

### Output
- `--output-format`: Audio format (wav/mp3, default: wav)
- `--keep-intermediate`: Keep temporary files for debugging

## 🏗️ Architecture

The pipeline consists of several integrated components:

### 1. Input Processing
- YouTube URL parsing and audio extraction
- Local file validation and processing
- Audio format standardization

### 2. Vocal Separation (UVR)
- **Primary Separation**: Vocals vs Instrumentals using UVR-MDX-NET
- **Quality Enhancement**: Denoising and post-processing
- **Format Standardization**: Consistent audio parameters

### 3. Voice Conversion (RVC)
- **Model Loading**: Automatic RVC model detection
- **Pitch Processing**: F0 extraction and adjustment
- **Voice Conversion**: High-quality voice synthesis
- **Quality Control**: Index rate and filtering options

### 4. Audio Effects Processing
- **Frequency Shaping**: High-pass filtering
- **Dynamic Processing**: Compression for consistency
- **Spatial Audio**: Reverb for natural sound
- **Format Conversion**: Output in requested format

### 5. Final Assembly
- **Track Mixing**: Balanced combination of elements
- **Quality Optimization**: Final audio processing
- **File Management**: Organized output structure

## 📁 File Structure

```
advanced_rvc_inference/infer/full_inference/
├── __init__.py                 # Package initialization
├── full_inference.py           # Main pipeline implementation
└── README.md                   # This documentation

advanced_rvc_inference/tabs/
└── full_inference_tab.py       # Gradio interface integration

launch_full_inference.py        # Standalone launcher script
```

## 🔧 Dependencies

### Core Dependencies
- **RVC Modules**: Voice conversion infrastructure
- **UVR5 Library**: Vocal separation models
- **Gradio**: User interface framework
- **yt-dlp**: YouTube audio extraction
- **Pedalboard**: Audio effects processing

### Audio Processing
- **librosa**: Audio analysis and manipulation
- **soundfile**: Audio I/O operations
- **pydub**: Audio file handling and mixing
- **numpy**: Numerical audio processing

### System Requirements
- **Python 3.9+**: Runtime environment
- **FFmpeg**: Audio format conversion
- **GPU**: Recommended for faster processing
- **RAM**: 8GB+ recommended for large files

## 🎛️ Configuration

### Model Paths
The pipeline automatically detects models in these directories:
- **RVC Models**: `advanced_rvc_inference/assets/weights/`
- **UVR Models**: `advanced_rvc_inference/assets/models/uvr5/`
- **Output Directory**: `advanced_rvc_inference/outputs/full_inference/`

### Default Settings
- **Pitch Change**: 0 octaves (no change)
- **Quality Settings**: Balanced defaults for most use cases
- **Output Format**: WAV for maximum quality
- **Cleanup**: Automatic intermediate file removal

## 🔍 Troubleshooting

### Common Issues

**1. "No RVC model found"**
- Ensure `.pth` files are in the weights directory
- Check model name spelling
- Verify file permissions

**2. "UVR separation failed"**
- Check UVR model files exist
- Verify audio file format compatibility
- Monitor system memory usage

**3. "YouTube download failed"**
- Check internet connection
- Verify YouTube URL format
- Update yt-dlp: `pip install --upgrade yt-dlp`

**4. "Audio effects failed"**
- Check input audio file integrity
- Verify Pedalboard dependencies
- Monitor disk space

### Performance Optimization

**GPU Acceleration**
- Use CUDA-compatible GPU for faster processing
- Adjust batch size for memory constraints
- Monitor GPU memory usage

**Memory Management**
- Enable intermediate file cleanup
- Process shorter audio segments for large files
- Monitor system RAM usage

## 📝 Example Workflows

### Basic Cover Generation
```bash
python launch_full_inference.py
# Use Web Interface:
# 1. Enter YouTube URL
# 2. Select RVC model
# 3. Click "Generate AI Cover"
```

### Command Line Example
```bash
python advanced_rvc_inference/infer/full_inference/full_inference.py \
    --input "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
    --model "female_voice_v2" \
    --pitch-change 1 \
    --reverb-size 0.2 \
    --vocals-gain 2
```

### Batch Processing
Create a script to process multiple songs:
```python
from advanced_rvc_inference.infer.full_inference import FullInferencePipeline

pipeline = FullInferencePipeline()
songs = ["song1.mp3", "song2.mp3", ...]

for song in songs:
    pipeline.full_inference_pipeline(
        song_input=song,
        voice_model="target_voice",
        output_format="mp3"
    )
```

## 🤝 Integration

The RVC X UVR Full Inference pipeline is designed to integrate seamlessly with the Advanced RVC Inference application:

- **Tab Integration**: Available as "RVC X UVR Full Inference" tab
- **Shared Resources**: Uses existing model directories and configurations
- **Unified Interface**: Consistent UI/UX with other modules
- **Extensible**: Easy to add new features and improvements

## 📄 License

This implementation follows the same licensing terms as the original AICoverGen project and the Advanced RVC Inference project.

## 🙏 Acknowledgments

- **AICoverGen**: Original inspiration and pipeline concept
- **UVR Team**: Ultimate Vocal Remover models and infrastructure
- **RVC Community**: Voice conversion technology and models
- **MiniMax**: Platform and development support

---

*Generated by MiniMax Agent - RVC X UVR Full Inference Pipeline v1.0.0*
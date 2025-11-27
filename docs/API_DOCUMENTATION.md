# 🎤 Advanced RVC Inference - API Documentation

## Overview

The Advanced RVC Inference API provides a simple and powerful interface for Real-time Voice Conversion (RVC) using pre-trained models. This documentation covers all available functions and their usage.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ArkanDash/Advanced-RVC-Inference.git
cd Advanced-RVC-Inference

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```python
from advanced_rvc_inference import full_inference_program

# Simple voice conversion
result = full_inference_program(
    model_path="path/to/your/model.pth",
    input_audio_path="path/to/input/audio.wav",
    output_path="path/to/output/converted.wav",
    pitch=2,  # Pitch shift in semitones
    f0_method="rmvpe"  # F0 extraction method
)

print(f"Conversion completed: {result}")
```

## Core Functions

### `full_inference_program()`

The main API function for voice conversion.

**Signature:**
```python
full_inference_program(
    model_path: str,
    input_audio_path: str,
    output_path: str,
    index_path: str = "",
    pitch: int = 0,
    f0_method: str = "rmvpe",
    index_rate: float = 0.5,
    rms_mix_rate: float = 1.0,
    protect: float = 0.33,
    hop_length: int = 64,
    f0_autotune: bool = False,
    f0_autotune_strength: float = 1.0,
    filter_radius: int = 3,
    clean_audio: bool = False,
    clean_strength: float = 0.7,
    export_format: str = "wav",
    resample_sr: int = 0,
    embedder_model: str = "contentvec",
    split_audio: bool = False,
    **kwargs
) -> str
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | Required | Path to the RVC model file (.pth) |
| `input_audio_path` | str | Required | Path to input audio file |
| `output_path` | str | Required | Path for output audio file |
| `index_path` | str | "" | Path to index file (auto-detected if empty) |
| `pitch` | int | 0 | Pitch shift in semitones (-12 to +12) |
| `f0_method` | str | "rmvpe" | F0 extraction method |
| `index_rate` | float | 0.5 | Index rate (0.0 to 1.0) |
| `rms_mix_rate` | float | 1.0 | RMS mix rate (0.0 to 1.0) |
| `protect` | float | 0.33 | Voice protection (0.0 to 0.5) |
| `hop_length` | int | 64 | Hop length for F0 extraction |
| `f0_autotune` | bool | False | Enable F0 autotune |
| `f0_autotune_strength` | float | 1.0 | F0 autotune strength |
| `filter_radius` | int | 3 | Filter radius for F0 |
| `clean_audio` | bool | False | Enable noise reduction |
| `clean_strength` | float | 0.7 | Noise reduction strength |
| `export_format` | str | "wav" | Output format |
| `resample_sr` | int | 0 | Resample rate (0 = no resampling) |
| `embedder_model` | str | "contentvec" | Embedder model |
| `split_audio` | bool | False | Split long audio files |

**Returns:**
- `str`: Path to the output file if successful, `None` if failed

**Example:**
```python
result = full_inference_program(
    model_path="models/singer_voice.pth",
    input_audio_path="input/speech.wav",
    output_path="output/converted_speech.wav",
    pitch=2,
    f0_method="rmvpe",
    index_rate=0.7,
    clean_audio=True,
    export_format="wav"
)
```

### `convert_audio()`

Lower-level audio conversion function.

**Signature:**
```python
convert_audio(
    input_path: str,
    output_path: str,
    model_path: str,
    index_path: str = "",
    **kwargs
) -> str
```

**Parameters:**
- `input_path`: Path to input audio file
- `output_path`: Path for output audio file  
- `model_path`: Path to RVC model file
- `index_path`: Path to index file (optional)
- `**kwargs`: Additional conversion parameters

**Returns:**
- `str`: Path to output file

### `import_voice_converter()`

Create a voice converter instance.

**Signature:**
```python
import_voice_converter(model_path: str, **kwargs) -> VoiceConverter
```

**Parameters:**
- `model_path`: Path to RVC model file
- `**kwargs`: Additional parameters

**Returns:**
- `VoiceConverter`: Voice converter instance

## Utility Functions

### `get_config()`

Get default system configuration.

**Returns:**
```python
{
    'device': 'cuda' or 'cpu',
    'is_half': bool,
    'sample_rate': 16000,
    'hop_length': 64,
    'f0_method': 'rmvpe',
    'embedder_model': 'contentvec'
}
```

### `check_fp16_support()`

Check if FP16 (half precision) is supported.

**Returns:**
- `bool`: True if FP16 is supported, False otherwise

### `download_file()`

Download a file from URL.

**Signature:**
```python
download_file(url: str, output_path: str) -> str
```

**Parameters:**
- `url`: URL to download from
- `output_path`: Local path to save file

**Returns:**
- `str`: Path to downloaded file, `None` if failed

## Model Information Functions

### `models_vocals()`

Get available vocal separation models.

**Returns:**
```python
{
    "Mel-Roformer by KimberleyJSN": "mel_roformer_vocals",
    "BS-Roformer by ViperX": "bs_roformer_vocals",
    "MDX23C": "mdx23c_vocals"
}
```

### `karaoke_models()`

Get available karaoke models.

### `denoise_models()`

Get available denoise models.

### `dereverb_models()`

Get available dereverb models.

### `deecho_models()`

Get available deecho models.

## Audio Processing Functions

### `add_audio_effects()`

Add audio effects using pedalboard.

**Signature:**
```python
add_audio_effects(
    audio_path: str,
    output_path: str,
    effects: dict = None
) -> str
```

### `merge_audios()`

Merge multiple audio files.

**Signature:**
```python
merge_audios(audio_paths: list, output_path: str) -> str
```

### `batch_convert()`

Convert multiple audio files in batch.

**Signature:**
```python
batch_convert(
    input_dir: str,
    output_dir: str,
    model_path: str,
    **kwargs
) -> list
```

## GPU Functions

### `gpu_optimizer()`

Get GPU optimizer if available.

### `gpu_settings()`

Get current GPU settings and information.

**Returns:**
```python
{
    'device_count': int,
    'current_device': int,
    'device_name': str,
    'memory_allocated': int,
    'memory_reserved': int
}
```

## F0 Methods

The following F0 (fundamental frequency) extraction methods are supported:

- **rmvpe**: Recommended method, best quality
- **harvest**: Good quality, slower
- **crepe**: High quality, requires more resources
- **dio**: Fast, lower quality
- **pm**: Basic method
- **fcpe**: Alternative method

## Embedder Models

Supported embedder models:

- **contentvec**: Default, good quality
- **hubert**: Alternative embedder
- **fairseq**: Fairseq-based embedder

## Error Handling

The API includes comprehensive error handling:

```python
try:
    result = full_inference_program(
        model_path="models/voice.pth",
        input_audio_path="input.wav",
        output_path="output.wav"
    )
    if result:
        print(f"Success: {result}")
    else:
        print("Conversion failed")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except Exception as e:
    print(f"Error: {e}")
```

## Google Colab Usage

For Google Colab, use the provided notebook:

1. Open `notebooks/Google_Colab_API_Example.ipynb`
2. Run the setup cell to install dependencies
3. Upload your model and audio files
4. Use the API functions as shown in the examples

## Troubleshooting

### Common Issues

1. **"full_inference_program not available due to missing dependencies"**
   - Install all requirements: `pip install -r requirements.txt`
   - Check if torch is installed: `python -c "import torch; print(torch.__version__)"`

2. **CUDA/GPU Issues**
   - The system automatically falls back to CPU if GPU is unavailable
   - Check GPU availability: `torch.cuda.is_available()`

3. **Model Loading Errors**
   - Ensure model file is a valid RVC .pth file
   - Check file permissions and path

4. **Audio Format Issues**
   - Supported formats: WAV, MP3, FLAC, OGG, M4A
   - Convert to WAV for best compatibility

### Performance Tips

1. **Use GPU when available** for faster processing
2. **Enable FP16** if supported: `check_fp16_support()`
3. **Split long audio files** using `split_audio=True`
4. **Use appropriate hop_length** (64 for quality, 128 for speed)
5. **Choose optimal F0 method** (rmvpe for quality, dio for speed)

## Examples

### Basic Conversion
```python
from advanced_rvc_inference import full_inference_program

result = full_inference_program(
    model_path="models/my_voice.pth",
    input_audio_path="input/speech.wav",
    output_path="output/converted.wav"
)
```

### Advanced Conversion
```python
result = full_inference_program(
    model_path="models/singer.pth",
    input_audio_path="input/song.wav",
    output_path="output/converted_song.wav",
    pitch=2,
    f0_method="rmvpe",
    index_rate=0.7,
    rms_mix_rate=0.8,
    protect=0.33,
    clean_audio=True,
    clean_strength=0.7,
    export_format="wav",
    resample_sr=44100
)
```

### Batch Processing
```python
from advanced_rvc_inference import batch_convert

results = batch_convert(
    input_dir="input_folder/",
    output_dir="output_folder/",
    model_path="models/voice.pth",
    pitch=0,
    f0_method="rmvpe"
)
```

## Support

- **GitHub Repository**: https://github.com/ArkanDash/Advanced-RVC-Inference
- **Issues**: Report bugs and feature requests on GitHub Issues
- **Discussions**: Join community discussions on GitHub

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

**Version**: 4.0.0  
**Last Updated**: November 2024  
**Authors**: ArkanDash & BF667
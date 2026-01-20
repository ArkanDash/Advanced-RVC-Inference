# Advanced RVC Inference CLI

A comprehensive command-line interface for the Advanced RVC Inference framework. This CLI provides access to all the powerful features of RVC including voice conversion, model training, audio separation, and more - all from the terminal.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command Reference](#command-reference)
  - [infer - Voice Conversion](#infer---voice-conversion)
  - [uvr - Audio Separation](#uvr---audio-separation)
  - [create-dataset - Create Training Data](#create-dataset---create-training-data)
  - [create-index - Create Model Index](#create-index---create-model-index)
  - [extract - Feature Extraction](#extract---feature-extraction)
  - [preprocess - Data Preprocessing](#preprocess---data-preprocessing)
  - [train - Model Training](#train---model-training)
  - [create-ref - Create Reference Set](#create-ref---create-reference-set)
  - [download - Download Models/Audio](#download---download-modelsaudio)
  - [serve - Web Interface](#serve---web-interface)
  - [info - System Information](#info---system-information)
  - [version - Version Info](#version---version-info)
  - [list-models - List Available Models](#list-models---list-available-models)
  - [list-f0-methods - List F0 Methods](#list-f0-methods---list-f0-methods)
- [Configuration](#configuration)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/ArkanDash/Advanced-RVC-Inference.git
cd Advanced-RVC-Inference

# Install the package
pip install -e .

# Or install with all dependencies
pip install -e .[all]
```

### Using the CLI Directly

```bash
# Run directly with Python
python rvc_cli.py --help

# Or use the wrapper script (Linux/macOS)
./rvc-cli --help

# Make the wrapper executable (Linux/macOS)
chmod +x rvc-cli
```

## Quick Start

### Basic Voice Conversion

```bash
# Convert voice with default settings
rvc-cli infer -m mymodel.pth -i input.wav -o output.wav

# Convert with pitch shift (12 semitones = octave up)
rvc-cli infer -m mymodel.pth -i input.wav -p 12

# Convert with custom F0 method
rvc-cli infer -m mymodel.pth -i input.wav --f0_method harvest
```

### Audio Separation

```bash
# Separate vocals from music
rvc-cli uvr -i music.wav

# Separate with specific model
rvc-cli uvr -i music.wav --model MDXNET_Main --output ./separated
```

### Launch Web Interface

```bash
# Start web interface on default port
rvc-cli serve

# Start on custom port with public share
rvc-cli serve --port 8080 --share
```

## Command Reference

### infer - Voice Conversion

Convert voice in an audio file using an RVC model.

```bash
rvc-cli infer -m <model> -i <input> [options]
```

**Required Arguments:**
- `-m, --model`: Path to RVC model file (.pth or .onnx)
- `-i, --input`: Path to input audio file

**Optional Arguments:**
- `-o, --output`: Output file path (auto-generated if not specified)
- `-p, --pitch`: Pitch shift in semitones (default: 0)
- `-f, --format`: Output format - wav, mp3, flac, ogg (default: wav)
- `--index`: Path to .index file for better quality
- `--f0_method`: F0 extraction method (default: rmvpe)
- `--filter_radius`: Filter radius for F0 smoothing (default: 3)
- `--index_rate`: Index strength 0.0-1.0 (default: 0.5)
- `--rms_mix_rate`: RMS mix rate (default: 1.0)
- `--protect`: Protect consonants 0.0-1.0 (default: 0.33)
- `--hop_length`: Hop length for processing (default: 64)
- `--embedder_model`: Embedding model (default: hubert_base)
- `--resample_sr`: Resample rate (0 = original, default: 0)
- `--split_audio`: Split audio before processing
- `--checkpointing`: Enable memory checkpointing
- `--f0_autotune`: Enable F0 autotune
- `--f0_autotune_strength`: Autotune strength (default: 1.0)
- `--formant_shifting`: Enable formant shifting
- `--formant_qfrency`: Formant frequency coefficient (default: 0.8)
- `--formant_timbre`: Formant timbre coefficient (default: 0.8)
- `--clean_audio`: Apply audio cleaning
- `--clean_strength`: Cleaning strength (default: 0.7)

**Example:**
```bash
rvc-cli infer -m artist_model.pth -i speech.wav -o converted.wav -p 2 \
    --index_rate 0.75 --f0_method rmvpe --clean_audio
```

### uvr - Audio Separation

Separate vocals from instrumentals using UVR5.

```bash
rvc-cli uvr -i <input> [options]
```

**Required Arguments:**
- `-i, --input`: Path to input audio file

**Optional Arguments:**
- `-o, --output`: Output directory (default: ./audios/uvr)
- `-f, --format`: Output format (default: wav)
- `--model`: Separation model (default: MDXNET_Main)
- `--karaoke_model`: Karaoke model (default: MDX-Version-1)
- `--reverb_model`: Reverb model (default: MDX-Reverb)
- `--denoise_model`: Denoise model (default: Normal)
- `--sample_rate`: Output sample rate (default: 44100)
- `--shifts`: Number of predictions (default: 2)
- `--batch_size`: Batch size (default: 1)
- `--overlap`: Overlap between segments (default: 0.25)
- `--aggression`: Extraction intensity (default: 5)
- `--hop_length`: Hop length (default: 1024)
- `--window_size`: Window size (default: 512)
- `--enable_tta`: Enable test-time augmentation
- `--enable_denoise`: Enable denoising
- `--separate_backing`: Separate backing vocals
- `--separate_reverb`: Separate reverb

**Available Models:**
- MDXNET_Main, MDXNET_9482
- HP-Vocal-1, HP-Vocal-2
- Inst_HQ_1 through Inst_HQ_5
- Kim_Vocal_1, Kim_Vocal_2

**Example:**
```bash
rvc-cli uvr -i song.wav --model HP-Vocal-2 --aggression 10 \
    --enable_denoise --output ./vocals
```

### create-dataset - Create Training Data

Create training dataset from YouTube videos or local audio files.

```bash
rvc-cli create-dataset -u <url> [options]
# or
rvc-cli create-dataset -i <directory> [options]
```

**Required Arguments (one of):**
- `-u, --url`: YouTube URL (separate multiple with commas)
- `-i, --input`: Input directory with audio files

**Optional Arguments:**
- `-o, --output`: Output directory (default: ./dataset)
- `--sample_rate`: Sample rate (default: 48000)
- `--clean_dataset`: Apply data cleaning
- `--clean_strength`: Cleaning strength (default: 0.7)
- `--separate`: Separate vocals (default: True)
- `--separator_model`: Separation model (default: MDXNET_Main)
- `--skip_start`: Seconds to skip at start (default: 0)
- `--skip_end`: Seconds to skip at end (default: 0)

**Example:**
```bash
rvc-cli create-dataset -u "https://youtube.com/watch?v=xxx" \
    --sample_rate 48000 --separate --output ./my_dataset
```

### create-index - Create Model Index

Create .index file for voice retrieval.

```bash
rvc-cli create-index <model_name> [options]
```

**Arguments:**
- `model_name`: Name of the model

**Optional Arguments:**
- `--version`: RVC version - v1 or v2 (default: v2)
- `--algorithm`: Index algorithm - Auto, Faiss, or KMeans (default: Auto)

**Example:**
```bash
rvc-cli create-index mymodel --version v2 --algorithm Faiss
```

### extract - Feature Extraction

Extract embeddings and F0 from training data.

```bash
rvc-cli extract <model_name> --sample_rate <rate> [options]
```

**Required Arguments:**
- `model_name`: Name of the model
- `--sample_rate`: Sample rate of input audio

**Optional Arguments:**
- `--version`: RVC version - v1 or v2 (default: v2)
- `--f0_method`: F0 extraction method (default: rmvpe)
- `--f0_onnx`: Use ONNX F0 predictor
- `--pitch_guidance`: Use pitch guidance (default: True)
- `--hop_length`: Hop length (default: 128)
- `--cpu_cores`: CPU cores (default: 2)
- `--gpu`: GPU index (default: - for CPU)
- `--embedder_model`: Embedder model (default: hubert_base)
- `--rms_extract`: Extract RMS energy

**Example:**
```bash
rvc-cli extract mymodel --sample_rate 48000 --f0_method rmvpe \
    --gpu 0 --pitch_guidance
```

### preprocess - Data Preprocessing

Slice and normalize training audio.

```bash
rvc-cli preprocess <model_name> --sample_rate <rate> [options]
```

**Required Arguments:**
- `model_name`: Name of the model
- `--sample_rate`: Sample rate

**Optional Arguments:**
- `--dataset_path`: Dataset path (default: ./dataset)
- `--cpu_cores`: CPU cores (default: 2)
- `--cut_method`: Cutting method - Automatic, Simple, or Skip (default: Automatic)
- `--process_effects`: Apply preprocessing effects
- `--clean_dataset`: Clean dataset
- `--chunk_len`: Chunk length for Simple method (default: 3.0)
- `--overlap_len`: Overlap length (default: 0.3)
- `--normalization`: Normalization mode - none, pre, or post (default: none)

**Example:**
```bash
rvc-cli preprocess mymodel --sample_rate 48000 --cut_method Automatic \
    --process_effects --normalization pre
```

### train - Model Training

Train a new RVC voice model.

```bash
rvc-cli train <model_name> [options]
```

**Required Arguments:**
- `model_name`: Name of the model

**Optional Arguments:**
- `--version`: RVC version - v1 or v2 (default: v2)
- `--author`: Model author name
- `--epochs`: Total training epochs (default: 300)
- `--batch_size`: Batch size (default: 8)
- `--save_every`: Save checkpoint every N epochs (default: 50)
- `--save_latest`: Save only latest checkpoint (default: True)
- `--save_weights`: Save all model weights (default: True)
- `--gpu`: GPU index (default: 0)
- `--cache_gpu`: Cache data in GPU
- `--pitch_guidance`: Use pitch guidance (default: True)
- `--pretrained_g`: Path to pre-trained G weights
- `--pretrained_d`: Path to pre-trained D weights
- `--vocoder`: Vocoder - Default, MRF-HiFi-GAN, or RefineGAN (default: Default)
- `--energy`: Use RMS energy
- `--overtrain_detect`: Enable overtraining detection
- `--optimizer`: Optimizer - AdamW, RAdam, or AnyPrecisionAdamW (default: AdamW)
- `--multiscale_loss`: Use multi-scale mel loss
- `--use_reference`: Use custom reference set
- `--reference_path`: Path to reference set

**Example:**
```bash
rvc-cli train mymodel --version v2 --epochs 500 --batch_size 8 \
    --gpu 0 --save_every 100 --vocoder "MRF-HiFi-GAN"
```

### create-ref - Create Reference Set

Create reference audio for better inference quality.

```bash
rvc-cli create-ref <audio_file> [options]
```

**Required Arguments:**
- `audio_file`: Path to audio file

**Optional Arguments:**
- `-n, --name`: Reference name (default: reference)
- `--version`: RVC version - v1 or v2 (default: v2)
- `--pitch_guidance`: Use pitch guidance (default: True)
- `--energy`: Use RMS energy
- `--embedder_model`: Embedder model (default: hubert_base)
- `--f0_method`: F0 extraction method (default: rmvpe)
- `--pitch_shift`: Pitch shift (default: 0)
- `--filter_radius`: Filter radius (default: 3)
- `--f0_autotune`: Enable F0 autotune
- `--alpha`: Alpha blending (default: 0.5)

**Example:**
```bash
rvc-cli create-ref reference_audio.wav -n myref --f0_method rmvpe
```

### download - Download Models/Audio

Download models from HuggingFace or audio from YouTube.

```bash
rvc-cli download -l <link> [options]
```

**Required Arguments:**
- `-l, --link`: Download link (HuggingFace or YouTube URL)

**Optional Arguments:**
- `-t, --type`: Download type - model, audio, or index (default: model)
- `-n, --name`: Name to save as

**Example:**
```bash
rvc-cli download -l "https://huggingface.co/user/model/resolve/main/model.pth"
```

### serve - Web Interface

Launch the Gradio web UI.

```bash
rvc-cli serve [options]
```

**Optional Arguments:**
- `--host`: Host to bind (default: 0.0.0.0)
- `--port`: Port to bind (default: 7860)
- `--share`: Create public share URL

**Example:**
```bash
rvc-cli serve --port 7860 --share
```

### info - System Information

Show system and environment information.

```bash
rvc-cli info
```

Displays:
- Operating system and version
- CPU information
- Memory and disk space
- GPU information (if available)
- Python and package versions

### version - Version Info

Show version and dependency information.

```bash
rvc-cli version
```

### list-models - List Available Models

List installed models in the weights folder.

```bash
rvc-cli list-models
```

### list-f0-methods - List F0 Methods

Show all available pitch extraction methods.

```bash
rvc-cli list-f0-methods
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ARVC_ASSETS_PATH` | Path to assets directory | `assets` |
| `ARVC_CONFIGS_PATH` | Path to configs directory | `configs` |
| `ARVC_WEIGHTS_PATH` | Path to weights directory | `assets/weights` |
| `ARVC_LOGS_PATH` | Path to logs directory | `assets/logs` |

### Model and Index Files

Place your model files (`.pth` or `.onnx`) in:
```
advanced_rvc_inference/assets/weights/
```

Place index files (`.index`) in:
```
advanced_rvc_inference/assets/logs/<model_name>/
```

## Examples

### Complete Training Workflow

```bash
# 1. Create dataset from YouTube
rvc-cli create-dataset -u "https://youtube.com/watch?v=xxx" \
    --output ./dataset --sample_rate 48000 --separate

# 2. Create index for the model (after training)
rvc-cli create-index mymodel --version v2

# 3. Extract features
rvc-cli extract mymodel --sample_rate 48000 --f0_method rmvpe --gpu 0

# 4. Preprocess data
rvc-cli preprocess mymodel --sample_rate 48000 --cut_method Automatic

# 5. Train model
rvc-cli train mymodel --version v2 --epochs 300 --batch_size 8 --gpu 0
```

### Voice Conversion with Different F0 Methods

```bash
# Using RMVPE (recommended)
rvc-cli infer -m model.pth -i input.wav -o output_rmvpe.wav --f0_method rmvpe

# Using Harvest (faster)
rvc-cli infer -m model.pth -i input.wav -o output_harvest.wav --f0_method harvest

# Using Crepe (most accurate but slow)
rvc-cli infer -m model.pth -i input.wav -o output_crepe.wav --f0_method crepe-medium
```

### Batch Processing

```bash
# Process all audio files in a directory
for file in ./inputs/*.wav; do
    rvc-cli infer -m model.pth -i "$file" -o "./outputs/$(basename $file)"
done
```

### Multiple Voice Models

```bash
# Convert with different models
rvc-cli infer -m model_a.pth -i voice.wav -o voice_model_a.wav -p 0
rvc-cli infer -m model_b.pth -i voice.wav -o voice_model_b.wav -p 2
rvc-cli infer -m model_c.pth -i voice.wav -o voice_model_c.wav -p -2
```

## Troubleshooting

### Common Issues

#### "Model file not found"
- Ensure the model path is correct
- Check that the model file has `.pth` or `.onnx` extension
- Verify file permissions

#### "CUDA out of memory"
- Reduce batch size: `--batch_size 4`
- Enable checkpointing: `--checkpointing`
- Use CPU: `--gpu -`

#### "Audio format not supported"
- Convert audio to WAV first: `ffmpeg -i input.mp3 output.wav`
- Supported formats: wav, mp3, flac, ogg, opus, m4a, aac

#### "F0 method not available"
- Install ONNX runtime for some methods
- Some methods require specific embedders

### Getting Help

```bash
# General help
rvc-cli --help

# Command-specific help
rvc-cli infer --help
rvc-cli uvr --help
rvc-cli train --help
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please read the contributing guidelines in CONTRIBUTING.md.

## Support

- GitHub Issues: https://github.com/ArkanDash/Advanced-RVC-Inference/issues
- Discord: https://discord.gg/hvmsukmBHE

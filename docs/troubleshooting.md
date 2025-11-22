# Troubleshooting Guide

This guide provides solutions to common issues and problems you may encounter when using Advanced RVC Inference.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [CUDA and GPU Problems](#cuda-and-gpu-problems)
3. [Memory Issues](#memory-issues)
4. [Audio Processing Problems](#audio-processing-problems)
5. [Model Loading Issues](#model-loading-issues)
6. [Training Problems](#training-problems)
7. [UI and Interface Issues](#ui-and-interface-issues)
8. [Performance Optimization](#performance-optimization)
9. [Environment-Specific Issues](#environment-specific-issues)

## Installation Issues

### Python Version Compatibility

**Problem**: "Python 3.8+ required" error during installation

**Solution**:
```bash
# Check Python version
python --version

# If using older version, upgrade
# On Ubuntu/Debian:
sudo apt update
sudo apt install python3.9 python3.9-venv python3.9-dev

# On macOS with Homebrew:
brew install python@3.9

# On Windows:
# Download from https://python.org/downloads/
```

### Dependency Installation Failures

**Problem**: pip install fails with dependency conflicts

**Solution**:
```bash
# Create fresh virtual environment
python -m venv fresh_rvc_env
source fresh_rvc_env/bin/activate  # Windows: fresh_rvc_env\Scripts\activate

# Upgrade pip first
pip install --upgrade pip

# Install PyTorch separately (CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# If still failing, try minimal installation
pip install gradio librosa soundfile numpy scipy
```

### FFmpeg Installation

**Problem**: "FFmpeg not found" error

**Solution**:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# CentOS/RHEL/Fedora
sudo yum install ffmpeg  # or
sudo dnf install ffmpeg

# macOS with Homebrew
brew install ffmpeg

# Windows with Chocolatey
choco install ffmpeg

# Windows with Scoop
scoop install ffmpeg

# Verify installation
ffmpeg -version
```

### CUDA Toolkit Issues

**Problem**: CUDA not detected or version mismatch

**Solution**:
```bash
# Check CUDA installation
nvidia-smi

# Check CUDA version
nvcc --version

# If CUDA not found, install CUDA Toolkit
# Download from: https://developer.nvidia.com/cuda-downloads

# Verify PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## CUDA and GPU Problems

### CUDA Out of Memory (OOM)

**Problem**: "CUDA out of memory" error during processing

**Solutions**:

**1. Reduce Batch Size**:
```python
from src.advanced_rvc_inference.config import Config

config = Config.get_instance()
config.set_batch_size(2)  # Reduce from default 4 or 8
```

**2. Switch to CPU**:
```bash
# Run with CPU only
python app.py --cpu-only

# Or in Python
from src.advanced_rvc_inference.config import Config
config = Config.get_instance()
config.set_device("cpu")
```

**3. Memory Cleanup**:
```python
from src.advanced_rvc_inference.core.memory_manager import MemoryManager

# Force memory cleanup
MemoryManager.cleanup_memory()
torch.cuda.empty_cache()

# Use memory-optimized context
with MemoryManager.memory_optimized(threshold=75):
    # Your processing code
    pass
```

**4. Process Smaller Files**:
- Split large audio files into smaller segments
- Use lower quality settings temporarily

### GPU Not Detected

**Problem**: "No CUDA-capable device is detected"

**Solutions**:

**1. Check GPU Installation**:
```bash
# Check if NVIDIA drivers are installed
nvidia-smi

# If not installed, install drivers
# Ubuntu: sudo apt install nvidia-driver-470
# Windows: Download from NVIDIA website
```

**2. Check CUDA Installation**:
```bash
# Verify CUDA toolkit
nvcc --version

# Check PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

**3. Force GPU Detection**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")

# If CUDA available but not detected by application
from src.advanced_rvc_inference.config import Config
config = Config.get_instance()
config.set_device("cuda")
```

### Incorrect GPU Memory Usage

**Problem**: GPU memory not being released after processing

**Solution**:
```python
# Add memory cleanup after processing
import torch
import gc

def process_with_cleanup(audio_path, model_path):
    try:
        result = process_audio(audio_path, model_path, output_path="output.wav")
        return result
    finally:
        # Force cleanup
        torch.cuda.empty_cache()
        gc.collect()
        print(f"GPU Memory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Use MemoryManager for automatic cleanup
from src.advanced_rvc_inference.core.memory_manager import MemoryManager

with MemoryManager.memory_optimized():
    result = process_audio(audio_path, model_path, output_path="output.wav")
```

## Memory Issues

### High System Memory Usage

**Problem**: System becomes unresponsive during processing

**Solutions**:

**1. Monitor Memory Usage**:
```python
from src.advanced_rvc_inference.core.memory_manager import MemoryManager

# Check current usage
usage = MemoryManager.get_memory_usage()
print(f"System Memory: {usage.system_memory_percent:.1f}%")
print(f"GPU Memory: {usage.gpu_memory_percent:.1f}%")

# Start monitoring
MemoryManager.start_monitoring(interval=5.0)
```

**2. Reduce Memory Usage**:
```python
# Reduce batch size
config.set_batch_size(2)

# Use smaller audio chunks
config.set_chunk_size(1024)  # Smaller processing chunks

# Enable aggressive memory cleanup
config.set_memory_threshold(70)  # Clean up at 70% usage
```

**3. Process Files Sequentially**:
```python
# Instead of batch processing
for audio_file in audio_files:
    with MemoryManager.memory_optimized():
        process_audio(audio_file, model_path, output_path)
```

### Memory Leaks During Long Sessions

**Problem**: Memory usage increases over time

**Solution**:
```python
# Periodic cleanup
import threading
import time

def periodic_cleanup():
    while True:
        time.sleep(300)  # Every 5 minutes
        MemoryManager.cleanup_memory()
        print("Periodic memory cleanup performed")

# Start cleanup thread
cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
cleanup_thread.start()
```

## Audio Processing Problems

### Audio File Format Issues

**Problem**: Unsupported audio format or codec errors

**Solutions**:

**1. Check Supported Formats**:
```python
# Supported: WAV, MP3, FLAC, OGG, M4A, AAC, ALAC
import os
from pathlib import Path

def check_audio_format(file_path):
    """Check if audio format is supported."""
    supported_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.alac']
    
    ext = Path(file_path).suffix.lower()
    if ext in supported_extensions:
        print(f"Format {ext} is supported")
        return True
    else:
        print(f"Format {ext} is not supported")
        return False

# Convert unsupported formats
def convert_audio(input_path, output_path):
    """Convert audio to supported format using FFmpeg."""
    import subprocess
    
    cmd = [
        'ffmpeg', '-i', input_path,
        '-acodec', 'pcm_s16le',  # WAV format
        '-ar', '44100',          # Sample rate
        '-ac', '1',              # Mono
        '-y',                    # Overwrite output
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Converted {input_path} to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed: {e}")
```

### Audio Quality Issues

**Problem**: Poor output audio quality

**Solutions**:

**1. Check Input Audio Quality**:
```python
# Ensure input audio meets requirements
def validate_audio_quality(audio_path):
    """Validate input audio quality."""
    import librosa
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=None)
    
    print(f"Sample Rate: {sr} Hz")
    print(f"Duration: {len(audio) / sr:.2f} seconds")
    print(f"Channels: {1 if len(audio.shape) == 1 else audio.shape[0]}")
    
    # Quality checks
    issues = []
    
    if sr < 22050:
        issues.append("Low sample rate (use 44100 Hz)")
    
    if len(audio) / sr < 1.0:
        issues.append("Audio too short (minimum 1 second)")
    
    if sr > 48000:
        issues.append("Unnecessarily high sample rate")
    
    if issues:
        print("Audio quality issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Audio quality is good")
    
    return len(issues) == 0
```

**2. Optimize Processing Parameters**:
```python
# Use high-quality F0 extraction
result = process_audio(
    audio_path="input.wav",
    model_path="model.pth",
    output_path="output.wav",
    f0_method="hybrid[crepe+rmvpe]",  # High quality method
    filter_radius=3,                  # Appropriate filtering
    rms_mix_rate=0.25,               # Good balance
    protect=0.33                     # Protect original characteristics
)
```

### Silent or Distorted Output

**Problem**: Output audio is silent or heavily distorted

**Solutions**:

**1. Check Model Compatibility**:
```python
from src.advanced_rvc_inference.models.manager import EnhancedModelManager

manager = EnhancedModelManager()

# Validate model
if not manager.validate_model("model.pth"):
    print("Model validation failed")
    # Try different model or check file integrity

# Check model info
model_info = manager.get_model_info("model.pth")
print(f"Model type: {model_info.model_type}")
print(f"Sample rate: {model_info.sample_rate}")
print(f"Channels: {model_info.channels}")
```

**2. Adjust Processing Parameters**:
```python
# Start with conservative settings
result = process_audio(
    audio_path="input.wav",
    model_path="model.pth",
    output_path="output.wav",
    f0_method="rmvpe",              # Stable method
    pitch_shift=0,                  # No pitch change initially
    filter_radius=3,                # Standard filtering
    rms_mix_rate=0.3,              # Moderate mixing
    protect=0.5                     # Higher protection
)

# Gradually adjust parameters
```

**3. Check Input/Output Paths**:
```python
import os

# Verify paths exist
input_file = "input.wav"
output_file = "output.wav"

if not os.path.exists(input_file):
    print(f"Input file not found: {input_file}")

if os.path.exists(output_file):
    # Check file size
    size = os.path.getsize(output_file)
    print(f"Output file size: {size} bytes")
    
    if size < 1000:  # Less than 1KB
        print("Output file suspiciously small - check processing")
```

## Model Loading Issues

### Model Not Found

**Problem**: "Model file not found" or "No models available"

**Solutions**:

**1. Check File Paths**:
```python
import os
from pathlib import Path

# Check if model files exist
model_dir = "weights/"
index_dir = "indexes/"

print("Checking model directory:")
if os.path.exists(model_dir):
    models = list(Path(model_dir).glob("*.pth")) + list(Path(model_dir).glob("*.onnx"))
    print(f"Found {len(models)} models:")
    for model in models:
        print(f"  - {model}")
else:
    print(f"Model directory not found: {model_dir}")

print("\nChecking index directory:")
if os.path.exists(index_dir):
    indexes = list(Path(index_dir).glob("*.index"))
    print(f"Found {len(indexes)} index files:")
    for index in indexes:
        print(f"  - {index}")
else:
    print(f"Index directory not found: {index_dir}")
```

**2. Download Models**:
```python
from src.advanced_rvc_inference.tabs.utilities.download_model import ModelDownloader

downloader = ModelDownloader()

# List available models
available_models = downloader.list_public_models()
print("Available models:")
for model in available_models[:10]:  # Show first 10
    print(f"  - {model['name']}: {model['description']}")

# Download a model
result = downloader.download_model(
    model_id="model_id_here",
    output_path="weights/",
    extract=True
)
```

**3. Validate Model Files**:
```python
from src.advanced_rvc_inference.models.manager import EnhancedModelManager

manager = EnhancedModelManager()

# Check all models in directory
model_files = ["model1.pth", "model2.onnx", "model3.pth"]

for model_file in model_files:
    if os.path.exists(model_file):
        try:
            is_valid = manager.validate_model(model_file)
            print(f"{model_file}: {'Valid' if is_valid else 'Invalid'}")
            
            if is_valid:
                info = manager.get_model_info(model_file)
                print(f"  Type: {info.model_type}")
                print(f"  Sample Rate: {info.sample_rate}")
        except Exception as e:
            print(f"Error checking {model_file}: {e}")
    else:
        print(f"File not found: {model_file}")
```

### Model Format Compatibility

**Problem**: PyTorch and ONNX model compatibility issues

**Solutions**:

**1. Check Model Format**:
```python
import torch

def check_model_format(model_path):
    """Check model format and compatibility."""
    
    try:
        if model_path.endswith('.pth') or model_path.endswith('.pt'):
            # PyTorch model
            model = torch.load(model_path, map_location='cpu')
            print(f"PyTorch model loaded successfully")
            print(f"Model type: {type(model)}")
            
        elif model_path.endswith('.onnx'):
            # ONNX model
            import onnx
            model = onnx.load(model_path)
            onnx.checker.check_model(model)
            print(f"ONNX model validated successfully")
            
        else:
            print(f"Unknown model format: {model_path}")
            
    except Exception as e:
        print(f"Error loading model: {e}")

# Check all models
for model_file in os.listdir("weights/"):
    if model_file.endswith(('.pth', '.pt', '.onnx')):
        print(f"\nChecking {model_file}:")
        check_model_format(f"weights/{model_file}")
```

**2. Convert Between Formats**:
```python
from src.advanced_rvc_inference.models.manager import EnhancedModelManager

manager = EnhancedModelManager()

# Convert PyTorch to ONNX
manager.convert_model(
    "model.pth", 
    "model.onnx", 
    use_dynamic_axes=True
)

# Convert ONNX to PyTorch
# Note: This requires custom conversion logic
# See conversion tools in the codebase
```

### Index File Issues

**Problem**: Missing or invalid index files

**Solutions**:

**1. Check Index File**:
```python
import numpy as np

def check_index_file(index_path):
    """Validate index file."""
    
    try:
        index_data = np.load(index_path)
        print(f"Index file loaded: {index_path}")
        print(f"Shape: {index_data.shape}")
        print(f"Data type: {index_data.dtype}")
        print(f"Min value: {index_data.min()}")
        print(f"Max value: {index_data.max()}")
        
        # Check for NaN or inf values
        if np.isnan(index_data).any():
            print("Warning: Index contains NaN values")
        
        if np.isinf(index_data).any():
            print("Warning: Index contains infinite values")
            
        return True
        
    except Exception as e:
        print(f"Error loading index file: {e}")
        return False

# Check all index files
for index_file in os.listdir("indexes/"):
    if index_file.endswith('.index'):
        print(f"\nChecking {index_file}:")
        check_index_file(f"indexes/{index_file}")
```

**2. Regenerate Index Files**:
```python
# If index files are missing, you may need to regenerate them
# This typically requires the original training dataset
# Check the training documentation for index generation
```

## Training Problems

### Training Fails to Start

**Problem**: Training crashes immediately or won't initialize

**Solutions**:

**1. Check Dataset**:
```python
from src.advanced_rvc_inference.training.data.dataset import RVCDataset

dataset = RVCDataset("path/to/dataset/")

# Validate dataset
validation_result = dataset.validate_dataset()
if not validation_result.is_valid:
    print("Dataset validation failed:")
    for error in validation_result.errors:
        print(f"  - {error}")

# Check dataset statistics
stats = dataset.get_statistics()
print(f"Total files: {stats.total_files}")
print(f"Average duration: {stats.avg_duration:.2f}s")
print(f"Total duration: {stats.total_duration:.2f}s")

# Common issues:
# - Not enough training files (< 10)
# - Audio files too short (< 1 second)
# - Sample rate mismatch
# - Corrupted audio files
```

**2. Check Training Configuration**:
```python
from src.advanced_rvc_inference.training.trainer import RVCTrainer

# Validate configuration
config = {
    "batch_size": 4,           # Adjust based on GPU memory
    "epochs": 100,
    "learning_rate": 0.001,
    "sample_rate": 44100,
    "save_frequency": 10
}

# Check if configuration is valid
trainer = RVCTrainer(config=config)

# Validate before training
try:
    trainer.validate_config(config)
    print("Configuration is valid")
except Exception as e:
    print(f"Configuration error: {e}")
```

**3. Check Disk Space**:
```python
import shutil

def check_disk_space(required_gb=5):
    """Check if enough disk space is available."""
    
    total, used, free = shutil.disk_usage(".")
    free_gb = free / (1024**3)
    
    print(f"Free disk space: {free_gb:.2f} GB")
    print(f"Required: {required_gb} GB")
    
    if free_gb < required_gb:
        print("Warning: Insufficient disk space for training")
        return False
    
    return True

# Check before training
if not check_disk_space(required_gb=10):
    print("Please free up disk space before training")
```

### Slow Training Progress

**Problem**: Training takes too long or seems stuck

**Solutions**:

**1. Optimize Training Parameters**:
```python
# Reduce batch size for faster processing
config = {
    "batch_size": 2,          # Smaller batches
    "epochs": 50,             # Fewer epochs
    "save_frequency": 5,      # Save more frequently
    "learning_rate": 0.01,    # Higher learning rate (faster convergence)
}

# Use mixed precision for faster training
config["use_mixed_precision"] = True

# Enable gradient checkpointing
config["gradient_checkpointing"] = True
```

**2. Use Faster F0 Extraction**:
```python
# Use faster F0 methods during training
config["f0_method"] = "rmvpe"  # Faster than hybrid methods

# Or use CPU for F0 extraction to save GPU memory
config["f0_device"] = "cpu"
```

**3. Monitor Training Progress**:
```python
# Check if training is actually progressing
import matplotlib.pyplot as plt

def plot_training_progress(log_file):
    """Plot training loss from log file."""
    
    import re
    
    losses = []
    epochs = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if 'epoch' in line and 'loss' in line:
                match = re.search(r'epoch (\d+).*loss ([0-9.]+)', line)
                if match:
                    epochs.append(int(match.group(1)))
                    losses.append(float(match.group(2)))
    
    if epochs and losses:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.show()
    else:
        print("No training data found in log file")

# Monitor progress
plot_training_progress("logs/training.log")
```

### Training OOM (Out of Memory)

**Problem**: Training runs out of GPU memory

**Solutions**:

**1. Reduce Memory Usage**:
```python
config = {
    "batch_size": 1,                    # Minimal batch size
    "gradient_accumulation_steps": 4,   # Accumulate gradients
    "max_grad_norm": 1.0,               # Gradient clipping
    "use_mixed_precision": True,        # Mixed precision training
    "gradient_checkpointing": True,     # Save memory at cost of speed
}

# Use CPU for F0 extraction
config["f0_device"] = "cpu"
config["embedder_device"] = "cpu"
```

**2. Monitor Memory During Training**:
```python
from src.advanced_rvc_inference.core.memory_manager import MemoryManager

def monitor_training_memory():
    """Monitor memory during training."""
    
    def check_memory():
        usage = MemoryManager.get_memory_usage()
        print(f"GPU Memory: {usage.gpu_memory_percent:.1f}%")
        
        if usage.gpu_memory_percent > 90:
            print("WARNING: High GPU memory usage")
            MemoryManager.cleanup_memory()
    
    return check_memory

# Use during training
memory_check = monitor_training_memory()
# Call memory_check() periodically during training
```

## UI and Interface Issues

### Gradio Interface Not Loading

**Problem**: Web interface fails to start or is inaccessible

**Solutions**:

**1. Check Port Availability**:
```python
import socket

def check_port_available(port=7860):
    """Check if port is available."""
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        result = s.connect_ex(('localhost', port))
        if result == 0:
            print(f"Port {port} is in use")
            return False
        else:
            print(f"Port {port} is available")
            return True

# Check port before starting
if not check_port_available(7860):
    print("Try a different port: python app.py --port 8080")
```

**2. Check Firewall/Security**:
```bash
# Ubuntu/Debian - allow port
sudo ufw allow 7860

# Check if interface is binding correctly
python app.py --host 0.0.0.0 --port 7860
```

**3. Check Dependencies**:
```bash
# Update Gradio
pip install --upgrade gradio

# Check for missing dependencies
pip install -r requirements.txt --force-reinstall
```

### Interface Freezing or Slow

**Problem**: UI becomes unresponsive or very slow

**Solutions**:

**1. Reduce UI Complexity**:
```python
# In app.py or configuration
ui_config = {
    "show_advanced_options": False,  # Hide advanced settings
    "theme": "compact",              # Use compact theme
    "auto_refresh": False,           # Disable auto-refresh
}
```

**2. Browser Issues**:
- Try different browser (Chrome, Firefox, Safari)
- Clear browser cache
- Disable browser extensions
- Try incognito/private mode

**3. Performance Optimization**:
```python
# Enable performance mode
config = Config.get_instance()
config.set_performance_mode(True)
config.set_ui_refresh_interval(2.0)  # Refresh every 2 seconds
```

## Performance Optimization

### Slow Processing Speed

**Problem**: Audio processing takes too long

**Solutions**:

**1. Optimize Processing Parameters**:
```python
# Use faster F0 methods
f0_method = "rmvpe"  # Faster than hybrid methods
# or
f0_method = "crepe_tiny"  # Fastest deep learning method

# Reduce quality for speed
result = process_audio(
    audio_path="input.wav",
    model_path="model.pth",
    output_path="output.wav",
    f0_method="rmvpe",           # Fast method
    filter_radius=1,             # Less filtering
    resample_sr=22050,           # Lower sample rate
)
```

**2. Batch Processing Optimization**:
```python
# Process multiple files together
from concurrent.futures import ThreadPoolExecutor

def batch_process_files(audio_files, model_path, max_workers=2):
    """Process multiple files with optimal worker count."""
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for audio_file in audio_files:
            future = executor.submit(
                process_audio,
                audio_file,
                model_path,
                f"output_{os.path.basename(audio_file)}"
            )
            futures.append(future)
        
        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing file: {e}")
        
        return results
```

**3. Hardware Optimization**:
```bash
# Monitor GPU usage during processing
nvidia-smi -l 1

# Optimize GPU settings
# Use nvidi-smi to check if GPU is being utilized

# Enable persistent GPU processes (Linux)
sudo nvidia-persistenced --persistence-mode
```

### Memory Optimization

**Problem**: High memory usage affecting performance

**Solutions**:

**1. Configure Memory Settings**:
```python
from src.advanced_rvc_inference.config import Config

config = Config.get_instance()

# Set aggressive memory cleanup
config.set_memory_threshold(70)  # Clean up at 70%
config.set_batch_size(2)         # Small batches
config.set_chunk_size(512)       # Small processing chunks
```

**2. Use Memory-Efficient Processing**:
```python
# Process files one at a time
for audio_file in audio_files:
    with MemoryManager.memory_optimized():
        process_audio(audio_file, model_path, output_path)
    
    # Force cleanup between files
    MemoryManager.cleanup_memory()
```

## Environment-Specific Issues

### Windows-Specific Problems

**Problem**: Issues specific to Windows environment

**Solutions**:

**1. Path Separators**:
```python
import os

# Use os.path.join for cross-platform compatibility
model_path = os.path.join("weights", "model.pth")
output_path = os.path.join("output", "audio.wav")

# Or use pathlib (recommended)
from pathlib import Path

model_path = Path("weights") / "model.pth"
output_path = Path("output") / "audio.wav"
```

**2. Windows Audio Drivers**:
```python
# Check Windows audio device
import pyaudio

p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(f"Device {i}: {info['name']} (Input: {info['maxInputChannels']}, Output: {info['maxOutputChannels']})")
p.terminate()
```

**3. Windows Firewall**:
```bash
# Allow Python through Windows Firewall
# Or run as administrator if needed
python app.py
```

### macOS-Specific Problems

**Problem**: Issues specific to macOS environment

**Solutions**:

**1. Audio Permissions**:
```bash
# Grant microphone access in System Preferences > Security & Privacy > Privacy > Microphone
```

**2. Metal Performance Shaders**:
```bash
# Install MPS-compatible PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**3. FFmpeg on macOS**:
```bash
# Install FFmpeg via Homebrew
brew install ffmpeg

# Verify installation
ffmpeg -version
```

### Linux-Specific Problems

**Problem**: Issues specific to Linux environment

**Solutions**:

**1. Audio Permissions**:
```bash
# Add user to audio group
sudo usermod -a -G audio $USER

# Or use PulseAudio/ALSA directly
```

**2. NVIDIA Drivers**:
```bash
# Install NVIDIA drivers
sudo apt install nvidia-driver-470  # Adjust version as needed

# Check installation
nvidia-smi
```

**3. Permission Issues**:
```bash
# Fix file permissions
chmod +x app.py
chmod 755 weights/
chmod 755 indexes/
```

### Google Colab Issues

**Problem**: Issues specific to Google Colab environment

**Solutions**:

**1. Runtime Disconnection**:
```python
# Save progress frequently
# Use checkpoint system
import pickle

def save_checkpoint(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Checkpoint saved: {filename}")

def load_checkpoint(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
```

**2. Mount Drive**:
```python
# Mount Google Drive for persistence
from google.colab import drive
drive.mount('/content/drive')

# Create symlinks
!ln -s /content/drive/MyDrive/RVC_Models /content/weights
!ln -s /content/drive/MyDrive/RVC_Indexes /content/indexes
```

**3. Dependency Caching**:
```python
# Check if dependencies are already installed
import os

marker_file = "/content/.dependencies_installed"

if not os.path.exists(marker_file):
    print("Installing dependencies...")
    # Your installation code here
    
    # Create marker file
    with open(marker_file, 'w') as f:
        f.write("Dependencies installed")
else:
    print("Dependencies already installed, skipping...")
```

## Getting Help

### Log File Analysis

**Problem**: Need to analyze logs for debugging

**Solution**:
```python
import re
from datetime import datetime

def analyze_log_file(log_file):
    """Analyze log file for common issues."""
    
    errors = []
    warnings = []
    memory_issues = []
    
    with open(log_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Check for errors
            if 'error' in line.lower() or 'exception' in line.lower():
                errors.append((line_num, line))
            
            # Check for warnings
            if 'warning' in line.lower():
                warnings.append((line_num, line))
            
            # Check for memory issues
            if 'memory' in line.lower() or 'oom' in line.lower():
                memory_issues.append((line_num, line))
    
    print(f"Log Analysis for {log_file}:")
    print(f"Errors found: {len(errors)}")
    for line_num, line in errors[-5:]:  # Show last 5 errors
        print(f"  Line {line_num}: {line}")
    
    print(f"Warnings found: {len(warnings)}")
    for line_num, line in warnings[-5:]:  # Show last 5 warnings
        print(f"  Line {line_num}: {line}")
    
    print(f"Memory issues found: {len(memory_issues)}")
    for line_num, line in memory_issues:
        print(f"  Line {line_num}: {line}")

# Analyze log files
analyze_log_file("logs/app.log")
analyze_log_file("logs/error.log")
```

### Diagnostic Script

**Problem**: Need comprehensive system diagnosis

**Solution**:
```python
#!/usr/bin/env python3
"""
Advanced RVC Inference Diagnostic Script
Run this script to gather system information for troubleshooting
"""

import sys
import os
import torch
import platform
import subprocess
from pathlib import Path

def run_diagnostics():
    """Run comprehensive system diagnostics."""
    
    print("=" * 60)
    print("Advanced RVC Inference - System Diagnostics")
    print("=" * 60)
    
    # System Information
    print(f"\nSystem Information:")
    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  Python: {sys.version}")
    print(f"  Platform: {platform.platform()}")
    
    # PyTorch Information
    print(f"\nPyTorch Information:")
    print(f"  Version: {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    
    # File System Check
    print(f"\nFile System Check:")
    required_dirs = ["src", "weights", "indexes", "logs"]
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            files = list(Path(dir_name).glob("*"))
            print(f"  {dir_name}/: {len(files)} files")
        else:
            print(f"  {dir_name}/: NOT FOUND")
    
    # Model Files Check
    print(f"\nModel Files Check:")
    weights_dir = Path("weights")
    if weights_dir.exists():
        models = list(weights_dir.glob("*.pth")) + list(weights_dir.glob("*.onnx"))
        print(f"  Models found: {len(models)}")
        for model in models[:5]:  # Show first 5
            size_mb = model.stat().st_size / (1024 * 1024)
            print(f"    {model.name}: {size_mb:.1f} MB")
    else:
        print(f"  weights/ directory not found")
    
    # Index Files Check
    print(f"\nIndex Files Check:")
    indexes_dir = Path("indexes")
    if indexes_dir.exists():
        indexes = list(indexes_dir.glob("*.index"))
        print(f"  Index files found: {len(indexes)}")
    else:
        print(f"  indexes/ directory not found")
    
    # Dependencies Check
    print(f"\nDependencies Check:")
    required_packages = ["gradio", "librosa", "soundfile", "numpy", "scipy"]
    for package in required_packages:
        try:
            __import__(package)
            print(f"  {package}: ✓")
        except ImportError:
            print(f"  {package}: ✗ MISSING")
    
    # Audio System Check
    print(f"\nAudio System Check:")
    try:
        import pyaudio
        print(f"  PyAudio: ✓")
        
        # List audio devices
        p = pyaudio.PyAudio()
        input_devices = []
        output_devices = []
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                input_devices.append(info['name'])
            if info['maxOutputChannels'] > 0:
                output_devices.append(info['name'])
        
        print(f"  Input devices: {len(input_devices)}")
        print(f"  Output devices: {len(output_devices)}")
        p.terminate()
    except ImportError:
        print(f"  PyAudio: ✗ MISSING")
    
    # FFmpeg Check
    print(f"\nFFmpeg Check:")
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  FFmpeg: ✓")
            version_line = result.stdout.split('\n')[0]
            print(f"    {version_line}")
        else:
            print(f"  FFmpeg: ✗ ERROR")
    except FileNotFoundError:
        print(f"  FFmpeg: ✗ NOT FOUND")
    
    # Configuration Check
    print(f"\nConfiguration Check:")
    config_file = Path("src/advanced_rvc_inference/assets/config.json")
    if config_file.exists():
        print(f"  Config file: ✓")
    else:
        print(f"  Config file: ✗ NOT FOUND")
    
    # Memory Check
    print(f"\nMemory Check:")
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"  System RAM: {memory.total / 1e9:.1f} GB")
        print(f"  Available: {memory.available / 1e9:.1f} GB")
        print(f"  Usage: {memory.percent:.1f}%")
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_used = torch.cuda.memory_allocated()
            print(f"  GPU Memory: {gpu_used / 1e9:.1f} / {gpu_memory / 1e9:.1f} GB")
    except ImportError:
        print(f"  psutil: ✗ MISSING - Cannot check memory")
    
    print(f"\n" + "=" * 60)
    print("Diagnostic Complete")
    print("=" * 60)

if __name__ == "__main__":
    run_diagnostics()
```

### Community Support

**Problem**: Need help from the community

**Solution**:
```markdown
When seeking help, please include:

1. **System Information**:
   - Operating system and version
   - Python version
   - PyTorch version
   - CUDA version (if applicable)

2. **Error Details**:
   - Complete error message
   - Steps to reproduce
   - What you were trying to do

3. **Log Files**:
   - Relevant log entries
   - Time when error occurred

4. **Configuration**:
   - Relevant configuration settings
   - Model files being used

5. **Diagnostic Output**:
   - Run the diagnostic script above
   - Include the output in your help request

### Where to Get Help:
- **GitHub Issues**: [https://github.com/ArkanDash/Advanced-RVC-Inference/issues](https://github.com/ArkanDash/Advanced-RVC-Inference/issues)
- **Discord Community**: [https://discord.gg/arkandash](https://discord.gg/arkandash)
- **Documentation**: [Complete documentation](docs/)
- **Email**: Contact maintainers through GitHub
```

This troubleshooting guide covers the most common issues users encounter. If you're still experiencing problems after trying these solutions, please gather the diagnostic information and reach out to the community for support.
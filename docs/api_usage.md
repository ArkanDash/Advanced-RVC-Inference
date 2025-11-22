# API Usage Documentation

This document provides comprehensive documentation for using the Advanced RVC Inference package in Python applications and scripts.

## Table of Contents

1. [Basic Setup](#basic-setup)
2. [Core Classes](#core-classes)
3. [F0 Extraction](#f0-extraction)
4. [Model Management](#model-management)
5. [Audio Processing](#audio-processing)
6. [Training API](#training-api)
7. [Configuration](#configuration)
8. [Memory Management](#memory-management)
9. [Advanced Usage](#advanced-usage)

## Basic Setup

### Installation for Development

```bash
# Clone and setup
git clone https://github.com/ArkanDash/Advanced-RVC-Inference.git
cd Advanced-RVC-Inference

# Create virtual environment
python -m venv rvc_env
source rvc_env/bin/activate  # Windows: rvc_env\Scripts\activate

# Install in development mode
pip install -e .
```

### Basic Import Structure

```python
# Core imports
from src.advanced_rvc_inference import (
    EnhancedF0Extractor,
    EnhancedAudioSeparator,
    RealtimeVoiceChanger,
    EnhancedModelManager,
    process_audio,
    Config,
    MemoryManager
)

# Tab imports (GUI modules)
from src.advanced_rvc_inference.tabs.inference.full_inference import InferenceTab
from src.advanced_rvc_inference.tabs.training.training_tab import TrainingTab

# Module availability flags
from src.advanced_rvc_inference import (
    APPLIO_AVAILABLE,
    TRAINING_AVAILABLE,
    SEPARATION_AVAILABLE
)
```

### Environment Check

```python
def check_environment():
    """Check if all required modules are available."""
    import torch
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name()}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print(f"Applio Available: {APPLIO_AVAILABLE}")
    print(f"Training Available: {TRAINING_AVAILABLE}")
    print(f"Separation Available: {SEPARATION_AVAILABLE}")

# Run environment check
check_environment()
```

## Core Classes

### Config (Singleton)

```python
from src.advanced_rvc_inference.config import Config

# Access global configuration (singleton pattern)
config = Config.get_instance()

# Device configuration
config.set_device("cuda")  # or "cpu"
device = config.get_device()

# Batch processing settings
config.set_batch_size(8)
batch_size = config.get_batch_size()

# Memory management
config.set_memory_threshold(85)  # 85% memory usage threshold
threshold = config.get_memory_threshold()

# Audio settings
config.set_sample_rate(44100)
config.set_channels(1)  # Mono

# Path configuration
config.set_weights_path("weights/")
config.set_indexes_path("indexes/")
config.set_logs_path("logs/")

# Export/import configuration
config.export_to_file("my_config.json")
config.load_from_file("my_config.json")

# Get all current settings
settings = config.get_all_settings()
```

### MemoryManager

```python
from src.advanced_rvc_inference.core.memory_manager import MemoryManager

# Get current memory usage
usage = MemoryManager.get_memory_usage()
print(f"GPU Memory: {usage.gpu_memory_percent:.1f}%")
print(f"System Memory: {usage.system_memory_percent:.1f}%")

# Force memory cleanup
MemoryManager.cleanup_memory()

# Use memory-optimized context manager
with MemoryManager.memory_optimized(threshold=80):
    # Memory-intensive operations here
    model = load_heavy_model()
    result = process_audio("input.wav", model)

# Monitor memory continuously
MemoryManager.start_monitoring(interval=5.0)  # Check every 5 seconds
# ... your operations ...
MemoryManager.stop_monitoring()

# Manual memory info
info = MemoryManager.get_detailed_memory_info()
print(f"GPU Memory: {info.gpu_memory / 1e9:.2f} GB")
print(f"GPU Reserved: {info.gpu_reserved / 1e9:.2f} GB")
```

## F0 Extraction

### Enhanced F0 Extractor

```python
from src.advanced_rvc_inference.core.f0_extractor import EnhancedF0Extractor

# Initialize extractor
extractor = EnhancedF0Extractor()

# Basic usage
f0_curve = extractor.extract_f0(
    audio_path="input.wav",
    method="hybrid[crepe+rmvpe]",  # 40+ methods available
    sample_rate=44100
)

# Available methods
methods = extractor.get_available_methods()
print("Available F0 methods:")
for method in methods:
    print(f"  - {method}")

# Method categories
traditional_methods = extractor.get_methods_by_category("traditional")
advanced_methods = extractor.get_methods_by_category("advanced")
hybrid_methods = extractor.get_methods_by_category("hybrid")

# Advanced usage with parameters
f0_curve = extractor.extract_f0(
    audio_path="input.wav",
    method="hybrid[crepe+rmvpe]",
    parameters={
        "crepe": {"threshold": 0.3},
        "rmvpe": {"hop_length": 512}
    },
    output_f0_path="output_f0.npy",
    save_parameters=True
)

# Batch processing
audio_files = ["file1.wav", "file2.wav", "file3.wav"]
f0_results = extractor.extract_f0_batch(
    audio_files=audio_files,
    method="rmvpe",
    batch_size=4,
    output_dir="f0_outputs/"
)

# Vietnamese-RVC specific methods
vietnamese_methods = [
    "hybrid[vietnamese_crepe+vietnamese_rmvpe]",
    "vietnamese_hubert_enhanced",
    "vietnamese_spin_v2"
]

for method in vietnamese_methods:
    f0 = extractor.extract_f0("input.wav", method=method)
```

### F0 Methods Reference

#### Traditional Methods
```python
# Basic F0 extraction
f0 = extractor.extract_f0("input.wav", method="parselmouth")
f0 = extractor.extract_f0("input.wav", method="pyin")
f0 = extractor.extract_f0("input.wav", method="swipe")
f0 = extractor.extract_f0("input.wav", method="yin")
```

#### Advanced Methods
```python
# Deep learning methods
f0 = extractor.extract_f0("input.wav", method="crepe")
f0 = extractor.extract_f0("input.wav", method="crepe_tiny")
f0 = extractor.extract_f0("input.wav", method="crepe_small")
f0 = extractor.extract_f0("input.wav", method="crepe_full")
f0 = extractor.extract_f0("input.wav", method="rmvpe")
f0 = extractor.extract_f0("input.wav", method="fcpe")
```

#### Hybrid Methods
```python
# Combined methods for better accuracy
f0 = extractor.extract_f0("input.wav", method="hybrid[crepe+rmvpe]")
f0 = extractor.extract_f0("input.wav", method="hybrid[fcpe+harvest]")
f0 = extractor.extract_f0("input.wav", method="hybrid[crepe+fcpe]")
f0 = extractor.extract_f0("input.wav", method="hybrid[rmvpe+pyin]")
```

#### Vietnamese-Specific Methods
```python
# Vietnamese-RVC integration
f0 = extractor.extract_f0("input.wav", method="vietnamese_hubert")
f0 = extractor.extract_f0("input.wav", method="vietnamese_spin_v1")
f0 = extractor.extract_f0("input.wav", method="vietnamese_spin_v2")
f0 = extractor.extract_f0("input.wav", method="vietnamese_whisper_tiny")
f0 = extractor.extract_f0("input.wav", method="vietnamese_whisper_large_v3")
```

## Model Management

### Enhanced Model Manager

```python
from src.advanced_rvc_inference.models.manager import EnhancedModelManager

# Initialize model manager
manager = EnhancedModelManager()

# Load a model
model = manager.load_model("path/to/model.pth")

# Model information
model_info = manager.get_model_info(model)
print(f"Model Type: {model_info.model_type}")  # 'v1' or 'v2'
print(f"Sampling Rate: {model_info.sample_rate}")
print(f"Channels: {model_info.channels}")
print(f"Embedder: {model_info.embedder}")

# List available models
available_models = manager.list_available_models()
for model_path in available_models:
    print(f"  - {model_path}")

# Model validation
is_valid = manager.validate_model("path/to/model.pth")
if not is_valid:
    print("Model validation failed!")

# Batch model loading
models = manager.load_models_batch([
    "model1.pth",
    "model2.pth",
    "model3.onnx"
])

# Model metadata management
manager.save_model_metadata("model.pth", {
    "description": "Female Japanese voice",
    "version": "v2.1",
    "language": "japanese",
    "voice_type": "female"
})

metadata = manager.get_model_metadata("model.pth")
```

### Model Format Support

```python
# PyTorch models (.pth/.pt)
model = manager.load_model("model.pth")
model = manager.load_model("model.pt")

# ONNX models (.onnx)
model = manager.load_model("model.onnx")

# Auto-detection of format
model = manager.load_model("model")  # Automatically detects .pth/.onnx

# Format conversion
manager.convert_model("model.pth", "model.onnx", use_dynamic_axes=True)
manager.convert_model("model.onnx", "model.pth")
```

## Audio Processing

### Basic Audio Processing

```python
from src.advanced_rvc_inference import process_audio

# Simple voice conversion
result = process_audio(
    audio_path="input.wav",
    model_path="voice_model.pth",
    output_path="output.wav",
    f0_method="hybrid[crepe+rmvpe]",
    pitch_shift=0,  # -12 to +12 semitones
    filter_radius=3,
    resample_sr=0,
    rms_mix_rate=0.25,
    protect=0.33,
    pitch_change_algo="pm"
)

# With additional parameters
result = process_audio(
    audio_path="input.wav",
    model_path="voice_model.pth",
    output_path="output.wav",
    f0_method="hybrid[crepe+rmvpe]",
    parameters={
        "pitch_shift": 2,
        "filter_radius": 5,
        "rms_mix_rate": 0.3,
        "protect": 0.25
    },
    index_path="model.index",  # Optional index file
    device="cuda"  # or "cpu"
)
```

### Advanced Audio Processing

```python
from src.advanced_rvc_inference.audio.voice_changer import RealtimeVoiceChanger

# Real-time voice changer
voice_changer = RealtimeVoiceChanger(
    model_path="voice_model.pth",
    device="cuda",
    chunk_size=1024
)

# Process audio file
result = voice_changer.process_file(
    input_path="input.wav",
    output_path="output.wav",
    f0_method="hybrid[crepe+rmvpe]",
    real_time=False
)

# Real-time processing (for streaming)
import pyaudio

# Setup audio stream
voice_changer.setup_audio_stream(
    input_device_index=None,  # Default input
    output_device_index=None,  # Default output
    sample_rate=44100,
    chunk_size=1024,
    channels=1
)

# Start real-time processing
voice_changer.start_real_time_processing(
    f0_method="rmvpe",
    pitch_shift=0
)

# Process audio chunks
with voice_changer.audio_stream() as stream:
    for chunk in stream:
        processed_chunk = voice_changer.process_chunk(chunk)
        stream.write(processed_chunk)

voice_changer.stop_real_time_processing()
```

### Audio Separation

```python
from src.advanced_rvc_inference.audio.separation import EnhancedAudioSeparator

# Initialize separator
separator = EnhancedAudioSeparator()

# Separate audio into vocals and accompaniment
result = separator.separate_audio(
    audio_path="song.wav",
    output_dir="separated/",
    method="mdx23c",  # Available: mdx23c, demucs, roformer
    output_format="wav"
)

# Batch separation
audio_files = ["song1.wav", "song2.wav", "song3.wav"]
results = separator.separate_batch(
    audio_files=audio_files,
    method="mdx23c",
    batch_size=2,
    output_dir="batch_separated/"
)

# Available separation methods
methods = separator.get_available_methods()
print("Available separation methods:")
for method in methods:
    print(f"  - {method}")

# Custom separation parameters
result = separator.separate_audio(
    audio_path="song.wav",
    method="roformer",
    parameters={
        "segment_size": 10,  # seconds
        "overlap": 0.25,
        "stem_count": 4  # vocals, drums, bass, other
    }
)
```

## Training API

### Training Setup

```python
from src.advanced_rvc_inference.training.trainer import RVCTrainer

# Initialize trainer
trainer = RVCTrainer(
    device="cuda",
    config={
        "batch_size": 4,
        "learning_rate": 0.001,
        "epochs": 100,
        "sample_rate": 44100
    }
)

# Training configuration
training_config = {
    "dataset_path": "path/to/dataset/",
    "output_path": "logs/training/",
    "model_name": "my_voice_model",
    
    # Audio parameters
    "sample_rate": 44100,
    "hop_length": 512,
    "win_length": 2048,
    
    # Training parameters
    "batch_size": 4,
    "epochs": 100,
    "learning_rate": 0.001,
    "save_frequency": 10,
    
    # Model parameters
    "embedder": "hubert",  # hubert, contentvec, spin
    "f0_method": "hybrid[crepe+rmvpe]",
    "use_pitch_augmentation": True,
    
    # Validation
    "validation_split": 0.2,
    "early_stopping_patience": 20
}
```

### Dataset Preparation

```python
from src.advanced_rvc_inference.training.data.dataset import RVCDataset

# Create dataset
dataset = RVCDataset(
    dataset_path="path/to/clean_audio/",
    sample_rate=44100,
    hop_length=512,
    win_length=2048,
    max_files=1000  # Limit for memory
)

# Dataset statistics
stats = dataset.get_statistics()
print(f"Total files: {stats.total_files}")
print(f"Average duration: {stats.avg_duration:.2f}s")
print(f"Total duration: {stats.total_duration:.2f}s")
print(f"Sample rate: {stats.sample_rate}Hz")

# Data validation
validation_result = dataset.validate_dataset()
if not validation_result.is_valid:
    print("Dataset validation failed:")
    for error in validation_result.errors:
        print(f"  - {error}")
```

### Training Process

```python
# Prepare training
trainer.prepare_training(training_config)

# Start training
training_history = trainer.train()

# Training monitoring
import matplotlib.pyplot as plt

def plot_training_history(history):
    """Plot training loss and validation metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Metrics plot
    for metric in ['accuracy', 'f1_score']:
        if metric in history:
            ax2.plot(history[metric], label=f'Validation {metric}')
    ax2.set_title('Validation Metrics')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

plot_training_history(training_history)

# Save final model
final_model_path = trainer.save_final_model("my_trained_model.pth")

# Export to ONNX
onnx_path = trainer.export_to_onnx(final_model_path, "my_model.onnx")
```

### Simple Training Interface

```python
from src.advanced_rvc_inference.training.simple_trainer import SimpleTrainer

# Quick training setup
simple_trainer = SimpleTrainer()

# Train with minimal configuration
result = simple_trainer.quick_train(
    dataset_path="path/to/audio/",
    model_name="quick_model",
    epochs=50,
    device="cuda"
)

# Advanced simple training
result = simple_trainer.train(
    dataset_path="path/to/audio/",
    output_path="logs/quick_training/",
    
    # Custom parameters
    sample_rate=44100,
    batch_size=4,
    learning_rate=0.001,
    
    # Augmentation
    use_augmentation=True,
    pitch_shift_range=(-2, 2),
    speed_shift_range=(0.9, 1.1),
    
    # Validation
    validation_split=0.2,
    
    # Early stopping
    early_stopping=True,
    patience=10
)

print(f"Training completed. Model saved to: {result.model_path}")
print(f"Final loss: {result.final_loss:.4f}")
print(f"Training time: {result.training_time:.2f}s")
```

## Configuration

### Global Configuration

```python
from src.advanced_rvc_inference.config import Config

# Initialize configuration
config = Config.get_instance()

# Device settings
config.set_device("cuda")  # or "cpu"
config.set_memory_threshold(85)
config.set_batch_size(4)

# Audio settings
audio_config = {
    "sample_rate": 44100,
    "channels": 1,
    "bit_depth": 16,
    "format": "wav"
}
config.set_audio_config(audio_config)

# Model settings
model_config = {
    "weights_path": "weights/",
    "index_path": "indexes/",
    "auto_load_models": True,
    "validation_mode": "strict"
}
config.set_model_config(model_config)

# UI settings
ui_config = {
    "theme": "dark",
    "language": "en_US",
    "auto_save_settings": True,
    "show_advanced_options": False
}
config.set_ui_config(ui_config)

# F0 extraction settings
f0_config = {
    "default_method": "hybrid[crepe+rmvpe]",
    "vietnamese_method": "vietnamese_hubert",
    "batch_processing": True,
    "save_f0_curves": False
}
config.set_f0_config(f0_config)

# Export configuration
config.export_to_file("my_config.json")
config.export_to_environment()  # Set as environment variables
```

### Configuration File Format

```json
{
  "version": "3.4.0",
  "device": {
    "primary": "cuda",
    "fallback": "cpu",
    "memory_threshold": 85,
    "batch_size": 4
  },
  "audio": {
    "sample_rate": 44100,
    "channels": 1,
    "bit_depth": 16,
    "format": "wav",
    "normalize": true,
    "noise_reduction": false
  },
  "models": {
    "weights_path": "weights/",
    "index_path": "indexes/",
    "auto_load": true,
    "validation": "strict",
    "cache_size": 100
  },
  "f0_extraction": {
    "default_method": "hybrid[crepe+rmvpe]",
    "vietnamese_method": "vietnamese_hubert",
    "batch_processing": true,
    "save_intermediate": false
  },
  "training": {
    "default_epochs": 100,
    "default_batch_size": 4,
    "learning_rate": 0.001,
    "save_frequency": 10,
    "early_stopping": true
  },
  "ui": {
    "theme": "dark",
    "language": "en_US",
    "auto_save": true,
    "advanced_mode": false
  },
  "logging": {
    "level": "INFO",
    "file": "logs/app.log",
    "max_size": "10MB",
    "backup_count": 5
  }
}
```

## Memory Management

### Automatic Memory Management

```python
from src.advanced_rvc_inference.core.memory_manager import MemoryManager

# Memory monitoring
def monitor_memory_during_processing():
    """Monitor memory usage during audio processing."""
    
    def check_memory():
        usage = MemoryManager.get_memory_usage()
        print(f"GPU Memory: {usage.gpu_memory_percent:.1f}%")
        print(f"System Memory: {usage.system_memory_percent:.1f}%")
    
    # Start monitoring
    MemoryManager.start_monitoring(interval=2.0)
    
    try:
        # Your processing code here
        for i in range(10):
            print(f"Processing batch {i+1}/10")
            # Simulate heavy processing
            check_memory()
            time.sleep(1)
            
    finally:
        # Always stop monitoring
        MemoryManager.stop_monitoring()

# Memory-optimized processing
def process_large_dataset(audio_files):
    """Process large dataset with automatic memory management."""
    
    results = []
    batch_size = 4
    
    for i in range(0, len(audio_files), batch_size):
        batch = audio_files[i:i+batch_size]
        
        with MemoryManager.memory_optimized(threshold=80):
            for audio_file in batch:
                result = process_audio(
                    audio_path=audio_file,
                    model_path="voice_model.pth",
                    output_path=f"output_{os.path.basename(audio_file)}"
                )
                results.append(result)
            
            # Force cleanup between batches
            MemoryManager.cleanup_memory()
    
    return results
```

### Manual Memory Control

```python
# Get detailed memory information
info = MemoryManager.get_detailed_memory_info()
print(f"GPU Memory: {info.gpu_memory / 1e9:.2f} GB")
print(f"GPU Reserved: {info.gpu_reserved / 1e9:.2f} GB")
print(f"GPU Cached: {info.gpu_cached / 1e9:.2f} GB")
print(f"System Memory: {info.system_memory / 1e9:.2f} GB")

# Check if memory is sufficient
is_sufficient = MemoryManager.check_memory_sufficient(
    required_gpu_gb=4.0,
    required_system_gb=8.0
)

if not is_sufficient:
    print("Insufficient memory for operation")
    # Reduce batch size or switch to CPU
    config.set_batch_size(2)
    config.set_device("cpu")

# Memory allocation tracking
MemoryManager.start_allocation_tracking()

# Your operations here
model = load_large_model()
process_audio("input.wav", model)

# Get allocation summary
allocations = MemoryManager.get_allocation_summary()
print("Memory allocations:")
for category, size in allocations.items():
    print(f"  {category}: {size / 1e9:.2f} GB")
```

## Advanced Usage

### Batch Processing Pipeline

```python
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

class BatchProcessor:
    def __init__(self, model_path, max_workers=4):
        self.model_path = model_path
        self.max_workers = max_workers
        self.config = Config.get_instance()
    
    def process_directory(self, input_dir, output_dir):
        """Process all audio files in a directory."""
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get all audio files
        audio_files = list(input_path.glob("*.wav")) + \
                     list(input_path.glob("*.mp3")) + \
                     list(input_path.glob("*.flac"))
        
        print(f"Found {len(audio_files)} audio files")
        
        # Process in batches
        results = []
        batch_size = min(self.max_workers, len(audio_files))
        
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.process_file, str(audio_file), str(output_path)): audio_file
                for audio_file in audio_files
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_file):
                audio_file = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"Processed: {audio_file.name}")
                except Exception as exc:
                    print(f"Error processing {audio_file.name}: {exc}")
        
        return results
    
    def process_file(self, input_file, output_dir):
        """Process a single audio file."""
        
        input_path = Path(input_file)
        output_file = Path(output_dir) / f"converted_{input_path.stem}.wav"
        
        with MemoryManager.memory_optimized():
            result = process_audio(
                audio_path=str(input_file),
                model_path=self.model_path,
                output_path=str(output_file),
                f0_method="hybrid[crepe+rmvpe]",
                device=self.config.get_device()
            )
        
        return str(output_file)

# Usage
processor = BatchProcessor("voice_model.pth", max_workers=2)
results = processor.process_directory("input_audio/", "output_audio/")
```

### Custom F0 Extractor

```python
from src.advanced_rvc_inference.core.f0_extractor import EnhancedF0Extractor
import numpy as np

class CustomF0Extractor(EnhancedF0Extractor):
    def __init__(self):
        super().__init__()
        self.custom_methods = {}
    
    def add_custom_method(self, name, method_function):
        """Add custom F0 extraction method."""
        self.custom_methods[name] = method_function
        self.registered_methods.append(name)
    
    def extract_with_custom_method(self, audio_path, method_name, **kwargs):
        """Extract F0 using custom method."""
        
        if method_name not in self.custom_methods:
            raise ValueError(f"Custom method '{method_name}' not found")
        
        # Load audio
        audio, sr = self.load_audio(audio_path)
        
        # Apply custom method
        f0_curve = self.custom_methods[method_name](audio, sr, **kwargs)
        
        # Post-process if needed
        f0_curve = self.postprocess_f0(f0_curve)
        
        return f0_curve
    
    def hybrid_custom_method(self, audio_path, method1, method2, weight1=0.5):
        """Combine two methods with custom weighting."""
        
        f0_1 = self.extract_f0(audio_path, method1)
        f0_2 = self.extract_f0(audio_path, method2)
        
        # Weighted combination
        f0_combined = weight1 * f0_1 + (1 - weight1) * f0_2
        
        # Smooth the result
        f0_smoothed = self.smooth_f0(f0_combined)
        
        return f0_smoothed

# Custom method example
def my_custom_method(audio, sr, window_length=2048, hop_length=512):
    """Example custom F0 extraction method."""
    
    # Your custom F0 extraction algorithm here
    # This is just a placeholder
    
    # For demonstration, let's use a simple autocorrelation
    frame_length = min(window_length, len(audio) // 4)
    hop_length = min(hop_length, frame_length // 4)
    
    f0_values = []
    
    for i in range(0, len(audio) - frame_length, hop_length):
        frame = audio[i:i + frame_length]
        
        # Simple autocorrelation for F0 estimation
        autocorr = np.correlate(frame, frame, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # Find peak (simplified)
        if len(autocorr) > 1:
            peak_idx = np.argmax(autocorr[1:]) + 1
            f0 = sr / peak_idx if peak_idx > 0 else 0
        else:
            f0 = 0
        
        f0_values.append(f0)
    
    return np.array(f0_values)

# Usage
custom_extractor = CustomF0Extractor()
custom_extractor.add_custom_method("my_method", my_custom_method)

# Use custom method
f0 = custom_extractor.extract_with_custom_method(
    "input.wav", 
    "my_method", 
    window_length=4096, 
    hop_length=512
)

# Hybrid custom method
f0_hybrid = custom_extractor.hybrid_custom_method(
    "input.wav",
    "my_method",
    "crepe",
    weight1=0.3
)
```

### Error Handling and Recovery

```python
import logging
from src.advanced_rvc_inference.core.memory_manager import MemoryManager

def robust_audio_processing(audio_path, model_path, max_retries=3):
    """Process audio with comprehensive error handling and recovery."""
    
    logger = logging.getLogger(__name__)
    
    for attempt in range(max_retries):
        try:
            # Check memory before processing
            usage = MemoryManager.get_memory_usage()
            if usage.gpu_memory_percent > 90:
                logger.warning("High GPU memory usage, cleaning up...")
                MemoryManager.cleanup_memory()
            
            # Attempt processing
            result = process_audio(
                audio_path=audio_path,
                model_path=model_path,
                output_path="output_temp.wav"
            )
            
            logger.info(f"Successfully processed {audio_path} on attempt {attempt + 1}")
            return result
            
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"CUDA OOM on attempt {attempt + 1}, switching to CPU")
            config = Config.get_instance()
            config.set_device("cpu")
            
            # Clean up GPU memory
            torch.cuda.empty_cache()
            MemoryManager.cleanup_memory()
            
            # Retry with CPU
            continue
            
        except Exception as e:
            logger.error(f"Error processing {audio_path} on attempt {attempt + 1}: {e}")
            
            if attempt == max_retries - 1:
                logger.error(f"Failed to process {audio_path} after {max_retries} attempts")
                raise
            
            # Wait before retry
            time.sleep(2 ** attempt)  # Exponential backoff
    
    return None

# Recovery strategy for batch processing
def resilient_batch_processing(audio_files, model_path):
    """Process batch with automatic recovery from failures."""
    
    successful = []
    failed = []
    
    for i, audio_file in enumerate(audio_files):
        try:
            logger.info(f"Processing {i+1}/{len(audio_files)}: {audio_file}")
            
            result = robust_audio_processing(audio_file, model_path)
            
            if result:
                successful.append((audio_file, result))
            else:
                failed.append(audio_file)
                
        except KeyboardInterrupt:
            logger.info("Batch processing interrupted by user")
            break
        except Exception as e:
            logger.error(f"Unexpected error processing {audio_file}: {e}")
            failed.append(audio_file)
    
    logger.info(f"Batch processing completed: {len(successful)} successful, {len(failed)} failed")
    return successful, failed
```

This API documentation provides comprehensive examples for all major components of the Advanced RVC Inference package. For more specific use cases and advanced features, refer to the individual module documentation and source code comments.
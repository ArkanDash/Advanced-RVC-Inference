# Advanced RVC Inference - Troubleshooting Guide

**Version:** 4.0.0  
**Authors:** ArkanDash & BF667

This comprehensive troubleshooting guide helps resolve common issues with Advanced RVC Inference.

## Quick Diagnosis

### System Check

```python
from advanced_rvc_inference import check_fp16_support, gpu_optimizer
import torch
import psutil

def quick_system_check():
    """Perform quick system diagnosis."""
    print("=== System Check ===")
    
    # Python version
    print(f"Python: {torch.__version__.split('.')[:2]}")
    
    # GPU status
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"FP16 support: {check_fp16_support()}")
    
    # Memory
    memory = psutil.virtual_memory()
    print(f"RAM: {memory.total / (1024**3):.1f}GB")
    print(f"Available RAM: {memory.available / (1024**3):.1f}GB")
    
    # Disk space
    disk = psutil.disk_usage('.')
    print(f"Free disk: {disk.free / (1024**3):.1f}GB")
    
    # Import test
    try:
        from advanced_rvc_inference import full_inference_program
        print("✅ Core imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")

if __name__ == "__main__":
    quick_system_check()
```

## Installation Issues

### 1. PyTorch Installation Problems

**Problem**: PyTorch installation fails or CUDA not detected

**Symptoms**:
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**Solutions**:

```bash
# Solution 1: Install correct PyTorch version for your CUDA
pip uninstall torch torchaudio
pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu118

# Solution 2: Verify CUDA installation
nvidia-smi
nvcc --version

# Solution 3: Test PyTorch with CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Solution 4: Install CPU version if no GPU
pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cpu
```

### 2. Audio Library Issues

**Problem**: Audio processing libraries missing or incompatible

**Symptoms**:
```
ImportError: No module named 'soundfile'
OSError: Error opening input file
```

**Solutions**:

```bash
# Install system audio libraries (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install ffmpeg libsndfile1-dev portaudio19-dev

# Install audio Python packages
pip install soundfile==0.13.0
pip install librosa>=0.10.2
pip install pydub>=0.25.1

# Alternative: Use conda
conda install -c conda-forge ffmpeg libsndfile portaudio
```

### 3. Missing Dependencies

**Problem**: Core dependencies not installed or version conflicts

**Symptoms**:
```
ModuleNotFoundError: No module named 'gradio'
pip install failed with dependency conflicts
```

**Solutions**:

```bash
# Solution 1: Fresh virtual environment
python -m venv fresh_env
source fresh_env/bin/activate  # Windows: fresh_env\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# Solution 2: Install specific versions
pip install gradio==5.23.3
pip install transformers==4.49.0

# Solution 3: Use requirements.txt
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --no-cache-dir
```

### 4. Permission Issues

**Problem**: File access or directory creation permissions

**Symptoms**:
```
PermissionError: [Errno 13] Permission denied
OSError: [Errno 30] Read-only file system
```

**Solutions**:

```bash
# Solution 1: Check directory permissions
ls -la logs/ temp/ assets/

# Solution 2: Create directories with proper permissions
mkdir -p logs temp assets/models
chmod 755 logs temp assets models

# Solution 3: Run with appropriate user permissions
sudo chown -R $USER:$USER .
```

## GPU Issues

### 1. CUDA Out of Memory

**Problem**: GPU runs out of memory during processing

**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions**:

```python
# Solution 1: Reduce batch size and enable optimizations
from advanced_rvc_inference import gpu_settings, KRVCFeatureExtractor

# Configure for memory efficiency
settings = gpu_settings(
    batch_size=1,
    precision="float16",
    memory_fraction=0.6
)

# Use memory-optimized KRVC
extractor = KRVCFeatureExtractor()
extractor.enable_memory_optimization()
extractor.enable_mixed_precision()

# Solution 2: Clear GPU memory manually
import torch
torch.cuda.empty_cache()

# Solution 3: Monitor memory usage
def monitor_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
```

### 2. GPU Not Detected

**Problem**: System has GPU but PyTorch doesn't detect it

**Symptoms**:
```
torch.cuda.is_available() returns False
No CUDA-capable device is detected
```

**Solutions**:

```bash
# Check GPU drivers
nvidia-smi  # For NVIDIA GPUs
# Check AMD GPU info for ROCm

# Solution 1: Install/reinstall GPU drivers
# NVIDIA: Download from nvidia.com/drivers
# AMD: Install ROCm from amd.com/rocm

# Solution 2: Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=0

# Solution 3: Check PyTorch installation
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"

# Solution 4: For laptops with integrated graphics
# Disable integrated graphics in BIOS or set CUDA_VISIBLE_DEVICES
```

### 3. Mixed Precision Issues

**Problem**: FP16 precision causes numerical instability

**Symptoms**```
torch.cuda.amp.autocast() is not enabled
RuntimeError: "clamp_min_cpu" not implemented for 'Half'
```

**Solutions**:

```python
# Solution 1: Check hardware FP16 support
from advanced_rpc_inference import check_fp16_support

if check_fp16_support():
    print("FP16 supported")
else:
    print("Using FP32 fallback")

# Solution 2: Configure mixed precision properly
import torch
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    # Your processing code here
    result = process_audio(audio_data)

# Solution 3: Fallback to FP32 if FP16 fails
try:
    with autocast():
        result = process_audio(audio_data)
except RuntimeError as e:
    if "Half" in str(e):
        print("FP16 not supported, using FP32")
        with torch.cuda.amp.autast(enabled=False):
            result = process_audio(audio_data)
```

## Audio Processing Issues

### 1. Audio Format Problems

**Problem**: Unsupported or corrupted audio formats

**Symptoms**:
```
OSError: Error opening input file 'input.wav'
ValueError: Audio file could not be decoded
```

**Solutions**:

```python
# Solution 1: Check audio format support
from advanced_rvc_inference.lib.utils import check_audio_format

# Check if format is supported
is_supported = check_audio_format("input.wav")
print(f"Format supported: {is_supported}")

# Solution 2: Convert audio format
import soundfile as sf
import librosa

# Convert to supported format
def convert_audio(input_path, output_path):
    try:
        audio, sr = librosa.load(input_path, sr=None)
        sf.write(output_path, audio, sr)
        print(f"Converted {input_path} to {output_path}")
    except Exception as e:
        print(f"Conversion failed: {e}")

# Solution 3: Validate audio file
def validate_audio_file(file_path):
    try:
        import librosa
        audio, sr = librosa.load(file_path, sr=None, duration=1.0)
        print(f"Valid audio: {len(audio)/sr:.2f}s at {sr}Hz")
        return True
    except Exception as e:
        print(f"Invalid audio: {e}")
        return False
```

### 2. Audio Quality Issues

**Problem**: Poor conversion quality or artifacts

**Symptoms**:
```
Distorted output audio
Robotic voice effect
Inconsistent volume
```

**Solutions**:

```python
# Solution 1: Optimize conversion parameters
result = full_inference_program(
    model_path="model.pth",
    input_audio_path="input.wav",
    output_path="output.wav",
    pitch_extract="rmvpe",           # Good F0 extraction
    embedder_model="contentvec",      # Reliable embedder
    protect=0.33,                     # Protect vocal quality
    rms_mix_rate=0.25,                # Volume envelope mixing
    enhancer=True,                    # Enable enhancement
    use_gpu=True                      # Use GPU for better quality
)

# Solution 2: Use high-quality settings
high_quality_settings = {
    'pitch_extract': 'hybrid[rmvpe+crepe+fcpe]',
    'embedder_model': 'hubert-base',
    'filter_radius': 5,
    'resample_sr': 48000,
    'db_level': -20.0,
    'enhancer': True,
    'autotune': False
}

# Solution 3: Pre/post-process audio
def preprocess_audio(input_path):
    import librosa
    import soundfile as sf
    
    # Load with high quality settings
    audio, sr = librosa.load(input_path, sr=44100, mono=False)
    
    # Preprocess: normalize, trim silence
    audio = librosa.util.normalize(audio, axis=-1)
    audio, _ = librosa.effects.trim(audio, top_db=20)
    
    return audio, sr
```

### 3. Real-time Processing Issues

**Problem**: High latency or dropped frames in real-time mode

**Symptoms**:
```
Audio buffer underrun
High latency (>100ms)
Dropped audio packets
```

**Solutions**:

```python
# Solution 1: Optimize real-time settings
from advanced_rvc_inference import KRVCRealTimeProcessor

processor = KRVCRealTimeProcessor(
    model_path="model.pth",
    buffer_size=512,           # Smaller buffer
    target_latency=0.03,       # 30ms target
    max_latency=0.05,          # 50ms max
    threading=True,            # Enable threading
    num_threads=2              # Worker threads
)

# Solution 2: Monitor real-time performance
def monitor_realtime_performance():
    import time
    start_time = time.time()
    
    # Process audio chunk
    result = processor.process_chunk(audio_chunk)
    
    processing_time = time.time() - start_time
    print(f"Processing time: {processing_time*1000:.1f}ms")
    
    if processing_time > 0.05:  # 50ms threshold
        print("⚠️ Performance issue detected")
        # Reduce quality settings or buffer size
```

## Model Issues

### 1. Model Loading Failures

**Problem**: Voice models fail to load or are incompatible

**Symptoms**:
```
FileNotFoundError: Model file not found
RuntimeError: Model architecture mismatch
```

**Solutions**:

```python
# Solution 1: Validate model files
from advanced_rvc_inference.lib.utils import validate_model

def check_model(model_path):
    try:
        is_valid = validate_model(model_path)
        if is_valid:
            print(f"✅ Model {model_path} is valid")
        else:
            print(f"❌ Model {model_path} is invalid")
        return is_valid
    except Exception as e:
        print(f"Model validation error: {e}")
        return False

# Solution 2: Handle model compatibility
try:
    model = import_voice_converter(
        model_path="model.pth",
        device="auto",
        use_fp16=True,
        krvc_type="v2"
    )
    print("Model loaded successfully")
except RuntimeError as e:
    if "architecture" in str(e).lower():
        print("Model architecture mismatch, trying different settings")
        model = import_voice_converter(
            model_path="model.pth",
            device="cpu",  # Fallback to CPU
            use_fp16=False,
            krvc_type="v1"
        )

# Solution 3: Auto-discover models
from advanced_rvc_inference import PathManager

pm = PathManager()
models = pm.find_models()
print(f"Found {len(models)} models:")
for name, path in models.items():
    print(f"  - {name}: {path}")
```

### 2. Index File Issues

**Problem**: Missing or corrupted feature index files

**Symptoms**:
```
FileNotFoundError: Index file not found
RuntimeError: Invalid index format
```

**Solutions**:

```python
# Solution 1: Generate missing index files
def generate_index(model_path, audio_sample_path, index_output_path):
    try:
        from advanced_rvc_inference.lib.index_generation import generate_feature_index
        
        generate_feature_index(
            model_path=model_path,
            reference_audio=audio_sample_path,
            output_path=index_output_path,
            num_features=256
        )
        print(f"Index generated: {index_output_path}")
    except Exception as e:
        print(f"Index generation failed: {e}")

# Solution 2: Use fallback index method
try:
    # Try to load with index
    result = full_inference_program(
        model_path="model.pth",
        index_path="model.index",
        input_audio_path="input.wav",
        output_path="output.wav"
    )
except FileNotFoundError:
    print("Index not found, running without index")
    # Run without index file (lower quality but functional)
    result = full_inference_program(
        model_path="model.pth",
        input_audio_path="input.wav",
        output_path="output.wav"
    )
```

### 3. Model Size Issues

**Problem**: Models too large for available memory

**Symptoms**:
```
RuntimeError: CUDA out of memory
MemoryError: Unable to allocate array
```

**Solutions**:

```python
# Solution 1: Use model optimization
from advanced_rvc_inference.lib.model_optimization import optimize_model_for_inference

optimized_model = optimize_model_for_inference(
    model_path="large_model.pth",
    optimization_level="memory_efficient",
    precision="float16"
)

# Solution 2: Load model in parts
def load_model_parts(model_path):
    import torch
    
    # Load model state dict
    state_dict = torch.load(model_path, map_location="cpu")
    
    # Split into parts if too large
    model_parts = {}
    for name, param in state_dict.items():
        if param.numel() > 1e7:  # Large parameter
            print(f"Large parameter: {name} ({param.numel()} elements)")
            # Process in chunks
        else:
            model_parts[name] = param
    
    return model_parts

# Solution 3: Use smaller models
def find_suitable_models():
    pm = PathManager()
    models = pm.find_models()
    
    suitable_models = {}
    for name, path in models.items():
        file_size = path.stat().st_size / (1024**3)  # Size in GB
        if file_size < 1.0:  # Smaller than 1GB
            suitable_models[name] = path
    
    return suitable_models
```

## Performance Issues

### 1. Slow Processing Speed

**Problem**: Processing takes too long

**Symptoms**:
```
Long processing time for short audio clips
High CPU/GPU utilization without results
```

**Solutions**:

```python
# Solution 1: Enable all optimizations
from advanced_rvc_inference import (
    krvc_speed_optimize, 
    krvc_inference_mode,
    gpu_optimizer
)

# Apply optimizations
krvc_speed_optimize()
krvc_inference_mode()
optimizer = gpu_optimizer()
optimal_settings = optimizer.get_optimal_settings()

# Configure for speed
speed_config = {
    'batch_size': optimal_settings.get('batch_size', 2),
    'precision': 'float16',
    'memory_fraction': 0.8,
    'torch_compile': True
}

# Solution 2: Profile performance
import time
from advanced_rvc_inference.lib.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.start()

result = full_inference_program(
    model_path="model.pth",
    input_audio_path="input.wav",
    output_path="output.wav",
    use_gpu=True,
    **speed_config
)

stats = monitor.stop()
print(f"Processing stats: {stats}")

# Solution 3: Use batch processing
def process_multiple_audio_files(file_list, batch_size=4):
    results = []
    for i in range(0, len(file_list), batch_size):
        batch = file_list[i:i+batch_size]
        batch_results = []
        
        for audio_file in batch:
            result = full_inference_program(
                model_path="model.pth",
                input_audio_path=audio_file,
                output_path=f"output_{audio_file}",
                use_gpu=True
            )
            batch_results.append(result)
        
        results.extend(batch_results)
    
    return results
```

### 2. High Memory Usage

**Problem**: System runs out of memory during processing

**Symptoms**:
```
System becomes unresponsive
High memory usage in task manager
Out of memory errors
```

**Solutions**:

```python
# Solution 1: Memory monitoring and cleanup
import gc
import psutil
import torch

def monitor_and_cleanup():
    process = psutil.Process()
    
    # Check memory usage
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / (1024**3):.1f}GB")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU memory: {gpu_memory:.1f}GB")
    
    # Force garbage collection
    gc.collect()
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Solution 2: Process in smaller chunks
def process_audio_in_chunks(input_path, chunk_duration=10.0):
    import librosa
    import soundfile as sf
    
    # Load audio
    audio, sr = librosa.load(input_path, sr=None)
    
    # Calculate chunk size
    chunk_samples = int(chunk_duration * sr)
    
    results = []
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i+chunk_samples]
        
        # Process chunk
        processed_chunk = full_inference_program(
            model_path="model.pth",
            input_audio_path=chunk,
            output_path=f"chunk_{i//chunk_samples}.wav",
            save_intermediate=False
        )
        
        # Clean up after each chunk
        monitor_and_cleanup()
        
        results.append(processed_chunk)
    
    return results

# Solution 3: Configure memory-efficient settings
memory_efficient_config = {
    'batch_size': 1,
    'precision': 'float16',
    'memory_fraction': 0.6,
    'enable_memory_optimization': True,
    'clear_cache_between_batches': True
}
```

## Configuration Issues

### 1. Configuration File Problems

**Problem**: Configuration files are corrupted or missing

**Symptoms**:
```
FileNotFoundError: config.json not found
JSONDecodeError: Expecting ',' delimiter
```

**Solutions**:

```python
# Solution 1: Reset to default configuration
import json
from pathlib import Path

def reset_configuration():
    default_config = {
        "gpu": {
            "enabled": True,
            "device_id": 0,
            "precision": "float16",
            "memory_fraction": 0.8
        },
        "krvc": {
            "enabled": True,
            "type": "v2",
            "optimization_level": "balanced"
        },
        "performance": {
            "batch_size": 1,
            "torch_compile": True
        }
    }
    
    config_path = Path("config.json")
    with open(config_path, 'w') as f:
        json.dump(default_config, f, indent=4)
    
    print("Configuration reset to defaults")

# Solution 2: Validate configuration
def validate_configuration(config_path):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Basic validation
        required_sections = ['gpu', 'krvc', 'performance']
        for section in required_sections:
            if section not in config:
                print(f"Missing section: {section}")
                return False
        
        print("Configuration is valid")
        return True
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False
```

### 2. Path Issues

**Problem**: File paths are incorrect or inaccessible

**Symptoms**:
```
FileNotFoundError: No such file or directory
PermissionError: Access denied
```

**Solutions**:

```python
# Solution 1: Use path manager for path resolution
from advanced_rvc_inference import PathManager

pm = PathManager()

# Validate and create paths
try:
    models_dir = pm.get_path('models_dir', create_if_missing=True)
    logs_dir = pm.get_path('logs_dir', create_if_missing=True)
    print("Paths validated and created")
except Exception as e:
    print(f"Path validation failed: {e}")

# Solution 2: Check path accessibility
def check_path_accessibility(file_path):
    path = Path(file_path)
    
    checks = {
        'exists': path.exists(),
        'is_file': path.is_file() if path.exists() else False,
        'is_readable': os.access(path, os.R_OK) if path.exists() else False,
        'is_writable': os.access(path.parent, os.W_OK) if path.parent.exists() else False
    }
    
    for check, result in checks.items():
        status = "✅" if result else "❌"
        print(f"{status} {check}: {result}")
    
    return all(checks.values())

# Solution 3: Create missing directories
def ensure_directories():
    directories = [
        'logs',
        'temp',
        'assets/models',
        'assets/weights',
        'assets/audios'
    ]
    
    for directory in directories:
        path = Path(directory)
        try:
            path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {directory}")
        except Exception as e:
            print(f"❌ Failed to create {directory}: {e}")
```

## Error Recovery

### Automatic Recovery

```python
def auto_recovery():
    """Attempt automatic error recovery."""
    try:
        # Test basic functionality
        from advanced_rvc_inference import full_inference_program
        print("✅ Core imports working")
        
        # Test GPU if available
        import torch
        if torch.cuda.is_available():
            print("✅ GPU detected")
        else:
            print("⚠️ No GPU detected, using CPU fallback")
        
        # Test configuration
        from advanced_rvc_inference.core import get_config
        config = get_config()
        print("✅ Configuration loaded")
        
        # Test path manager
        from advanced_rvc_inference import PathManager
        pm = PathManager()
        validation = pm.validate_project_structure()
        missing = [k for k, v in validation.items() if not v]
        if missing:
            print(f"⚠️ Missing directories: {missing}")
        else:
            print("✅ All directories present")
        
        print("🎉 System ready for processing")
        return True
        
    except Exception as e:
        print(f"❌ Recovery failed: {e}")
        return False

if __name__ == "__main__":
    auto_recovery()
```

### Manual Recovery Steps

1. **Reset virtual environment**
   ```bash
   rm -rf venv/
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Clear caches and temporary files**
   ```bash
   rm -rf .cache/ temp/
   python -c "from advanced_rvc_inference import PathManager; pm = PathManager(); pm.cleanup_temp_files()"
   ```

3. **Reinstall GPU drivers**
   - NVIDIA: Download and install latest drivers
   - AMD: Install/update ROCm stack

4. **Reset configuration**
   - Delete `config.json`
   - Run application to generate default config

5. **Check system resources**
   - Ensure sufficient RAM and disk space
   - Verify GPU has enough VRAM

## Getting Help

### Before Asking for Help

1. **Run system check**
   ```python
   from advanced_rvc_inference import quick_system_check
   quick_system_check()
   ```

2. **Check log files**
   - `logs/application.log`
   - `logs/error.log`

3. **Gather system information**
   ```bash
   python --version
   nvidia-smi  # If NVIDIA GPU
   pip list | grep -E "(torch|gradio|transformers)"
   ```

4. **Reproduce the issue**
   - Note exact steps to reproduce
   - Include input files and configuration
   - Capture full error messages

### Information to Include

When reporting issues, include:

- **System Information**: OS, Python version, hardware specs
- **Error Messages**: Full traceback and error logs
- **Configuration**: Relevant config settings
- **Steps to Reproduce**: Exact sequence of actions
- **Expected vs Actual**: What should happen vs what happens
- **Environment**: Virtual environment setup, dependencies

### Community Support

- **GitHub Issues**: [Create detailed bug report](https://github.com/ArkanDash/Advanced-RVC-Inference/issues)
- **Discord Community**: [Ask questions in real-time](https://discord.gg/hvmsukmBHE)
- **Documentation**: [Check complete docs](https://github.com/ArkanDash/Advanced-RVC-Inference/wiki)

---

*This troubleshooting guide is maintained by ArkanDash & BF667 for Advanced RVC Inference V4.0.0*
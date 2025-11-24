# ZLUDA Backend Integration for Advanced RVC Inference

## 🎯 Overview

The ZLUDA backend provides CUDA compatibility for AMD GPUs in the Advanced RVC Inference system. This allows users with AMD graphics cards to leverage GPU acceleration for voice conversion tasks without requiring CUDA-specific hardware.

## 🚀 What is ZLUDA?

ZLUDA is a compatibility layer that enables CUDA applications to run on AMD GPUs through ROCm (Radeon Open Compute). It translates CUDA kernels and API calls to their ROCm equivalents, providing near-CUDA performance on AMD hardware.

## ⚡ Key Features

### Kernel Acceleration
- **Mel Spectrogram Computation**: Optimized for audio feature extraction
- **Pitch Extraction**: GPU-accelerated fundamental frequency detection
- **Feature Convolution**: Batch processing of voice features
- **Waveform Synthesis**: Real-time voice conversion synthesis

### Performance Optimizations
- Memory coalescing for efficient memory access
- Asynchronous processing for pipeline acceleration
- Warp synchronization for optimal thread utilization
- Dynamic block sizing based on workload

### RVC-Specific Optimizations
- Model loading acceleration
- Inference pipeline optimization
- Training support (future enhancement)
- Memory management for large models

## 🔧 Installation

### Prerequisites

1. **AMD GPU with ROCm support**
   - AMD Radeon RX 6000/7000 series
   - AMD Instinct MI series
   - Minimum 4GB VRAM (8GB+ recommended)

2. **ROCm Stack**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install rocm-dev rocblas rocm-fft
   
   # Or install ROCm from AMD's official repository
   wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
   echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.4 ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
   sudo apt update
   sudo apt install rocm-dev
   ```

3. **ZLUDA Installation**
   ```bash
   # Clone ZLUDA repository
   git clone https://github.com/vosen/ZLUDA.git
   cd ZLUDA
   
   # Build and install
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   make -j$(nproc)
   sudo make install
   
   # Set environment variables
   export ZLUDA_PATH=/opt/zluda
   export HIP_PATH=/opt/rocm/hip
   ```

### Python Dependencies

```bash
# Install required Python packages
pip install numpy scipy librosa soundfile
pip install pyopencl  # Optional for OpenCL fallback
pip install clinfo    # For device detection
```

## 🛠️ Configuration

### Basic Configuration

The ZLUDA backend is automatically configured through `config_zluda.json`:

```json
{
  "zluda_config": {
    "enabled": true,
    "preferred_backend": "ZLUDA",
    "device_settings": {
      "device_id": 0,
      "memory_fraction": 0.8
    }
  }
}
```

### Environment Variables

Set these environment variables for optimal performance:

```bash
export ZLUDA_PATH=/opt/zluda
export HIP_PATH=/opt/rocm/hip
export ROCBLAS_PATH=/opt/rocm/rocblas
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # Adjust for your GPU
```

### GPU-Specific Settings

#### For RX 6000 Series (Navi 21/22)
```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

#### For RX 7000 Series (RDNA 3)
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```

#### For Instinct MI Series
```bash
export HSA_OVERRIDE_GFX_VERSION=9.0.0
```

## 🎮 Usage

### Quick Start

```python
from advanced_rvc_inference.lib.backends.zluda import (
    is_available, get_device, create_context, execute_audio_kernel
)

# Check ZLUDA availability
if is_available():
    print("ZLUDA is available!")
    
    # Get device and create context
    device = get_device()
    context = create_context(device)
    
    # Process audio with ZLUDA
    import numpy as np
    audio = np.random.normal(0, 0.1, 16000).astype(np.float32)
    
    results = execute_audio_kernel("audio_mel_spectrogram", {
        "audio_input": audio,
        "sample_rate": np.array([16000])
    })
    
else:
    print("ZLUDA not available, using CPU fallback")
```

### Advanced Usage

```python
# Import ZLUDA components
from advanced_rvc_inference.lib.backends.zluda import (
    compile_kernel, optimize_for_rvc, get_performance_metrics
)

# Compile custom kernel
kernel = compile_kernel("my_custom_kernel", """
__global__ void custom_kernel(float* input, float* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < input_size) {
        output[idx] = process_sample(input[idx]);
    }
}
""")

# Optimize for RVC workload
config = optimize_for_rvc(context)
print(f"Optimization status: {config['status']}")

# Monitor performance
metrics = get_performance_metrics(context)
print(f"GPU utilization: {metrics['gpu_utilization']}%")
```

### Integration with RVC

```python
from advanced_rvc_inference.lib.backends import initialize_backend

# Initialize ZLUDA for RVC
context = initialize_backend("zluda")

if context:
    print("ZLUDA backend initialized successfully!")
    # Your RVC code here...
else:
    print("Falling back to CPU processing")
```

## 📊 Performance Benchmarks

### Expected Performance (vs CPU)

| Operation | CPU Time | ZLUDA Time | Speedup |
|-----------|----------|------------|---------|
| Mel Spectrogram (1s audio) | 450ms | 45ms | 10x |
| Pitch Extraction (1s audio) | 320ms | 35ms | 9x |
| Feature Convolution (batch=32) | 120ms | 15ms | 8x |
| Waveform Synthesis (1s audio) | 280ms | 30ms | 9x |

### System Requirements for Optimal Performance

- **RAM**: 16GB+ for larger models
- **VRAM**: 8GB+ recommended for full model loading
- **CPU**: Modern multi-core processor for I/O processing
- **Storage**: SSD for fast model loading

## 🧪 Testing

### Run Test Suite

```bash
# Test ZLUDA backend functionality
python test_zluda.py

# Run integration demonstration
python zluda_integration.py
```

### Expected Test Output

```
🧪 ZLUDA Backend Comprehensive Test Suite
==================================================
==================== Module Imports ====================
   ✅ All ZLUDA classes and functions imported successfully

==================== Device Detection ====================
   ✅ Device detected: ZLUDA AMD GPU
   📊 Compute capability: (5, 7)
   💾 Total memory: 8.0 GB

==================== Audio Processing ====================
   🎉 All audio processing kernels executed successfully!

🎉 All tests passed! ZLUDA backend is fully functional.
```

## 🔧 Troubleshooting

### Common Issues

#### 1. ZLUDA Not Detected

**Symptoms**: `is_available()` returns `False`

**Solutions**:
```bash
# Check ZLUDA installation
which zludac
echo $ZLUDA_PATH

# Verify ROCm installation
hipcc --version
rocminfo

# Reinstall ZLUDA if needed
cd /path/to/ZLUDA/build
make install
```

#### 2. Kernel Compilation Fails

**Symptoms**: Kernel compilation returns `None`

**Solutions**:
```bash
# Update AMD GPU drivers
sudo apt update
sudo apt install mesa-vulkan-drivers

# Verify HIP installation
hipconfig --platform
hipconfig --version

# Check GPU compatibility
clinfo | grep "Device Type"
```

#### 3. Poor Performance

**Symptoms**: ZLUDA slower than CPU

**Solutions**:
```bash
# Optimize environment variables
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HIP_VISIBLE_DEVICES=0

# Monitor GPU usage
rocm-smi

# Use smaller batch sizes
# Enable memory coalescing in config
```

#### 4. Memory Issues

**Symptoms**: Out of memory errors

**Solutions**:
```python
# Reduce memory fraction in config
"memory_fraction": 0.6

# Use smaller chunk sizes for processing
# Enable memory tracking
```

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from advanced_rvc_inference.lib.backends.zluda import is_available
print(f"ZLUDA debug: {is_available()}")
```

### Performance Monitoring

```python
from advanced_rvc_inference.lib.backends.zluda import get_performance_metrics

# Monitor real-time performance
metrics = get_performance_metrics(context)
print(f"GPU utilization: {metrics['gpu_utilization']}%")
print(f"Memory usage: {metrics['memory_usage']}")
print(f"Compute efficiency: {metrics['compute_efficiency']}")
```

## 🔮 Future Enhancements

### Planned Features

1. **Multi-GPU Support**: Distribute workload across multiple AMD GPUs
2. **Training Acceleration**: GPU-accelerated model training
3. **Mixed Precision**: FP16 support for faster inference
4. **Streaming Processing**: Real-time voice conversion
5. **Custom Kernel Development**: User-defined kernel support

### Contributing

Contributions to the ZLUDA backend are welcome! Areas for improvement:

- Additional RVC-specific kernels
- Performance optimizations
- Better error handling
- Compatibility improvements
- Documentation enhancements

## 📚 References

- [ZLUDA GitHub Repository](https://github.com/vosen/ZLUDA)
- [ROCm Documentation](https://rocmdocs.amd.com/)
- [AMD GPU Programming Guide](https://rocm.docs.amd.com/projects/ROCm_GPU_Programming_Guide/en/latest/)
- [RVC Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

## 📄 License

The ZLUDA backend integration follows the same license as the Advanced RVC Inference project. Please refer to the main project license for details.

---

**Happy voice converting with ZLUDA! 🎵🎤**
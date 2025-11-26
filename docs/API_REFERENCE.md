# Advanced RVC Inference - API Reference

**Version:** 4.0.0  
**Authors:** ArkanDash & BF667

This document provides comprehensive API reference for the Advanced RVC Inference project.

## Core Functions

### `full_inference_program()`

The main function for voice conversion with full parameter support.

```python
def full_inference_program(
    model_path: str,
    index_path: str,
    input_audio_path: str,
    output_path: str,
    pitch_extract: str = "rmvpe",
    embedder_model: str = "contentvec",
    filter_radius: int = 3,
    resample_sr: int = 0,
    rms_mix_rate: float = 0.25,
    protect: float = 0.33,
    hop_length: int = 512,
    krvc_type: str = "v2",
    direct_io: str = None,
    db_level: float = -18.0,
    embedder_path: str = "assets/models/embedders/",
    enhancer: bool = False,
    autotune: bool = False,
    autotune_strength: float = 0.5,
    use_gpu: bool = True,
    gpu_id: int = 0,
    **kwargs
) -> Optional[str]:
    """
    Complete voice conversion program with advanced parameters.
    
    This function performs end-to-end voice conversion using the RVC technology
    with KRVC kernel optimizations for enhanced performance.
    
    Args:
        model_path: Path to the voice model file (.pth/.onnx)
        index_path: Path to the feature index file (.index)
        input_audio_path: Path to input audio file
        output_path: Path for output audio file
        pitch_extract: F0 extraction method (60+ available methods)
        embedder_model: Content extraction model (60+ available models)
        filter_radius: Spectral envelope filtering radius
        resample_sr: Target sample rate (0 = keep original)
        rms_mix_rate: Mix rate for RMS volume envelope
        protect: Protection parameter for voice quality
        hop_length: Hop length for pitch extraction
        krvc_type: KRVC kernel type ("v1", "v2", "custom")
        direct_io: Direct I/O mode (None, "convert", "enhance")
        db_level: Output dB level
        embedder_path: Path to embedder models
        enhancer: Enable audio enhancer
        autotune: Enable autotune effect
        autotune_strength: Autotune effect strength (0.0-1.0)
        use_gpu: Enable GPU acceleration
        gpu_id: GPU device ID
        **kwargs: Additional parameters
        
    Returns:
        Path to output file if successful, None if failed
        
    Raises:
        FileNotFoundError: If input files don't exist
        ImportError: If required dependencies are missing
        RuntimeError: If processing fails
        
    Examples:
        Basic usage:
        >>> full_inference_program(
        ...     model_path="models/voice_model.pth",
        ...     index_path="models/voice_index.index",
        ...     input_audio_path="input.wav",
        ...     output_path="output.wav"
        ... )
        
        Advanced usage with optimization:
        >>> full_inference_program(
        ...     model_path="models/voice_model.pth",
        ...     index_path="models/voice_index.index",
        ...     input_audio_path="input.wav",
        ...     output_path="output.wav",
        ...     pitch_extract="hybrid[rmvpe+crepe+fcpe]",
        ...     embedder_model="vietnamese-hubert-base",
        ...     krvc_type="v2",
        ...     use_gpu=True,
        ...     enhancer=True
        ... )
    """
```

### `import_voice_converter()`

Load and initialize a voice conversion model.

```python
def import_voice_converter(
    model_path: str,
    device: str = "auto",
    use_fp16: bool = True,
    use_jit: bool = False,
    krvc_type: str = "v2"
) -> Union[torch.nn.Module, Any]:
    """
    Import and configure voice conversion model.
    
    Args:
        model_path: Path to model file
        device: Device to use ("auto", "cpu", "cuda")
        use_fp16: Enable FP16 precision
        use_jit: Enable PyTorch JIT compilation
        krvc_type: KRVC kernel type
        
    Returns:
        Loaded and configured model
    """
```

### `get_config()`

Get current configuration settings.

```python
def get_config() -> Dict[str, Any]:
    """
    Get current runtime configuration.
    
    Returns:
        Dictionary containing current configuration
    """
```

## KRVC Kernel Functions

### `KRVCFeatureExtractor`

Advanced feature extraction with KRVC optimizations.

```python
class KRVCFeatureExtractor:
    """KRVC-enhanced feature extractor for voice conversion."""
    
    def __init__(self, krvc_type: str = "v2", use_gpu: bool = True):
        """
        Initialize KRVC feature extractor.
        
        Args:
            krvc_type: KRVC kernel type ("v1", "v2", "custom")
            use_gpu: Enable GPU acceleration
        """
    
    def extract_features(
        self,
        audio: torch.Tensor,
        sample_rate: int = 44100,
        extract_f0: bool = True,
        extract_embedding: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Extract voice features using KRVC optimization.
        
        Args:
            audio: Input audio tensor
            sample_rate: Audio sample rate
            extract_f0: Extract F0 features
            extract_embedding: Extract embedding features
            
        Returns:
            Dictionary of extracted features
        """
    
    def set_inference_mode(self) -> None:
        """Configure extractor for inference mode."""
    
    def set_training_mode(self) -> None:
        """Configure extractor for training mode."""
    
    def enable_mixed_precision(self) -> None:
        """Enable mixed precision for memory efficiency."""
    
    def enable_memory_optimization(self) -> None:
        """Enable memory optimization techniques."""
    
    def benchmark_performance(self, test_audio: torch.Tensor) -> Dict[str, float]:
        """
        Benchmark extraction performance.
        
        Args:
            test_audio: Test audio for benchmarking
            
        Returns:
            Performance metrics dictionary
        """
```

### Performance Optimization Functions

#### `krvc_speed_optimize()`

Apply KRVC speed optimizations globally.

```python
def krvc_speed_optimize() -> None:
    """
    Apply KRVC speed optimizations to improve processing performance.
    """
```

#### `krvc_inference_mode()`

Configure KRVC for inference workloads.

```python
def krvc_inference_mode() -> None:
    """
    Configure KRVC for inference-specific optimizations.
    """
```

#### `krvc_training_mode()`

Configure KRVC for training workloads.

```python
def krvc_training_mode() -> None:
    """
    Configure KRVC for training-specific optimizations.
    """
```

#### `krvc_mixed_precision_training()`

Enable mixed precision training.

```python
def krvc_mixed_precision_training(
    enable: bool = True,
    opt_level: str = "O1"
) -> None:
    """
    Enable/disable mixed precision training.
    
    Args:
        enable: Enable mixed precision
        opt_level: Optimization level ("O0", "O1", "O2", "O3")
    """
```

## GPU Optimization Functions

### `gpu_optimizer()`

Get GPU optimizer instance.

```python
def gpu_optimizer() -> "GPUOptimizer":
    """
    Get GPU optimizer instance.
    
    Returns:
        GPUOptimizer instance
    """
```

### `gpu_settings()`

Configure GPU settings.

```python
def gpu_settings(
    batch_size: int = 1,
    precision: str = "float16",
    memory_fraction: float = 0.8,
    allow_growth: bool = True
) -> Dict[str, Any]:
    """
    Configure GPU settings.
    
    Args:
        batch_size: Processing batch size
        precision: Compute precision ("float16", "float32", "bfloat16")
        memory_fraction: GPU memory usage fraction
        allow_growth: Enable memory growth
        
    Returns:
        Configuration dictionary
    """
```

### `check_fp16_support()`

Check if FP16 is supported on current hardware.

```python
def check_fp16_support() -> bool:
    """
    Check if FP16 precision is supported.
    
    Returns:
        True if FP16 is supported
    """
```

## Audio Processing Functions

### `add_audio_effects()`

Apply audio effects to processed audio.

```python
def add_audio_effects(
    audio: Union[AudioSegment, torch.Tensor],
    effects: List[Dict[str, Any]],
    sample_rate: int = 44100
) -> Union[AudioSegment, torch.Tensor]:
    """
    Add audio effects to audio.
    
    Args:
        audio: Input audio
        effects: List of effect configurations
        sample_rate: Audio sample rate
        
    Returns:
        Audio with effects applied
    """
```

### `merge_audios()`

Merge multiple audio files.

```python
def merge_audios(
    audio_list: List[Union[str, Path, torch.Tensor]],
    output_path: str,
    fade_duration: float = 0.1,
    crossfade: bool = True
) -> str:
    """
    Merge multiple audio files.
    
    Args:
        audio_list: List of audio file paths or tensors
        output_path: Output file path
        fade_duration: Fade duration in seconds
        crossfade: Enable crossfading
        
    Returns:
        Path to merged audio file
    """
```

## Real-time Processing Functions

### `real_time_voice_conversion()`

Real-time voice conversion processing.

```python
def real_time_voice_conversion(
    model_path: str,
    input_stream,
    output_stream,
    buffer_size: int = 1024,
    target_latency: float = 0.05
) -> None:
    """
    Real-time voice conversion processing.
    
    Args:
        model_path: Path to voice model
        input_stream: Input audio stream
        output_stream: Output audio stream
        buffer_size: Processing buffer size
        target_latency: Target latency in seconds
    """
```

## Model Management Functions

### Model Discovery Functions

#### `models_vocals()`

Get list of vocal models.

```python
def models_vocals() -> Dict[str, Dict[str, Any]]:
    """
    Get list of available vocal models.
    
    Returns:
        Dictionary of vocal models with metadata
    """
```

#### `karaoke_models()`

Get list of karaoke/separation models.

```python
def karaoke_models() -> Dict[str, Dict[str, Any]]:
    """
    Get list of available karaoke/separation models.
    
    Returns:
        Dictionary of separation models with metadata
    """
```

#### `denoise_models()`

Get list of denoising models.

```python
def denoise_models() -> Dict[str, Dict[str, Any]]:
    """
    Get list of available denoising models.
    
    Returns:
        Dictionary of denoising models with metadata
    """
```

#### `dereverb_models()`

Get list of dereverberation models.

```python
def dereverb_models() -> Dict[str, Dict[str, Any]]:
    """
    Get list of available dereverberation models.
    
    Returns:
        Dictionary of dereverberation models with metadata
    """
```

#### `deecho_models()`

Get list of de-echoing models.

```python
def deecho_models() -> Dict[str, Dict[str, Any]]:
    """
    Get list of available de-echoing models.
    
    Returns:
        Dictionary of de-echoing models with metadata
    """
```

## Utility Functions

### `download_file()`

Download files with progress tracking.

```python
def download_file(
    url: str,
    output_path: Union[str, Path],
    chunk_size: int = 8192,
    show_progress: bool = True
) -> None:
    """
    Download file from URL.
    
    Args:
        url: Download URL
        output_path: Output file path
        chunk_size: Download chunk size
        show_progress: Show download progress
    """
```

## Advanced Classes

### `KRVCAdvancedOptimizer`

Advanced optimization for KRVC processing.

```python
class KRVCAdvancedOptimizer:
    """Advanced optimizer for KRVC processing."""
    
    def __init__(self, optimization_level: str = "balanced"):
        """
        Initialize advanced optimizer.
        
        Args:
            optimization_level: "speed", "memory", "balanced", "maximum"
        """
    
    def optimize_model(self, model: torch.nn.Module, **kwargs) -> torch.nn.Module:
        """
        Optimize model for performance.
        
        Args:
            model: PyTorch model to optimize
            
        Returns:
            Optimized model
        """
    
    def optimize_inference(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        optimization_mode: str = "max-autotune"
    ) -> torch.nn.Module:
        """
        Optimize model for inference.
        
        Args:
            model: Model to optimize
            input_shape: Expected input shape
            optimization_mode: torch.compile optimization mode
            
        Returns:
            Optimized model for inference
        """
```

### `KRVCPerformanceMonitor`

Performance monitoring for KRVC operations.

```python
class KRVCPerformanceMonitor:
    """Performance monitor for KRVC operations."""
    
    def __init__(self, enable_monitoring: bool = True):
        """
        Initialize performance monitor.
        
        Args:
            enable_monitoring: Enable performance monitoring
        """
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current performance statistics.
        
        Returns:
            Performance statistics dictionary
        """
    
    def reset_stats(self) -> None:
        """Reset performance statistics."""
    
    def benchmark_complete_optimization(
        self,
        test_data: torch.Tensor,
        optimized_model: torch.nn.Module,
        iterations: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark complete optimization pipeline.
        
        Args:
            test_data: Test data for benchmarking
            optimized_model: Model to benchmark
            iterations: Number of benchmark iterations
            
        Returns:
            Benchmark results
        """
```

## Constants and Enums

### Availability Flags

```python
# KRVC availability
KRVC_AVAILABLE: bool = True

# GPU optimization availability
GPU_OPTIMIZATION_AVAILABLE: bool = True

# Performance optimization features
TORCH_COMPILE_AVAILABLE: bool = True
TORCHFX_AVAILABLE: bool = True
TORCH_AUDIOMENTATIONS_AVAILABLE: bool = True
```

### Default Values

```python
# Default F0 extraction methods
DEFAULT_PITCH_EXTRACT = "rmvpe"
DEFAULT_EMBEDDER_MODEL = "contentvec"
DEFAULT_KRVC_TYPE = "v2"

# Default audio parameters
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_HOP_LENGTH = 512
DEFAULT_FILTER_RADIUS = 3

# Default quality parameters
DEFAULT_RMS_MIX_RATE = 0.25
DEFAULT_PROTECT = 0.33
DEFAULT_DB_LEVEL = -18.0
```

## Error Handling

### Common Exceptions

```python
class RVCError(Exception):
    """Base exception for RVC operations."""
    pass

class ModelNotFoundError(RVCError):
    """Raised when model file is not found."""
    pass

class UnsupportedFormatError(RVCError):
    """Raised when audio format is not supported."""
    pass

class GPUError(RVCError):
    """Raised when GPU operations fail."""
    pass

class OptimizationError(RVCError):
    """Raised when optimization fails."""
    pass
```

### Error Handling Example

```python
try:
    result = full_inference_program(
        model_path="model.pth",
        input_audio_path="input.wav",
        output_path="output.wav"
    )
except FileNotFoundError as e:
    print(f"Model file not found: {e}")
except RVCError as e:
    print(f"RVC processing error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Tips

### Memory Optimization

1. **Use mixed precision**: `use_fp16=True`
2. **Enable memory optimization**: `enable_memory_optimization()`
3. **Clear cache regularly**: `cleanup_krvc_memory()`

### Speed Optimization

1. **Use appropriate batch sizes**: `batch_size=2-4`
2. **Enable KRVC optimizations**: `krvc_type="v2"`
3. **Use torch.compile**: Set `use_jit=True`

### Quality Optimization

1. **Use hybrid F0 methods**: `pitch_extract="hybrid[rmvpe+crepe+fcpe]"`
2. **Enable audio enhancement**: `enhancer=True`
3. **Use appropriate protection**: `protect=0.33`

---

*This API reference is maintained by ArkanDash & BF667 for Advanced RVC Inference V4.0.0*
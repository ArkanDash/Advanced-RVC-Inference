"""
KADVC Configuration
Advanced kernel optimization settings for voice conversion
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import torch


@dataclass
class KADVCConfig:
    """Configuration class for KADVC optimization settings"""
    
    # General optimization settings
    enable_mixed_precision: bool = True
    enable_tensor_cores: bool = True
    benchmark_mode: bool = True
    memory_efficient_algorithms: bool = True
    
    # CUDA optimization settings
    cuda_allow_tf32: bool = True
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False
    memory_fraction: float = 0.95
    
    # Performance settings
    optimal_batch_size_colab: int = 4
    chunk_size_for_large_audio: int = 32768
    max_audio_length_seconds: int = 300
    
    # F0 extraction settings
    f0_method: str = "hybrid"  # hybrid, librosa, crepe
    f0_hop_length_factor: float = 200  # 200Hz frame rate
    f0_min_freq: float = 50
    f0_max_freq: float = 500
    
    # Feature extraction settings
    n_fft: int = 2048
    hop_length: int = 256
    win_length: int = 2048
    window_type: str = "hann"
    
    # Kernel optimization settings
    use_custom_kernels: bool = True
    parallel_processing: bool = True
    cache_optimizations: bool = True
    
    # Memory management
    gc_threshold: float = 0.8
    clear_cache_frequency: int = 100  # Clear cache every N iterations
    memory_growth: bool = True
    
    # GPU-specific optimizations
    enable_graph_capture: bool = True if torch.cuda.is_available() else False
    warmup_iterations: int = 10
    
    # Compatibility settings
    colab_optimization: bool = True
    fallback_to_cpu: bool = True
    timeout_seconds: int = 300
    
    @classmethod
    def create_colab_config(cls) -> 'KADVCConfig':
        """Create KADVC configuration optimized for Google Colab"""
        return cls(
            enable_mixed_precision=True,
            enable_tensor_cores=torch.cuda.is_bf16_supported(),
            benchmark_mode=True,
            memory_efficient_algorithms=True,
            cuda_allow_tf32=True,
            cudnn_benchmark=True,
            memory_fraction=0.95 if torch.cuda.is_available() else 1.0,
            optimal_batch_size_colab=4,
            chunk_size_for_large_audio=16384,  # Smaller chunks for Colab memory
            max_audio_length_seconds=180,  # Shorter for Colab timeouts
            use_custom_kernels=True,
            parallel_processing=True,
            cache_optimizations=True,
            colab_optimization=True,
            fallback_to_cpu=True,
            clear_cache_frequency=50,  # More frequent for Colab memory constraints
            warmup_iterations=5
        )
    
    @classmethod
    def create_local_config(cls) -> 'KADVCConfig':
        """Create KADVC configuration for local high-end GPUs"""
        return cls(
            enable_mixed_precision=True,
            enable_tensor_cores=True,
            benchmark_mode=True,
            memory_efficient_algorithms=True,
            cuda_allow_tf32=True,
            cudnn_benchmark=True,
            memory_fraction=0.95,
            optimal_batch_size_colab=8,  # Larger batch size for local GPUs
            chunk_size_for_large_audio=65536,
            max_audio_length_seconds=600,  # Longer audio support
            use_custom_kernels=True,
            parallel_processing=True,
            cache_optimizations=True,
            colab_optimization=False,
            fallback_to_cpu=False,
            clear_cache_frequency=100,
            warmup_iterations=20
        )
    
    def get_cuda_kernel_config(self) -> Dict[str, Any]:
        """Get CUDA kernel configuration parameters"""
        return {
            "block_size": self._get_optimal_block_size(),
            "grid_size": self._get_optimal_grid_size(),
            "shared_memory_size": 49152,  # 48KB shared memory
            "max_threads_per_block": 1024,
            "warp_size": 32,
            "compute_capability": torch.cuda.get_device_capability(),
            "enable_tensor_cores": self.enable_tensor_cores,
            "mixed_precision": self.enable_mixed_precision
        }
    
    def _get_optimal_block_size(self) -> int:
        """Get optimal CUDA block size for current GPU"""
        device_props = torch.cuda.get_device_properties(0)
        
        if device_props.major >= 7:  # Modern GPUs with high compute capability
            return 1024  # Maximum threads per block
        elif device_props.major >= 6:  # Pascal/Turing GPUs
            return 768
        else:  # Older GPUs
            return 512
    
    def _get_optimal_grid_size(self) -> int:
        """Get optimal CUDA grid size"""
        device_props = torch.cuda.get_device_properties(0)
        max_threads_per_multiprocessor = device_props.max_threads_per_multi_processor
        
        # Use 75% of available multiprocessors to avoid oversubscription
        return int(device_props.multi_processor_count * 0.75 * max_threads_per_multiprocessor // self._get_optimal_block_size())
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return status"""
        issues = []
        warnings = []
        
        # Validate memory settings
        if self.memory_fraction > 1.0:
            issues.append("memory_fraction cannot be > 1.0")
        elif self.memory_fraction > 0.95:
            warnings.append("High memory fraction may cause OOM errors")
        
        # Validate performance settings
        if self.optimal_batch_size_colab > 16:
            warnings.append("Large batch size may cause memory issues in Colab")
        
        # Validate audio settings
        if self.n_fft > 4096:
            warnings.append("Large FFT size may impact real-time performance")
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            warnings.append("CUDA not available - some optimizations disabled")
            self.fallback_to_cpu = True
        
        # Validate F0 settings
        if self.f0_min_freq >= self.f0_max_freq:
            issues.append("f0_min_freq must be < f0_max_freq")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "config": self.__dict__
        }
    
    def apply_to_torch(self):
        """Apply KADVC configuration to PyTorch runtime settings"""
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(self.memory_fraction)
        
        torch.backends.cudnn.benchmark = self.benchmark_mode
        torch.backends.cudnn.deterministic = self.cudnn_deterministic
        torch.backends.cuda.matmul.allow_tf32 = self.cuda_allow_tf32
        
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_bf16 = self.enable_tensor_cores
            torch.backends.cuda.enable_tensor_core_math = self.enable_tensor_cores


# Default configurations
DEFAULT_KADVC_CONFIG = KADVCConfig.create_colab_config()
LOCAL_KADVC_CONFIG = KADVCConfig.create_local_config()


# Compatibility matrix for different GPU types
GPU_COMPATIBILITY = {
    "T4": {
        "memory_gb": 16,
        "compute_capability": (7, 5),
        "tensor_cores": True,
        "max_batch_size": 8,
        "recommended_f0_method": "hybrid"
    },
    "V100": {
        "memory_gb": 32,
        "compute_capability": (7, 0),
        "tensor_cores": True,
        "max_batch_size": 16,
        "recommended_f0_method": "hybrid"
    },
    "A100": {
        "memory_gb": 80,
        "compute_capability": (8, 0),
        "tensor_cores": True,
        "max_batch_size": 32,
        "recommended_f0_method": "crepe"
    },
    "K80": {
        "memory_gb": 12,
        "compute_capability": (3, 7),
        "tensor_cores": False,
        "max_batch_size": 4,
        "recommended_f0_method": "librosa"
    },
    "P4": {
        "memory_gb": 8,
        "compute_capability": (6, 1),
        "tensor_cores": False,
        "max_batch_size": 4,
        "recommended_f0_method": "librosa"
    },
    "RTX_3090": {
        "memory_gb": 24,
        "compute_capability": (8, 6),
        "tensor_cores": True,
        "max_batch_size": 16,
        "recommended_f0_method": "crepe"
    },
    "RTX_4090": {
        "memory_gb": 24,
        "compute_capability": (8, 9),
        "tensor_cores": True,
        "max_batch_size": 32,
        "recommended_f0_method": "crepe"
    }
}


def get_gpu_type() -> str:
    """Detect current GPU type"""
    if not torch.cuda.is_available():
        return "CPU"
    
    gpu_name = torch.cuda.get_device_name(0).upper()
    
    for gpu_type in GPU_COMPATIBILITY.keys():
        if gpu_type in gpu_name:
            return gpu_type
    
    # Fallback to compute capability detection
    capability = torch.cuda.get_device_capability(0)
    if capability[0] >= 8:
        return "A100"
    elif capability[0] >= 7:
        return "T4"
    elif capability[0] >= 6:
        return "P4"
    else:
        return "Unknown"


def create_optimized_config() -> KADVCConfig:
    """Create KADVC configuration optimized for detected hardware"""
    gpu_type = get_gpu_type()
    
    if gpu_type in GPU_COMPATIBILITY:
        gpu_info = GPU_COMPATIBILITY[gpu_type]
        
        if "Colab" in gpu_info.get("recommended_f0_method", ""):
            # Colab optimization
            return KADVCConfig.create_colab_config()
        else:
            # Local GPU optimization
            config = KADVCConfig.create_local_config()
            
            # Apply GPU-specific optimizations
            config.optimal_batch_size_colab = min(
                config.optimal_batch_size_colab,
                gpu_info["max_batch_size"]
            )
            config.enable_tensor_cores = gpu_info["tensor_cores"]
            config.f0_method = gpu_info["recommended_f0_method"]
            
            return config
    else:
        # Fallback to default
        return DEFAULT_KADVC_CONFIG
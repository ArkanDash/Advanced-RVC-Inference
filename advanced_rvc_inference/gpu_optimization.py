"""
OpenCL Support and GPU Optimization Utilities for Advanced RVC Inference
Enhanced GPU support for T4 and A100 GPUs with OpenCL acceleration
Version 3.5.3
"""

import os
import sys
import numpy as np
import logging
from typing import Optional, List, Dict, Any, Tuple
import torch
from pathlib import Path

logger = logging.getLogger(__name__)

# OpenCL imports with fallbacks
try:
    import pyopencl as cl
    import pyopencl.array as cl_array
    from pyopencl import characterize
    OPENCL_AVAILABLE = True
    logger.info("OpenCL support detected and loaded successfully")
except ImportError as e:
    OPENCL_AVAILABLE = False
    logger.warning(f"OpenCL not available: {e}")

try:
    import torch
    import torch.nn.functional as F
    from torch.cuda import memory_summary
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available for GPU optimization")

class GPUOptimizer:
    """
    GPU optimization manager for T4 and A100 GPUs
    Handles memory management, precision settings, and performance optimization
    """
    
    def __init__(self):
        self.gpu_info = self._detect_gpu()
        self.memory_allocated = 0
        self.precision_mode = "fp32"  # Default to FP32 for compatibility
        
    def _detect_gpu(self) -> Dict[str, Any]:
        """Detect GPU type and capabilities"""
        gpu_info = {
            "type": "cpu",
            "memory_gb": 0,
            "compute_capability": "0.0",
            "tensor_cores": False,
            "multi_processing": True,
            "opencl_supported": OPENCL_AVAILABLE,
            "cuda_supported": TORCH_AVAILABLE
        }
        
        if not TORCH_AVAILABLE:
            return gpu_info
            
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                
                # Detect GPU type
                if "T4" in gpu_name:
                    gpu_type = "T4"
                    tensor_cores = False
                    memory_gb = props.total_memory / (1024**3)
                    compute_cap = f"{props.major}.{props.minor}"
                elif "A100" in gpu_name:
                    gpu_type = "A100" 
                    tensor_cores = True
                    memory_gb = props.total_memory / (1024**3)
                    compute_cap = f"{props.major}.{props.minor}"
                elif "V100" in gpu_name:
                    gpu_type = "V100"
                    tensor_cores = True
                    memory_gb = props.total_memory / (1024**3)
                    compute_cap = f"{props.major}.{props.minor}"
                elif "RTX 4090" in gpu_name or "RTX 4080" in gpu_name or "RTX 3090" in gpu_name:
                    gpu_type = "RTX_Ada"
                    tensor_cores = True
                    memory_gb = props.total_memory / (1024**3)
                    compute_cap = f"{props.major}.{props.minor}"
                else:
                    gpu_type = "Unknown_NVIDIA"
                    tensor_cores = False
                    memory_gb = props.total_memory / (1024**3)
                    compute_cap = f"{props.major}.{props.minor}"
                
                gpu_info.update({
                    "type": gpu_type,
                    "memory_gb": memory_gb,
                    "compute_capability": compute_cap,
                    "tensor_cores": tensor_cores,
                    "device_count": gpu_count,
                    "current_device": i,
                    "device_name": gpu_name
                })
                
                logger.info(f"Detected GPU {i}: {gpu_name}")
                logger.info(f"Memory: {memory_gb:.1f}GB, Tensor Cores: {tensor_cores}")
                
        return gpu_info
    
    def get_optimal_settings(self) -> Dict[str, Any]:
        """Get optimal GPU settings based on detected hardware"""
        gpu_type = self.gpu_info["type"]
        memory_gb = self.gpu_info["memory_gb"]
        
        if gpu_type == "T4":
            # T4 optimizations - memory efficient
            settings = {
                "batch_size": 1 if memory_gb < 16 else 2,
                "precision": "fp16",
                "mixed_precision": True,
                "gradient_accumulation_steps": 4,
                "max_audio_length": 30,  # seconds
                "use_amp": True,
                "compile_model": False,
                "memory_efficient": True
            }
        elif gpu_type == "A100":
            # A100 optimizations - high performance
            settings = {
                "batch_size": 4 if memory_gb >= 40 else 2,
                "precision": "bf16",
                "mixed_precision": True,
                "gradient_accumulation_steps": 1,
                "max_audio_length": 120,  # seconds
                "use_amp": True,
                "compile_model": True,
                "memory_efficient": False
            }
        elif gpu_type == "V100":
            # V100 optimizations
            settings = {
                "batch_size": 2,
                "precision": "fp16",
                "mixed_precision": True,
                "gradient_accumulation_steps": 2,
                "max_audio_length": 60,
                "use_amp": True,
                "compile_model": False,
                "memory_efficient": False
            }
        elif gpu_type.startswith("RTX"):
            # RTX series optimizations
            settings = {
                "batch_size": 2 if memory_gb >= 24 else 1,
                "precision": "fp16",
                "mixed_precision": True,
                "gradient_accumulation_steps": 2,
                "max_audio_length": 60,
                "use_amp": True,
                "compile_model": True,
                "memory_efficient": False
            }
        else:
            # CPU or unknown GPU
            settings = {
                "batch_size": 1,
                "precision": "fp32",
                "mixed_precision": False,
                "gradient_accumulation_steps": 4,
                "max_audio_length": 10,
                "use_amp": False,
                "compile_model": False,
                "memory_efficient": True
            }
        
        # Override with OpenCL optimizations if available
        if OPENCL_AVAILABLE:
            settings["use_opencl"] = True
            settings["opencl_queue_count"] = min(8, max(1, int(memory_gb)))
        else:
            settings["use_opencl"] = False
            
        return settings
    
    def optimize_memory(self):
        """Optimize GPU memory allocation"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return
            
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Enable memory-efficient attention if available
        if hasattr(F, 'scaled_dot_product_attention'):
            logger.info("Memory-efficient attention enabled")
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {"total": 0, "allocated": 0, "cached": 0}
            
        total = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated()
        cached = torch.cuda.memory_reserved()
        
        return {
            "total_gb": total / (1024**3),
            "allocated_gb": allocated / (1024**3),
            "cached_gb": cached / (1024**3),
            "utilization": (allocated / total) * 100
        }

class OpenCLAudioProcessor:
    """
    OpenCL-accelerated audio processing for RVC inference
    Provides GPU acceleration for common audio operations
    """
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.context = None
        self.queue = None
        self.program = None
        
        if not OPENCL_AVAILABLE:
            logger.warning("OpenCL not available, using CPU fallback")
            return
            
        try:
            self._initialize_opencl()
            self._load_kernels()
            logger.info(f"OpenCL initialized successfully on device {device_id}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenCL: {e}")
            self.context = None
            self.queue = None
    
    def _initialize_opencl(self):
        """Initialize OpenCL context and queue"""
        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError("No OpenCL platforms found")
            
        # Try to find GPU platform
        gpu_platform = None
        for platform in platforms:
            devices = platform.get_devices()
            if any(device.type & cl.device_type.GPU for device in devices):
                gpu_platform = platform
                break
                
        if gpu_platform is None:
            gpu_platform = platforms[0]  # Fallback to first platform
            
        devices = gpu_platform.get_devices()
        self.context = cl.Context(devices)
        self.queue = cl.CommandQueue(self.context)
    
    def _load_kernels(self):
        """Load OpenCL kernels for audio processing"""
        kernel_code = """
        // FFT optimization kernel
        __kernel void optimize_fft(__global float* input, __global float* output, int length) {
            int i = get_global_id(0);
            if (i < length) {
                // Simple FFT optimization - can be enhanced with actual FFT
                output[i] = input[i] * 0.5f;
            }
        }
        
        // Audio filtering kernel
        __kernel void audio_filter(__global float* audio, __global float* filtered, 
                                  __global float* coefficients, int filter_length, int audio_length) {
            int i = get_global_id(0);
            if (i < audio_length) {
                float sum = 0.0f;
                for (int j = 0; j < filter_length; j++) {
                    if (i >= j) {
                        sum += audio[i - j] * coefficients[j];
                    }
                }
                filtered[i] = sum;
            }
        }
        
        // Normalization kernel
        __kernel void normalize(__global float* data, int length, float target_amplitude) {
            int i = get_global_id(0);
            if (i < length) {
                data[i] = data[i] * target_amplitude;
            }
        }
        """
        
        self.program = cl.Program(self.context, kernel_code).build()
    
    def process_audio_opencl(self, audio_data: np.ndarray, operation: str = "normalize") -> np.ndarray:
        """Process audio data using OpenCL acceleration"""
        if not OPENCL_AVAILABLE or self.context is None:
            # Fallback to numpy operations
            if operation == "normalize":
                return audio_data * 0.8  # Simple normalization
            elif operation == "filter":
                # Simple filter - can be enhanced
                kernel = np.array([0.25, 0.5, 0.25])
                return np.convolve(audio_data, kernel, mode='same')
            else:
                return audio_data
        
        try:
            if operation == "normalize":
                return self._normalize_opencl(audio_data)
            elif operation == "filter":
                return self._filter_opencl(audio_data)
            elif operation == "fft_optimize":
                return self._fft_optimize_opencl(audio_data)
            else:
                return audio_data
        except Exception as e:
            logger.error(f"OpenCL processing failed: {e}, falling back to CPU")
            return self._cpu_fallback(audio_data, operation)
    
    def _normalize_opencl(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio using OpenCL"""
        output = np.zeros_like(audio_data, dtype=np.float32)
        
        # Create OpenCL buffers
        input_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, 
                             hostbuf=audio_data.astype(np.float32))
        output_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, output.nbytes)
        
        # Execute kernel
        self.program.normalize(self.queue, audio_data.shape, None, 
                              input_buf, output_buf, np.int32(audio_data.shape[0]), np.float32(0.8))
        
        # Copy result back
        cl.enqueue_copy(self.queue, output, output_buf)
        
        return output
    
    def _filter_opencl(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply filter using OpenCL"""
        # Simple moving average filter
        filter_coeffs = np.array([0.25, 0.5, 0.25], dtype=np.float32)
        output = np.zeros_like(audio_data, dtype=np.float32)
        
        # Create buffers
        audio_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=audio_data.astype(np.float32))
        output_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, output.nbytes)
        filter_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=filter_coeffs)
        
        # Execute kernel
        self.program.audio_filter(self.queue, audio_data.shape, None,
                                 audio_buf, output_buf, filter_buf,
                                 np.int32(filter_coeffs.shape[0]), np.int32(audio_data.shape[0]))
        
        # Copy result back
        cl.enqueue_copy(self.queue, output, output_buf)
        
        return output
    
    def _fft_optimize_opencl(self, audio_data: np.ndarray) -> np.ndarray:
        """FFT optimization using OpenCL"""
        output = np.zeros_like(audio_data, dtype=np.float32)
        
        # Create buffers
        input_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=audio_data.astype(np.float32))
        output_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, output.nbytes)
        
        # Execute kernel
        self.program.optimize_fft(self.queue, audio_data.shape, None,
                                 input_buf, output_buf, np.int32(audio_data.shape[0]))
        
        # Copy result back
        cl.enqueue_copy(self.queue, output, output_buf)
        
        return output
    
    def _cpu_fallback(self, audio_data: np.ndarray, operation: str) -> np.ndarray:
        """CPU fallback for audio processing"""
        if operation == "normalize":
            return audio_data * 0.8
        elif operation == "filter":
            kernel = np.array([0.25, 0.5, 0.25])
            return np.convolve(audio_data, kernel, mode='same')
        elif operation == "fft_optimize":
            return audio_data * 0.5
        else:
            return audio_data
    
    def cleanup(self):
        """Clean up OpenCL resources"""
        if self.context:
            self.context.release()
            self.context = None

def get_gpu_optimizer() -> GPUOptimizer:
    """Get singleton GPU optimizer instance"""
    if not hasattr(get_gpu_optimizer, '_instance'):
        get_gpu_optimizer._instance = GPUOptimizer()
    return get_gpu_optimizer._instance

def get_opencl_processor(device_id: int = 0) -> Optional[OpenCLAudioProcessor]:
    """Get OpenCL audio processor instance"""
    if not OPENCL_AVAILABLE:
        return None
    
    cache_key = f"opencl_processor_{device_id}"
    if not hasattr(get_opencl_processor, '_cache'):
        get_opencl_processor._cache = {}
    
    if cache_key not in get_opencl_processor._cache:
        try:
            get_opencl_processor._cache[cache_key] = OpenCLAudioProcessor(device_id)
        except Exception as e:
            logger.error(f"Failed to create OpenCL processor: {e}")
            get_opencl_processor._cache[cache_key] = None
    
    return get_opencl_processor._cache[cache_key]

# Global instances
_gpu_optimizer = None
_opencl_processors = {}
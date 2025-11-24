# Backends module for GPU acceleration
# Supports DirectML, OpenCL, and ZLUDA backends

# DirectML backend
try:
    from .directml import *
    DIRECTML_AVAILABLE = True
except ImportError:
    DIRECTML_AVAILABLE = False

# OpenCL backend
try:
    from .opencl import *
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False

# ZLUDA backend with enhanced kernel functionality
try:
    from .zluda import *
    ZLUDA_AVAILABLE = True
except ImportError:
    ZLUDA_AVAILABLE = False

# Backend utilities
def get_available_backends():
    """Get list of available backends"""
    backends = []
    if DIRECTML_AVAILABLE:
        backends.append('DirectML')
    if OPENCL_AVAILABLE:
        backends.append('OpenCL')
    if ZLUDA_AVAILABLE:
        backends.append('ZLUDA')
    return backends

def get_recommended_backend():
    """Get recommended backend based on system configuration"""
    if ZLUDA_AVAILABLE:
        return 'ZLUDA'
    elif DIRECTML_AVAILABLE:
        return 'DirectML'
    elif OPENCL_AVAILABLE:
        return 'OpenCL'
    else:
        return 'CPU'

def initialize_backend(backend_name: str):
    """Initialize specified backend"""
    try:
        if backend_name.lower() == 'zluda':
            from .zluda import create_context, get_device
            device = get_device()
            context = create_context(device)
            return context
        elif backend_name.lower() == 'directml':
            from .directml import create_context
            return create_context()
        elif backend_name.lower() == 'opencl':
            from .opencl import create_context
            return create_context()
        else:
            raise ValueError(f"Unknown backend: {backend_name}")
    except Exception as e:
        print(f"Failed to initialize {backend_name}: {e}")
        return None

__all__ = [
    'DIRECTML_AVAILABLE',
    'OPENCL_AVAILABLE', 
    'ZLUDA_AVAILABLE',
    'get_available_backends',
    'get_recommended_backend',
    'initialize_backend',
    # ZLUDA exports
    'ZLUDAError', 'ZLUDAKernel', 'ZLUDADevice', 'ZLUDAContext', 'ZLUDAMemoryManager',
    'compile_kernel', 'execute_audio_kernel', 'optimize_for_rvc', 'get_performance_metrics'
]
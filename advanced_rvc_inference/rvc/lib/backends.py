"""
Backend modules for GPU acceleration support
"""

# Try to import DirectML if available
try:
    import torch_directml as directml_lib
    class directml:
        @staticmethod
        def is_available():
            try:
                device = directml_lib.device()
                return True
            except:
                return False
        
        @staticmethod
        def empty_cache():
            # DirectML doesn't typically need cache clearing like CUDA
            pass
except ImportError:
    class directml:
        @staticmethod
        def is_available():
            return False
        
        @staticmethod
        def empty_cache():
            pass

# Try to import OpenCL if available
try:
    import pyopencl as cl
    import torch
    
    class _OpenCLBackend:
        def __init__(self):
            self.pytorch_ocl = self  # To match the expected interface
            try:
                # Initialize OpenCL context if possible
                platforms = cl.get_platforms()
                self._available = len(platforms) > 0
            except:
                self._available = False
        
        def is_available(self):
            return self._available
        
        def empty_cache(self):
            # PyTorch OpenCL cache clearing
            if torch.cuda.is_available() and torch.version.cuda:
                # If CUDA is available, we might have mixed contexts
                torch.cuda.empty_cache()
    
    opencl = _OpenCLBackend()
except ImportError:
    class _OpenCLBackend:
        def __init__(self):
            self.pytorch_ocl = self
        
        def is_available(self):
            return False
        
        def empty_cache(self):
            pass
    
    opencl = _OpenCLBackend()
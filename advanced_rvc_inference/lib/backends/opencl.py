# OpenCL backend fallback
import warnings

def is_available():
    """Check if OpenCL is available"""
    return False

def get_device():
    """Get OpenCL device"""
    warnings.warn("OpenCL not available")
    return None

def create_context():
    """Create OpenCL context"""
    warnings.warn("OpenCL not available")
    return None
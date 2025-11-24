# DirectML backend fallback
import warnings

def is_available():
    """Check if DirectML is available"""
    return False

def get_device():
    """Get DirectML device"""
    warnings.warn("DirectML not available")
    return None

def create_context():
    """Create DirectML context"""
    warnings.warn("DirectML not available")
    return None
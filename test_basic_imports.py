"""
Test script to check basic imports without triggering the problematic imports
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_imports():
    """Test basic imports to identify issues"""
    print("Testing basic imports...")
    
    # Test importing the package itself
    try:
        import advanced_rvc_inference
        print("✓ Package import works")
    except ImportError as e:
        print(f"✗ Package import error: {e}")
        return False

    # Test importing core components
    try:
        from advanced_rvc_inference import app, core
        print("✓ Core module imports work")
    except ImportError as e:
        print(f"✗ Core module import error: {e}")
        return False

    # Test importing RVC config
    try:
        from advanced_rvc_inference.rvc.configs.config import Config
        print("✓ RVC config import works")
    except ImportError as e:
        print(f"✗ RVC config import error: {e}")
        
    # Test importing tabs
    try:
        from advanced_rvc_inference.tabs.inference.inference import inference_tab
        print("✓ Tab import works")
    except ImportError as e:
        print(f"✗ Tab import error: {e}")
        
    return True

if __name__ == "__main__":
    test_basic_imports()
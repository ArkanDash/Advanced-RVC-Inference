#!/usr/bin/env python3
"""
Minimal test for core functions without dependencies
This script creates a minimal core module for testing imports
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, '/workspace/Advanced-RVC-Inference')

def test_import_alternative():
    """Test import using alternative method"""
    print("Testing alternative import method...")
    
    # Method 1: Try importing directly with error handling
    try:
        # Try to import just what we need, handling missing dependencies
        import importlib.util
        spec = importlib.util.spec_from_file_location("core", "/workspace/Advanced-RVC-Inference/core.py")
        core = importlib.util.module_from_spec(spec)
        
        # Mock missing dependencies to allow loading
        sys.modules['torch'] = type('MockTorch', (), {
            'cuda': type('MockCuda', (), {'is_available': lambda: False})(),
            'load': lambda x: {},
            'tensor': lambda x: x
        })()
        
        sys.modules['numpy'] = type('MockNumpy', (), {
            'array': lambda x: x,
            'float32': float,
            'int32': int
        })()
        
        sys.modules['librosa'] = type('MockLibrosa', (), {})()
        sys.modules['soundfile'] = type('MockSoundfile', (), {
            'read': lambda x: ([], 44100),
            'write': lambda x, y, z: None
        })()
        sys.modules['torchaudio'] = type('MockTorchaudio', (), {})()
        sys.modules['noisereduce'] = type('MockNoisereduce', (), {
            'reduce_noise': lambda y, z: y
        })()
        sys.modules['demucs'] = type('MockDemucs', (), {})()
        sys.modules['yt_dlp'] = type('MockYtDlp', (), {})()
        
        spec.loader.exec_module(core)
        
        # Test if functions exist
        if hasattr(core, 'full_inference_program'):
            print("✓ full_inference_program found in core module")
        else:
            print("✗ full_inference_program NOT found in core module")
            
        if hasattr(core, 'download_music'):
            print("✓ download_music found in core module")
        else:
            print("✗ download_music NOT found in core module")
            
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def create_simple_test():
    """Create a simple test file that bypasses dependencies"""
    test_content = '''#!/usr/bin/env python3
"""
Simple test for core functions - version that works around dependencies
"""

# Add current directory to path
import sys
import os
sys.path.insert(0, '/workspace/Advanced-RVC-Inference')

# Mock missing dependencies before importing
class MockTorch:
    cuda = type('MockCuda', (), {'is_available': lambda: False})()
    @staticmethod
    def load(x): return {}
    @staticmethod
    def tensor(x): return x

class MockModule:
    pass

# Set up mock modules
sys.modules['torch'] = MockTorch()
sys.modules['numpy'] = type('MockNumpy', (), {'array': lambda x: x, 'float32': float})()
sys.modules['librosa'] = MockModule()
sys.modules['soundfile'] = MockModule()
sys.modules['torchaudio'] = MockModule()
sys.modules['noisereduce'] = MockModule()
sys.modules['demucs'] = MockModule()
sys.modules['yt_dlp'] = type('MockYtDlp', (), {'YoutubeDL': object, 'extract_info': lambda self, url, **kwargs: {}})()

try:
    from core import full_inference_program, download_music
    print("SUCCESS: Both functions imported successfully!")
    print(f"full_inference_program: {full_inference_program}")
    print(f"download_music: {download_music}")
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
except Exception as e:
    print(f"OTHER ERROR: {e}")
'''
    
    with open('/workspace/Advanced-RVC-Inference/simple_test.py', 'w') as f:
        f.write(test_content)
    
    print("Created simple_test.py - you can run this to test imports")

if __name__ == "__main__":
    print("Core Functions Verification")
    print("=" * 40)
    
    # Run alternative import test
    test_import_alternative()
    
    # Create simple test file
    create_simple_test()
    
    print("\n" + "=" * 40)
    print("If the import still fails, try:")
    print("1. Install missing dependencies: pip install torch librosa soundfile torchaudio yt-dlp")
    print("2. Run: python3 simple_test.py")
    print("3. Check if you're using the correct Python environment")
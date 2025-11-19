#!/usr/bin/env python3
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

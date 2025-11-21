#!/usr/bin/env python3
"""
Test script to verify F0 extraction works with updated pipeline.py using auto-loader
"""
import os
import sys
import numpy as np
import torch

# Add the project root to Python path
sys.path.append('/workspace/Advanced-RVC-Inference')

from programs.applio_code.rvc.infer.pipeline import Pipeline
from programs.applio_code.rvc.configs.config import Config

def test_f0_methods():
    """Test different F0 extraction methods using the updated pipeline"""
    
    print("Testing F0 extraction methods with updated pipeline...")
    
    # Create config and pipeline
    config = Config()
    pipeline = Pipeline(44100, config)
    
    # Create synthetic test audio (1 second of 440Hz sine wave)
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    test_audio = np.sin(2 * np.pi * frequency * t)
    
    print(f"Created test audio: {len(test_audio)} samples at {sample_rate}Hz")
    
    # Test parameters
    p_len = len(test_audio) // 160  # hop_length = 160
    pitch = 0
    f0_autotune = False
    filter_radius = 3
    hop_length = 160
    inp_f0 = None
    
    # Test methods
    methods_to_test = ["rmvpe", "fcpe", "yin"]
    
    for method in methods_to_test:
        try:
            print(f"\n--- Testing {method.upper()} method ---")
            
            # Test F0 extraction
            f0_coarse, f0_bak = pipeline.get_f0(
                input_audio_path="test_audio.wav",
                x=test_audio,
                p_len=p_len,
                pitch=pitch,
                f0_method=method,
                filter_radius=filter_radius,
                hop_length=hop_length,
                f0_autotune=f0_autotune,
                inp_f0=inp_f0
            )
            
            print(f"✅ {method.upper()}: SUCCESS")
            print(f"   - F0 coarse shape: {f0_coarse.shape}")
            print(f"   - F0 bak shape: {f0_bak.shape}")
            print(f"   - F0 bak range: {f0_bak.min():.2f} - {f0_bak.max():.2f} Hz")
            print(f"   - Non-zero F0 values: {np.sum(f0_bak > 0)} / {len(f0_bak)}")
            
        except Exception as e:
            print(f"❌ {method.upper()}: FAILED - {str(e)}")
            continue
    
    print("\n--- Test completed ---")

if __name__ == "__main__":
    test_f0_methods()
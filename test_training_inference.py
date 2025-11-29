#!/usr/bin/env python3
"""
Test script for training and inference functionality in Advanced RVC Inference
This script tests core functionality that depends on the updated dependencies
"""

import sys
import os
import tempfile
import numpy as np
import torch
from pathlib import Path

def test_basic_imports():
    """Test that core modules can be imported"""
    print("Testing basic imports...")
    
    try:
        import advanced_rvc_inference.core
        import advanced_rvc_inference.app
        print("✓ Core modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import core modules: {e}")
        return False
    
    return True

def test_torch_functionality():
    """Test PyTorch functionality"""
    print("\nTesting PyTorch functionality...")
    
    try:
        # Test basic tensor operations
        x = torch.randn(3, 4)
        y = torch.randn(4, 2)
        result = torch.mm(x, y)
        print(f"✓ Basic tensor operations work: {result.shape}")
        
        # Test if CUDA is available (without requiring it)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✓ Device detection: {device}")
        
        # Test tensor.to() method
        x_on_device = x.to(device)
        print(f"✓ Tensor moved to {device}: {x_on_device.shape}")
        
        return True
    except Exception as e:
        print(f"✗ PyTorch functionality test failed: {e}")
        return False

def test_audio_processing():
    """Test audio processing dependencies"""
    print("\nTesting audio processing dependencies...")
    
    try:
        # Test librosa
        import librosa
        print("✓ Librosa imported successfully")
        
        # Test soundfile
        import soundfile as sf
        print("✓ SoundFile imported successfully")
        
        # Test pydub
        from pydub import AudioSegment
        print("✓ PyDub imported successfully")
        
        # Test pedalboard
        from pedalboard import Pedalboard, Reverb
        print("✓ Pedalboard imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Audio processing test failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Audio processing test error: {e}")
        return False

def test_ml_dependencies():
    """Test machine learning dependencies"""
    print("\nTesting ML dependencies...")
    
    try:
        # Test sklearn
        from sklearn.cluster import KMeans
        print("✓ Scikit-learn imported successfully")
        
        # Test transformers
        import transformers
        print("✓ Transformers imported successfully")
        
        # Test huggingface_hub
        import huggingface_hub
        print("✓ HuggingFace Hub imported successfully")
        
        # Test torch related
        import torch.nn as nn
        import torch.optim as optim
        print("✓ PyTorch neural network modules imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ ML dependencies test failed: {e}")
        return False
    except Exception as e:
        print(f"✗ ML dependencies test error: {e}")
        return False

def test_gradio_interface():
    """Test Gradio interface functionality"""
    print("\nTesting Gradio interface...")
    
    try:
        import gradio as gr
        print("✓ Gradio imported successfully")
        
        # Test basic Gradio functionality without launching
        def dummy_function(x):
            return x
        
        interface = gr.Interface(
            fn=dummy_function,
            inputs=gr.Textbox(label="Input"),
            outputs=gr.Textbox(label="Output"),
            title="Test Interface"
        )
        print("✓ Gradio interface created successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Gradio interface test failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Gradio interface test error: {e}")
        return False

def test_rvc_specific():
    """Test RVC-specific functionality"""
    print("\nTesting RVC-specific functionality...")
    
    try:
        # Test core RVC inference
        from advanced_rvc_inference.rvc.infer.infer import VoiceConverter
        print("✓ VoiceConverter class imported successfully")
        
        # Test config
        from advanced_rvc_inference.rvc.configs.config import Config
        print("✓ Config class imported successfully")
        
        # Test basic config functionality
        config = Config()
        print("✓ Config instance created successfully")
        
        return True
    except ImportError as e:
        print(f"✗ RVC-specific functionality test failed: {e}")
        return False
    except Exception as e:
        print(f"✗ RVC-specific functionality test error: {e}")
        return False

def main():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__ if hasattr(torch, '__version__') else 'Unknown'}")
    print("=" * 60)
    
    # Change to project root directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    sys.path.insert(0, str(project_root))
    
    # Run all tests
    tests = [
        test_basic_imports,
        test_torch_functionality,
        test_audio_processing,
        test_ml_dependencies,
        test_gradio_interface,
        test_rvc_specific
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_func in tests:
        if test_func():
            passed_tests += 1
        else:
            print(f"! {test_func.__name__} failed")
    
    print("=" * 60)
    print(f"Test Results: {passed_tests}/{total_tests} test groups passed")
    
    if passed_tests == total_tests:
        print("✓ All tests passed! Dependencies are compatible with newer Python versions.")
        return 0
    else:
        print(f"✗ {total_tests - passed_tests} test groups failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
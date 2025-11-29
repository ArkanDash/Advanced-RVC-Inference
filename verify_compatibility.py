#!/usr/bin/env python3
"""
Compatibility verification script for Advanced RVC Inference
This script checks if all required dependencies can be imported properly
"""

import sys
import importlib

REQUIRED_MODULES = [
    # Core dependencies
    'wget',
    'yaml',
    'tiktoken',
    'hyperpyyaml',
    'torch',
    'torchvision', 
    'torchaudio',
    'julius',
    'omegaconf',
    'httpx',
    'contextlib2',
    'faiss',
    'audio_separator',
    'audiomentations',
    'auraloss',
    'noisereduce',
    'pystoi',
    'pyworld',
    'torchlibrosa',
    'torchmetrics',
    'torchseg',
    'transformers',
    'sklearn',
    'einops',
    'ml_collections',
    'segmentation_models_pytorch',
    'huggingface_hub',
    'librosa',
    'parselmouth',
    'soundfile',
    'pedalboard',
    'numpy',
    'numba',
    'scipy',
    'matplotlib',
    'gradio',
    'requests',
    'aiohttp',
    'pysrt',
    'yt_dlp',
    'edge_tts',
    'ffmpy',
    'ffmpeg',
    'bs4',
    'pyopencl',
    'tensorboard',
    'onnx',
    'onnxslim',
    'onnx2torch',
    'Crypto',
    'sounddevice',
    'webrtcvad',
    'audioread',
    'resampy',
    'spectralcluster',
    'rich',
    'click',
    'colorama',
    'tqdm',
    # Additional audio processing
    'pydub',
    'audio_separation',
]

def check_module(module_name):
    """Try to import a module and return success status"""
    try:
        importlib.import_module(module_name)
        print(f"✓ {module_name} - OK")
        return True
    except ImportError as e:
        print(f"✗ {module_name} - FAILED: {e}")
        return False
    except Exception as e:
        print(f"! {module_name} - ERROR: {e}")
        return False

def main():
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print("=" * 50)
    
    print("Checking required modules for Advanced RVC Inference...")
    print("=" * 50)
    
    failed_modules = []
    success_count = 0
    total_count = len(REQUIRED_MODULES)
    
    for module in REQUIRED_MODULES:
        if check_module(module):
            success_count += 1
        else:
            failed_modules.append(module)
    
    print("=" * 50)
    print(f"Results: {success_count}/{total_count} modules imported successfully")
    
    if failed_modules:
        print(f"Failed modules: {len(failed_modules)}")
        for module in failed_modules:
            print(f"  - {module}")
        return 1
    else:
        print("All modules imported successfully!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
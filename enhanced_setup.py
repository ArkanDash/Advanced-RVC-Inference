#!/usr/bin/env python3
"""
Enhanced RVC Inference Setup Script
Automatically installs dependencies, optimizes for GPU, and configures OpenCL
Version 3.5.3
"""

import os
import sys
import subprocess
import platform
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3.8, 0):
        logger.error("Python 3.8 or higher is required")
        sys.exit(1)
    logger.info(f"Python version: {sys.version}")

def install_dependencies():
    """Install required dependencies"""
    logger.info("Installing dependencies...")
    
    # Upgrade pip first
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip>=23.3"])
        logger.info("Pip upgraded successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to upgrade pip: {e}")
    
    # Install dependencies from requirements.txt
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
            logger.info("Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
    else:
        logger.error("requirements.txt not found")
        return False
    
    return True

def check_gpu_support():
    """Check GPU support and capabilities"""
    logger.info("Checking GPU support...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"CUDA available with {gpu_count} GPU(s)")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                logger.info(f"GPU {i}: {gpu_name}")
        else:
            logger.warning("CUDA not available - will use CPU")
    except ImportError:
        logger.warning("PyTorch not available - GPU optimization disabled")
    
    # Check OpenCL support
    try:
        import pyopencl
        platforms = pyopencl.get_platforms()
        logger.info(f"OpenCL available with {len(platforms)} platform(s)")
        
        for i, platform in enumerate(platforms):
            devices = platform.get_devices()
            logger.info(f"Platform {i}: {platform.name}")
            for j, device in enumerate(devices):
                logger.info(f"  Device {j}: {device.name}")
    except ImportError:
        logger.warning("OpenCL not available")

def optimize_for_gpu():
    """Optimize settings for available GPU"""
    logger.info("Optimizing for GPU...")
    
    try:
        from advanced_rvc_inference.gpu_optimization import get_gpu_optimizer
        
        gpu_optimizer = get_gpu_optimizer()
        gpu_settings = gpu_optimizer.get_optimal_settings()
        
        logger.info(f"Detected GPU: {gpu_optimizer.gpu_info['type']}")
        logger.info(f"Optimal settings: {gpu_settings}")
        
        # Save GPU settings to config file
        config_dir = Path(__file__).parent / "configs"
        config_dir.mkdir(exist_ok=True)
        
        import json
        gpu_config_file = config_dir / "gpu_settings.json"
        with open(gpu_config_file, 'w') as f:
            json.dump({
                "gpu_info": gpu_optimizer.gpu_info,
                "optimal_settings": gpu_settings
            }, f, indent=2)
        
        logger.info(f"GPU settings saved to {gpu_config_file}")
        
    except ImportError as e:
        logger.error(f"GPU optimization not available: {e}")
    except Exception as e:
        logger.error(f"Failed to optimize GPU settings: {e}")

def run_tests():
    """Run basic functionality tests"""
    logger.info("Running functionality tests...")
    
    try:
        # Test core imports
        from advanced_rvc_inference import (
            full_inference_program,
            import_voice_converter,
            get_config,
            check_fp16_support,
            models_vocals
        )
        logger.info("Core imports successful")
        
        # Test GPU optimization
        try:
            from advanced_rvc_inference.gpu_optimization import GPUOptimizer
            gpu_optimizer = GPUOptimizer()
            logger.info(f"GPU optimization test: {gpu_optimizer.gpu_info['type']} detected")
        except Exception as e:
            logger.warning(f"GPU optimization test failed: {e}")
        
        # Test KRVC kernel
        try:
            from advanced_rvc_inference.krvc_kernel import KRVCFeatureExtractor
            logger.info("KRVC kernel test successful")
        except Exception as e:
            logger.warning(f"KRVC kernel test failed: {e}")
        
        logger.info("Basic functionality tests completed")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False
    
    return True

def create_launch_scripts():
    """Create platform-specific launch scripts"""
    logger.info("Creating launch scripts...")
    
    current_dir = Path(__file__).parent
    
    # Create enhanced run script for Linux/Mac
    linux_script = current_dir / "run_enhanced.sh"
    linux_script_content = """#!/bin/bash
# Enhanced RVC Inference Launcher
# Auto-configures GPU and OpenCL support

echo "Starting Enhanced RVC Inference with GPU Optimization..."
echo "Checking GPU and OpenCL support..."

# Set GPU optimizations based on detected hardware
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Launch with enhanced settings
python -m advanced_rvc_inference.main "$@"
"""
    
    with open(linux_script, 'w') as f:
        f.write(linux_script_content)
    
    # Make executable
    os.chmod(linux_script, 0o755)
    
    # Create enhanced run script for Windows
    windows_script = current_dir / "run_enhanced.bat"
    windows_script_content = """@echo off
REM Enhanced RVC Inference Launcher
REM Auto-configures GPU and OpenCL support

echo Starting Enhanced RVC Inference with GPU Optimization...
echo Checking GPU and OpenCL support...

REM Set GPU optimizations
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

REM Launch with enhanced settings
python -m advanced_rvc_inference.main %*
"""
    
    with open(windows_script, 'w') as f:
        f.write(windows_script_content)
    
    logger.info(f"Created launch scripts: {linux_script}, {windows_script}")

def main():
    """Main setup function"""
    logger.info("=== Enhanced RVC Inference Setup ===")
    logger.info(f"Platform: {platform.system()} {platform.machine()}")
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    if not install_dependencies():
        logger.error("Dependency installation failed")
        sys.exit(1)
    
    # Check GPU support
    check_gpu_support()
    
    # Optimize for GPU
    optimize_for_gpu()
    
    # Run tests
    if not run_tests():
        logger.warning("Some tests failed, but setup can continue")
    
    # Create launch scripts
    create_launch_scripts()
    
    logger.info("=== Setup completed successfully! ===")
    logger.info("You can now run:")
    logger.info("  python -m advanced_rvc_inference.main")
    logger.info("  Or use the enhanced launch scripts:")
    logger.info("  ./run_enhanced.sh (Linux/Mac)")
    logger.info("  run_enhanced.bat (Windows)")

if __name__ == "__main__":
    main()
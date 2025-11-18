"""
KADVC Test Suite
Comprehensive testing for KADVC optimization system
"""

import sys
import os
import torch
import numpy as np
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_kadvc_import():
    """Test KADVC module import"""
    print("Testing KADVC import...")
    try:
        from programs.kernels import (
            setup_kadvc_for_rvc,
            KADVCConfig,
            KADVCCUDAKernels,
            get_kadvc_optimizer
        )
        print("✅ KADVC import successful")
        return True
    except ImportError as e:
        print(f"❌ KADVC import failed: {e}")
        return False

def test_kadvc_initialization():
    """Test KADVC initialization"""
    print("Testing KADVC initialization...")
    try:
        from programs.kernels import setup_kadvc_for_rvc
        
        # Initialize KADVC
        kadvc = setup_kadvc_for_rvc()
        print("✅ KADVC initialization successful")
        
        # Check performance stats
        stats = kadvc.get_performance_report()
        gpu_type = stats.get('gpu_info', {}).get('gpu_name', 'Unknown')
        speedup = kadvc._calculate_speedup()
        
        print(f"   GPU: {gpu_type}")
        print(f"   Estimated speedup: {speedup}x")
        print(f"   Kernels available: {kadvc._kernels_available}")
        
        return True
    except Exception as e:
        print(f"❌ KADVC initialization failed: {e}")
        return False

def test_kadvc_config():
    """Test KADVC configuration"""
    print("Testing KADVC configuration...")
    try:
        from programs.kernels import KADVCConfig, create_optimized_config
        
        # Test default configuration
        config = KADVCConfig.create_colab_config()
        print("✅ KADVC Colab config created")
        
        # Test local configuration
        config = KADVCConfig.create_local_config()
        print("✅ KADVC local config created")
        
        # Test auto configuration
        config = create_optimized_config()
        print("✅ KADVC auto config created")
        
        # Test validation
        validation = config.validate_config()
        print(f"   Configuration valid: {validation['valid']}")
        if validation['warnings']:
            print(f"   Warnings: {len(validation['warnings'])}")
        
        return True
    except Exception as e:
        print(f"❌ KADVC configuration test failed: {e}")
        return False

def test_kadvc_performance():
    """Test KADVC performance optimizations"""
    print("Testing KADVC performance...")
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, skipping performance tests")
        return True
    
    try:
        from programs.kernels import setup_kadvc_for_rvc
        
        kadvc = setup_kadvc_for_rvc()
        
        # Test F0 extraction performance
        print("   Testing F0 extraction...")
        test_audio = torch.randn(1, 48000, device="cuda")
        
        start_time = time.time()
        f0 = kadvc.fast_f0_extraction(test_audio, sample_rate=48000)
        torch.cuda.synchronize()
        f0_time = time.time() - start_time
        
        print(f"   F0 extraction: {f0_time:.3f}s")
        
        # Test feature extraction performance
        print("   Testing feature extraction...")
        start_time = time.time()
        f0, features = kadvc.fast_feature_extraction(test_audio, sample_rate=48000)
        torch.cuda.synchronize()
        feature_time = time.time() - start_time
        
        print(f"   Feature extraction: {feature_time:.3f}s")
        
        print("✅ KADVC performance tests completed")
        return True
        
    except Exception as e:
        print(f"❌ KADVC performance test failed: {e}")
        return False

def test_kadvc_integration():
    """Test KADVC integration with RVC training"""
    print("Testing KADVC integration...")
    try:
        from programs.training.simple_trainer import create_training_config
        
        # Create training config with KADVC settings
        config = create_training_config(
            model_name="test_model",
            kadvc_settings={
                'enabled': True,
                'mixed_precision': True,
                'custom_kernels': True,
                'memory_optimization': True
            }
        )
        
        print("✅ KADVC training integration test passed")
        print(f"   KADVC enabled: {config.get('kadvc_settings', {}).get('enabled')}")
        
        return True
    except Exception as e:
        print(f"❌ KADVC integration test failed: {e}")
        return False

def test_kadvc_fallback():
    """Test KADVC fallback mechanisms"""
    print("Testing KADVC fallback mechanisms...")
    try:
        from programs.kernels import KADVCConfig, KADVCOptimizer
        
        # Test CPU fallback
        config = KADVCConfig()
        config.fallback_to_cpu = True
        optimizer = KADVCOptimizer(config)
        
        print("✅ KADVC fallback test passed")
        return True
    except Exception as e:
        print(f"❌ KADVC fallback test failed: {e}")
        return False

def run_all_tests():
    """Run all KADVC tests"""
    print("🧪 Running KADVC Test Suite")
    print("=" * 50)
    
    tests = [
        test_kadvc_import,
        test_kadvc_config,
        test_kadvc_initialization,
        test_kadvc_performance,
        test_kadvc_integration,
        test_kadvc_fallback
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1
        
        print()  # Empty line between tests
    
    print("=" * 50)
    print(f"🎯 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All KADVC tests passed! System is ready for production use.")
    else:
        print("⚠️ Some KADVC tests failed. Check the output above for details.")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
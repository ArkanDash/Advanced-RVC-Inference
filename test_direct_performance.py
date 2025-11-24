"""
Direct Performance Integration Testing
======================================

Tests performance optimizations directly without importing the full project.

Author: MiniMax Agent
Date: 2025-11-24
Version: 1.0.0
"""

import torch
import torch.nn as nn
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_torchfx_direct():
    """Test TorchFX directly without project imports."""
    print("\n🔧 Testing TorchFX Direct Integration...")
    
    try:
        import torchfx
        import torchfx.filter as filter_module
        
        print("✓ TorchFX imported successfully")
        
        # Test filter creation
        try:
            # Create a simple filter using TorchFX
            lowpass = filter_module.Butterworth(
                kind='lowpass',
                order=4,
                cutoff_freq=8000,
                sample_rate=44100
            )
            print("✓ TorchFX Butterworth filter created")
            
            # Test with sample audio
            test_audio = torch.randn(2, 1, 44100)  # 2 samples, 1 channel, 1 second
            
            # Apply filter
            processed = lowpass(test_audio)
            print(f"✓ TorchFX processing: {test_audio.shape} → {processed.shape}")
            
            return True, "TorchFX working with Butterworth filters"
            
        except Exception as e:
            print(f"⚠ TorchFX filter creation failed: {e}")
            # Test alternative approach
            try:
                # Try FIR filter
                fir_filter = filter_module.FIR(
                    taps=101,
                    cutoff_freq=8000,
                    sample_rate=44100
                )
                print("✓ TorchFX FIR filter created")
                
                test_audio = torch.randn(2, 1, 44100)
                processed = fir_filter(test_audio)
                print(f"✓ TorchFX FIR processing: {test_audio.shape} → {processed.shape}")
                
                return True, "TorchFX working with FIR filters"
                
            except Exception as e2:
                print(f"✗ Both filter types failed: {e}, {e2}")
                return False, f"TorchFX filter operations failed: {e}, {e2}"
        
    except ImportError:
        return False, "TorchFX not installed"
    except Exception as e:
        return False, f"TorchFX error: {e}"

def test_audiomentations_direct():
    """Test torch-audiomentations directly without project imports."""
    print("\n🎵 Testing torch-audiomentations Direct Integration...")
    
    try:
        import torch_audiomentations as ta
        
        print("✓ torch-audiomentations imported successfully")
        
        # Test transform creation
        transforms_to_test = []
        
        try:
            # Test AddColoredNoise
            noise_transform = ta.AddColoredNoise(min_snr_in_db=3, max_snr_in_db=15, p=0.5)
            transforms_to_test.append(('AddColoredNoise', noise_transform))
            print("✓ AddColoredNoise created")
        except Exception as e:
            print(f"⚠ AddColoredNoise failed: {e}")
        
        try:
            # Test PitchShift
            pitch_transform = ta.PitchShift(min_transpose_semitones=-2, max_transpose_semitones=2, p=0.5)
            transforms_to_test.append(('PitchShift', pitch_transform))
            print("✓ PitchShift created")
        except Exception as e:
            print(f"⚠ PitchShift failed: {e}")
        
        try:
            # Test Gain
            gain_transform = ta.Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.5)
            transforms_to_test.append(('Gain', gain_transform))
            print("✓ Gain created")
        except Exception as e:
            print(f"⚠ Gain failed: {e}")
        
        try:
            # Test PeakNormalization
            norm_transform = ta.PeakNormalization(p=0.5)
            transforms_to_test.append(('PeakNormalization', norm_transform))
            print("✓ PeakNormalization created")
        except Exception as e:
            print(f"⚠ PeakNormalization failed: {e}")
        
        if not transforms_to_test:
            return False, "No transforms could be created"
        
        # Test with sample audio
        test_audio = torch.randn(2, 1, 44100)  # 2 samples, 1 channel, 1 second
        
        successful_transforms = 0
        for name, transform in transforms_to_test:
            try:
                processed = transform(test_audio)
                print(f"✓ {name} processing: {test_audio.shape} → {processed.shape}")
                successful_transforms += 1
            except Exception as e:
                print(f"⚠ {name} processing failed: {e}")
        
        if successful_transforms > 0:
            return True, f"torch-audiomentations working with {successful_transforms}/{len(transforms_to_test)} transforms"
        else:
            return False, "All transforms failed"
        
    except ImportError:
        return False, "torch-audiomentations not installed"
    except Exception as e:
        return False, f"torch-audiomentations error: {e}"

def test_torch_compile_direct():
    """Test torch.compile directly without project imports."""
    print("\n⚡ Testing torch.compile Direct Integration...")
    
    try:
        # Test torch.compile availability
        @torch.compile
        def test_function(x):
            return torch.relu(x * 2.0)
        
        test_input = torch.randn(5, 5)
        result = test_function(test_input)
        print("✓ torch.compile decorator working")
        
        # Test model compilation
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(100, 64)
                self.linear2 = nn.Linear(64, 32)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.relu(self.linear1(x))
                x = self.linear2(x)
                return x
        
        model = TestModel()
        example_input = torch.randn(1, 100)
        
        print("✓ Test model created")
        
        # Test different compilation modes
        modes_to_test = ['default', 'reduce-overhead']
        working_modes = []
        
        for mode in modes_to_test:
            try:
                compiled_model = torch.compile(model, mode=mode)
                result = compiled_model(example_input)
                print(f"✓ Model compiled with mode '{mode}': {example_input.shape} → {result.shape}")
                working_modes.append(mode)
            except Exception as e:
                print(f"⚠ Mode '{mode}' failed: {e}")
        
        if working_modes:
            # Performance comparison
            model.eval()
            compiled_model = torch.compile(model, mode=working_modes[0])
            compiled_model.eval()
            
            # Benchmark original model
            times_original = []
            for _ in range(30):
                start = time.time()
                with torch.no_grad():
                    _ = model(example_input)
                times_original.append(time.time() - start)
            
            # Benchmark compiled model (warm up first)
            for _ in range(10):
                with torch.no_grad():
                    _ = compiled_model(example_input)
            
            times_compiled = []
            for _ in range(30):
                start = time.time()
                with torch.no_grad():
                    _ = compiled_model(example_input)
                times_compiled.append(time.time() - start)
            
            avg_original = sum(times_original) / len(times_original)
            avg_compiled = sum(times_compiled) / len(times_compiled)
            speedup = avg_original / avg_compiled if avg_compiled > 0 else 1.0
            
            print(f"✓ Performance comparison:")
            print(f"  Original: {avg_original*1000:.2f}ms")
            print(f"  Compiled: {avg_compiled*1000:.2f}ms")
            print(f"  Speedup: {speedup:.2f}x")
            
            return True, f"torch.compile working with {len(working_modes)}/{len(modes_to_test)} modes, {speedup:.2f}x speedup"
        else:
            return False, "No compilation modes worked"
        
    except Exception as e:
        return False, f"torch.compile error: {e}"

def run_direct_performance_tests():
    """Run direct performance tests for all optimization libraries."""
    print("="*80)
    print("ADVANCED RVC V4.0.0 DIRECT PERFORMANCE TESTING")
    print("="*80)
    
    # Environment check
    print("\n🔍 Environment Check:")
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    print(f"✓ Device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    
    # Test each component
    results = {}
    
    results['torchfx'] = test_torchfx_direct()
    results['audiomentations'] = test_audiomentations_direct()
    results['compile'] = test_torch_compile_direct()
    
    # Summary
    print("\n" + "="*80)
    print("DIRECT PERFORMANCE TEST SUMMARY")
    print("="*80)
    
    total_components = len(results)
    working_components = sum(1 for success, _ in results.values() if success)
    
    for component, (success, message) in results.items():
        status = "✓ WORKING" if success else "✗ FAILED"
        print(f"{component.replace('_', ' ').title():20}: {status} - {message}")
    
    print(f"\nOverall Status: {working_components}/{total_components} components working")
    
    success_rate = (working_components / total_components) * 100
    print(f"Success Rate: {success_rate:.1f}%")
    
    # Overall assessment
    if success_rate >= 75:
        print("\n🎉 EXCELLENT! Performance optimization integration is ready.")
        print("   Advanced RVC V4.0.0 can provide significant performance benefits.")
    elif success_rate >= 50:
        print("\n✅ GOOD! Core performance optimizations are working.")
        print("   Some advanced features may be limited, but benefits are available.")
    elif success_rate >= 25:
        print("\n⚠️  MODERATE! Some performance optimizations are working.")
        print("   Limited benefits available, consider troubleshooting remaining issues.")
    else:
        print("\n❌ POOR! Most performance optimizations are not working.")
        print("   Significant performance improvements may not be available.")
    
    # Specific recommendations
    print("\n💡 RECOMMENDATIONS:")
    if not results['torchfx'][0]:
        print("- Install TorchFX: pip install torchfx>=0.2.0")
    if not results['audiomentations'][0]:
        print("- Install torch-audiomentations: pip install torch-audiomentations>=0.12.0")
    if not results['compile'][0]:
        print("- Upgrade PyTorch: pip install torch>=2.9.1")
    
    if all(success for success, _ in results.values()):
        print("- All optimizations working! Consider using them for maximum performance.")
        print("- Test with your specific use cases for optimal configuration.")
    
    print("\n" + "="*80)
    
    return results

if __name__ == "__main__":
    run_direct_performance_tests()
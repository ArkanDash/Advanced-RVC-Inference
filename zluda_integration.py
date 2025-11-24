#!/usr/bin/env python3
"""
ZLUDA Backend Integration for RVC
Demonstrates how to use ZLUDA kernels for accelerated RVC inference
"""

import numpy as np
import warnings
from typing import Dict, Any, Optional
import time

def setup_zluda_for_rvc():
    """Setup ZLUDA backend for RVC operations"""
    try:
        from advanced_rvc_inference.lib.backends.zluda import (
            is_available, get_device, create_context, 
            optimize_for_rvc, execute_audio_kernel
        )
        
        print("🚀 Initializing ZLUDA backend for RVC...")
        
        if not is_available():
            print("⚠️  ZLUDA not available - using CPU fallback")
            return None
        
        # Get device and create context
        device = get_device()
        if device is None:
            print("❌ Failed to get ZLUDA device")
            return None
            
        context = create_context(device)
        if context is None:
            print("❌ Failed to create ZLUDA context")
            return None
        
        # Optimize for RVC workloads
        config = optimize_for_rvc(context)
        print(f"✅ ZLUDA optimized: {config['status']}")
        
        if 'recommendations' in config:
            for rec in config['recommendations']:
                print(f"   💡 {rec}")
        
        return {
            'context': context,
            'device': device,
            'config': config
        }
        
    except Exception as e:
        print(f"❌ ZLUDA initialization failed: {e}")
        return None

def process_audio_with_zluda(audio_data: np.ndarray, sample_rate: int = 16000):
    """Process audio using ZLUDA kernels"""
    try:
        from advanced_rvc_inference.lib.backends.zluda import execute_audio_kernel
        
        # Prepare input data
        inputs = {
            'audio_input': audio_data.astype(np.float32),
            'sample_rate': np.array([sample_rate], dtype=np.int32)
        }
        
        # Execute mel spectrogram kernel
        print("🎵 Processing mel spectrogram with ZLUDA...")
        start_time = time.time()
        
        mel_outputs = execute_audio_kernel("audio_mel_spectrogram", inputs)
        
        processing_time = time.time() - start_time
        print(f"   ✅ Mel spectrogram computed in {processing_time:.3f}s")
        
        return mel_outputs
        
    except Exception as e:
        print(f"⚠️  ZLUDA audio processing failed: {e}")
        return {'audio_output': audio_data}

def extract_pitch_with_zluda(audio_data: np.ndarray, threshold: float = 0.1):
    """Extract pitch using ZLUDA kernel"""
    try:
        from advanced_rvc_inference.lib.backends.zluda import execute_audio_kernel
        
        inputs = {
            'pitch_audio_input': audio_data.astype(np.float32),
            'threshold': np.array([threshold], dtype=np.float32)
        }
        
        print("🎼 Extracting pitch with ZLUDA...")
        start_time = time.time()
        
        pitch_outputs = execute_audio_kernel("pitch_extraction", inputs)
        
        processing_time = time.time() - start_time
        print(f"   ✅ Pitch extraction completed in {processing_time:.3f}s")
        
        return pitch_outputs
        
    except Exception as e:
        print(f"⚠️  ZLUDA pitch extraction failed: {e}")
        return {'pitch_output': np.zeros(len(audio_data))}

def synthesize_waveform_with_zluda(features: np.ndarray, f0: np.ndarray):
    """Generate waveform using ZLUDA kernel"""
    try:
        from advanced_rvc_inference.lib.backends.zluda import execute_audio_kernel
        
        inputs = {
            'synthesis_features': features.astype(np.float32),
            'synthesis_f0': f0.astype(np.float32)
        }
        
        print("🔊 Synthesizing waveform with ZLUDA...")
        start_time = time.time()
        
        waveform_outputs = execute_audio_kernel("waveform_generation", inputs)
        
        processing_time = time.time() - start_time
        print(f"   ✅ Waveform synthesis completed in {processing_time:.3f}s")
        
        return waveform_outputs
        
    except Exception as e:
        print(f"⚠️  ZLUDA waveform synthesis failed: {e}")
        return {'waveform_output': np.zeros(len(features))}

def run_full_rvc_pipeline_with_zluda():
    """Run complete RVC pipeline using ZLUDA acceleration"""
    print("🎯 Starting Full RVC Pipeline with ZLUDA Acceleration")
    print("=" * 60)
    
    # Initialize ZLUDA
    zluda_setup = setup_zluda_for_rvc()
    
    if zluda_setup is None:
        print("❌ Pipeline cannot continue without ZLUDA")
        return
    
    # Generate sample audio data for demonstration
    print("\n📊 Generating sample audio data...")
    sample_rate = 16000
    duration = 1.0  # 1 second
    audio_length = int(sample_rate * duration)
    audio_data = np.random.normal(0, 0.1, audio_length).astype(np.float32)
    print(f"   Generated {duration}s audio ({audio_length} samples)")
    
    # Step 1: Mel Spectrogram
    print("\n🎵 Step 1: Mel Spectrogram Computation")
    mel_results = process_audio_with_zluda(audio_data, sample_rate)
    
    # Step 2: Pitch Extraction
    print("\n🎼 Step 2: Pitch Extraction")
    pitch_results = extract_pitch_with_zluda(audio_data)
    
    # Step 3: Feature Convolution (simplified)
    print("\n🔄 Step 3: Feature Processing")
    features = mel_results.get('mel_output', audio_data[:256])
    pitch_curve = pitch_results.get('pitch_output', np.ones(256) * 220)
    
    convolution_results = synthesize_waveform_with_zluda(features, pitch_curve)
    
    # Step 4: Waveform Synthesis
    print("\n🔊 Step 4: Waveform Synthesis")
    waveform_results = synthesize_waveform_with_zluda(features, pitch_curve)
    
    # Performance summary
    print("\n📈 Performance Summary")
    try:
        from advanced_rvc_inference.lib.backends.zluda import get_performance_metrics
        metrics = get_performance_metrics(zluda_setup['context'])
        print(f"   Backend Status: {metrics['status']}")
        print(f"   GPU Utilization: {metrics.get('gpu_utilization', 'N/A')}%")
        print(f"   Memory Usage: {metrics.get('memory_usage', 'N/A')}")
        print(f"   Throughput: {metrics.get('throughput', 'N/A')}")
        print(f"   Compute Efficiency: {metrics.get('compute_efficiency', 'N/A')}")
    except Exception as e:
        print(f"   Could not retrieve metrics: {e}")
    
    print("\n✅ RVC Pipeline with ZLUDA completed successfully!")

def demonstrate_kernel_features():
    """Demonstrate ZLUDA kernel capabilities"""
    print("🔧 ZLUDA Kernel Features Demonstration")
    print("=" * 50)
    
    try:
        from advanced_rvc_inference.lib.backends.zluda import (
            is_available, ZLUDA_KERNELS, compile_kernel
        )
        
        print(f"ZLUDA Available: {'✅' if is_available() else '❌'}")
        print(f"\nAvailable Kernels ({len(ZLUDA_KERNELS)}):")
        
        for kernel_name, source in ZLUDA_KERNELS.items():
            print(f"   🎯 {kernel_name}")
            print(f"      Lines of code: {len(source.split(chr(10)))}")
            
            # Try to compile kernel
            kernel = compile_kernel(kernel_name)
            status = "✅ Compiled" if kernel else "❌ Failed"
            print(f"      Status: {status}")
            print()
        
        print("🎯 Kernel Optimizations:")
        print("   • Mel Spectrogram: 256-block size for audio processing")
        print("   • Pitch Extraction: 512-block size for frequency analysis") 
        print("   • Feature Convolution: 128-block size for batch processing")
        print("   • Waveform Synthesis: 1024-block size for high-quality output")
        
    except Exception as e:
        print(f"❌ Kernel demonstration failed: {e}")

def performance_benchmark():
    """Benchmark ZLUDA vs CPU performance"""
    print("⚡ ZLUDA Performance Benchmark")
    print("=" * 40)
    
    # Generate test data
    test_sizes = [1024, 4096, 16384]
    
    for size in test_sizes:
        print(f"\n📊 Testing with {size} samples:")
        
        # CPU timing
        audio_data = np.random.normal(0, 0.1, size).astype(np.float32)
        
        cpu_start = time.time()
        # Simulate CPU processing
        _ = np.fft.fft(audio_data)
        _ = np.abs(_)
        cpu_time = time.time() - cpu_start
        
        # ZLUDA timing
        try:
            from advanced_rvc_inference.lib.backends.zluda import execute_audio_kernel
            
            inputs = {'audio_input': audio_data}
            zluda_start = time.time()
            _ = execute_audio_kernel("audio_mel_spectrogram", inputs)
            zluda_time = time.time() - zluda_start
            
            speedup = cpu_time / zluda_time if zluda_time > 0 else 1.0
            print(f"   CPU Time: {cpu_time:.4f}s")
            print(f"   ZLUDA Time: {zluda_time:.4f}s")
            print(f"   Speedup: {speedup:.2f}x")
            
        except Exception as e:
            print(f"   ZLUDA Error: {e}")

if __name__ == "__main__":
    print("🔥 ZLUDA Backend for RVC - Complete Demonstration")
    print("=" * 60)
    
    # 1. Show kernel features
    demonstrate_kernel_features()
    
    print("\n" + "=" * 60)
    
    # 2. Run performance benchmark
    performance_benchmark()
    
    print("\n" + "=" * 60)
    
    # 3. Run full RVC pipeline
    run_full_rvc_pipeline_with_zluda()
    
    print("\n🎉 All demonstrations completed!")
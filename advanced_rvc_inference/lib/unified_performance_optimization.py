"""
Unified Performance Optimization Integration
===========================================

This module provides a unified interface for all performance optimization libraries:
- TorchFX for high-performance DSP operations
- torch-audiomentations for GPU-accelerated audio augmentation  
- torch.compile for JIT compilation optimization

Combines all optimizations for maximum performance in RVC inference and training.

Author: MiniMax Agent
Date: 2025-11-24
Version: 1.0.0
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, Union, Dict, Any, List, Tuple
import time

from .torchfx_integration import TorchFXProcessor, TorchFXMelSpectrogramProcessor
from .torch_audiomentations_integration import TorchAudioMentationsProcessor, RVCAudioAugmenter
from .torch_compile_optimization import TorchCompileOptimizer, compile_function

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedPerformanceOptimizer:
    """
    Unified performance optimizer combining TorchFX, torch-audiomentations, and torch.compile.
    
    Provides:
    - High-performance audio processing with TorchFX
    - GPU-accelerated augmentation with torch-audiomentations
    - JIT compilation optimization with torch.compile
    - Seamless integration and coordination between all systems
    """
    
    def __init__(self, 
                 device: Optional[Union[str, torch.device]] = None,
                 enable_torchfx: bool = True,
                 enable_augmentation: bool = True,
                 enable_compilation: bool = True,
                 sample_rate: int = 44100):
        """
        Initialize unified performance optimizer.
        
        Args:
            device: Target device for all operations
            enable_torchfx: Enable TorchFX DSP processing
            enable_augmentation: Enable torch-audiomentations augmentation
            enable_compilation: Enable torch.compile optimization
            sample_rate: Audio sample rate
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.sample_rate = sample_rate
        
        # Enable/disable individual components
        self.enable_torchfx = enable_torchfx
        self.enable_augmentation = enable_augmentation
        self.enable_compilation = enable_compilation
        
        # Initialize components
        self.torchfx_processor = None
        self.audio_augmenter = None
        self.compile_optimizer = None
        
        # Initialize based on availability and settings
        self._initialize_components()
        
        # Performance statistics
        self.performance_stats = {
            'torchfx_processed_samples': 0,
            'augmented_samples': 0,
            'compiled_models': 0,
            'total_optimization_time': 0.0
        }
        
        logger.info("Unified Performance Optimizer initialized")
    
    def _initialize_components(self):
        """Initialize all enabled components."""
        try:
            if self.enable_torchfx:
                self.torchfx_processor = TorchFXProcessor(self.device)
                if self.torchfx_processor._is_torchfx_available:
                    logger.info("TorchFX DSP processor initialized")
                else:
                    logger.warning("TorchFX not available, DSP optimization disabled")
                    self.enable_torchfx = False
            
            if self.enable_augmentation:
                self.audio_augmenter = RVCAudioAugmenter(self.sample_rate, self.device)
                if self.audio_augmenter.processor._is_available:
                    logger.info("Torch-audiomentations processor initialized")
                else:
                    logger.warning("Torch-audiomentations not available, augmentation disabled")
                    self.enable_augmentation = False
            
            if self.enable_compilation:
                self.compile_optimizer = TorchCompileOptimizer(self.device)
                if self.compile_optimizer.compile_available:
                    logger.info("torch.compile optimizer initialized")
                else:
                    logger.warning("torch.compile not available, JIT optimization disabled")
                    self.enable_compilation = False
                    
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
    
    def process_audio_batch(self, 
                          audio_batch: torch.Tensor,
                          dsp_filters: Optional[List[str]] = None,
                          augmentation_preset: Optional[str] = None,
                          augmentation_probability: float = 0.5) -> torch.Tensor:
        """
        Process audio batch using all available optimizations.
        
        Args:
            audio_batch: Input audio tensor (batch_size, samples) or (batch_size, channels, samples)
            dsp_filters: List of DSP filters to apply
            augmentation_preset: Augmentation preset to use
            augmentation_probability: Probability of applying augmentation
            
        Returns:
            Optimally processed audio tensor
        """
        start_time = time.time()
        processed_audio = audio_batch.to(self.device)
        
        # Apply TorchFX DSP processing
        if self.enable_torchfx and self.torchfx_processor:
            try:
                dsp_pipeline = self.torchfx_processor.create_audio_pipeline(dsp_filters)
                processed_audio = self.torchfx_processor.process_audio_batch(
                    processed_audio, self.sample_rate, dsp_pipeline
                )
                self.performance_stats['torchfx_processed_samples'] += processed_audio.shape[0]
                logger.debug("Applied TorchFX DSP processing")
            except Exception as e:
                logger.warning(f"TorchFX processing failed: {e}")
        
        # Apply audio augmentation
        if self.enable_augmentation and self.audio_augmenter and augmentation_preset:
            try:
                if augmentation_preset in self.audio_augmenter.presets:
                    processed_audio = self.audio_augmenter.apply_preset(
                        processed_audio, augmentation_preset
                    )
                    self.performance_stats['augmented_samples'] += processed_audio.shape[0]
                    logger.debug(f"Applied augmentation preset: {augmentation_preset}")
            except Exception as e:
                logger.warning(f"Augmentation failed: {e}")
        
        # Record processing time
        processing_time = time.time() - start_time
        self.performance_stats['total_optimization_time'] += processing_time
        
        return processed_audio
    
    def optimize_model(self, 
                      model: nn.Module,
                      example_inputs: Union[torch.Tensor, List[torch.Tensor]],
                      optimization_level: str = 'balanced') -> nn.Module:
        """
        Optimize model using torch.compile with optimal settings.
        
        Args:
            model: Model to optimize
            example_inputs: Example inputs for compilation
            optimization_level: 'speed', 'memory', 'balanced'
            
        Returns:
            Optimized model
        """
        if not (self.enable_compilation and self.compile_optimizer):
            logger.warning("Compilation optimization not available")
            return model
        
        try:
            # Determine compilation mode based on optimization level
            if optimization_level == 'speed':
                mode = 'max-autotune'
                fullgraph = True
            elif optimization_level == 'memory':
                mode = 'reduce-overhead'
                fullgraph = False
            else:  # balanced
                mode = 'default'
                fullgraph = True
            
            start_time = time.time()
            
            optimized_model = self.compile_optimizer.compile_model(
                model,
                example_inputs,
                mode=mode,
                fullgraph=fullgraph,
                name=f"optimized_{optimization_level}"
            )
            
            optimization_time = time.time() - start_time
            self.performance_stats['compiled_models'] += 1
            
            logger.info(f"Model optimized in {optimization_time:.2f}s with {mode} mode")
            return optimized_model
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            return model
    
    def get_optimized_pipeline(self, 
                             input_shape: Tuple[int, ...] = (1, 80, 100),
                             dsp_config: Optional[Dict] = None,
                             augmentation_config: Optional[Dict] = None) -> 'OptimizedRVCPipeline':
        """
        Get complete optimized RVC processing pipeline.
        
        Args:
            input_shape: Expected input shape
            dsp_config: DSP processing configuration
            augmentation_config: Augmentation configuration
            
        Returns:
            Optimized pipeline instance
        """
        return OptimizedRVCPipeline(
            torchfx_processor=self.torchfx_processor if self.enable_torchfx else None,
            audio_augmenter=self.audio_augmenter if self.enable_augmentation else None,
            compile_optimizer=self.compile_optimizer if self.enable_compilation else None,
            input_shape=input_shape,
            dsp_config=dsp_config or {},
            augmentation_config=augmentation_config or {},
            device=self.device
        )
    
    def benchmark_complete_optimization(self, 
                                      test_audio: torch.Tensor,
                                      model: Optional[nn.Module] = None,
                                      iterations: int = 100) -> Dict[str, Any]:
        """
        Benchmark complete optimization pipeline.
        
        Args:
            test_audio: Test audio tensor
            model: Optional model to include in benchmark
            iterations: Number of benchmark iterations
            
        Returns:
            Comprehensive performance statistics
        """
        logger.info("Running complete optimization benchmark...")
        
        # Baseline performance (no optimization)
        baseline_times = []
        for _ in range(min(20, iterations)):  # Fewer iterations for baseline
            start_time = time.time()
            if model:
                with torch.no_grad():
                    _ = model(test_audio)
            _ = test_audio  # Just copy operation
            baseline_times.append(time.time() - start_time)
        
        baseline_stats = {
            'mean_time': sum(baseline_times) / len(baseline_times),
            'min_time': min(baseline_times),
            'max_time': max(baseline_times)
        }
        
        # Optimized performance
        optimized_times = []
        for _ in range(iterations):
            start_time = time.time()
            processed = self.process_audio_batch(
                test_audio, 
                dsp_filters=['lowpass', 'highpass', 'normalize'],
                augmentation_preset='voice_preservation',
                augmentation_probability=0.0  # Disable augmentation for fair comparison
            )
            if model:
                with torch.no_grad():
                    _ = model(processed)
            optimized_times.append(time.time() - start_time)
        
        optimized_stats = {
            'mean_time': sum(optimized_times) / len(optimized_times),
            'min_time': min(optimized_times),
            'max_time': max(optimized_times)
        }
        
        # Calculate improvements
        speedup = baseline_stats['mean_time'] / optimized_stats['mean_time']
        
        benchmark_results = {
            'baseline': baseline_stats,
            'optimized': optimized_stats,
            'speedup': speedup,
            'performance_stats': self.performance_stats.copy(),
            'available_features': {
                'torchfx': self.enable_torchfx and self.torchfx_processor is not None,
                'augmentation': self.enable_augmentation and self.audio_augmenter is not None,
                'compilation': self.enable_compilation and self.compile_optimizer is not None
            }
        }
        
        return benchmark_results
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system and optimization information."""
        info = {
            'device': str(self.device),
            'sample_rate': self.sample_rate,
            'enabled_features': {
                'torchfx_dsp': self.enable_torchfx,
                'audio_augmentation': self.enable_augmentation,
                'jit_compilation': self.enable_compilation
            },
            'performance_stats': self.performance_stats.copy()
        }
        
        # Add component-specific info
        if self.torchfx_processor:
            info['torchfx_info'] = self.torchfx_processor.get_processing_info()
        
        if self.audio_augmenter:
            info['augmentation_info'] = {
                'presets': list(self.audio_augmenter.presets.keys()),
                'available_transforms': self.audio_augmenter.processor.get_available_transforms()
            }
        
        if self.compile_optimizer:
            info['compilation_info'] = self.compile_optimizer.get_optimization_info()
        
        return info
    
    def cleanup(self):
        """Clean up all optimization resources."""
        if self.compile_optimizer:
            self.compile_optimizer.cleanup()
        
        # Clear performance stats
        self.performance_stats = {
            'torchfx_processed_samples': 0,
            'augmented_samples': 0,
            'compiled_models': 0,
            'total_optimization_time': 0.0
        }
        
        logger.info("Unified Performance Optimizer cleanup completed")


class OptimizedRVCPipeline:
    """
    Complete optimized RVC processing pipeline.
    """
    
    def __init__(self, 
                 torchfx_processor: Optional[TorchFXProcessor],
                 audio_augmenter: Optional[RVCAudioAugmenter],
                 compile_optimizer: Optional[TorchCompileOptimizer],
                 input_shape: Tuple[int, ...],
                 dsp_config: Dict,
                 augmentation_config: Dict,
                 device: Union[str, torch.device]):
        """
        Initialize optimized RVC pipeline.
        """
        self.torchfx_processor = torchfx_processor
        self.audio_augmenter = audio_augmenter
        self.compile_optimizer = compile_optimizer
        self.input_shape = input_shape
        self.dsp_config = dsp_config
        self.augmentation_config = augmentation_config
        self.device = device
        
        # Optimized components
        self.processors = []
        self._build_pipeline()
    
    def _build_pipeline(self):
        """Build the complete processing pipeline."""
        logger.info("Building optimized RVC pipeline...")
        
        # Add TorchFX DSP processing
        if self.torchfx_processor:
            dsp_filters = self.dsp_config.get('filters', ['lowpass', 'highpass', 'normalize'])
            dsp_pipeline = self.torchfx_processor.create_audio_pipeline(dsp_filters)
            self.processors.append(('dsp', dsp_pipeline))
        
        # Add augmentation (training mode)
        if self.audio_augmenter and self.augmentation_config.get('enable_training_augmentation', False):
            self.processors.append(('augmentation', self.audio_augmenter))
        
        logger.info(f"Pipeline built with {len(self.processors)} components")
    
    def process(self, 
               audio: torch.Tensor,
               mode: str = 'inference') -> torch.Tensor:
        """
        Process audio through the optimized pipeline.
        
        Args:
            audio: Input audio tensor
            mode: Processing mode ('inference' or 'training')
            
        Returns:
            Processed audio tensor
        """
        processed = audio.to(self.device)
        
        for processor_name, processor in self.processors:
            try:
                if processor_name == 'dsp':
                    processed = self.torchfx_processor.process_audio_batch(
                        processed, self.dsp_config.get('sample_rate', 44100), processor
                    )
                elif processor_name == 'augmentation' and mode == 'training':
                    preset = self.augmentation_config.get('preset', 'voice_preservation')
                    processed = processor.apply_preset(processed, preset)
                    
            except Exception as e:
                logger.warning(f"Pipeline processing failed at {processor_name}: {e}")
                continue
        
        return processed
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get pipeline information."""
        return {
            'input_shape': self.input_shape,
            'components': [name for name, _ in self.processors],
            'dsp_config': self.dsp_config,
            'augmentation_config': self.augmentation_config,
            'device': str(self.device)
        }


# Global instance
unified_optimizer = None

def get_unified_optimizer(device: Optional[Union[str, torch.device]] = None,
                        enable_torchfx: bool = True,
                        enable_augmentation: bool = True,
                        enable_compilation: bool = True,
                        sample_rate: int = 44100) -> UnifiedPerformanceOptimizer:
    """Get global unified optimizer instance."""
    global unified_optimizer
    if unified_optimizer is None:
        unified_optimizer = UnifiedPerformanceOptimizer(
            device=device,
            enable_torchfx=enable_torchfx,
            enable_augmentation=enable_augmentation,
            enable_compilation=enable_compilation,
            sample_rate=sample_rate
        )
    return unified_optimizer


def create_optimized_rvc_system(model: Optional[nn.Module] = None,
                              input_shape: Tuple[int, ...] = (1, 80, 100),
                              optimization_level: str = 'balanced') -> Dict[str, Any]:
    """
    Create complete optimized RVC system.
    
    Args:
        model: RVC model to optimize
        input_shape: Input shape for the model
        optimization_level: Level of optimization to apply
        
    Returns:
        Dictionary containing optimized components
    """
    # Get unified optimizer
    optimizer = get_unified_optimizer()
    
    # Create optimized pipeline
    pipeline = optimizer.get_optimized_pipeline(input_shape)
    
    # Optimize model if provided
    optimized_model = model
    if model:
        example_input = torch.randn(input_shape, device=optimizer.device)
        optimized_model = optimizer.optimize_model(model, example_input, optimization_level)
    
    return {
        'optimizer': optimizer,
        'pipeline': pipeline,
        'optimized_model': optimized_model,
        'system_info': optimizer.get_system_info(),
        'pipeline_info': pipeline.get_pipeline_info()
    }


if __name__ == "__main__":
    # Test unified optimization system
    print("Testing Unified Performance Optimization System...")
    
    # Create test audio and model
    test_audio = torch.randn(4, 1, 44100)
    
    class SimpleRVCModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(44100, 44100)
            
        def forward(self, x):
            return self.linear(x)
    
    test_model = SimpleRVCModel()
    
    # Create optimized system
    system = create_optimized_rvc_system(test_model, (1, 44100), 'balanced')
    
    # Get system info
    info = system['system_info']
    print(f"System Info: {info}")
    
    # Test processing
    optimizer = system['optimizer']
    processed = optimizer.process_audio_batch(
        test_audio,
        dsp_filters=['normalize'],
        augmentation_preset='voice_preservation',
        augmentation_probability=0.0
    )
    print(f"Processed audio shape: {processed.shape}")
    
    # Benchmark
    benchmark_results = optimizer.benchmark_complete_optimization(test_audio, test_model, 20)
    print(f"Benchmark Results: {benchmark_results}")
    
    print("Unified optimization system test completed!")
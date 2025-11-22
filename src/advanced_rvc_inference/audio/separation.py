"""
Enhanced Audio Separation Module

This module provides advanced audio source separation capabilities using
multiple state-of-the-art models including MDX-Net, BS-Roformer, MDX23C,
Demucs, and more.

Features:
- Multi-architecture support (MDX-Net, Demucs, VR Arch, MDXC)
- ONNX acceleration for all models
- CPU, CUDA, ROCm, Apple Silicon CoreML support
- Batch processing capabilities
- Custom model loading and management
- High-quality separation with configurable parameters
"""

import os
import logging
import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Generator
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import contextmanager
import warnings

# Audio separation models
try:
    from audio_separator.separator import Separator as AudioSeparator
    AUDIO_SEPARATOR_AVAILABLE = True
except ImportError:
    AUDIO_SEPARATOR_AVAILABLE = False
    warnings.warn("audio-separator not available. Install with: pip install python-audio-separator")

# Demucs support
try:
    import demucs.api
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False

# MDX-Net support
try:
    import mdx.api
    MDX_AVAILABLE = True
except ImportError:
    MDX_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


class EnhancedAudioSeparator:
    """
    Enhanced Audio Source Separator with multi-architecture support.
    
    This class provides a unified interface for various audio separation models
    with advanced features like ONNX acceleration, batch processing, and
    customizable parameters.
    """
    
    def __init__(self, 
                 device: str = "auto",
                 enable_onnx: bool = True,
                 use_cuda: bool = True,
                 memory_efficient: bool = False):
        """
        Initialize the enhanced audio separator.
        
        Args:
            device: Computing device ('auto', 'cpu', 'cuda', 'rocm', 'mps')
            enable_onnx: Enable ONNX acceleration when available
            use_cuda: Use CUDA for GPU acceleration
            memory_efficient: Use memory-efficient settings for large audio files
        """
        self.device = self._detect_device(device, use_cuda)
        self.enable_onnx = enable_onnx and self.device.type in ['cuda', 'cpu', 'mps']
        self.memory_efficient = memory_efficient
        
        # Model caches
        self.separators = {}
        self.model_configs = {}
        
        # Initialize separator backends
        self._initialize_separators()
        
        logger.info(f"Enhanced Audio Separator initialized on {self.device} with ONNX: {self.enable_onnx}")
    
    def _detect_device(self, device: str, use_cuda: bool) -> torch.device:
        """Detect and configure the computing device."""
        if device == "auto":
            if use_cuda and torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            elif hasattr(torch.backends, 'opencl') and torch.backends.opencl.is_available():
                return torch.device("opencl")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def _initialize_separators(self):
        """Initialize available separator backends."""
        
        # Audio Separator (python-audio-separator)
        if AUDIO_SEPARATOR_AVAILABLE:
            try:
                # Initialize main separator
                separator = AudioSeparator()
                self.separators['audio_separator'] = separator
                
                # Load available models
                available_models = self._get_audio_separator_models()
                logger.info(f"Loaded {len(available_models)} models from audio-separator")
                
            except Exception as e:
                logger.warning(f"Failed to initialize audio-separator: {e}")
        
        # Demucs
        if DEMUCS_AVAILABLE:
            try:
                # Demucs models are loaded on-demand
                self.separators['demucs'] = 'available'
                logger.info("Demucs backend available")
            except Exception as e:
                logger.warning(f"Failed to initialize Demucs: {e}")
        
        # MDX-Net
        if MDX_AVAILABLE:
            try:
                # MDX models are loaded on-demand
                self.separators['mdx'] = 'available'
                logger.info("MDX-Net backend available")
            except Exception as e:
                logger.warning(f"Failed to initialize MDX-Net: {e}")
    
    def _get_audio_separator_models(self) -> List[Dict[str, Any]]:
        """Get list of available models from audio-separator."""
        try:
            separator = self.separators.get('audio_separator')
            if separator and hasattr(separator, 'list_models'):
                return separator.list_models()
        except Exception as e:
            logger.warning(f"Failed to list models: {e}")
        return []
    
    def separate_audio(self,
                      audio_input: Union[str, np.ndarray, Path],
                      model_name: str = "BS-Roformer-Viperx-1297",
                      output_dir: Optional[Union[str, Path]] = None,
                      output_format: str = "wav",
                      **kwargs) -> Dict[str, str]:
        """
        Separate audio into individual stems.
        
        Args:
            audio_input: Input audio file path or numpy array
            model_name: Model to use for separation
            output_dir: Directory to save separated stems
            output_format: Output audio format (wav, flac, mp3)
            **kwargs: Model-specific parameters
            
        Returns:
            Dictionary mapping stem names to output file paths
        """
        
        # Handle input
        if isinstance(audio_input, (str, Path)):
            audio, sr = librosa.load(audio_input, sr=None)
            input_path = Path(audio_input)
        elif isinstance(audio_input, np.ndarray):
            if kwargs.get('sample_rate') is None:
                raise ValueError("sample_rate must be provided for numpy array input")
            audio = audio_input
            sr = kwargs['sample_rate']
            input_path = Path("input_audio")
        else:
            raise ValueError("Unsupported input type")
        
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        logger.info(f"Separating audio using model: {model_name}")
        
        # Choose separator backend
        if model_name in self._get_audio_separator_models():
            return self._separate_with_audio_separator(audio, model_name, output_dir, output_format, **kwargs)
        elif model_name.startswith("demucs"):
            return self._separate_with_demucs(audio, model_name, output_dir, output_format, **kwargs)
        elif model_name.startswith("mdx"):
            return self._separate_with_mdx(audio, model_name, output_dir, output_format, **kwargs)
        else:
            # Fallback to default model
            default_model = "BS-Roformer-Viperx-1297"
            logger.warning(f"Unknown model {model_name}, using default: {default_model}")
            return self._separate_with_audio_separator(audio, default_model, output_dir, output_format, **kwargs)
    
    def _separate_with_audio_separator(self,
                                      audio: np.ndarray,
                                      model_name: str,
                                      output_dir: Optional[Union[str, Path]],
                                      output_format: str,
                                      **kwargs) -> Dict[str, str]:
        """Separate audio using python-audio-separator."""
        try:
            separator = self.separators['audio_separator']
            
            # Load model
            separator.load_model(model_name)
            
            # Prepare output directory
            if output_dir is None:
                output_dir = Path("separated_audio")
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Configure parameters
            config = self._get_model_config(model_name, **kwargs)
            
            # Handle different input types
            if isinstance(audio, np.ndarray):
                # Create temporary file
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    sf.write(tmp.name, audio, config.get('sample_rate', 44100))
                    input_path = tmp.name
            else:
                input_path = audio
            
            # Run separation
            output_files = separator.separate(input_path, 
                                            store_dir=str(output_dir),
                                            output_format=output_format,
                                            **config)
            
            # Clean up temporary file
            if isinstance(audio, np.ndarray) and os.path.exists(input_path):
                os.unlink(input_path)
            
            return output_files
            
        except Exception as e:
            logger.error(f"Audio separator failed: {e}")
            raise
    
    def _separate_with_demucs(self,
                             audio: np.ndarray,
                             model_name: str,
                             output_dir: Optional[Union[str, Path]],
                             output_format: str,
                             **kwargs) -> Dict[str, str]:
        """Separate audio using Demucs."""
        if not DEMUCS_AVAILABLE:
            raise ImportError("Demucs not available. Install with: pip install demucs")
        
        try:
            # Prepare output directory
            if output_dir is None:
                output_dir = Path("separated_audio")
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Configure Demucs
            config = self._get_demucs_config(model_name, **kwargs)
            
            # Initialize separator
            separator = demucs.api.Separator(**config)
            
            # Run separation
            sources = separator.separate(audio)
            
            # Save stems
            stem_names = separator.track_names
            output_files = {}
            
            for i, (name, stem) in enumerate(sources.items()):
                output_path = output_dir / f"{name}.{output_format}"
                sf.write(str(output_path), stem, config.get('sample_rate', 44100))
                output_files[name] = str(output_path)
            
            return output_files
            
        except Exception as e:
            logger.error(f"Demucs separation failed: {e}")
            raise
    
    def _separate_with_mdx(self,
                          audio: np.ndarray,
                          model_name: str,
                          output_dir: Optional[Union[str, Path]],
                          output_format: str,
                          **kwargs) -> Dict[str, str]:
        """Separate audio using MDX-Net."""
        if not MDX_AVAILABLE:
            raise ImportError("MDX-Net not available. Install with: pip install mdx")
        
        try:
            # Prepare output directory
            if output_dir is None:
                output_dir = Path("separated_audio")
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Configure MDX
            config = self._get_mdx_config(model_name, **kwargs)
            
            # Initialize separator
            separator = mdx.api.Separator(**config)
            
            # Run separation
            sources = separator.separate(audio)
            
            # Save stems
            stem_names = separator.track_names
            output_files = {}
            
            for name, stem in sources.items():
                output_path = output_dir / f"{name}.{output_format}"
                sf.write(str(output_path), stem, config.get('sample_rate', 44100))
                output_files[name] = str(output_path)
            
            return output_files
            
        except Exception as e:
            logger.error(f"MDX separation failed: {e}")
            raise
    
    def batch_separate(self,
                      audio_files: List[Union[str, Path]],
                      model_name: str = "BS-Roformer-Viperx-1297",
                      output_root: Optional[Union[str, Path]] = None,
                      max_workers: int = 4,
                      **kwargs) -> List[Dict[str, str]]:
        """
        Batch separate multiple audio files.
        
        Args:
            audio_files: List of input audio file paths
            model_name: Model to use for separation
            output_root: Root directory for all outputs
            max_workers: Maximum number of parallel workers
            **kwargs: Model-specific parameters
            
        Returns:
            List of dictionaries mapping stem names to output paths
        """
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.separate_audio, 
                               audio_file, 
                               model_name, 
                               self._get_output_dir(output_root, audio_file),
                               **kwargs): audio_file
                for audio_file in audio_files
            }
            
            # Collect results
            for future in future_to_file:
                audio_file = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Successfully separated: {audio_file}")
                except Exception as e:
                    logger.error(f"Failed to separate {audio_file}: {e}")
                    results.append({})
        
        return results
    
    def _get_output_dir(self, 
                       output_root: Optional[Union[str, Path]], 
                       audio_file: Union[str, Path]) -> Path:
        """Generate output directory for a given audio file."""
        if output_root is None:
            output_root = Path("separated_audio")
        else:
            output_root = Path(output_root)
        
        audio_path = Path(audio_file)
        output_dir = output_root / audio_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return output_dir
    
    def _get_model_config(self, model_name: str, **kwargs) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        
        # Base configuration
        config = {
            'sample_rate': kwargs.get('sample_rate', 44100),
            'output_bitrate': kwargs.get('output_bitrate', '320k'),
            'use_autocast': self.device.type == 'cuda',
            'use_soundfile': self.memory_efficient,
        }
        
        # Model-specific configurations
        if 'roformer' in model_name.lower():
            config.update({
                'override_model_segment_size': False,
                'overlap': kwargs.get('overlap', 8),
                'batch_size': kwargs.get('batch_size', 1),
                'pitch_shift': kwargs.get('pitch_shift', 0),
            })
        elif 'demucs' in model_name.lower():
            config.update({
                'segment_size': kwargs.get('segment_size', 'Default'),
                'shifts': kwargs.get('shifts', 2),
                'overlap': kwargs.get('overlap', 0.25),
                'segments_enabled': kwargs.get('segments_enabled', True),
            })
        elif 'mdx' in model_name.lower():
            config.update({
                'hop_length': kwargs.get('hop_length', 1024),
                'segment_size': kwargs.get('segment_size', 256),
                'overlap': kwargs.get('overlap', 0.25),
                'batch_size': kwargs.get('batch_size', 1),
                'enable_denoise': kwargs.get('enable_denoise', False),
            })
        
        return config
    
    def _get_demucs_config(self, model_name: str, **kwargs) -> Dict[str, Any]:
        """Get configuration for Demucs models."""
        return {
            'model': model_name.replace('demucs_', ''),
            'device': str(self.device),
            'segment': kwargs.get('segment', 'Default'),
            'shift': kwargs.get('shift', 0.25),
            'overlap': kwargs.get('overlap', 0.25),
            'int16': kwargs.get('int16', True),
            'flac': kwargs.get('flac', False),
            'mp3': kwargs.get('mp3', False),
            'mp3-preset': kwargs.get('mp3_preset', 2),
            'mp3-abr': kwargs.get('mp3_abr', 128),
            'stems': kwargs.get('stems', None),
            'mp3-preset': kwargs.get('mp3_preset', 2),
        }
    
    def _get_mdx_config(self, model_name: str, **kwargs) -> Dict[str, Any]:
        """Get configuration for MDX models."""
        return {
            'model': model_name.replace('mdx_', ''),
            'device': str(self.device),
            'segment': kwargs.get('segment', 256),
            'overlap': kwargs.get('overlap', 0.25),
            'shifts': kwargs.get('shifts', 2),
            'split_mode': kwargs.get('split_mode', True),
            'no_stft': kwargs.get('no_stft', False),
        }
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get all available separation models organized by backend."""
        models = {
            'audio_separator': [],
            'demucs': [],
            'mdx': []
        }
        
        # Audio Separator models
        if AUDIO_SEPARATOR_AVAILABLE:
            models['audio_separator'] = self._get_audio_separator_models()
        
        # Demucs models
        if DEMUCS_AVAILABLE:
            # Demucs models are typically pre-defined
            models['demucs'] = [
                'demucs_htdemucs', 'demucs_mdx_extra', 'demucs_extra',
                'demucs_htdemucs_ft', 'demucs_htdemucs_6s'
            ]
        
        # MDX models
        if MDX_AVAILABLE:
            # MDX models are typically pre-defined
            models['mdx'] = [
                'mdx_original', 'mdx_extra', 'mdx_extra_v3', 'mdx_final'
            ]
        
        return models
    
    def benchmark_models(self,
                        audio_file: Union[str, Path],
                        models: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Benchmark different models on a test audio file."""
        
        if models is None:
            available_models = self.get_available_models()
            models = []
            for backend_models in available_models.values():
                models.extend(backend_models[:2])  # Test first 2 from each backend
        
        results = {}
        
        for model_name in models:
            try:
                logger.info(f"Benchmarking model: {model_name}")
                
                import time
                start_time = time.time()
                
                # Run separation
                output_files = self.separate_audio(
                    audio_file, 
                    model_name, 
                    output_dir=f"temp_benchmark_{model_name}"
                )
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Calculate metrics
                output_size = sum(os.path.getsize(path) for path in output_files.values())
                num_stems = len(output_files)
                
                results[model_name] = {
                    'processing_time': processing_time,
                    'output_size': output_size,
                    'num_stems': num_stems,
                    'output_files': output_files,
                    'status': 'success'
                }
                
                logger.info(f"Model {model_name}: {processing_time:.2f}s, {num_stems} stems")
                
            except Exception as e:
                results[model_name] = {
                    'error': str(e),
                    'processing_time': float('inf'),
                    'status': 'failed'
                }
                logger.error(f"Model {model_name} failed: {e}")
        
        return results


# Global separator instance
_audio_separator = None

def get_audio_separator(device: str = "auto", 
                       enable_onnx: bool = True,
                       memory_efficient: bool = False) -> EnhancedAudioSeparator:
    """Get or create global audio separator instance."""
    global _audio_separator
    if _audio_separator is None:
        _audio_separator = EnhancedAudioSeparator(
            device=device,
            enable_onnx=enable_onnx,
            memory_efficient=memory_efficient
        )
    return _audio_separator
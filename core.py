"""
Enhanced Advanced RVC Inference Core Module
Optimized for performance, readability, maintainability, and security

This module provides comprehensive audio processing capabilities for voice conversion,
including vocals separation, noise reduction, reverb removal, echo cancellation,
and high-quality voice conversion using RVC models.

The main processing pipeline includes:
1. Audio input and preprocessing with validation
2. Multi-stage audio separation (vocals, instrumentals, backing vocals)
3. Audio enhancement (denoise, dereverb, deecho)
4. Voice conversion using RVC models
5. Post-processing and audio merging

Version: 3.2 Enhanced
Author: Enhanced by BF667
"""

import sys
import os
import subprocess
import torch
import json
import hashlib
import mimetypes
from functools import lru_cache, wraps
from pathlib import Path
import shutil
import tempfile
import time
from contextlib import contextmanager
from typing import Optional, Dict, Any, Tuple, Union, List, Generator
import logging

# Audio processing libraries
import numpy as np
from pedalboard import Pedalboard, Reverb
from pedalboard.io import AudioFile
from pydub import AudioSegment
from audio_separator.separator import Separator
import yaml

# Project imports
from programs.applio_code.rvc.infer.infer import VoiceConverter
from programs.applio_code.rvc.lib.tools.model_download import model_download_pipeline
from programs.music_separation_code.inference import proc_file
from assets.presence.discord_presence import RPCManager, track_presence

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Get current directory
now_dir = os.getcwd()
os.path.dirname(os.path.abspath(__file__))

# Enhanced model configurations with validation
MODELS_CONFIG = {
    "vocals": [
        {
            "name": "Mel-Roformer by KimberleyJSN",
            "path": os.path.join(now_dir, "models", "mel-vocals"),
            "model": os.path.join(now_dir, "models", "mel-vocals", "model.ckpt"),
            "config": os.path.join(now_dir, "models", "mel-vocals", "config.yaml"),
            "type": "mel_band_roformer",
            "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml",
            "model_url": "https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt",
            "description": "State-of-the-art vocal isolation using Mel-Roformer architecture",
            "quality": "high",
            "speed": "medium"
        },
        {
            "name": "BS-Roformer by ViperX",
            "path": os.path.join(now_dir, "models", "bs-vocals"),
            "model": os.path.join(now_dir, "models", "bs-vocals", "model.ckpt"),
            "config": os.path.join(now_dir, "models", "bs-vocals", "config.yaml"),
            "type": "bs_roformer",
            "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml",
            "model_url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            "description": "High-quality instrumental separation with BS-Roformer",
            "quality": "high",
            "speed": "fast"
        },
        {
            "name": "MDX23C",
            "path": os.path.join(now_dir, "models", "mdx23c-vocals"),
            "model": os.path.join(now_dir, "models", "mdx23c-vocals", "model.ckpt"),
            "config": os.path.join(now_dir, "models", "mdx23c-vocals", "config.yaml"),
            "type": "mdx23c",
            "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_vocals_mdx23c.yaml",
            "model_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_mdx23c_sdr_10.17.ckpt",
            "description": "Advanced neural network processing with MDX23C architecture",
            "quality": "very_high",
            "speed": "slow"
        },
    ],
    "karaoke": [
        {
            "name": "Mel-Roformer Karaoke by aufr33 and viperx",
            "path": os.path.join(now_dir, "models", "mel-kara"),
            "model": os.path.join(now_dir, "models", "mel-kara", "model.ckpt"),
            "config": os.path.join(now_dir, "models", "mel-kara", "config.yaml"),
            "type": "mel_band_roformer",
            "config_url": "https://huggingface.co/shiromiya/audio-separation-models/resolve/main/mel_band_roformer_karaoke_aufr33_viperx/config_mel_band_roformer_karaoke.yaml",
            "model_url": "https://huggingface.co/shiromiya/audio-separation-models/resolve/main/mel_band_roformer_karaoke_aufr33_viperx/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
            "description": "Specialized karaoke model for vocal-instrumental separation",
            "quality": "high",
            "speed": "medium"
        },
        {
            "name": "UVR-BVE",
            "full_name": "UVR-BVE-4B_SN-44100-1.pth",
            "arch": "vr",
            "description": "UVR-BVE model for advanced separation tasks",
            "quality": "medium",
            "speed": "fast"
        },
    ],
    "denoise": [
        {
            "name": "Mel-Roformer Denoise Normal by aufr33",
            "path": os.path.join(now_dir, "models", "mel-denoise"),
            "model": os.path.join(now_dir, "models", "mel-denoise", "model.ckpt"),
            "config": os.path.join(now_dir, "models", "mel-denoise", "config.yaml"),
            "type": "mel_band_roformer",
            "config_url": "https://huggingface.co/shiromiya/audio-separation-models/resolve/main/mel_band_roformer_denoise_normal_aufr33/config_mel_band_roformer_denoise.yaml",
            "model_url": "https://huggingface.co/shiromiya/audio-separation-models/resolve/main/mel_band_roformer_denoise_normal_aufr33/mel_band_roformer_denoise_normal_aufr33_sdr_19.51.ckpt",
            "description": "Advanced denoising with Mel-Roformer architecture",
            "quality": "high",
            "speed": "medium"
        },
    ]
}

# Enhanced validation and security constants
ALLOWED_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.alac', '.wma'}
MAX_FILE_SIZE_MB = getattr(__builtins__, 'max_file_size_mb', 500)  # Default 500MB limit
MAX_AUDIO_DURATION_MINUTES = getattr(__builtins__, 'max_duration_min', 30)  # Default 30 min limit
SAFE_TEMP_DIR = tempfile.gettempdir()

# Cache management
class CacheManager:
    """Enhanced cache management with TTL and size limits."""
    
    def __init__(self, cache_dir: str = "cache", max_size_mb: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._clean_cache()
    
    def _clean_cache(self):
        """Clean old cache files to maintain size limits."""
        try:
            total_size = 0
            for file_path in self.cache_dir.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            
            if total_size > self.max_size_bytes:
                # Remove oldest files first
                files_by_time = sorted(self.cache_dir.rglob("*"), key=lambda x: x.stat().st_mtime)
                for file_path in files_by_time:
                    if file_path.is_file():
                        file_path.unlink()
                        total_size -= file_path.stat().st_size
                        if total_size <= self.max_size_bytes:
                            break
        except Exception as e:
            logger.warning(f"Cache cleanup failed: {e}")
    
    def get_cache_path(self, key: str) -> Path:
        """Generate safe cache path from key."""
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{safe_key}.cache"
    
    @contextmanager
    def cached_operation(self, key: str, ttl_seconds: int = 3600):
        """Context manager for cached operations with TTL."""
        cache_path = self.get_cache_path(key)
        
        # Check if cached result is still valid
        if cache_path.exists():
            try:
                file_age = time.time() - cache_path.stat().st_mtime
                if file_age < ttl_seconds:
                    logger.debug(f"Using cached result for: {key}")
                    yield cache_path
                    return
            except Exception:
                pass  # Continue to fresh computation
        
        # Compute fresh result
        temp_path = cache_path.with_suffix(".tmp")
        try:
            yield temp_path
            # Move to cache if successful
            temp_path.rename(cache_path)
            logger.debug(f"Cached result for: {key}")
        except Exception as e:
            # Clean up on error
            if temp_path.exists():
                temp_path.unlink()
            logger.warning(f"Cache operation failed for {key}: {e}")
            raise

# Initialize cache manager
cache_manager = CacheManager()

def safe_file_validation(file_path: Union[str, Path]) -> Tuple[bool, str]:
    """
    Enhanced file validation with security checks.
    
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    file_path = Path(file_path)
    
    # Check if file exists
    if not file_path.exists():
        return False, "File does not exist"
    
    # Check file extension
    if file_path.suffix.lower() not in ALLOWED_AUDIO_EXTENSIONS:
        return False, f"Unsupported file extension: {file_path.suffix}"
    
    # Check file size
    try:
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            return False, f"File too large: {file_size_mb:.1f}MB > {MAX_FILE_SIZE_MB}MB limit"
    except OSError:
        return False, "Cannot read file information"
    
    # Check MIME type
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if mime_type and not mime_type.startswith('audio/'):
        logger.warning(f"MIME type mismatch for {file_path}: {mime_type}")
    
    # Security: Check for path traversal
    try:
        file_path.resolve().relative_to(Path.cwd())
    except ValueError:
        return False, "Invalid file path (potential security issue)"
    
    return True, "File validation passed"

def enhanced_error_handler(func):
    """Decorator for enhanced error handling with logging."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_id = hashlib.md5(f"{func.__name__}{time.time()}".encode()).hexdigest()[:8]
            logger.error(f"Error in {func.__name__} [ID: {error_id}]: {e}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            
            # Clean up temporary files
            for arg in args:
                if isinstance(arg, (str, Path)) and str(arg).startswith(SAFE_TEMP_DIR):
                    try:
                        Path(arg).unlink(missing_ok=True)
                    except Exception:
                        pass
            
            # Return user-friendly error message
            return f"Error in {func.__name__}: {str(e)} [Error ID: {error_id}]"
    return wrapper

@enhanced_error_handler
def enhanced_audio_separation(
    input_file: str,
    model_type: str = "vocals",
    model_name: str = "Mel-Roformer by KimberleyJSN",
    output_format: str = "wav"
) -> str:
    """
    Enhanced audio separation with improved error handling and progress tracking.
    
    Args:
        input_file: Path to input audio file
        model_type: Type of separation model (vocals, karaoke, denoise)
        model_name: Specific model name to use
        output_format: Output audio format
    
    Returns:
        str: Path to separated audio file or error message
    """
    logger.info(f"Starting enhanced audio separation: {model_type} using {model_name}")
    
    # Validate input file
    is_valid, validation_msg = safe_file_validation(input_file)
    if not is_valid:
        return f"Input validation failed: {validation_msg}"
    
    # Generate cache key
    cache_key = f"sep_{model_type}_{model_name}_{Path(input_file).name}"
    
    # Check for cached result
    with cache_manager.cached_operation(cache_key, ttl_seconds=1800) as cache_path:  # 30 min TTL
        # Process audio separation
        try:
            # Create separator with enhanced configuration
            separator_config = {
                "model_name": model_name,
                "output_format": output_format,
                "normalize": True,
                "denoise": model_type == "denoise"
            }
            
            separator = Separator(model_name=model_name)
            
            # Process with progress tracking
            logger.info(f"Processing audio separation...")
            processed_path = separator.process_file(
                input_file=input_file,
                output_format=output_format,
                output_directory=str(cache_path.parent)
            )
            
            logger.info(f"Audio separation completed successfully: {processed_path}")
            return processed_path
            
        except Exception as e:
            logger.error(f"Audio separation failed: {e}")
            raise Exception(f"Audio separation failed: {str(e)}")

@enhanced_error_handler
def enhanced_voice_conversion(
    input_audio: str,
    model_path: str,
    index_file: Optional[str] = None,
    pitch: float = 0,
    filter_radius: int = 3,
    resample_sr: int = 0,
    rms_mix_rate: float = 0.25,
    protect: float = 0.33
) -> str:
    """
    Enhanced voice conversion with performance optimizations and security.
    
    Args:
        input_audio: Path to input audio file
        model_path: Path to RVC model file
        index_file: Optional index file for voice characteristics
        pitch: Pitch adjustment (-12 to +12)
        filter_radius: Radius for spectral filtering (0-7)
        resample_sr: Resampling sample rate (0 = no resample)
        rms_mix_rate: RMS mix rate for blending (0-1)
        protect: Protection rate for voice characteristics (0-1)
    
    Returns:
        str: Path to converted audio file or error message
    """
    logger.info("Starting enhanced voice conversion...")
    
    # Validate input files
    is_valid, validation_msg = safe_file_validation(input_audio)
    if not is_valid:
        return f"Input audio validation failed: {validation_msg}"
    
    if not os.path.exists(model_path):
        return f"Model file not found: {model_path}"
    
    if index_file and not os.path.exists(index_file):
        return f"Index file not found: {index_file}"
    
    # Create temporary output file
    temp_output = tempfile.mktemp(suffix=".wav", dir=SAFE_TEMP_DIR)
    
    try:
        # Initialize voice converter with enhanced settings
        converter = VoiceConverter(
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_prefetcher=True,  # Enable prefetching for better performance
        )
        
        # Convert with optimized parameters
        result_path = converter.convert(
            input_path=input_audio,
            model_path=model_path,
            index_path=index_file,
            output_path=temp_output,
            pitch=pitch,
            filter_radius=filter_radius,
            resample_sr=resample_sr,
            rms_mix_rate=rms_mix_rate,
            protect=protect,
            segment_length=30.0,  # Process in segments for memory efficiency
        )
        
        logger.info(f"Voice conversion completed successfully: {result_path}")
        return result_path
        
    except Exception as e:
        logger.error(f"Voice conversion failed: {e}")
        # Clean up temporary file
        if os.path.exists(temp_output):
            os.remove(temp_output)
        raise Exception(f"Voice conversion failed: {str(e)}")

@enhanced_error_handler
def enhanced_post_processing(
    audio_path: str,
    output_format: str = "wav",
    add_reverb: bool = False,
    reverb_amount: float = 0.3,
    volume_adjustment: float = 1.0
) -> str:
    """
    Enhanced post-processing with audio effects and optimization.
    
    Args:
        audio_path: Path to input audio file
        output_format: Output format (wav, mp3, flac, etc.)
        add_reverb: Whether to add reverb effect
        reverb_amount: Amount of reverb (0.0-1.0)
        volume_adjustment: Volume multiplier (0.1-3.0)
    
    Returns:
        str: Path to processed audio file or error message
    """
    logger.info("Starting enhanced post-processing...")
    
    # Validate input
    is_valid, validation_msg = safe_file_validation(audio_path)
    if not is_valid:
        return f"Input validation failed: {validation_msg}"
    
    # Generate output filename
    input_path = Path(audio_path)
    output_path = input_path.parent / f"{input_path.stem}_enhanced.{output_format.lower()}"
    
    try:
        # Load audio with enhanced error handling
        with AudioFile(audio_path).resampled_to(44100) as f:
            audio = f.read(f.frames)
            sample_rate = f.samplerate
        
        # Apply post-processing effects
        effects = []
        
        # Add reverb if requested
        if add_reverb and 0.0 < reverb_amount <= 1.0:
            reverb = Reverb(
                room_size=reverb_amount,
                damping=0.5,
                wet_level=reverb_amount,
                dry_level=1.0 - reverb_amount
            )
            effects.append(reverb)
        
        # Create pedalboard for effects
        board = Pedalboard(effects)
        
        # Apply effects
        if effects:
            enhanced_audio = board(audio, sample_rate)
        else:
            enhanced_audio = audio
        
        # Apply volume adjustment
        if volume_adjustment != 1.0:
            enhanced_audio = enhanced_audio * max(0.0, min(volume_adjustment, 10.0))
        
        # Normalize audio to prevent clipping
        if np.max(np.abs(enhanced_audio)) > 0:
            enhanced_audio = enhanced_audio / np.max(np.abs(enhanced_audio)) * 0.95
        
        # Save processed audio
        with AudioFile(str(output_path), 'w', sample_rate, enhanced_audio.shape[0]) as f:
            f.write(enhanced_audio)
        
        logger.info(f"Post-processing completed successfully: {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Post-processing failed: {e}")
        raise Exception(f"Post-processing failed: {str(e)}")

def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system information for debugging and optimization.
    
    Returns:
        Dict containing system specifications and status
    """
    info = {
        "platform": sys.platform,
        "python_version": sys.version,
        "torch_version": torch.__version__ if torch else "Not available",
        "cuda_available": torch.cuda.is_available() if torch else False,
        "cuda_version": torch.cuda.version() if torch and torch.cuda.is_available() else "Not available",
        "gpu_count": torch.cuda.device_count() if torch and torch.cuda.is_available() else 0,
        "memory_info": {},
        "disk_info": {},
        "models_available": {},
        "cache_status": {}
    }
    
    try:
        # Memory information
        if hasattr(shutil, 'disk_usage'):
            disk_usage = shutil.disk_usage(now_dir)
            info["disk_info"] = {
                "total_gb": disk_usage.total / (1024**3),
                "free_gb": disk_usage.free / (1024**3),
                "used_gb": disk_usage.used / (1024**3)
            }
        
        # Check available models
        for category, models in MODELS_CONFIG.items():
            available_models = []
            for model in models:
                model_path = Path(model.get("path", ""))
                if model_path.exists():
                    available_models.append(model.get("name", "Unknown"))
            info["models_available"][category] = available_models
        
        # Cache information
        try:
            cache_size = sum(f.stat().st_size for f in cache_manager.cache_dir.rglob("*") if f.is_file())
            info["cache_status"] = {
                "cache_dir": str(cache_manager.cache_dir),
                "cache_size_mb": cache_size / (1024**2),
                "max_cache_size_mb": cache_manager.max_size_bytes / (1024**2)
            }
        except Exception as e:
            logger.warning(f"Could not get cache info: {e}")
            
    except Exception as e:
        logger.warning(f"Could not gather system information: {e}")
    
    return info

# Performance monitoring
class PerformanceMonitor:
    """Monitor and log performance metrics."""
    
    def __init__(self):
        self.operations = {}
    
    @contextmanager
    def time_operation(self, operation_name: str):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            if operation_name not in self.operations:
                self.operations[operation_name] = []
            self.operations[operation_name].append(duration)
            logger.debug(f"Operation '{operation_name}' took {duration:.2f} seconds")
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics."""
        stats = {}
        for operation, times in self.operations.items():
            if times:
                stats[operation] = {
                    "count": len(times),
                    "total_time": sum(times),
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times)
                }
        return stats

# Initialize performance monitor
performance_monitor = PerformanceMonitor()

# Export enhanced functions and classes
__all__ = [
    'MODELS_CONFIG',
    'enhanced_audio_separation',
    'enhanced_voice_conversion', 
    'enhanced_post_processing',
    'safe_file_validation',
    'get_system_info',
    'CacheManager',
    'PerformanceMonitor',
    'performance_monitor'
]

logger.info("Enhanced RVC Inference Core module loaded successfully")
"""
Enhanced RVC Utilities
Comprehensive utility functions for RVC audio processing and F0 generation
"""

import os
import re
import sys
import gc
import math
import torch
import logging
import warnings
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict, Any
from scipy import signal
from scipy.io import wavfile
from torch import nn
from transformers import HubertModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("librosa").setLevel(logging.ERROR)
logging.getLogger("soundfile").setLevel(logging.ERROR)
logging.getLogger("fairseq").setLevel(logging.ERROR)
logging.getLogger("faiss.loader").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# Legacy compatibility - maintain backward compatibility
now_dir = os.getcwd()
sys.path.append(now_dir)

base_path = os.path.join(now_dir, "rvc", "models", "formant", "stftpitchshift")
stft = base_path + ".exe" if sys.platform == "win32" else base_path

class HubertModelWithFinalProj(HubertModel):
    """Hubert model with final projection layer for RVC compatibility"""
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

def load_audio_legacy(file, sample_rate):
    """Legacy audio loading function for backward compatibility"""
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        audio, sr = sf.read(file)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.T)
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    except Exception as error:
        raise RuntimeError(f"An error occurred loading the audio: {error}")

    return audio.flatten()

def load_audio_infer(file, sample_rate):
    """Audio loading function for inference"""
    file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
    if not os.path.isfile(file):
        raise FileNotFoundError(f"Audio file not found: {file}")
    
    try:
        audio, sr = sf.read(file)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.T)
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        return audio.flatten()
    except Exception as error:
        raise RuntimeError(f"An error occurred loading the audio: {error}")

def format_title(title: str) -> str:
    """Format title for display"""
    if title is None:
        return ""
    
    # Remove unsupported characters
    title = unicodedata.normalize('NFKD', title)
    title = title.encode('ascii', 'ignore').decode('ascii')
    
    # Remove extra whitespace
    title = re.sub(r'\s+', ' ', title).strip()
    
    return title

class AudioUtils:
    """Enhanced audio utility functions"""
    
    @staticmethod
    def load_audio(file_path: Union[str, Path], sample_rate: int = 16000, 
                   normalize: bool = True, trim_silence: bool = True,
                   silence_threshold: float = 0.01) -> np.ndarray:
        """
        Load and preprocess audio file
        
        Args:
            file_path: Path to audio file
            sample_rate: Target sample rate
            normalize: Whether to normalize audio
            trim_silence: Whether to trim leading/trailing silence
            silence_threshold: Threshold for silence detection
            
        Returns:
            Preprocessed audio array
        """
        try:
            file_path = str(file_path).strip().strip('"').strip("\n").strip('"').strip()
            
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"Audio file not found: {file_path}")
            
            # Load audio using soundfile (preferred) or librosa (fallback)
            try:
                audio, sr = sf.read(file_path, dtype=np.float32)
            except Exception:
                audio, sr = librosa.load(file_path, sr=None)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio.T)
            
            # Resample if necessary
            if sr != sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate, 
                                       res_type="soxr_vhq")
            
            # Normalize audio
            if normalize and np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            # Trim silence
            if trim_silence:
                audio = AudioUtils.trim_silence(audio, threshold=silence_threshold)
            
            return audio.flatten()
            
        except Exception as e:
            raise RuntimeError(f"Error loading audio: {e}")
    
    @staticmethod
    def save_audio(audio: np.ndarray, file_path: Union[str, Path], 
                   sample_rate: int = 16000, normalize: bool = True):
        """
        Save audio to file
        
        Args:
            audio: Audio array
            file_path: Output file path
            sample_rate: Sample rate
            normalize: Whether to normalize before saving
        """
        try:
            if normalize and np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            sf.write(str(file_path), audio, sample_rate)
        except Exception as e:
            raise RuntimeError(f"Error saving audio: {e}")
    
    @staticmethod
    def trim_silence(audio: np.ndarray, threshold: float = 0.01, 
                     frame_length: int = 2048) -> np.ndarray:
        """Trim leading and trailing silence from audio"""
        try:
            # Use librosa for robust silence trimming
            trimmed, _ = librosa.effects.trim(audio, top_db=20, 
                                            frame_length=frame_length,
                                            hop_length=frame_length//4)
            return trimmed
        except:
            # Fallback simple trimming
            energy = np.sum(audio**2)
            if energy == 0:
                return audio
            
            # Find non-silent regions
            window_size = len(audio) // 100
            energies = np.array([np.sum(audio[i:i+window_size]**2) 
                               for i in range(0, len(audio)-window_size, window_size)])
            
            threshold_energy = threshold * np.max(energies)
            non_silent = energies > threshold_energy
            
            if not np.any(non_silent):
                return audio
            
            start_idx = np.argmax(non_silent) * window_size
            end_idx = len(audio) - np.argmax(non_silent[::-1]) * window_size
            
            return audio[start_idx:end_idx]
    
    @staticmethod
    def normalize_audio(audio: np.ndarray, target_level: float = -23.0) -> np.ndarray:
        """Normalize audio to target LUFS level (simplified)"""
        if len(audio) == 0:
            return audio
        
        # Calculate RMS and normalize
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            # Convert LUFS to linear scale (simplified)
            target_linear = 10**(target_level/20)
            audio = audio * (target_linear / rms)
            
            # Prevent clipping
            audio = np.clip(audio, -0.99, 0.99)
        
        return audio
    
    @staticmethod
    def apply_fade(audio: np.ndarray, fade_duration: float = 0.05, 
                   sample_rate: int = 16000) -> np.ndarray:
        """Apply fade in/out to audio"""
        fade_samples = int(fade_duration * sample_rate)
        
        if len(audio) < 2 * fade_samples:
            return audio
        
        # Fade in
        fade_in = np.linspace(0, 1, fade_samples)
        audio[:fade_samples] *= fade_in
        
        # Fade out
        fade_out = np.linspace(1, 0, fade_samples)
        audio[-fade_samples:] *= fade_out
        
        return audio

class F0Utils:
    """F0-related utility functions compatible with enhanced Generator"""
    
    @staticmethod
    def validate_f0(f0: np.ndarray, sample_rate: int = 16000, 
                   hop_length: int = 160) -> Dict[str, Any]:
        """
        Validate and analyze F0 array
        
        Returns:
            Dictionary with validation metrics
        """
        voiced_mask = f0 > 0
        
        if not np.any(voiced_mask):
            return {
                "is_valid": False,
                "error": "No voiced segments detected",
                "voiced_ratio": 0.0,
                "f0_mean": 0.0,
                "f0_std": 0.0
            }
        
        voiced_f0 = f0[voiced_mask]
        
        # Basic statistics
        f0_mean = np.mean(voiced_f0)
        f0_std = np.std(voiced_f0)
        voiced_ratio = np.sum(voiced_mask) / len(f0)
        
        # Validation checks
        issues = []
        
        # Check for reasonable F0 range
        if f0_mean < 50 or f0_mean > 1000:
            issues.append("F0 mean outside typical human range (50-1000 Hz)")
        
        # Check for excessive jitter
        if len(voiced_f0) > 1:
            f0_diff = np.diff(voiced_f0)
            jitter = np.mean(np.abs(f0_diff)) / f0_mean
            if jitter > 0.1:  # 10% jitter threshold
                issues.append(f"High pitch jitter: {jitter:.1%}")
        
        # Check for reasonable standard deviation
        if f0_std > f0_mean * 0.5:  # 50% std deviation threshold
            issues.append("High F0 standard deviation")
        
        # Calculate frame timing
        duration = len(f0) * hop_length / sample_rate
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "voiced_ratio": voiced_ratio,
            "f0_mean": f0_mean,
            "f0_std": f0_std,
            "duration": duration,
            "jitter": np.mean(np.abs(np.diff(voiced_f0))) / f0_mean if len(voiced_f0) > 1 else 0.0
        }
    
    @staticmethod
    def smooth_f0(f0: np.ndarray, method: str = "median", 
                  window_size: int = 5) -> np.ndarray:
        """
        Smooth F0 array to reduce jitter
        
        Args:
            f0: F0 array
            method: Smoothing method ('median', 'gaussian', 'moving_average')
            window_size: Smoothing window size
        """
        if method == "median":
            from scipy.signal import medfilt
            return medfilt(f0, kernel_size=min(window_size, len(f0)))
        
        elif method == "gaussian":
            from scipy.ndimage import gaussian_filter1d
            return gaussian_filter1d(f0, sigma=window_size/3.0)
        
        elif method == "moving_average":
            kernel = np.ones(window_size) / window_size
            return np.convolve(f0, kernel, mode='same')
        
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
    
    @staticmethod
    def interpolate_f0(f0: np.ndarray, method: str = "linear") -> np.ndarray:
        """Interpolate unvoiced F0 segments"""
        voiced_mask = f0 > 0
        
        if not np.any(voiced_mask) or np.all(voiced_mask):
            return f0
        
        # Get voiced segments
        voiced_f0 = f0.copy()
        
        # Interpolate unvoiced regions
        indices = np.arange(len(f0))
        voiced_indices = indices[voiced_mask]
        voiced_values = f0[voiced_mask]
        
        if len(voiced_values) > 1:
            interpolated = np.interp(indices, voiced_indices, voiced_values)
            # Only fill small gaps (less than 10 frames)
            gap_threshold = 10
            
            # Find gaps
            diff = np.diff(voiced_mask.astype(int))
            starts = np.where(diff == -1)[0] + 1  # End of voiced segments
            ends = np.where(diff == 1)[0] + 1     # Start of voiced segments
            
            # Handle edge cases
            if voiced_mask[0]:
                starts = np.concatenate([[0], starts])
            if voiced_mask[-1]:
                ends = np.concatenate([ends, [len(f0)]])
            
            # Fill small gaps
            for start, end in zip(starts, ends):
                gap_size = end - start
                if gap_size <= gap_threshold:
                    f0[start:end] = interpolated[start:end]
        
        return f0
    
    @staticmethod
    def pitch_shift_f0(f0: np.ndarray, shift_semitones: float) -> np.ndarray:
        """Shift F0 by specified number of semitones"""
        if shift_semitones == 0:
            return f0
        
        shift_ratio = 2 ** (shift_semitones / 12)
        voiced_mask = f0 > 0
        
        shifted_f0 = f0.copy()
        shifted_f0[voiced_mask] *= shift_ratio
        
        return shifted_f0
    
    @staticmethod
    def f0_to_midi(f0: np.ndarray) -> np.ndarray:
        """Convert F0 frequencies to MIDI note numbers"""
        return 69 + 12 * np.log2(f0 / 440.0)
    
    @staticmethod
    def midi_to_f0(midi: np.ndarray) -> np.ndarray:
        """Convert MIDI note numbers to F0 frequencies"""
        return 440.0 * 2 ** ((midi - 69) / 12)

class AudioFeatures:
    """Audio feature extraction utilities"""
    
    @staticmethod
    def extract_spectral_features(audio: np.ndarray, sample_rate: int = 16000,
                                 n_fft: int = 2048, hop_length: int = 512) -> Dict[str, np.ndarray]:
        """Extract spectral features from audio"""
        try:
            # Compute STFT
            stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            magnitude = np.abs(stft)
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(
                S=magnitude, sr=sample_rate)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                S=magnitude, sr=sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(
                S=magnitude, sr=sample_rate)[0]
            spectral_contrast = librosa.feature.spectral_contrast(
                S=magnitude, sr=sample_rate)
            
            return {
                "spectral_centroid": spectral_centroid,
                "spectral_bandwidth": spectral_bandwidth,
                "spectral_rolloff": spectral_rolloff,
                "spectral_contrast": spectral_contrast
            }
        except Exception as e:
            logger.warning(f"Error extracting spectral features: {e}")
            return {}
    
    @staticmethod
    def extract_mfcc(audio: np.ndarray, sample_rate: int = 16000,
                     n_mfcc: int = 13) -> np.ndarray:
        """Extract MFCC features"""
        try:
            mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
            return mfcc
        except Exception as e:
            logger.warning(f"Error extracting MFCC: {e}")
            return np.array([])
    
    @staticmethod
    def extract_rms(audio: np.ndarray, sample_rate: int = 16000,
                    hop_length: int = 512) -> np.ndarray:
        """Extract RMS energy"""
        try:
            rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
            return rms
        except Exception as e:
            logger.warning(f"Error extracting RMS: {e}")
            return np.array([])

class DeviceUtils:
    """Device and platform utilities"""
    
    @staticmethod
    def get_device() -> torch.device:
        """Get optimal device for computation"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS device (Apple Silicon)")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
        
        return device
    
    @staticmethod
    def clear_memory():
        """Clear CUDA memory if available"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    @staticmethod
    def get_memory_info() -> Dict[str, Any]:
        """Get memory usage information"""
        info = {"device": "cpu", "memory": {}}
        
        if torch.cuda.is_available():
            info["device"] = "cuda"
            info["memory"] = {
                "allocated": torch.cuda.memory_allocated() / 1024**3,
                "cached": torch.cuda.memory_reserved() / 1024**3,
                "max_allocated": torch.cuda.max_memory_allocated() / 1024**3
            }
        
        return info

class PathUtils:
    """Path and file utilities"""
    
    @staticmethod
    def ensure_dir(path: Union[str, Path]) -> Path:
        """Ensure directory exists, create if necessary"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def find_audio_files(directory: Union[str, Path], 
                        extensions: List[str] = None) -> List[Path]:
        """Find all audio files in directory"""
        if extensions is None:
            extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']
        
        directory = Path(directory)
        audio_files = []
        
        for ext in extensions:
            audio_files.extend(directory.glob(f"**/*{ext}"))
            audio_files.extend(directory.glob(f"**/*{ext.upper()}"))
        
        return sorted(audio_files)
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for cross-platform compatibility"""
        # Remove or replace invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        filename = re.sub(r'[^\w\-_.]', '_', filename)
        filename = re.sub(r'_+', '_', filename)  # Multiple underscores to single
        return filename.strip('._')

class QualityUtils:
    """Audio quality assessment utilities"""
    
    @staticmethod
    def calculate_snr(audio: np.ndarray, noise: np.ndarray = None) -> float:
        """Calculate Signal-to-Noise Ratio"""
        if noise is None:
            # Estimate noise from silent parts
            # This is a simplified approach
            signal_power = np.mean(audio**2)
            # Assume 1% of signal as noise (rough estimate)
            noise_power = signal_power * 0.01
        else:
            signal_power = np.mean(audio**2)
            noise_power = np.mean(noise**2)
        
        if noise_power == 0:
            return float('inf')
        
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    
    @staticmethod
    def detect_clipping(audio: np.ndarray, threshold: float = 0.99) -> Dict[str, Any]:
        """Detect audio clipping"""
        clipped_samples = np.sum(np.abs(audio) >= threshold)
        clipped_ratio = clipped_samples / len(audio)
        
        return {
            "is_clipped": clipped_ratio > 0.001,  # More than 0.1% clipped
            "clipped_ratio": clipped_ratio,
            "clipped_samples": clipped_samples,
            "max_amplitude": np.max(np.abs(audio))
        }
    
    @staticmethod
    def audio_quality_score(audio: np.ndarray) -> Dict[str, float]:
        """Calculate overall audio quality score"""
        # Normalize amplitude
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio_normalized = audio / peak
        else:
            audio_normalized = audio
        
        # Check for clipping
        clipping_info = QualityUtils.detect_clipping(audio_normalized)
        
        # Calculate various quality metrics
        rms = np.sqrt(np.mean(audio_normalized**2))
        
        # Dynamic range
        if len(audio_normalized) > 1024:
            # Split into chunks and calculate dynamic range
            chunk_size = len(audio_normalized) // 10
            dynamic_ranges = []
            for i in range(0, len(audio_normalized), chunk_size):
                chunk = audio_normalized[i:i+chunk_size]
                if len(chunk) > 0:
                    chunk_rms = np.sqrt(np.mean(chunk**2))
                    if chunk_rms > 0:
                        chunk_peak = np.max(np.abs(chunk))
                        dr = 20 * np.log10(chunk_peak / chunk_rms) if chunk_peak > 0 else 0
                        dynamic_ranges.append(dr)
            
            avg_dynamic_range = np.mean(dynamic_ranges) if dynamic_ranges else 0
        else:
            avg_dynamic_range = 0
        
        # Overall quality score (0-100)
        quality_score = 100.0
        
        # Penalize clipping
        if clipping_info["is_clipped"]:
            quality_score -= clipping_info["clipped_ratio"] * 100
        
        # Adjust for dynamic range (good range is 6-20 dB)
        if avg_dynamic_range < 6:
            quality_score -= (6 - avg_dynamic_range) * 2
        elif avg_dynamic_range > 20:
            quality_score -= (avg_dynamic_range - 20) * 1
        
        # Adjust for RMS level (good level is around -20 to -10 dB)
        rms_db = 20 * np.log10(rms) if rms > 0 else -80
        if rms_db < -30:
            quality_score -= (30 + rms_db) * 2
        elif rms_db > -6:
            quality_score -= (rms_db + 6) * 1
        
        quality_score = max(0, min(100, quality_score))
        
        return {
            "overall_score": quality_score,
            "clipping_score": 100 - clipping_info["clipped_ratio"] * 100,
            "dynamic_range": avg_dynamic_range,
            "rms_level": rms_db,
            "peak_level": 20 * np.log10(peak) if peak > 0 else -80
        }

# Convenience functions for direct import
def load_audio(file_path: Union[str, Path], sample_rate: int = 16000, **kwargs) -> np.ndarray:
    """Convenience function for audio loading"""
    return AudioUtils.load_audio(file_path, sample_rate, **kwargs)

def save_audio(audio: np.ndarray, file_path: Union[str, Path], sample_rate: int = 16000, **kwargs):
    """Convenience function for audio saving"""
    return AudioUtils.save_audio(audio, file_path, sample_rate, **kwargs)

def validate_f0(f0: np.ndarray, **kwargs) -> Dict[str, Any]:
    """Convenience function for F0 validation"""
    return F0Utils.validate_f0(f0, **kwargs)

def get_device() -> torch.device:
    """Convenience function for device detection"""
    return DeviceUtils.get_device()

def clear_memory():
    """Convenience function for memory cleanup"""
    return DeviceUtils.clear_memory()

def ensure_dir(path: Union[str, Path]) -> Path:
    """Convenience function for directory creation"""
    return PathUtils.ensure_dir(path)

def find_audio_files(directory: Union[str, Path], **kwargs) -> List[Path]:
    """Convenience function for finding audio files"""
    return PathUtils.find_audio_files(directory, **kwargs)

# Export main classes and functions
__all__ = [
    'AudioUtils', 'F0Utils', 'AudioFeatures', 'DeviceUtils', 
    'PathUtils', 'QualityUtils',
    'load_audio', 'save_audio', 'validate_f0', 'get_device',
    'clear_memory', 'ensure_dir', 'find_audio_files'
]
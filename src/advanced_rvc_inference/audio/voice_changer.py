"""
Enhanced Real-time Voice Changer Module

This module provides advanced real-time voice conversion capabilities with
support for multiple backends, low-latency processing, and comprehensive
audio device management.

Features:
- Multi-backend support (CPU, CUDA, ROCm, DirectML, Apple Silicon)
- Low-latency processing (< 256ms)
- VAD (Voice Activity Detection)
- Audio device management (ASIO, WASAPI, CoreAudio)
- Real-time parameter adjustment
- Chunk-based processing
- Crossfade and silence detection
- Multiple F0 extraction methods
"""

import os
import logging
import torch
import numpy as np
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
from contextlib import contextmanager
import queue
import warnings

# Audio handling
try:
    import pyaudio
    import sounddevice as sd
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    warnings.warn("Audio libraries not available. Install with: pip install pyaudio sounddevice")

# VAD support
try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False
    warnings.warn("WebRTC VAD not available. Install with: pip install webrtcvad")

# Core RVC modules
from ..core.f0_extractor import get_f0_extractor
from ..models.rvc_inference import RVCInference

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class AudioDeviceConfig:
    """Configuration for audio input/output devices."""
    input_device: Optional[str] = None
    output_device: Optional[str] = None
    sample_rate: int = 44100
    channels: int = 1
    chunk_size: int = 1024
    buffer_size: int = 2048


@dataclass
class VoiceChangerConfig:
    """Configuration for voice changer parameters."""
    model_path: str = ""
    index_path: str = ""
    pitch_shift: int = 0
    filter_radius: int = 7
    resample_sr: int = 0
    rms_mix_rate: float = 0.25
    protect: float = 0.33
    hop_length: int = 160
    pitch_extract_method: str = "rmvpe"
    f0_min: float = 50.0
    f0_max: float = 1200.0
    vad_sensitivity: int = 3
    silence_threshold: float = -60.0
    chunk_size: float = 0.1
    crossfade_size: float = 0.05
    extra_conversion: float = 1.0
    enable_vad: bool = True
    enable_auto_gain: bool = True


class EnhancedRealTimeVoiceChanger:
    """
    Enhanced Real-time Voice Changer with multi-backend support.
    
    This class provides low-latency voice conversion with comprehensive
    audio device management and real-time parameter adjustment.
    """
    
    def __init__(self,
                 config: VoiceChangerConfig,
                 device_config: AudioDeviceConfig,
                 backend: str = "auto"):
        """
        Initialize the real-time voice changer.
        
        Args:
            config: Voice changer configuration parameters
            device_config: Audio device configuration
            backend: Processing backend ('auto', 'cpu', 'cuda', 'rocm', 'directml', 'coreml')
        """
        self.config = config
        self.device_config = device_config
        self.backend = self._detect_backend(backend)
        
        # Initialize components
        self.rvc_inference = None
        self.f0_extractor = None
        self.vad = None
        
        # Audio handling
        self.audio = None
        self.input_stream = None
        self.output_stream = None
        
        # Processing
        self.is_running = False
        self.is_paused = False
        self.processing_thread = None
        self.audio_queue = queue.Queue(maxsize=10)
        self.output_queue = queue.Queue(maxsize=10)
        
        # Buffers
        self.input_buffer = []
        self.output_buffer = []
        self.f0_buffer = []
        self.crossfade_buffer = []
        
        # Statistics
        self.stats = {
            'processing_time': 0.0,
            'latency': 0.0,
            'frames_processed': 0,
            'frames_dropped': 0,
            'vad_triggered': 0
        }
        
        logger.info(f"Enhanced Real-time Voice Changer initialized with backend: {self.backend}")
    
    def _detect_backend(self, backend: str) -> str:
        """Detect the best available backend."""
        if backend == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            elif hasattr(torch.backends, 'directml') and torch.backends.directml.is_available():
                return "directml"
            elif hasattr(torch.backends, 'opencl') and torch.backends.opencl.is_available():
                return "rocm"
            else:
                return "cpu"
        return backend
    
    def initialize(self) -> bool:
        """Initialize all components."""
        try:
            # Initialize RVC inference
            self.rvc_inference = RVCInference(
                model_path=self.config.model_path,
                index_path=self.config.index_path,
                device=self.backend,
                f0_method=self.config.pitch_extract_method,
                f0_min=self.config.f0_min,
                f0_max=self.config.f0_max
            )
            
            # Initialize F0 extractor
            self.f0_extractor = get_f0_extractor(
                device=self.backend,
                enable_onnx=True
            )
            
            # Initialize VAD if available
            if VAD_AVAILABLE and self.config.enable_vad:
                self.vad = webrtcvad.VAD(self.config.vad_sensitivity)
            
            # Initialize audio
            if AUDIO_AVAILABLE:
                self._initialize_audio()
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            return False
    
    def _initialize_audio(self):
        """Initialize audio streams."""
        if not AUDIO_AVAILABLE:
            return
        
        try:
            self.audio = pyaudio.PyAudio()
            
            # Get available audio devices
            self.available_devices = self._get_audio_devices()
            
            # Create input stream
            self.input_stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=self.device_config.channels,
                rate=self.device_config.sample_rate,
                input=True,
                input_device_index=self._get_device_index(self.device_config.input_device),
                frames_per_buffer=self.device_config.chunk_size,
                stream_callback=self._input_callback
            )
            
            # Create output stream
            self.output_stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=self.device_config.channels,
                rate=self.device_config.sample_rate,
                output=True,
                output_device_index=self._get_device_index(self.device_config.output_device),
                frames_per_buffer=self.device_config.chunk_size,
                stream_callback=self._output_callback
            )
            
            logger.info("Audio streams initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize audio: {e}")
            raise
    
    def _get_audio_devices(self) -> Dict[str, Any]:
        """Get list of available audio devices."""
        if not AUDIO_AVAILABLE:
            return {}
        
        devices = {
            'input': [],
            'output': []
        }
        
        try:
            for i in range(self.audio.get_device_count()):
                info = self.audio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    devices['input'].append({
                        'index': i,
                        'name': info['name'],
                        'channels': info['maxInputChannels']
                    })
                if info['maxOutputChannels'] > 0:
                    devices['output'].append({
                        'index': i,
                        'name': info['name'],
                        'channels': info['maxOutputChannels']
                    })
        except Exception as e:
            logger.warning(f"Failed to get audio devices: {e}")
        
        return devices
    
    def _get_device_index(self, device_name: Optional[str]) -> Optional[int]:
        """Get device index by name."""
        if not device_name or not self.available_devices:
            return None
        
        for device_type in ['input', 'output']:
            for device in self.available_devices[device_type]:
                if device_name.lower() in device['name'].lower():
                    return device['index']
        
        return None
    
    def _input_callback(self, in_data, frame_count, time_info, status) -> Tuple[bytes, int]:
        """Callback for audio input."""
        if status:
            logger.warning(f"Input stream status: {status}")
        
        if not self.is_running:
            return (None, pyaudio.paAbort)
        
        try:
            # Convert audio data
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # Add to input queue
            if not self.audio_queue.full():
                self.audio_queue.put(audio_data)
            else:
                logger.warning("Input queue full, dropping audio frame")
            
            return (None, pyaudio.paContinue)
            
        except Exception as e:
            logger.error(f"Input callback error: {e}")
            return (None, pyaudio.paAbort)
    
    def _output_callback(self, out_data, frame_count, time_info, status) -> Tuple[bytes, int]:
        """Callback for audio output."""
        if status:
            logger.warning(f"Output stream status: {status}")
        
        if not self.is_running:
            return (b'\x00' * len(out_data), pyaudio.paAbort)
        
        try:
            # Get processed audio from output queue
            if not self.output_queue.empty():
                processed_audio = self.output_queue.get_nowait()
                
                # Ensure correct length
                if len(processed_audio) < len(out_data):
                    padding = np.zeros(len(out_data) - len(processed_audio), dtype=np.float32)
                    processed_audio = np.concatenate([processed_audio, padding])
                elif len(processed_audio) > len(out_data):
                    processed_audio = processed_audio[:len(out_data)]
                
                return (processed_audio.tobytes(), pyaudio.paContinue)
            else:
                # Return silence if no processed audio available
                return (b'\x00' * len(out_data), pyaudio.paContinue)
                
        except Exception as e:
            logger.error(f"Output callback error: {e}")
            return (b'\x00' * len(out_data), pyaudio.paContinue)
    
    def start(self) -> bool:
        """Start the voice changer."""
        if self.is_running:
            logger.warning("Voice changer is already running")
            return True
        
        try:
            # Initialize if not already done
            if not self.rvc_inference or not self.f0_extractor:
                if not self.initialize():
                    return False
            
            # Start audio streams
            if self.input_stream and self.output_stream:
                self.input_stream.start_stream()
                self.output_stream.start_stream()
            
            # Start processing thread
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.start()
            
            logger.info("Voice changer started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start voice changer: {e}")
            self.stop()
            return False
    
    def stop(self):
        """Stop the voice changer."""
        if not self.is_running:
            return
        
        try:
            self.is_running = False
            
            # Stop processing thread
            if self.processing_thread:
                self.processing_thread.join(timeout=1.0)
            
            # Stop audio streams
            if self.input_stream:
                self.input_stream.stop_stream()
                self.input_stream.close()
            
            if self.output_stream:
                self.output_stream.stop_stream()
                self.output_stream.close()
            
            if self.audio:
                self.audio.terminate()
            
            # Clear queues
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            
            while not self.output_queue.empty():
                try:
                    self.output_queue.get_nowait()
                except queue.Empty:
                    break
            
            logger.info("Voice changer stopped")
            
        except Exception as e:
            logger.error(f"Error stopping voice changer: {e}")
    
    def pause(self):
        """Pause voice conversion (continue processing but use original audio)."""
        self.is_paused = True
        logger.info("Voice changer paused")
    
    def resume(self):
        """Resume voice conversion."""
        self.is_paused = False
        logger.info("Voice changer resumed")
    
    def _processing_loop(self):
        """Main processing loop."""
        while self.is_running:
            try:
                # Get audio data
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get_nowait()
                    start_time = time.time()
                    
                    # Apply VAD if enabled
                    if self.vad and self.config.enable_vad:
                        if self._is_voiced(audio_data):
                            # Process with voice conversion
                            processed_audio = self._process_audio(audio_data)
                            self.stats['vad_triggered'] += 1
                        else:
                            # Pass through original audio
                            processed_audio = audio_data
                    else:
                        # Process with voice conversion
                        processed_audio = self._process_audio(audio_data)
                    
                    # Apply crossfade
                    processed_audio = self._apply_crossfade(processed_audio)
                    
                    # Add to output queue
                    if not self.output_queue.full():
                        self.output_queue.put(processed_audio)
                    
                    # Update statistics
                    processing_time = time.time() - start_time
                    self.stats['processing_time'] = processing_time
                    self.stats['frames_processed'] += 1
                    
                else:
                    time.sleep(0.001)  # Small delay to prevent busy waiting
                    
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                time.sleep(0.1)  # Prevent tight error loop
    
    def _process_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Process audio with voice conversion."""
        if self.is_paused:
            return audio_data
        
        try:
            # Apply pre-processing
            processed_audio = self._preprocess_audio(audio_data)
            
            # Extract F0
            f0, _ = self.f0_extractor.extract_f0(
                processed_audio,
                method=self.config.pitch_extract_method,
                sample_rate=self.device_config.sample_rate
            )
            
            # Apply voice conversion
            converted_audio = self.rvc_inference.convert_audio(
                processed_audio,
                f0,
                pitch_shift=self.config.pitch_shift,
                filter_radius=self.config.filter_radius,
                rms_mix_rate=self.config.rms_mix_rate,
                protect=self.config.protect
            )
            
            # Apply post-processing
            converted_audio = self._postprocess_audio(converted_audio)
            
            return converted_audio
            
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return audio_data  # Return original audio on error
    
    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Preprocess audio before voice conversion."""
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Apply auto-gain if enabled
        if self.config.enable_auto_gain:
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                target_rms = 0.1  # Target RMS level
                audio = audio * (target_rms / rms)
        
        return audio
    
    def _postprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Postprocess audio after voice conversion."""
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Apply gain control
        audio = audio * 0.8  # Prevent clipping
        
        return audio
    
    def _is_voiced(self, audio_data: np.ndarray) -> bool:
        """Check if audio contains voice using VAD."""
        if not self.vad or not self.config.enable_vad:
            return True
        
        try:
            # Convert to 16-bit PCM
            audio_16bit = (audio_data * 32767).astype(np.int16)
            
            # Apply VAD (expects 16kHz, mono, 16-bit)
            audio_vad = librosa.resample(audio_16bit, 
                                       orig_sr=self.device_config.sample_rate,
                                       target_sr=16000)
            
            # WebRTC VAD requires specific frame sizes
            frame_duration = 30  # ms
            frame_size = int(16000 * frame_duration / 1000)
            
            if len(audio_vad) < frame_size:
                return True  # Not enough data, treat as voiced
            
            # Check first frame
            is_speech = self.vad.is_speech(
                audio_vad[:frame_size].tobytes(),
                sample_rate=16000
            )
            
            return is_speech
            
        except Exception as e:
            logger.warning(f"VAD error: {e}")
            return True  # Default to voiced on error
    
    def _apply_crossfade(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply crossfade to smooth transitions."""
        if len(self.crossfade_buffer) == 0:
            return audio_data
        
        crossfade_length = int(self.config.crossfade_size * self.device_config.sample_rate)
        
        if len(audio_data) <= crossfade_length:
            # If audio is too short, just concatenate
            result = np.concatenate([self.crossfade_buffer, audio_data])
            self.crossfade_buffer = result[-crossfade_length:] if len(result) > crossfade_length else result
            return result
        
        # Apply crossfade
        crossfade_audio = audio_data[:crossfade_length]
        main_audio = audio_data[crossfade_length:]
        
        # Create crossfade envelope
        fade_in = np.linspace(0, 1, crossfade_length)
        fade_out = np.linspace(1, 0, crossfade_length)
        
        # Apply crossfade
        crossfade_result = (self.crossfade_buffer[-crossfade_length:] * fade_out + 
                          crossfade_audio * fade_in)
        
        # Combine with previous buffer and main audio
        result = np.concatenate([
            self.crossfade_buffer[:-crossfade_length] if len(self.crossfade_buffer) > crossfade_length else [],
            crossfade_result,
            main_audio
        ])
        
        # Update crossfade buffer
        self.crossfade_buffer = result[-crossfade_length:] if len(result) > crossfade_length else result
        
        return result
    
    def update_config(self, **kwargs):
        """Update configuration parameters in real-time."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated {key} to {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        stats = self.stats.copy()
        stats.update({
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'backend': self.backend,
            'audio_queue_size': self.audio_queue.qsize(),
            'output_queue_size': self.output_queue.qsize(),
            'buffer_sizes': {
                'input': len(self.input_buffer),
                'output': len(self.output_buffer),
                'crossfade': len(self.crossfade_buffer)
            }
        })
        return stats
    
    def save_settings(self, filepath: Union[str, Path]):
        """Save current settings to file."""
        settings = {
            'config': self.config.__dict__,
            'device_config': self.device_config.__dict__,
            'backend': self.backend
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(settings, f, indent=2)
        
        logger.info(f"Settings saved to {filepath}")
    
    def load_settings(self, filepath: Union[str, Path]):
        """Load settings from file."""
        import json
        
        with open(filepath, 'r') as f:
            settings = json.load(f)
        
        # Update configurations
        if 'config' in settings:
            for key, value in settings['config'].items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        if 'device_config' in settings:
            for key, value in settings['device_config'].items():
                if hasattr(self.device_config, key):
                    setattr(self.device_config, key, value)
        
        if 'backend' in settings:
            self.backend = settings['backend']
        
        logger.info(f"Settings loaded from {filepath}")


# Global voice changer instance
_voice_changer = None

def get_voice_changer(config: VoiceChangerConfig,
                     device_config: AudioDeviceConfig,
                     backend: str = "auto") -> EnhancedRealTimeVoiceChanger:
    """Get or create global voice changer instance."""
    global _voice_changer
    if _voice_changer is None:
        _voice_changer = EnhancedRealTimeVoiceChanger(
            config=config,
            device_config=device_config,
            backend=backend
        )
    return _voice_changer


import os
import sys
import torch
import logging
import warnings
import time
import numpy as np
from pathlib import Path
from functools import lru_cache
from contextlib import nullcontext

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Import Rich logging system
from .lib.rich_logging import logger as rich_logger, RICH_AVAILABLE

# Import path manager
from .lib.path_manager import path

# Import Vietnamese-RVC utilities with fallbacks
from .lib.utils import (
    load_audio, 
    check_assets, 
    clear_gpu_cache, 
    extract_median_f0,
    proposal_f0_up_key,
    autotune_f0,
    circular_write
)

# Import Vietnamese-RVC inference components
try:
    from .rvc.infer.conversion.pipeline import Pipeline
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

try:
    from .rvc.infer.conversion.audio_processing import preprocess, postprocess
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False

# Import KRVC kernel for performance optimization
try:
    from .krvc_kernel import (
        KRVCFeatureExtractor,
        krvc_speed_optimize,
        KRVCInferenceOptimizer,
        KRVCPerformanceMonitor
    )
    KRVC_AVAILABLE = True
except ImportError:
    KRVC_AVAILABLE = False

# Import GPU optimization
try:
    from .gpu_optimization import get_gpu_optimizer
    GPU_OPTIMIZATION_AVAILABLE = True
except ImportError:
    GPU_OPTIMIZATION_AVAILABLE = False

# Initialize optimizations
def _initialize_performance():
    """Initialize performance optimizations without circular imports"""
    if KRVC_AVAILABLE:
        try:
            krvc_speed_optimize()
            print("KRVC Kernel loaded successfully - Enhanced performance mode active")
        except Exception as e:
            print(f"KRVC initialization failed: {e}")

    if GPU_OPTIMIZATION_AVAILABLE:
        try:
            gpu_optimizer = get_gpu_optimizer()
            gpu_settings = gpu_optimizer.get_optimal_settings()
            gpu_optimizer.optimize_memory()
            print(f"GPU Optimization initialized - {gpu_optimizer.gpu_info['type']} detected")
        except Exception as e:
            print(f"GPU optimization failed: {e}")

# Initialize performance optimizations
_initialize_performance()

# Audio processing imports with fallbacks
try:
    from pedalboard import Pedalboard, Reverb
    from pedalboard.io import AudioFile
    PEDALBOARD_AVAILABLE = True
except ImportError:
    PEDALBOARD_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYTHONWAV_AVAILABLE = True
except ImportError:
    PYTHONWAV_AVAILABLE = False

try:
    from audio_separator.separator import Separator
    SEPARATOR_AVAILABLE = True
except ImportError:
    SEPARATOR_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

class VoiceConverter:
    """
    Enhanced Voice Converter based on Vietnamese-RVC implementation
    Integrates Rich logging, KRVC optimizations, and enhanced error handling
    """
    
    def __init__(self, model_path: str, sid: int = 0, config=None):
        self.model_path = model_path
        self.sid = sid
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_half = torch.cuda.is_available()
        
        # Vietnamese-RVC configuration
        self.config = config or self._get_default_config()
        
        # Initialize model components
        self.hubert_model = None
        self.tgt_sr = 40000  # Default target sample rate
        self.net_g = None
        self.vc = None
        self.cpt = None
        self.version = "v2"  # Default to v2
        self.n_spk = 1
        self.use_f0 = 1
        self.loaded_model = None
        self.vocoder = "Default"
        self.checkpointing = False
        self.sample_rate = 16000
        self.energy = False
        
        # Performance optimizations
        self.performance_mode = KRVC_AVAILABLE or GPU_OPTIMIZATION_AVAILABLE
        
        # Initialize the voice converter
        self._initialize()
    
    def _get_default_config(self):
        """Get default configuration matching Vietnamese-RVC structure"""
        class Config:
            def __init__(self):
                self.x_pad = 1.25
                self.x_query = 10
                self.x_center = 60
                self.x_max = 65
                self.device = self.device
                self.is_half = self.is_half
                
        return Config()
    
    def _initialize(self):
        """Initialize the voice converter"""
        try:
            # Use print instead of logger to avoid circular imports during initialization
            with self._get_status_context("🎵 Initializing Voice Converter..."):
                self.get_vc(self.model_path, self.sid)
                print(f"✅ Voice Converter initialized successfully")
                print(f"📊 Device: {self.device}, Half precision: {self.is_half}")
                print(f"🤖 Model: {Path(self.model_path).name}")
        except Exception as e:
            print(f"❌ Failed to initialize Voice Converter: {e}")
            raise
    
    def _get_status_context(self, message):
        """Get status context - returns dummy context manager to avoid circular imports"""
        class DummyStatus:
            def __enter__(self):
                print(message)
                return self
            def __exit__(self, *args):
                pass
        return DummyStatus()
    
    def convert_audio(self, 
                     audio_input_path: str, 
                     audio_output_path: str,
                     index_path: str = "",
                     embedder_model: str = "contentvec",
                     pitch: int = 0,
                     f0_method: str = "rmvpe",
                     index_rate: float = 0.5,
                     rms_mix_rate: float = 1.0,
                     protect: float = 0.33,
                     hop_length: int = 64,
                     f0_autotune: bool = False,
                     f0_autotune_strength: float = 1.0,
                     filter_radius: int = 3,
                     clean_audio: bool = False,
                     clean_strength: float = 0.7,
                     export_format: str = "wav",
                     resample_sr: int = 0,
                     checkpointing: bool = False,
                     f0_file: str = "",
                     f0_onnx: bool = False,
                     embedders_mode: str = "fairseq",
                     formant_shifting: bool = False,
                     formant_qfrency: float = 0.8,
                     formant_timbre: float = 0.8,
                     split_audio: bool = False,
                     proposal_pitch: bool = False,
                     proposal_pitch_threshold: float = 255.0,
                     audio_processing: bool = False,
                     alpha: float = 0.5,
                     batch_processing: bool = False):
        """
        Convert audio using Vietnamese-RVC pipeline with enhanced logging
        """
        
        start_time = time.time()
        self.checkpointing = checkpointing
        
        # Log conversion parameters
        rich_logger.header("🎯 Audio Conversion Parameters")
        rich_logger.info(f"Input: {Path(audio_input_path).name}")
        rich_logger.info(f"Output: {Path(audio_output_path).name}")
        rich_logger.info(f"Model: {Path(self.model_path).name}")
        rich_logger.info(f"Pitch shift: {pitch} semitones")
        rich_logger.info(f"F0 method: {f0_method}")
        rich_logger.info(f"Index rate: {index_rate}")
        rich_logger.info(f"Quality protection: {protect}")
        
        try:
            with rich_logger.status("🎵 Loading and processing audio..."):
                # Load audio
                audio = load_audio(audio_input_path, self.sample_rate, 
                                 formant_shifting=formant_shifting,
                                 formant_qfrency=formant_qfrency,
                                 formant_timbre=formant_timbre)
                
                if audio_processing and AUDIO_PROCESSING_AVAILABLE:
                    audio = preprocess(audio, self.sample_rate, device=self.device)
                
                # Normalize audio
                audio_max = np.abs(audio).max() / 0.95
                if audio_max > 1:
                    audio /= audio_max
                
                # Load embedder model if not already loaded
                if not self.hubert_model:
                    rich_logger.info(f"Loading embedder model: {embedder_model}")
                    from .lib.utils import ensure_embedder_available
                    embedder_path = ensure_embedder_available(embedder_model, auto_download=True)
                    
                    if embedder_path:
                        rich_logger.info(f"Embedder model found at: {embedder_path}")
                        # Load embedder model logic here
                        from .lib.utils import load_embedders_model
                        models = load_embedders_model(embedder_model, embedders_mode)
                    else:
                        rich_logger.warning(f"Embedder model '{embedder_model}' not available")
                        models = None
                    
                    if isinstance(models, torch.nn.Module):
                        models = models.to(
                            torch.float16 if self.is_half else torch.float32
                        ).eval().to(self.device)
                    self.hubert_model = models
                
            # Handle audio splitting for long files
            if split_audio:
                rich_logger.info("Splitting audio for processing...")
                from .lib.utils import cut
                chunks = cut(audio, self.sample_rate, db_thresh=-60, min_interval=500)
                rich_logger.info(f"Split into {len(chunks)} chunks")
            else:
                chunks = [(audio, 0, 0)]
            
            # Process audio chunks
            converted_chunks = []
            total_chunks = len(chunks)
            
            if total_chunks > 1:
                rich_logger.info(f"Processing {total_chunks} audio chunks...")
            
            for i, (waveform, start, end) in enumerate(chunks):
                if total_chunks > 1:
                    rich_logger.info(f"Processing chunk {i+1}/{total_chunks}")
                
                # Convert chunk
                converted_chunk = self._convert_chunk(
                    waveform, start, end,
                    pitch, f0_method, index_rate, rms_mix_rate, protect,
                    hop_length, f0_autotune, f0_autotune_strength, filter_radius,
                    f0_file, f0_onnx, embedders_mode,
                    proposal_pitch, proposal_pitch_threshold,
                    index_path, alpha
                )
                
                converted_chunks.append((start, end, converted_chunk))
            
            # Restore audio chunks
            with rich_logger.status("🔄 Restoring audio chunks..."):
                if len(chunks) > 1:
                    from .lib.utils import restore
                    audio_output = restore(converted_chunks, len(audio), 
                                         dtype=converted_chunks[0][2].dtype)
                else:
                    audio_output = converted_chunks[0][2]
            
            # Post-process audio
            if audio_processing and AUDIO_PROCESSING_AVAILABLE:
                audio_output = postprocess(audio_output, self.tgt_sr, audio, 
                                         self.sample_rate, device=self.device)
            
            # Resample if needed
            if self.tgt_sr != resample_sr and resample_sr > 0:
                rich_logger.info(f"Resampling from {self.tgt_sr}Hz to {resample_sr}Hz")
                audio_output = librosa.resample(audio_output, orig_sr=self.tgt_sr, 
                                              target_sr=resample_sr, res_type="soxr_vhq")
                self.tgt_sr = resample_sr
            
            # Apply noise reduction
            if clean_audio:
                rich_logger.info("Applying noise reduction...")
                from .lib.tools.noisereduce import TorchGate
                if not hasattr(self, "tg"):
                    self.tg = TorchGate(self.tgt_sr, prop_decrease=clean_strength).to(self.device)
                audio_output = self.tg(
                    torch.from_numpy(audio_output).unsqueeze(0).to(self.device).float()
                ).squeeze(0).cpu().detach().numpy()
            
            # Ensure output length matches input
            if len(audio) / self.sample_rate > len(audio_output) / self.tgt_sr:
                padding_length = int(np.round(len(audio) / self.sample_rate * self.tgt_sr) - len(audio_output))
                padding = np.zeros(padding_length, dtype=audio_output.dtype)
                audio_output = np.concatenate([audio_output, padding])
            
            # Save audio file
            with rich_logger.status("💾 Saving audio file..."):
                try:
                    import soundfile as sf
                    sf.write(audio_output_path, audio_output, self.tgt_sr, format=export_format)
                except Exception as e:
                    rich_logger.warning(f"Failed to save with soundfile, using librosa: {e}")
                    audio_output = librosa.resample(audio_output, orig_sr=self.tgt_sr, 
                                                  target_sr=48000, res_type="soxr_vhq")
                    sf.write(audio_output_path, audio_output, 48000, format=export_format)
            
            elapsed_time = time.time() - start_time
            rich_logger.success(f"Conversion completed successfully!")
            rich_logger.info(f"Time taken: {elapsed_time:.2f} seconds")
            rich_logger.info(f"Output: {audio_output_path}")
            
            return audio_output_path
            
        except Exception as e:
            import traceback
            rich_logger.error(f"Conversion failed: {e}")
            rich_logger.debug(traceback.format_exc())
            raise
    
    def _convert_chunk(self, waveform, start, end, pitch, f0_method, index_rate, 
                      rms_mix_rate, protect, hop_length, f0_autotune, 
                      f0_autotune_strength, filter_radius, f0_file, f0_onnx,
                      embedders_mode, proposal_pitch, proposal_pitch_threshold,
                      index_path, alpha):
        """Convert a single audio chunk"""
        
        if not PIPELINE_AVAILABLE:
            raise RuntimeError("Vietnamese-RVC pipeline not available")
        
        # Get F0 extraction generator
        from .lib.predictors.Generator import Generator
        f0_generator = Generator(
            sample_rate=self.sample_rate,
            hop_length=hop_length,
            f0_min=50,
            f0_max=1100,
            device=self.device,
            f0_onnx_mode=f0_onnx,
            auto_download_models=True
        )
        
        # Extract features and F0
        with torch.no_grad():
            # Extract hubert features
            from .lib.utils import extract_features
            feats = extract_features(self.hubert_model, 
                                   torch.from_numpy(waveform).to(self.device).float(), 
                                   self.version, self.device)
            
            # Calculate F0
            if f0_file and os.path.exists(f0_file):
                # Load F0 from file
                f0 = np.load(f0_file)
            else:
                # Extract F0 using generator
                pitch, pitchf = f0_generator.calculator(
                    x_pad=0, f0_method=f0_method, x=waveform,
                    f0_up_key=pitch, p_len=len(waveform)//hop_length,
                    filter_radius=filter_radius,
                    f0_autotune=f0_autotune,
                    f0_autotune_strength=f0_autotune_strength,
                    proposal_pitch=proposal_pitch,
                    proposal_pitch_threshold=proposal_pitch_threshold
                )
                f0 = pitchf
            
            # Apply the Vietnamese-RVC conversion pipeline
            audio_output = self.vc.pipeline(
                logger=rich_logger,
                model=self.hubert_model,
                net_g=self.net_g,
                sid=self.sid,
                audio=waveform,
                f0_up_key=pitch,
                f0_method=f0_method,
                file_index=index_path.strip().strip('"').strip("\n").strip('"'),
                index_rate=index_rate,
                pitch_guidance=self.use_f0,
                filter_radius=filter_radius,
                rms_mix_rate=rms_mix_rate,
                version=self.version,
                protect=protect,
                hop_length=hop_length,
                f0_autotune=f0_autotune,
                f0_autotune_strength=f0_autotune_strength,
                f0_file=f0_file,
                f0_onnx=f0_onnx,
                proposal_pitch=proposal_pitch,
                proposal_pitch_threshold=proposal_pitch_threshold,
                energy_use=self.energy,
                del_onnx=True,
                alpha=alpha
            )
        
        return audio_output
    
    def get_vc(self, weight_root: str, sid: int):
        """Load the voice conversion model"""
        if sid == "" or sid == []:
            self.cleanup()
            clear_gpu_cache()
        
        if not self.loaded_model or self.loaded_model != weight_root:
            self.loaded_model = weight_root
            from .lib.utils import load_model
            self.cpt = load_model(weight_root)
            if self.cpt is not None:
                self.setup()
    
    def cleanup(self):
        """Clean up GPU memory and models"""
        if self.hubert_model is not None:
            del self.net_g, self.n_spk, self.vc, self.hubert_model, self.tgt_sr
            self.hubert_model = self.net_g = self.n_spk = self.vc = self.tgt_sr = None
            clear_gpu_cache()
        
        del self.net_g, self.cpt
        clear_gpu_cache()
        self.cpt = None
    
    def setup(self):
        """Setup the voice conversion model"""
        if self.cpt is not None:
            if self.loaded_model.endswith(".pth"):
                # Load PyTorch model
                from .lib.algorithm.synthesizers import Synthesizer
                
                self.tgt_sr = self.cpt["config"][-1]
                self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]
                
                self.use_f0 = self.cpt.get("f0", 1)
                self.version = self.cpt.get("version", "v2")
                self.vocoder = self.cpt.get("vocoder", "Default")
                self.energy = self.cpt.get("energy", False)
                
                if self.vocoder != "Default":
                    self.is_half = False
                
                self.net_g = Synthesizer(
                    *self.cpt["config"], 
                    use_f0=self.use_f0, 
                    text_enc_hidden_dim=768 if self.version == "v2" else 256,
                    vocoder=self.vocoder,
                    checkpointing=self.checkpointing,
                    energy=self.energy
                )
                del self.net_g.enc_q
                
                self.net_g.load_state_dict(self.cpt["weight"], strict=False)
                self.net_g.eval().to(self.device)
                self.net_g = self.net_g.to(torch.float16 if self.is_half else torch.float32)
                self.n_spk = self.cpt["config"][-3]
            else:
                # Load ONNX model
                self.net_g = self.cpt.to(self.device)
                self.tgt_sr = self.cpt.cpt.get("tgt_sr", 40000)
                self.use_f0 = self.cpt.cpt.get("f0", 1)
                self.version = self.cpt.cpt.get("version", "v2")
                self.energy = self.cpt.cpt.get("energy", False)
            
            if PIPELINE_AVAILABLE:
                self.vc = Pipeline(self.tgt_sr, self.config)

# Enhanced conversion functions
def convert_audio(input_path: str, 
                 output_path: str,
                 model_path: str,
                 index_path: str = "",
                 **kwargs) -> str:
    """
    Enhanced audio conversion function with Rich logging
    
    Args:
        input_path: Path to input audio file
        output_path: Path for output audio file
        model_path: Path to RVC model file
        **kwargs: Additional conversion parameters
        
    Returns:
        str: Path to output file
    """
    rich_logger.header("🎵 Advanced RVC Voice Conversion")
    rich_logger.info(f"Input: {Path(input_path).name}")
    rich_logger.info(f"Model: {Path(model_path).name}")
    
    # Check and auto-download required assets following Vietnamese-RVC patterns
    f0_method = kwargs.get('f0_method', 'rmvpe')
    embedder_model = kwargs.get('embedder_model', 'contentvec')
    f0_onnx = kwargs.get('f0_onnx', False)
    embedders_mode = kwargs.get('embedders_mode', 'fairseq')
    
    # Import enhanced auto-download functions
    try:
        from .lib.utils import ensure_f0_model_available, ensure_embedder_available
        f0_model_path = ensure_f0_model_available(f0_method, auto_download=True)
        embedder_path = ensure_embedder_available(embedder_model, auto_download=True)
        
        if f0_model_path:
            rich_logger.info(f"✅ F0 model ready: {Path(f0_model_path).name}")
        else:
            rich_logger.warning(f"⚠️ F0 model '{f0_method}' not available")
            
        if embedder_path:
            rich_logger.info(f"✅ Embedder model ready: {Path(embedder_path).name}")
        else:
            rich_logger.warning(f"⚠️ Embedder model '{embedder_model}' not available")
            
    except ImportError:
        # Fallback to basic asset checking
        check_assets(f0_method, embedder_model, f0_onnx, embedders_mode)
    
    # Create voice converter
    converter = VoiceConverter(model_path)
    
    # Perform conversion
    return converter.convert_audio(
        audio_input_path=input_path,
        audio_output_path=output_path,
        index_path=index_path,
        **kwargs
    )

def batch_convert(input_dir: str, 
                 output_dir: str,
                 model_path: str,
                 **kwargs) -> list:
    """
    Batch convert multiple audio files
    
    Args:
        input_dir: Directory containing input audio files
        output_dir: Directory for output audio files
        model_path: Path to RVC model file
        **kwargs: Additional conversion parameters
        
    Returns:
        list: List of converted file paths
    """
    from pathlib import Path
    import shutil
    
    rich_logger.header("📁 Batch Audio Conversion")
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Supported audio formats
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.opus', '.m4a', '.mp4', '.aac', '.alac', '.wma', '.aiff', '.webm', '.ac3'}
    
    # Find audio files
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(input_path.glob(f"*{ext}"))
        audio_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    if not audio_files:
        rich_logger.warning("No audio files found in directory")
        return []
    
    rich_logger.info(f"Found {len(audio_files)} audio files to convert")
    
    # Convert files
    converted_files = []
    for audio_file in audio_files:
        try:
            output_file = output_path / f"{audio_file.stem}_converted.wav"
            
            rich_logger.info(f"Converting: {audio_file.name}")
            
            result = convert_audio(
                input_path=str(audio_file),
                output_path=str(output_file),
                model_path=model_path,
                **kwargs
            )
            
            converted_files.append(result)
            rich_logger.success(f"Completed: {audio_file.name}")
            
        except Exception as e:
            rich_logger.error(f"Failed to convert {audio_file.name}: {e}")
            continue
    
    rich_logger.success(f"Batch conversion completed: {len(converted_files)}/{len(audio_files)} files converted")
    
    return converted_files

# Export functions
__all__ = [
    'VoiceConverter', 
    'convert_audio', 
    'batch_convert',
    'rich_logger',
    'RICH_AVAILABLE'
]

# Initialize optimizations after all imports are complete
try:
    # Initialize performance optimizations first
    _initialize_performance()
    
    # Initialize performance monitor if KRVC is available
    if KRVC_AVAILABLE:
        try:
            from .krvc_kernel import KRVCPerformanceMonitor
            global performance_monitor
            performance_monitor = KRVCPerformanceMonitor()
        except:
            pass
    
    # Initialize application logging (after other systems are ready)
    rich_logger.header("🎯 Advanced RVC Inference - Enhanced Vietnamese-RVC Implementation")
    rich_logger.info("Rich logging system initialized")
    rich_logger.info(f"Vietnamese-RVC pipeline available: {PIPELINE_AVAILABLE}")
    rich_logger.info(f"KRVC optimizations available: {KRVC_AVAILABLE}")
    rich_logger.info(f"GPU optimizations available: {GPU_OPTIMIZATION_AVAILABLE}")
    
except Exception as e:
    # Fallback logging if Rich is not available
    print(f"Initialization warning: {e}")
    print("Advanced RVC Inference initialized with fallback logging")

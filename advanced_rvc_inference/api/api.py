"""
High-level API for Advanced RVC Inference.

Provides programmatic access to voice conversion, training,
and audio processing functionality.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
from dataclasses import dataclass

# Initialize logger
logger = logging.getLogger(__name__)

# Lazy imports for heavy dependencies
_TORCH_AVAILABLE = False
_GRADIO_AVAILABLE = False

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    import gradio as gr

    _GRADIO_AVAILABLE = True
except ImportError:
    pass


@dataclass
class RVCConfig:
    """Configuration for RVC inference."""

    device: str = "cuda:0"
    half_precision: bool = True
    num_threads: int = 0
    is_half: bool = True

    # Device configuration
    x_pad: int = 3
    x_query: int = 10
    x_center: int = 60
    x_max: int = 65

    # Paths
    weights_path: str = "assets/weights"
    logs_path: str = "assets/logs"
    assets_path: str = "assets"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RVCConfig":
        """Create config from dictionary."""
        return cls(
            device=config_dict.get("device", "cuda:0"),
            half_precision=config_dict.get("fp16", True),
            num_threads=config_dict.get("num_threads", 0),
            is_half=config_dict.get("is_half", True),
            weights_path=config_dict.get("weights_path", "assets/weights"),
            logs_path=config_dict.get("logs_path", "assets/logs"),
            assets_path=config_dict.get("assets_path", "assets"),
        )


class RVCModel:
    """Represents a loaded RVC model."""

    def __init__(
        self,
        model_path: str,
        index_path: Optional[str] = None,
        device: str = "cuda:0",
        half: bool = True,
    ):
        """
        Initialize an RVC model.

        Args:
            model_path: Path to the model file (.pth or .onnx)
            index_path: Path to the index file
            device: Device to run inference on
            half: Whether to use half precision
        """
        self.model_path = model_path
        self.index_path = index_path
        self.device = device
        self.half = half
        self.model = None
        self.hubert = None
        self.index = None

    def load(self):
        """Load the model into memory."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Import heavy dependencies only when needed
        import torch

        from advanced_rvc_inference.core.model_utils import load_model

        self.model = load_model(self.model_path, device=self.device, half=self.half)
        logger.info(f"Loaded model: {self.model_path}")

    def unload(self):
        """Unload the model and clear GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None

        # Clear GPU cache
        if "cuda" in self.device:
            torch.cuda.empty_cache()

        logger.info(f"Unloaded model: {self.model_path}")


class RVCInference:
    """
    High-level interface for RVC voice conversion.

    Example:
        >>> rvc = RVCInference(device="cuda:0")
        >>> rvc.load_model("path/to/model.pth")
        >>> audio = rvc.infer("input.wav", pitch_change=0)
        >>> rvc.unload()
    """

    def __init__(
        self,
        device: Optional[str] = None,
        half_precision: bool = True,
        config: Optional[RVCConfig] = None,
    ):
        """
        Initialize the RVC inference engine.

        Args:
            device: Device to use (e.g., "cuda:0", "cpu")
            half_precision: Whether to use half precision
            config: Configuration object
        """
        self.config = config or RVCConfig()
        self.device = device or self.config.device
        self.half_precision = half_precision and self.config.half_precision
        self.model: Optional[RVCModel] = None

        # Set device
        if device is None:
            if _TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = "cuda:0"
            elif _TORCH_AVAILABLE and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        logger.info(f"Initialized RVCInference on device: {self.device}")

    def load_model(
        self,
        model_path: str,
        index_path: Optional[str] = None,
    ) -> RVCModel:
        """
        Load an RVC model.

        Args:
            model_path: Path to the model file
            index_path: Path to the index file (optional)

        Returns:
            RVCModel: The loaded model
        """
        self.model = RVCModel(
            model_path=model_path,
            index_path=index_path,
            device=self.device,
            half=self.half_precision,
        )
        self.model.load()
        return self.model

    def unload_model(self):
        """Unload the current model."""
        if self.model is not None:
            self.model.unload()
            self.model = None

    def infer(
        self,
        input_audio: str,
        pitch_change: int = 0,
        output_path: Optional[str] = None,
        format: str = "wav",
        f0_method: str = "rmvpe",
        protect: float = 0.33,
        index_rate: float = 0.5,
        rms_mix_rate: float = 1.0,
        filter_radius: int = 3,
        hop_length: int = 128,
        **kwargs,
    ) -> str:
        """
        Run voice conversion inference.

        Args:
            input_audio: Path to the input audio file
            pitch_change: Pitch shift in semitones
            output_path: Path to save the output (auto-generated if None)
            format: Output audio format
            f0_method: Pitch extraction method
            protect: Protection parameter
            index_rate: Index rate for retrieval
            rms_mix_rate: RMS mix rate
            filter_radius: Filter radius
            hop_length: Hop length for pitch extraction
            **kwargs: Additional parameters

        Returns:
            str: Path to the output audio file
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")

        # Generate output path if not specified
        if output_path is None:
            input_path = Path(input_audio)
            output_path = str(
                input_path.parent / f"{input_path.stem}_converted{input_path.suffix}"
            )

        # Run conversion
        from advanced_rvc_inference.rvc.infer.inference import convert

        convert(
            pitch=pitch_change,
            filter_radius=filter_radius,
            index_rate=index_rate,
            rms_mix_rate=rms_mix_rate,
            protect=protect,
            hop_length=hop_length,
            f0_method=f0_method,
            input_path=input_audio,
            output_path=output_path,
            pth_path=self.model.model_path,
            index_path=self.model.index_path,
            f0_autotune=False,
            clean_audio=False,
            clean_strength=0.5,
            export_format=format,
            embedder_model="contentvec_base",
            resample_sr=0,
            split_audio=False,
            f0_autotune_strength=0.5,
            checkpointing=False,
            f0_onnx=False,
            embedders_mode="fairseq",
            formant_shifting=False,
            formant_qfrency=0,
            formant_timbre=0,
            f0_file="",
            proposal_pitch=False,
            proposal_pitch_threshold=0.05,
            audio_processing=False,
            alpha=0.5,
        )

        return output_path

    def infer_batch(
        self,
        input_dir: str,
        output_dir: str,
        pitch_change: int = 0,
        format: str = "wav",
        f0_method: str = "rmvpe",
        **kwargs,
    ) -> List[str]:
        """
        Run batch voice conversion on all audio files in a directory.

        Args:
            input_dir: Directory containing input audio files
            output_dir: Directory for output files
            pitch_change: Pitch shift in semitones
            format: Output audio format
            f0_method: Pitch extraction method
            **kwargs: Additional parameters

        Returns:
            List[str]: Paths to output audio files
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")

        from advanced_rvc_inference.rvc.infer.inference import convert

        convert(
            pitch=pitch_change,
            filter_radius=3,
            index_rate=0.5,
            rms_mix_rate=1.0,
            protect=0.33,
            hop_length=128,
            f0_method=f0_method,
            input_path=input_dir,
            output_path=output_dir,
            pth_path=self.model.model_path,
            index_path=self.model.index_path,
            f0_autotune=False,
            clean_audio=False,
            clean_strength=0.5,
            export_format=format,
            embedder_model="contentvec_base",
            resample_sr=0,
            split_audio=False,
            f0_autotune_strength=0.5,
            checkpointing=False,
            f0_onnx=False,
            embedders_mode="fairseq",
            formant_shifting=False,
            formant_qfrency=0,
            formant_timbre=0,
            f0_file="",
            proposal_pitch=False,
            proposal_pitch_threshold=0.05,
            audio_processing=False,
            alpha=0.5,
        )

        # Return list of output files
        output_path = Path(output_dir)
        return list(output_path.glob(f"*.{format}"))

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.unload_model()


class RVCTrainer:
    """Interface for RVC model training."""

    def __init__(self, config: Optional[RVCConfig] = None):
        """
        Initialize the trainer.

        Args:
            config: Configuration object
        """
        self.config = config or RVCConfig()
        logger.info("RVCTrainer initialized")

    def train(
        self,
        name: str,
        training_data: str,
        validation_data: Optional[str] = None,
        epochs: int = 100,
        batch_size: int = 4,
        save_interval: int = 10,
        **kwargs,
    ):
        """
        Start training a model.

        Note: Full training requires the web interface.

        Args:
            name: Name for the training experiment
            training_data: Path to training data
            validation_data: Path to validation data
            epochs: Number of training epochs
            batch_size: Batch size
            save_interval: Save checkpoint every N epochs
            **kwargs: Additional parameters
        """
        logger.info(f"Starting training: {name}")
        logger.info("Note: Full training is best performed via the web interface")
        logger.info("Run: rvc-gui")
        logger.info("Then navigate to the Training tab")


class RVCRealtime:
    """Interface for real-time voice conversion."""

    def __init__(self, config: Optional[RVCConfig] = None):
        """
        Initialize real-time processing.

        Args:
            config: Configuration object
        """
        self.config = config or RVCConfig()
        self.model: Optional[RVCModel] = None
        logger.info("RVCRealtime initialized")

    def start(
        self,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
        sample_rate: int = 48000,
        chunk_size: int = 960,
    ):
        """
        Start real-time voice conversion.

        Note: Full real-time processing requires the web interface.

        Args:
            input_device: Input device index
            output_device: Output device index
            sample_rate: Sample rate
            chunk_size: Processing chunk size
        """
        logger.info("Starting real-time mode")
        logger.info("Note: Full real-time processing requires the web interface")
        logger.info("Run: rvc-gui")
        logger.info("Then navigate to the Realtime tab")

    def stop(self):
        """Stop real-time processing."""
        if self.model is not None:
            self.model.unload()
            self.model = None

logger = logging.getLogger(__name__)

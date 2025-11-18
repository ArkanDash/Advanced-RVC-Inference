"""
Training Configuration for Advanced RVC Inference
Handles model training, data preprocessing, and configuration management
"""

import os
import json
import torch
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pathlib import Path

@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    
    # Model settings
    model_name: str = "rvc_model"
    sample_rate: int = 48000
    model_version: str = "v2"  # v1 or v2
    pitch_guidance: bool = True
    
    # Training settings
    total_epochs: int = 300
    save_frequency: int = 50
    batch_size: int = 8
    learning_rate: float = 0.001
    gradient_accumulation_steps: int = 1
    
    # Feature extraction
    f0_method: str = "rmvpe"  # rmvpe, crepe, hybrid
    embedder_model: str = "hubert_base"  # hubert_base, contentvec
    hop_length: int = 160
    
    # Data preprocessing
    split_audio_mode: str = "Automatic"  # Automatic, Simple, Skip
    normalization_mode: str = "none"  # none, pre, post
    clean_dataset: bool = False
    process_effects: bool = False
    
    # Advanced settings
    use_pretrain: bool = True
    cache_in_gpu: bool = True
    save_only_latest: bool = True
    save_every_weights: bool = True
    custom_dataset: bool = False
    dataset_path: str = "dataset"
    
    # GPU settings
    gpu_ids: str = "0"
    use_mixed_precision: bool = True
    
    # Output settings
    output_dir: str = "logs"
    weights_dir: str = "weights"
    index_dir: str = "index"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "model_name": self.model_name,
            "sample_rate": self.sample_rate,
            "model_version": self.model_version,
            "pitch_guidance": self.pitch_guidance,
            "total_epochs": self.total_epochs,
            "save_frequency": self.save_frequency,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "f0_method": self.f0_method,
            "embedder_model": self.embedder_model,
            "hop_length": self.hop_length,
            "split_audio_mode": self.split_audio_mode,
            "normalization_mode": self.normalization_mode,
            "clean_dataset": self.clean_dataset,
            "process_effects": self.process_effects,
            "use_pretrain": self.use_pretrain,
            "cache_in_gpu": self.cache_in_gpu,
            "save_only_latest": self.save_only_latest,
            "save_every_weights": self.save_every_weights,
            "custom_dataset": self.custom_dataset,
            "dataset_path": self.dataset_path,
            "gpu_ids": self.gpu_ids,
            "use_mixed_precision": self.use_mixed_precision,
            "output_dir": self.output_dir,
            "weights_dir": self.weights_dir,
            "index_dir": self.index_dir
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})
    
    def save(self, config_path: str):
        """Save configuration to JSON file"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, config_path: str) -> 'TrainingConfig':
        """Load configuration from JSON file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)


class TrainingLogger:
    """Training logger with progress tracking"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.output_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log epoch start"""
        self.logger.info(f"Starting epoch {epoch}/{total_epochs}")
    
    def log_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        """Log epoch end with metrics"""
        self.logger.info(f"Epoch {epoch} completed:")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.6f}")
    
    def log_training_progress(self, step: int, total_steps: int, metrics: Dict[str, float]):
        """Log training step progress"""
        self.logger.info(f"Step {step}/{total_steps} - " + 
                        " | ".join([f"{k}: {v:.6f}" for k, v in metrics.items()]))
    
    def log_model_save(self, epoch: int, model_path: str):
        """Log model save event"""
        self.logger.info(f"Model saved at epoch {epoch}: {model_path}")
    
    def log_error(self, error_msg: str, exception: Exception = None):
        """Log error with optional exception"""
        if exception:
            self.logger.error(f"{error_msg}: {str(exception)}", exc_info=True)
        else:
            self.logger.error(error_msg)


def get_device() -> torch.device:
    """Get the best available device for training"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU for training")
    
    return device


def check_dependencies():
    """Check if required dependencies for training are available"""
    try:
        import torch
        import torchvision
        import torchaudio
        import librosa
        import soundfile
        import numpy as np
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        return False


def validate_training_data(dataset_path: str) -> bool:
    """Validate training dataset"""
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"Dataset path does not exist: {dataset_path}")
        return False
    
    # Check for audio files
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac'}
    audio_files = [f for f in dataset_path.iterdir() 
                   if f.suffix.lower() in audio_extensions]
    
    if len(audio_files) < 5:
        print(f"Insufficient audio files for training. Found {len(audio_files)}, need at least 5")
        return False
    
    print(f"Found {len(audio_files)} audio files for training")
    return True

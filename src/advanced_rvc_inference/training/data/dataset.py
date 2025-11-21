"""
RVC Dataset Handler for Advanced RVC Inference
Handles data loading, preprocessing, and feature extraction for training
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import librosa
import soundfile as sf
import pickle
from typing import Dict, List, Optional, Tuple, Any
import logging

class RVCDataSet(Dataset):
    """RVC Dataset for voice conversion training"""
    
    def __init__(
        self,
        dataset_path: str,
        sample_rate: int = 48000,
        hop_length: int = 160,
        f0_method: str = "rmvpe",
        embedder_model: str = "hubert_base",
        cache_features: bool = True
    ):
        self.dataset_path = Path(dataset_path)
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.f0_method = f0_method
        self.embedder_model = embedder_model
        self.cache_features = cache_features
        
        # Cache directories
        self.features_dir = self.dataset_path / "features"
        self.features_dir.mkdir(exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Load file list
        self.audio_files = self._load_audio_files()
        
        # Feature cache
        self.feature_cache = {}
        
        # Initialize feature extractors
        self._init_extractors()
        
        self.logger.info(f"RVC Dataset initialized with {len(self.audio_files)} audio files")
    
    def _load_audio_files(self) -> List[Path]:
        """Load list of audio files from dataset directory"""
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
        
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(self.dataset_path.glob(f"*{ext}"))
            audio_files.extend(self.dataset_path.glob(f"**/*{ext}"))
        
        # Remove duplicates and sort
        audio_files = sorted(list(set(audio_files)))
        
        if not audio_files:
            raise ValueError(f"No audio files found in {self.dataset_path}")
        
        return audio_files
    
    def _init_extractors(self):
        """Initialize feature extractors"""
        try:
            # Initialize hubert/contentvec embedder
            if self.embedder_model == "hubert_base":
                from transformers import Wav2Vec2Model, Wav2Vec2Config
                self.embedder = Wav2Vec2Model.from_pretrained("facebook/hubert-base-ls960")
            elif self.embedder_model == "contentvec":
                # ContentVec would be initialized here
                self.embedder = None  # Placeholder
            else:
                self.embedder = None  # Custom model
            
            if self.embedder:
                self.embedder.eval()
                
        except Exception as e:
            self.logger.warning(f"Failed to initialize embedder: {e}")
            self.embedder = None
        
        try:
            # Initialize F0 extractor
            if self.f0_method == "rmvpe":
                # RMVPE F0 extractor would be initialized here
                self.f0_extractor = None  # Placeholder
            elif self.f0_method == "crepe":
                # CREPE F0 extractor would be initialized here
                self.f0_extractor = None  # Placeholder
            else:
                self.f0_extractor = None  # Default or other methods
                
        except Exception as e:
            self.logger.warning(f"Failed to initialize F0 extractor: {e}")
            self.f0_extractor = None
    
    def _load_audio(self, audio_path: Path) -> np.ndarray:
        """Load audio file"""
        try:
            # Load audio with specified sample rate
            audio, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
            
            # Normalize audio
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.9
            
            return audio
            
        except Exception as e:
            self.logger.error(f"Failed to load audio {audio_path}: {e}")
            # Return silence as fallback
            return np.zeros(self.sample_rate * 10)  # 10 seconds of silence
    
    def _extract_features(self, audio: np.ndarray, file_id: int) -> Dict[str, Any]:
        """Extract features from audio"""
        features = {}
        
        try:
            # Compute STFT
            stft = librosa.stft(
                audio, 
                hop_length=self.hop_length, 
                n_fft=2048,
                win_length=2048
            )
            
            # Compute magnitude and phase
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                S=magnitude,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                n_mels=80
            )
            
            # Convert to log scale
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            features['stft_magnitude'] = magnitude
            features['stft_phase'] = phase
            features['mel_spectrogram'] = mel_spec
            
            # Extract F0 if F0 extractor is available
            if self.f0_extractor:
                f0 = self.f0_extractor.extract_f0(audio, self.sample_rate)
                features['f0'] = f0
            
            # Extract embeddings if embedder is available
            if self.embedder:
                # Convert audio to tensor and run through embedder
                audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
                with torch.no_grad():
                    embeddings = self.embedder(audio_tensor)
                    if hasattr(embeddings, 'last_hidden_state'):
                        features['embeddings'] = embeddings.last_hidden_state.squeeze(0).numpy()
            
            # Store original audio
            features['audio'] = audio
            
            # Save features to cache file if caching is enabled
            if self.cache_features:
                cache_file = self.features_dir / f"{file_id:04d}.pkl"
                with open(cache_file, 'wb') as f:
                    pickle.dump(features, f)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Failed to extract features: {e}")
            return {
                'stft_magnitude': np.zeros((1025, 100)),
                'stft_phase': np.zeros((1025, 100)),
                'mel_spectrogram': np.zeros((80, 100)),
                'f0': np.zeros(100),
                'embeddings': np.zeros((100, 768)),
                'audio': audio
            }
    
    def _get_cached_features(self, file_id: int) -> Optional[Dict[str, Any]]:
        """Get cached features if available"""
        if not self.cache_features:
            return None
        
        cache_file = self.features_dir / f"{file_id:04d}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cached features {cache_file}: {e}")
        
        return None
    
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single item from the dataset"""
        file_id = idx
        
        # Try to load cached features
        features = self._get_cached_features(file_id)
        
        if features is None:
            # Load audio file
            audio_path = self.audio_files[idx]
            audio = self._load_audio(audio_path)
            
            # Extract features
            features = self._extract_features(audio, file_id)
        
        # Convert to tensors
        item = {}
        
        # Audio tensor
        if 'audio' in features:
            item['audio'] = torch.from_numpy(features['audio']).float()
        
        # Mel spectrogram tensor
        if 'mel_spectrogram' in features:
            mel_spec = features['mel_spectrogram']
            # Pad or truncate to consistent length
            if mel_spec.shape[1] < 1000:
                mel_spec = np.pad(mel_spec, ((0, 0), (0, 1000 - mel_spec.shape[1])), mode='constant')
            else:
                mel_spec = mel_spec[:, :1000]
            item['mel_spectrogram'] = torch.from_numpy(mel_spec).float()
        
        # F0 tensor
        if 'f0' in features:
            f0 = features['f0']
            # Pad or truncate F0 to match mel spec length
            if len(f0) < 1000:
                f0 = np.pad(f0, (0, 1000 - len(f0)), mode='constant')
            else:
                f0 = f0[:1000]
            item['f0'] = torch.from_numpy(f0).float()
        
        # Embeddings tensor
        if 'embeddings' in features:
            embeddings = features['embeddings']
            # Truncate embeddings to match mel spec length
            embeddings = embeddings[:1000, :]
            item['embeddings'] = torch.from_numpy(embeddings).float()
        
        # Add metadata
        item['file_path'] = str(self.audio_files[idx])
        item['file_id'] = file_id
        item['sample_rate'] = self.sample_rate
        
        return item


class BatchSampler:
    """Batch sampler for RVC training with variable length sequences"""
    
    def __init__(self, dataset: RVCDataSet, batch_size: int = 8, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Create sequence length indices
        self.sequence_indices = self._create_sequence_indices()
        
    def _create_sequence_indices(self) -> List[Tuple[int, int, int]]:
        """Create indices for variable-length sequences"""
        indices = []
        
        for file_id, file_path in enumerate(self.dataset.audio_files):
            try:
                # Load audio file to get length
                audio, sr = librosa.load(str(file_path), sr=self.dataset.sample_rate)
                
                # Calculate sequence length (frames)
                sequence_length = len(audio) // self.dataset.hop_length
                
                # Create segments for this file
                segment_size = 8192 // self.dataset.hop_length  # Approximate segment size
                
                for start in range(0, sequence_length - segment_size + 1, segment_size // 2):
                    end = min(start + segment_size, sequence_length)
                    if end - start >= segment_size // 2:  # Minimum segment length
                        indices.append((file_id, start, end))
                        
            except Exception as e:
                logging.warning(f"Failed to process {file_path}: {e}")
                continue
        
        return indices
    
    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sequence_indices) // self.batch_size
        else:
            return (len(self.sequence_indices) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        # Shuffle indices
        indices = list(range(len(self.sequence_indices)))
        np.random.shuffle(indices)
        
        # Create batches
        for start_idx in range(0, len(indices), self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]
            
            if len(batch_indices) == self.batch_size or not self.drop_last:
                batch_files = []
                batch_starts = []
                batch_ends = []
                
                for idx in batch_indices:
                    file_id, start, end = self.sequence_indices[idx]
                    batch_files.append(file_id)
                    batch_starts.append(start)
                    batch_ends.append(end)
                
                yield (batch_files, batch_starts, batch_ends)


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate function for RVC training batches"""
    # This would handle batching of variable-length sequences
    # For now, return the batch as-is
    
    collated_batch = {}
    
    # Stack tensors
    tensor_keys = ['audio', 'mel_spectrogram', 'f0', 'embeddings']
    
    for key in tensor_keys:
        if key in batch[0]:
            tensors = [item[key] for item in batch]
            collated_batch[key] = torch.stack(tensors)
    
    # Handle string and other non-tensor data
    for key in batch[0]:
        if key not in tensor_keys:
            collated_batch[key] = [item[key] for item in batch]
    
    return collated_batch

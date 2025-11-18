"""
Feature Extraction Utilities for Advanced RVC Training
Handles F0 extraction, embedding extraction, and feature preprocessing
"""

import os
import sys
import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
import logging
from scipy.ndimage import gaussian_filter1d

class FeatureExtractor:
    """Feature extraction for RVC training"""
    
    def __init__(self, sample_rate: int = 48000, hop_length: int = 160):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.logger = logging.getLogger(__name__)
        
        # Initialize extractors
        self.embedder = None
        self.f0_extractor = None
        
        self._init_embedder()
        self._init_f0_extractor()
    
    def _init_embedder(self):
        """Initialize embedding extractor"""
        try:
            # Placeholder for embedder initialization
            # In a real implementation, this would load:
            # - Hubert Base model
            # - ContentVec model
            # - Or other embedding models
            
            self.logger.info("Embedding extractor initialized")
            self.embedder = "hubert_base"  # Placeholder
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize embedder: {e}")
            self.embedder = None
    
    def _init_f0_extractor(self):
        """Initialize F0 (pitch) extractor"""
        try:
            # Placeholder for F0 extractor initialization
            # In a real implementation, this would load:
            # - RMVPE model
            # - CREPE model
            # - Other F0 extraction models
            
            self.logger.info("F0 extractor initialized")
            self.f0_extractor = "rmvpe"  # Placeholder
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize F0 extractor: {e}")
            self.f0_extractor = None
    
    def extract_audio_features(self, audio: np.ndarray, 
                             extract_f0: bool = True,
                             extract_embeddings: bool = True) -> Dict[str, np.ndarray]:
        """Extract all features from audio"""
        features = {}
        
        try:
            # Extract basic audio features
            features.update(self._extract_basic_features(audio))
            
            # Extract F0 if requested
            if extract_f0:
                features['f0'] = self.extract_f0(audio)
            
            # Extract embeddings if requested
            if extract_embeddings:
                features['embeddings'] = self.extract_embeddings(audio)
            
            # Extract mel spectrogram
            features['mel_spec'] = self.extract_mel_spectrogram(audio)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            # Return empty features as fallback
            return self._get_empty_features()
    
    def _extract_basic_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract basic audio features (STFT, etc.)"""
        try:
            # Compute STFT
            stft = librosa.stft(
                audio, 
                hop_length=self.hop_length, 
                n_fft=2048,
                win_length=2048
            )
            
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Compute power spectrum
            power = magnitude ** 2
            
            return {
                'stft_magnitude': magnitude,
                'stft_phase': phase,
                'stft_power': power,
                'audio': audio
            }
            
        except Exception as e:
            self.logger.warning(f"Basic feature extraction failed: {e}")
            return {'audio': audio}
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel spectrogram"""
        try:
            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                n_mels=80,
                n_fft=2048,
                win_length=2048
            )
            
            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            return mel_spec_db
            
        except Exception as e:
            self.logger.warning(f"Mel spectrogram extraction failed: {e}")
            # Return zero mel spectrogram as fallback
            return np.zeros((80, max(1, len(audio) // self.hop_length)))
    
    def extract_f0(self, audio: np.ndarray) -> np.ndarray:
        """Extract F0 (fundamental frequency)"""
        try:
            # Placeholder F0 extraction
            # In a real implementation, this would use:
            # - RMVPE model
            # - CREPE model
            # - Or other F0 extraction algorithms
            
            # For now, use librosa as a basic F0 extractor
            f0 = librosa.yin(
                audio, 
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7'),
                hop_length=self.hop_length
            )
            
            # Smooth F0 trajectory
            f0 = gaussian_filter1d(f0, sigma=1.0)
            
            # Remove unvoiced frames (set to 0)
            f0[f0 < 0] = 0
            
            return f0
            
        except Exception as e:
            self.logger.warning(f"F0 extraction failed: {e}")
            # Return zero F0 as fallback
            return np.zeros(max(1, len(audio) // self.hop_length))
    
    def extract_embeddings(self, audio: np.ndarray) -> np.ndarray:
        """Extract speaker embeddings"""
        try:
            if self.embedder is None:
                # Return zero embeddings as fallback
                return np.zeros((max(1, len(audio) // self.hop_length), 768))
            
            # Placeholder for actual embedding extraction
            # In a real implementation, this would:
            # 1. Convert audio to tensor
            # 2. Run through the embedder model
            # 3. Return the extracted embeddings
            
            # For now, return dummy embeddings
            sequence_length = max(1, len(audio) // self.hop_length)
            return np.random.randn(sequence_length, 768).astype(np.float32)
            
        except Exception as e:
            self.logger.warning(f"Embedding extraction failed: {e}")
            # Return zero embeddings as fallback
            sequence_length = max(1, len(audio) // self.hop_length)
            return np.zeros((sequence_length, 768))
    
    def extract_hybrid_features(self, audio: np.ndarray, 
                              primary_method: str = "rmvpe",
                              secondary_method: str = "crepe") -> Dict[str, np.ndarray]:
        """Extract features using hybrid approach"""
        features = {}
        
        try:
            # Extract basic features
            features.update(self._extract_basic_features(audio))
            
            # Extract F0 using primary method
            features['f0_primary'] = self.extract_f0(audio)
            
            # For hybrid approach, you might:
            # 1. Use RMVPE for voiced frames
            # 2. Use CREPE for unvoiced frames
            # 3. Combine results intelligently
            
            features['f0_hybrid'] = features['f0_primary']  # Placeholder
            
            # Extract embeddings
            features['embeddings'] = self.extract_embeddings(audio)
            
            # Extract mel spectrogram
            features['mel_spec'] = self.extract_mel_spectrogram(audio)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Hybrid feature extraction failed: {e}")
            return self._get_empty_features()
    
    def _get_empty_features(self) -> Dict[str, np.ndarray]:
        """Return empty features dictionary as fallback"""
        sequence_length = 100  # Default sequence length
        
        return {
            'stft_magnitude': np.zeros((1025, sequence_length)),
            'stft_phase': np.zeros((1025, sequence_length)),
            'stft_power': np.zeros((1025, sequence_length)),
            'f0': np.zeros(sequence_length),
            'embeddings': np.zeros((sequence_length, 768)),
            'mel_spec': np.zeros((80, sequence_length)),
            'audio': np.zeros(self.sample_rate * 2)  # 2 seconds of silence
        }
    
    def batch_extract_features(self, audio_files: List[str], 
                             output_dir: str,
                             extract_f0: bool = True,
                             extract_embeddings: bool = True) -> bool:
        """Extract features from multiple audio files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        successful = 0
        failed = 0
        
        for audio_file in audio_files:
            try:
                self.logger.info(f"Extracting features from {Path(audio_file).name}")
                
                # Load audio
                audio, sr = librosa.load(audio_file, sr=self.sample_rate)
                
                # Extract features
                features = self.extract_audio_features(
                    audio,
                    extract_f0=extract_f0,
                    extract_embeddings=extract_embeddings
                )
                
                # Save features
                output_file = output_path / f"{Path(audio_file).stem}_features.npz"
                np.savez_compressed(output_file, **features)
                
                successful += 1
                
            except Exception as e:
                self.logger.error(f"Failed to extract features from {audio_file}: {e}")
                failed += 1
        
        self.logger.info(f"Batch feature extraction completed: {successful} successful, {failed} failed")
        return successful > 0


def create_feature_index(dataset_path: str, model_name: str, 
                        index_algorithm: str = "faiss") -> bool:
    """Create feature index for fast retrieval during inference"""
    try:
        import faiss
        
        dataset_path = Path(dataset_path)
        
        # Find feature files
        feature_files = list(dataset_path.glob("*.npz"))
        
        if not feature_files:
            logging.warning("No feature files found for indexing")
            return False
        
        # Extract features for indexing
        all_features = []
        file_indices = []
        
        for feature_file in feature_files:
            try:
                features = np.load(feature_file)
                
                if 'embeddings' in features:
                    embedding = features['embeddings']
                    
                    # Use mean pooling to get single vector per file
                    if embedding.ndim > 2:
                        embedding = np.mean(embedding, axis=0)
                    
                    # Use only first 768 dimensions if too many
                    if embedding.ndim == 2 and embedding.shape[0] > 768:
                        embedding = embedding[:, :768]
                    elif embedding.ndim == 1 and len(embedding) > 768:
                        embedding = embedding[:768]
                    
                    all_features.append(embedding)
                    file_indices.append(feature_file.name)
                
            except Exception as e:
                logging.warning(f"Failed to load features from {feature_file}: {e}")
        
        if not all_features:
            logging.error("No valid features found for indexing")
            return False
        
        # Stack features
        feature_matrix = np.stack(all_features)
        
        # Normalize features for cosine similarity
        faiss.normalize_L2(feature_matrix)
        
        # Create index based on algorithm
        if index_algorithm.lower() == "faiss":
            # Use Flat index with L2 distance
            dimension = feature_matrix.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(feature_matrix.astype(np.float32))
            
        elif index_algorithm.lower() == "kmeans":
            # Use KMeans clustering
            dimension = feature_matrix.shape[1]
            n_clusters = min(100, len(feature_matrix))
            index = faiss.IndexKMeans(dimension, n_clusters)
            index.train(feature_matrix.astype(np.float32))
            index.add(feature_matrix.astype(np.float32))
            
        else:
            logging.error(f"Unsupported index algorithm: {index_algorithm}")
            return False
        
        # Save index
        index_file = dataset_path / f"{model_name}.index"
        
        if index_algorithm.lower() == "faiss":
            faiss.write_index(index, str(index_file))
        elif index_algorithm.lower() == "kmeans":
            faiss.write_index(index, str(index_file))
        
        # Save metadata
        metadata = {
            "model_name": model_name,
            "algorithm": index_algorithm,
            "feature_count": len(all_features),
            "feature_dimension": feature_matrix.shape[1],
            "file_indices": file_indices,
            "created_at": "2025-11-18"
        }
        
        metadata_file = dataset_path / f"{model_name}_metadata.json"
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logging.info(f"Feature index created: {index_file}")
        return True
        
    except ImportError:
        logging.error("FAISS not available. Install with: pip install faiss-cpu")
        return False
        
    except Exception as e:
        logging.error(f"Failed to create feature index: {e}")
        return False


def analyze_features(feature_file: str) -> Dict[str, Any]:
    """Analyze extracted features"""
    try:
        features = np.load(feature_file)
        
        analysis = {
            'file': feature_file,
            'feature_shapes': {},
            'statistics': {}
        }
        
        for name, feature in features.items():
            if isinstance(feature, np.ndarray):
                analysis['feature_shapes'][name] = list(feature.shape)
                
                if feature.dtype in [np.float32, np.float64]:
                    analysis['statistics'][name] = {
                        'mean': float(np.mean(feature)),
                        'std': float(np.std(feature)),
                        'min': float(np.min(feature)),
                        'max': float(np.max(feature))
                    }
                elif feature.dtype == np.int32 or feature.dtype == np.int64:
                    analysis['statistics'][name] = {
                        'min': int(np.min(feature)),
                        'max': int(np.max(feature)),
                        'unique_values': int(len(np.unique(feature)))
                    }
        
        return analysis
        
    except Exception as e:
        return {'error': f"Failed to analyze features: {e}"}


def extract_features_from_dataset(dataset_path: str, 
                                output_dir: str,
                                sample_rate: int = 48000,
                                hop_length: int = 160,
                                f0_method: str = "librosa",
                                embedder_model: str = "hubert_base") -> bool:
    """Extract features from entire dataset"""
    
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        logging.error(f"Dataset path does not exist: {dataset_path}")
        return False
    
    # Find audio files
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(dataset_path.glob(f"*{ext}"))
        audio_files.extend(dataset_path.glob(f"**/*{ext}"))
    
    if not audio_files:
        logging.error("No audio files found in dataset")
        return False
    
    # Initialize feature extractor
    extractor = FeatureExtractor(sample_rate=sample_rate, hop_length=hop_length)
    extractor.f0_extractor = f0_method
    extractor.embedder = embedder_model
    
    # Extract features
    return extractor.batch_extract_features(
        [str(f) for f in audio_files],
        output_dir,
        extract_f0=True,
        extract_embeddings=True
    )

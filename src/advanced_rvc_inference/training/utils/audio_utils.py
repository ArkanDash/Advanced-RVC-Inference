"""
Audio Processing Utilities for Advanced RVC Training
Handles audio preprocessing, segmentation, and enhancement
"""

import os
import sys
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import noisereduce as nr

class AudioPreprocessor:
    """Audio preprocessing for RVC training data"""
    
    def __init__(self, target_sr: int = 48000, hop_length: int = 160):
        self.target_sr = target_sr
        self.hop_length = hop_length
        self.logger = logging.getLogger(__name__)
    
    def load_and_preprocess(self, audio_path: str, normalize: bool = True, 
                          trim_silence: bool = True, enhance: bool = True) -> np.ndarray:
        """Load and preprocess audio file"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
            
            # Trim silence if requested
            if trim_silence:
                audio = self.trim_silence(audio, sr)
            
            # Normalize if requested
            if normalize and np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.9
            
            # Apply enhancement if requested
            if enhance:
                audio = self.enhance_audio(audio, sr)
            
            return audio
            
        except Exception as e:
            self.logger.error(f"Failed to preprocess audio {audio_path}: {e}")
            return np.zeros(self.target_sr)  # Return silence as fallback
    
    def trim_silence(self, audio: np.ndarray, sr: int, 
                    threshold_db: float = -40) -> np.ndarray:
        """Trim silence from audio"""
        try:
            # Use librosa for silence trimming
            audio_trimmed, _ = librosa.effects.trim(
                audio, 
                top_db=-threshold_db, 
                frame_length=2048, 
                hop_length=512
            )
            
            # Also trim very quiet segments at the beginning and end
            energy = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
            
            # Find non-silent segments
            non_silent_indices = np.where(energy > np.max(energy) * 0.01)[0]
            
            if len(non_silent_indices) > 0:
                start_frame = non_silent_indices[0]
                end_frame = non_silent_indices[-1]
                
                start_sample = start_frame * 512
                end_sample = min(end_frame * 512 + 1024, len(audio))  # Add some padding
                
                audio = audio[start_sample:end_sample]
            
            return audio
            
        except Exception as e:
            self.logger.warning(f"Silence trimming failed: {e}")
            return audio
    
    def enhance_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Enhance audio quality"""
        try:
            # Noise reduction
            audio = nr.reduce_noise(y=audio, sr=sr, stationary=False, prop_decrease=0.8)
            
            # High-pass filter to remove low-frequency noise
            nyquist = sr // 2
            cutoff = 80  # Hz
            b, a = signal.butter(4, cutoff / nyquist, btype='high')
            audio = signal.filtfilt(b, a, audio)
            
            # Normalize again after processing
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.9
            
            return audio
            
        except Exception as e:
            self.logger.warning(f"Audio enhancement failed: {e}")
            return audio
    
    def segment_audio(self, audio: np.ndarray, sr: int, 
                     segment_length: float = 10.0, 
                     overlap: float = 2.0) -> List[np.ndarray]:
        """Segment audio into chunks"""
        segment_length_samples = int(segment_length * sr)
        overlap_samples = int(overlap * sr)
        stride = segment_length_samples - overlap_samples
        
        segments = []
        
        for start in range(0, len(audio) - segment_length_samples + 1, stride):
            end = start + segment_length_samples
            segment = audio[start:end]
            
            # Pad if needed
            if len(segment) < segment_length_samples:
                segment = np.pad(segment, (0, segment_length_samples - len(segment)))
            
            segments.append(segment)
        
        return segments
    
    def validate_audio(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Validate audio quality"""
        validation = {
            'duration': len(audio) / sr,
            'sample_rate': sr,
            'max_amplitude': float(np.max(np.abs(audio))),
            'rms_level': float(np.sqrt(np.mean(audio**2))),
            'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(audio))),
            'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))),
            'is_silent': np.max(np.abs(audio)) < 0.01,
            'is_too_quiet': np.sqrt(np.mean(audio**2)) < 0.01,
            'is_too_loud': np.max(np.abs(audio)) > 0.95
        }
        
        return validation


def preprocess_audio_files(dataset_path: str, target_sr: int = 48000) -> bool:
    """Preprocess all audio files in the dataset"""
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        logging.error(f"Dataset path does not exist: {dataset_path}")
        return False
    
    # Initialize preprocessor
    preprocessor = AudioPreprocessor(target_sr=target_sr)
    
    # Find audio files
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(dataset_path.glob(f"*{ext}"))
        audio_files.extend(dataset_path.glob(f"**/*{ext}"))
    
    if not audio_files:
        logging.error(f"No audio files found in {dataset_path}")
        return False
    
    # Create processed directory
    processed_dir = dataset_path / "processed"
    processed_dir.mkdir(exist_ok=True)
    
    # Process each file
    successful = 0
    failed = 0
    
    for audio_file in audio_files:
        try:
            logging.info(f"Processing {audio_file.name}...")
            
            # Load and preprocess
            audio = preprocessor.load_and_preprocess(str(audio_file))
            
            # Validate audio
            validation = preprocessor.validate_audio(audio, target_sr)
            
            if validation['is_silent'] or validation['is_too_quiet']:
                logging.warning(f"Audio file {audio_file.name} is too quiet or silent, skipping")
                continue
            
            if validation['is_too_loud']:
                logging.warning(f"Audio file {audio_file.name} is too loud, normalizing")
            
            # Segment if audio is too long
            if len(audio) > target_sr * 30:  # Longer than 30 seconds
                segments = preprocessor.segment_audio(audio, target_sr, segment_length=15.0, overlap=2.0)
                
                for i, segment in enumerate(segments):
                    output_file = processed_dir / f"{audio_file.stem}_seg_{i:03d}.wav"
                    sf.write(output_file, segment, target_sr)
            else:
                # Save processed audio
                output_file = processed_dir / f"{audio_file.stem}.wav"
                sf.write(output_file, audio, target_sr)
            
            successful += 1
            
        except Exception as e:
            logging.error(f"Failed to process {audio_file.name}: {e}")
            failed += 1
    
    logging.info(f"Audio preprocessing completed: {successful} successful, {failed} failed")
    return successful > 0


def batch_audio_preprocessing(input_dir: str, output_dir: str, 
                            target_sr: int = 48000,
                            normalize: bool = True,
                            trim_silence: bool = True,
                            enhance: bool = True) -> bool:
    """Batch preprocess audio files"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        logging.error(f"Input directory does not exist: {input_dir}")
        return False
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = AudioPreprocessor(target_sr=target_sr)
    
    # Find audio files
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(input_path.glob(f"*{ext}"))
        audio_files.extend(input_path.glob(f"**/*{ext}"))
    
    if not audio_files:
        logging.error(f"No audio files found in {input_dir}")
        return False
    
    # Process files
    successful = 0
    
    for audio_file in audio_files:
        try:
            logging.info(f"Preprocessing {audio_file.name}...")
            
            # Load and preprocess
            audio = preprocessor.load_and_preprocess(
                str(audio_file),
                normalize=normalize,
                trim_silence=trim_silence,
                enhance=enhance
            )
            
            # Validate
            validation = preprocessor.validate_audio(audio, target_sr)
            
            if validation['is_silent']:
                logging.warning(f"Audio file {audio_file.name} is silent, skipping")
                continue
            
            # Save to output directory
            output_file = output_path / f"{audio_file.stem}_processed.wav"
            sf.write(output_file, audio, target_sr)
            
            # Save validation report
            validation_file = output_path / f"{audio_file.stem}_validation.json"
            import json
            with open(validation_file, 'w') as f:
                json.dump(validation, f, indent=2)
            
            successful += 1
            
        except Exception as e:
            logging.error(f"Failed to preprocess {audio_file.name}: {e}")
    
    logging.info(f"Batch preprocessing completed: {successful} files processed")
    return successful > 0


def analyze_dataset_quality(dataset_path: str) -> Dict[str, Any]:
    """Analyze dataset quality and provide recommendations"""
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        return {"error": f"Dataset path does not exist: {dataset_path}"}
    
    # Find audio files
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(dataset_path.glob(f"*{ext}"))
        audio_files.extend(dataset_path.glob(f"**/*{ext}"))
    
    if not audio_files:
        return {"error": "No audio files found in dataset"}
    
    # Analyze each file
    stats = {
        'total_files': len(audio_files),
        'total_duration': 0.0,
        'durations': [],
        'sample_rates': [],
        'rms_levels': [],
        'max_amplitudes': [],
        'zero_crossing_rates': [],
        'spectral_centroids': [],
        'quality_issues': [],
        'recommendations': []
    }
    
    preprocessor = AudioPreprocessor()
    
    for audio_file in audio_files:
        try:
            audio, sr = librosa.load(str(audio_file), sr=None)
            validation = preprocessor.validate_audio(audio, sr)
            
            stats['total_duration'] += validation['duration']
            stats['durations'].append(validation['duration'])
            stats['sample_rates'].append(validation['sample_rate'])
            stats['rms_levels'].append(validation['rms_level'])
            stats['max_amplitudes'].append(validation['max_amplitude'])
            stats['zero_crossing_rates'].append(validation['zero_crossing_rate'])
            stats['spectral_centroids'].append(validation['spectral_centroid'])
            
            # Check for issues
            if validation['is_silent']:
                stats['quality_issues'].append(f"{audio_file.name}: Silent audio")
            elif validation['is_too_quiet']:
                stats['quality_issues'].append(f"{audio_file.name}: Too quiet")
            elif validation['is_too_loud']:
                stats['quality_issues'].append(f"{audio_file.name}: Clipping detected")
            
        except Exception as e:
            stats['quality_issues'].append(f"{audio_file.name}: Loading error - {e}")
    
    # Generate recommendations
    if stats['total_duration'] < 600:  # Less than 10 minutes
        stats['recommendations'].append("Dataset duration is low. Consider adding more audio files for better training results.")
    
    avg_duration = np.mean(stats['durations'])
    if avg_duration < 5.0:  # Less than 5 seconds average
        stats['recommendations'].append("Audio files are quite short. Longer utterances typically provide better training results.")
    
    if len(stats['quality_issues']) > len(audio_files) * 0.1:  # More than 10% have issues
        stats['recommendations'].append("High percentage of audio files have quality issues. Consider preprocessing the dataset.")
    
    # Add general recommendations
    stats['recommendations'].extend([
        "Ensure audio files are from the same speaker for best results",
        "Use high-quality, noise-free recordings",
        "Include a variety of speaking styles and emotions if desired",
        "Aim for at least 10-30 minutes of clean audio data"
    ])
    
    return stats

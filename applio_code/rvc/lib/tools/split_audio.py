"""
Audio splitting and merging utilities for voice conversion.
"""
import os
import numpy as np
import librosa
import soundfile as sf


def process_audio(audio_path, output_dir=None):
    """
    Process audio file for voice conversion.
    
    Args:
        audio_path: Path to input audio file
        output_dir: Directory to save processed audio files
        
    Returns:
        tuple: (status, output_path) where status is "Success" or "Error"
    """
    try:
        if output_dir is None:
            output_dir = os.path.dirname(audio_path)
            
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load audio file
        audio, sr = librosa.load(audio_path, sr=None)
        
        # Basic audio processing
        # Normalize audio
        audio = librosa.util.normalize(audio)
        
        # Save processed audio
        output_path = os.path.join(output_dir, "processed_" + os.path.basename(audio_path))
        sf.write(output_path, audio, sr)
        
        return "Success", output_path
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        return "Error", None


def merge_audio(timestamps_file):
    """
    Merge audio segments using timestamps file.
    
    Args:
        timestamps_file: Path to file containing merge timestamps
        
    Returns:
        tuple: (sample_rate, merged_audio_array)
    """
    try:
        # Read timestamps file
        with open(timestamps_file, 'r') as f:
            timestamps = f.readlines()
            
        # Parse timestamps and merge audio segments
        # This is a simplified implementation
        # In practice, this would reconstruct audio from segments
        
        # For now, return default values
        sample_rate = 44100  # Default sample rate
        merged_audio = np.array([])  # Empty array as placeholder
        
        return sample_rate, merged_audio
        
    except Exception as e:
        print(f"Error merging audio: {e}")
        return 44100, np.array([])


def split_audio_long(audio_path, segment_length=30, overlap=5):
    """
    Split long audio into smaller segments.
    
    Args:
        audio_path: Path to input audio file
        segment_length: Length of each segment in seconds
        overlap: Overlap between segments in seconds
        
    Returns:
        list: Paths to segment files
    """
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None)
        
        # Calculate segment lengths
        segment_samples = segment_length * sr
        overlap_samples = overlap * sr
        
        segments = []
        start = 0
        
        while start < len(audio):
            end = min(start + segment_samples, len(audio))
            segment = audio[start:end]
            
            # Save segment
            segment_path = f"segment_{len(segments):03d}.wav"
            sf.write(segment_path, segment, sr)
            segments.append(segment_path)
            
            # Move to next segment with overlap
            start = end - overlap_samples
            if start >= len(audio):
                break
                
        return segments
        
    except Exception as e:
        print(f"Error splitting audio: {e}")
        return []
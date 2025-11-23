#!/usr/bin/env python3
"""
RVC Dataset Preprocessing Script
Handles dataset preparation for RVC training
"""

import os
import sys
import argparse
import json
import librosa
import soundfile as sf
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def setup_directories():
    """Create necessary directories for training"""
    dirs = [
        "logs",
        "logs/44k",
        "logs/48k", 
        "logs/40k",
        "logs/32k",
        "logs/models",
        "logs/pretraineds",
        "logs/tensorboard",
        "audio_files",
        "audio_files/records",
        "audio_files/voices",
        "audio_files/preprocess",
        "audio_files/preprocess/separate",
        "audio_files/preprocess/deecho",
        "audio_files/preprocess/denoise"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def preprocess_audio_file(audio_path, target_sr, output_path):
    """Preprocess a single audio file"""
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None)
        
        # Resample if needed
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # Normalize audio
        audio = audio / np.max(np.abs(audio))
        
        # Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Save preprocessed audio
        sf.write(output_path, audio, target_sr)
        
        return {
            "original": str(audio_path),
            "preprocessed": str(output_path),
            "duration": len(audio) / target_sr,
            "sample_rate": target_sr,
            "status": "success"
        }
    except Exception as e:
        return {
            "original": str(audio_path),
            "error": str(e),
            "status": "error"
        }

def process_dataset(dataset_path, model_name, sample_rate, cpu_cores, process_effects=False):
    """Process entire dataset"""
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    # Setup directories
    setup_directories()
    
    # Create model directory
    model_dir = Path("logs") / sample_rate / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all audio files
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(dataset_path.rglob(f"*{ext}"))
        audio_files.extend(dataset_path.rglob(f"*{ext.upper()}"))
    
    if not audio_files:
        raise ValueError(f"No audio files found in {dataset_path}")
    
    print(f"Found {len(audio_files)} audio files")
    
    # Create preprocessing results
    results = []
    sr = int(sample_rate)
    
    with ThreadPoolExecutor(max_workers=cpu_cores) as executor:
        futures = []
        
        for audio_file in audio_files:
            # Create relative path structure
            rel_path = audio_file.relative_to(dataset_path)
            output_path = model_dir / "preprocess" / rel_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            future = executor.submit(preprocess_audio_file, audio_file, sr, output_path)
            futures.append((future, audio_file, output_path))
        
        for future, original_file, output_path in futures:
            try:
                result = future.result()
                results.append(result)
                print(f"Processed: {original_file.name}")
            except Exception as e:
                print(f"Failed to process {original_file}: {e}")
                results.append({
                    "original": str(original_file),
                    "error": str(e),
                    "status": "error"
                })
    
    # Save preprocessing log
    log_path = model_dir / "preprocess_log.json"
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Generate summary
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful
    
    summary = {
        "total_files": len(results),
        "successful": successful,
        "failed": failed,
        "model_name": model_name,
        "sample_rate": sample_rate,
        "cpu_cores": cpu_cores,
        "process_effects": process_effects
    }
    
    summary_path = model_dir / "preprocess_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nPreprocessing completed!")
    print(f"Total files: {summary['total_files']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Results saved to: {log_path}")
    
    return summary

def main():
    parser = argparse.ArgumentParser(description="RVC Dataset Preprocessing")
    parser.add_argument("--dataset_path", required=True, help="Path to dataset directory")
    parser.add_argument("--model_name", required=True, help="Name for the model")
    parser.add_argument("--sample_rate", default="40000", help="Target sample rate")
    parser.add_argument("--cpu_cores", type=int, default=4, help="Number of CPU cores")
    parser.add_argument("--process_effects", action="store_true", help="Apply audio effects")
    
    args = parser.parse_args()
    
    try:
        summary = process_dataset(
            args.dataset_path,
            args.model_name,
            args.sample_rate,
            args.cpu_cores,
            args.process_effects
        )
        print("Preprocessing completed successfully!")
        
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
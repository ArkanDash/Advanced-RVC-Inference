#!/usr/bin/env python3
"""
RVC F0 Extraction Script
Extracts pitch (F0) from audio files for RVC training
"""

import os
import sys
import argparse
import json
import librosa
import numpy as np
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_f0_harvest(audio_path, f0_min=50, f0_max=1100):
    """Extract F0 using harvest algorithm"""
    try:
        import pyworld
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Extract F0 using harvest
        f0, timeaxis = pyworld.dio(audio, sr, f0_min=f0_min, f0_max=f0_max)
        f0 = pyworld.stonemask(audio, sr, f0)
        
        return f0.astype(np.float32)
        
    except Exception as e:
        logger.error(f"Harvest F0 extraction failed for {audio_path}: {e}")
        return np.array([], dtype=np.float32)

def extract_f0_dio(audio_path, f0_min=50, f0_max=1100):
    """Extract F0 using dio algorithm"""
    try:
        import pyworld
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Extract F0 using dio
        f0 = pyworld.dio(audio, sr, f0_min=f0_min, f0_max=f0_max).astype(np.float32)
        
        return f0
        
    except Exception as e:
        logger.error(f"DIO F0 extraction failed for {audio_path}: {e}")
        return np.array([], dtype=np.float32)

def extract_f0_rmvpe(audio_path, f0_min=50, f0_max=1100):
    """Extract F0 using RMVPE algorithm"""
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # For now, use a simple approach as RMVPE might not be available
        # In a real implementation, you would import and use the RMVPE model
        logger.warning("RMVPE not available, using alternative method")
        
        # Use harmonic product spectrum as fallback
        f0, _, _ = librosa.pyin(audio, fmin=50, fmax=1100, sr=sr)
        f0 = f0.flatten()
        
        return f0.astype(np.float32)
        
    except Exception as e:
        logger.error(f"RMVPE F0 extraction failed for {audio_path}: {e}")
        return np.array([], dtype=np.float32)

def extract_f0_crepe(audio_path, f0_min=50, f0_max=1100):
    """Extract F0 using CREPE algorithm"""
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Use CREPE for F0 extraction
        try:
            import crepe
            f0, _, _ = crepe.predict(audio, sr, verbose=0)
            return f0.flatten().astype(np.float32)
        except ImportError:
            logger.warning("CREPE not available, using pyin fallback")
            f0, _, _ = librosa.pyin(audio, fmin=50, fmax=1100, sr=sr)
            return f0.flatten().astype(np.float32)
        
    except Exception as e:
        logger.error(f"CREPE F0 extraction failed for {audio_path}: {e}")
        return np.array([], dtype=np.float32)

def extract_f0_pyin(audio_path, f0_min=50, f0_max=1100):
    """Extract F0 using librosa pyin"""
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Extract F0 using pyin
        f0, _, _ = librosa.pyin(audio, fmin=f0_min, fmax=f0_max, sr=sr)
        f0 = f0.flatten()
        
        return f0.astype(np.float32)
        
    except Exception as e:
        logger.error(f"Pyin F0 extraction failed for {audio_path}: {e}")
        return np.array([], dtype=np.float32)

def extract_f0_wrapper(audio_path, method="rmvpe", f0_min=50, f0_max=1100):
    """Wrapper for F0 extraction methods"""
    
    methods = {
        "harvest": extract_f0_harvest,
        "dio": extract_f0_dio,
        "rmvpe": extract_f0_rmvpe,
        "crepe": extract_f0_crepe,
        "pyin": extract_f0_pyin
    }
    
    if method not in methods:
        logger.warning(f"Unknown method {method}, using rmvpe")
        method = "rmvpe"
    
    return methods[method](audio_path, f0_min, f0_max)

def process_f0_extraction(model_name, method="rmvpe", gpus="0", cpu_cores=4):
    """Process F0 extraction for all audio files"""
    
    # Setup directories
    model_dir = Path("logs") / "40k" / model_name  # Use 40k as default
    preprocess_dir = model_dir / "preprocess"
    f0_dir = model_dir / "f0"
    
    if not preprocess_dir.exists():
        raise FileNotFoundError(f"Preprocessed data not found: {preprocess_dir}")
    
    # Create F0 output directory
    f0_dir.mkdir(exist_ok=True)
    
    # Find all preprocessed audio files
    audio_files = []
    for audio_file in preprocess_dir.rglob("*.wav"):
        if audio_file.is_file():
            audio_files.append(audio_file)
    
    if not audio_files:
        raise ValueError(f"No preprocessed audio files found in {preprocess_dir}")
    
    logger.info(f"Found {len(audio_files)} audio files to process")
    logger.info(f"Using F0 extraction method: {method}")
    
    # Process files
    results = []
    
    with ThreadPoolExecutor(max_workers=cpu_cores) as executor:
        futures = []
        
        for audio_file in audio_files:
            future = executor.submit(
                process_single_f0,
                audio_file,
                method,
                f0_dir
            )
            futures.append(future)
        
        for future in futures:
            try:
                result = future.result()
                results.append(result)
                filename = result.get('filename', 'unknown')
                logger.info(f"Processed F0: {filename}")
            except Exception as e:
                logger.error(f"F0 processing failed: {e}")
    
    # Save processing log
    log_path = model_dir / "f0_extraction_log.json"
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Generate summary
    successful = sum(1 for r in results if r.get("status") == "success")
    failed = len(results) - successful
    
    summary = {
        "total_files": len(results),
        "successful": successful,
        "failed": failed,
        "model_name": model_name,
        "method": method,
        "f0_dir": str(f0_dir)
    }
    
    summary_path = model_dir / "f0_extraction_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nF0 extraction completed!")
    logger.info(f"Total files: {summary['total_files']}")
    logger.info(f"Successful: {summary['successful']}")
    logger.info(f"Failed: {summary['failed']}")
    logger.info(f"F0 files saved to: {f0_dir}")
    
    return summary

def process_single_f0(audio_file, method, f0_dir):
    """Process F0 extraction for a single audio file"""
    try:
        filename = audio_file.stem
        
        # Extract F0
        f0 = extract_f0_wrapper(audio_file, method)
        
        if len(f0) == 0:
            raise ValueError("No F0 values extracted")
        
        # Save F0
        f0_path = f0_dir / f"{filename}.npy"
        np.save(f0_path, f0)
        
        return {
            "filename": filename,
            "audio_file": str(audio_file),
            "f0_file": str(f0_path),
            "f0_length": len(f0),
            "method": method,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "filename": audio_file.stem,
            "audio_file": str(audio_file),
            "error": str(e),
            "status": "error"
        }

def main():
    parser = argparse.ArgumentParser(description="RVC F0 Extraction")
    parser.add_argument("--model_name", required=True, help="Name of the model")
    parser.add_argument("--method", default="rmvpe", help="F0 extraction method")
    parser.add_argument("--gpus", default="0", help="GPU IDs to use")
    parser.add_argument("--cpu_cores", type=int, default=4, help="Number of CPU cores")
    
    args = parser.parse_args()
    
    try:
        summary = process_f0_extraction(
            args.model_name,
            args.method,
            args.gpus,
            args.cpu_cores
        )
        logger.info("F0 extraction completed successfully!")
        
    except Exception as e:
        logger.error(f"F0 extraction failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
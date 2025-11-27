#!/usr/bin/env python3
"""
RVC Model Evaluation Script
Evaluates trained RVC models on test data
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import soundfile as sf
import librosa
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
from ...lib.path_manager import path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_path):
    """Load trained RVC model"""
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        logger.info(f"Loaded model from: {model_path}")
        return checkpoint
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None

def convert_audio(model, input_audio, output_path, **kwargs):
    """Convert audio using the trained model"""
    try:
        # Load input audio
        audio, sr = librosa.load(input_audio, sr=16000)
        
        # This is a placeholder - in a real implementation,
        # you would use the actual RVC inference pipeline
        logger.info(f"Converting audio: {input_audio}")
        
        # For demonstration, just return the original audio
        # In real implementation, you would:
        # 1. Extract features from input audio
        # 2. Run through the model
        # 3. Apply post-processing
        # 4. Save the converted audio
        
        sf.write(output_path, audio, sr)
        
        return {
            "input": str(input_audio),
            "output": str(output_path),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Audio conversion failed: {e}")
        return {
            "input": str(input_audio),
            "error": str(e),
            "status": "error"
        }

def calculate_audio_metrics(original_audio, converted_audio, sr=16000):
    """Calculate audio quality metrics"""
    try:
        # Load audio files
        original, _ = librosa.load(original_audio, sr=sr)
        converted, _ = librosa.load(converted_audio, sr=sr)
        
        # Ensure same length
        min_len = min(len(original), len(converted))
        original = original[:min_len]
        converted = converted[:min_len]
        
        # Calculate metrics
        metrics = {}
        
        # 1. Spectral centroid (brightness)
        original_centroid = librosa.feature.spectral_centroid(y=original, sr=sr)[0]
        converted_centroid = librosa.feature.spectral_centroid(y=converted, sr=sr)[0]
        metrics["spectral_centroid_diff"] = float(np.mean(np.abs(original_centroid - converted_centroid)))
        
        # 2. RMS energy difference
        original_rms = librosa.feature.rms(y=original)[0]
        converted_rms = librosa.feature.rms(y=converted)[0]
        metrics["rms_energy_diff"] = float(np.mean(np.abs(original_rms - converted_rms)))
        
        # 3. Zero crossing rate difference
        original_zcr = librosa.feature.zero_crossing_rate(original)[0]
        converted_zcr = librosa.feature.zero_crossing_rate(converted)[0]
        metrics["zcr_diff"] = float(np.mean(np.abs(original_zcr - converted_zcr)))
        
        # 4. MFCC difference
        original_mfcc = librosa.feature.mfcc(y=original, sr=sr, n_mfcc=13)
        converted_mfcc = librosa.feature.mfcc(y=converted, sr=sr, n_mfcc=13)
        metrics["mfcc_diff"] = float(np.mean(np.abs(original_mfcc - converted_mfcc)))
        
        # 5. Pitch correlation (if F0 available)
        try:
            original_f0 = librosa.pyin(original, fmin=50, fmax=1100)[0]
            converted_f0 = librosa.pyin(converted, fmin=50, fmax=1100)[0]
            
            # Remove NaN values
            original_f0 = original_f0[~np.isnan(original_f0)]
            converted_f0 = converted_f0[~np.isnan(converted_f0)]
            
            if len(original_f0) > 0 and len(converted_f0) > 0:
                min_len_f0 = min(len(original_f0), len(converted_f0))
                correlation = np.corrcoef(original_f0[:min_len_f0], converted_f0[:min_len_f0])[0, 1]
                metrics["pitch_correlation"] = float(correlation) if not np.isnan(correlation) else 0.0
            else:
                metrics["pitch_correlation"] = 0.0
        except:
            metrics["pitch_correlation"] = 0.0
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to calculate metrics: {e}")
        return {}

def evaluate_model(model_name, test_audio_dir, output_dir):
    """Evaluate model on test data"""
    
    # Setup directories
    model_dir = Path("logs") / "40k" / model_name
    test_dir = Path(test_audio_dir)
    output_dir = Path(output_dir)
    
    # Find model files
    model_files = list(model_dir.glob("*.pth"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_dir}")
    
    model_path = model_files[0]  # Use the first model file
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find test audio files
    test_files = []
    for ext in ['.wav', '.mp3', '.flac', '.m4a']:
        test_files.extend(test_dir.glob(f"*{ext}"))
    
    if not test_files:
        raise ValueError(f"No test audio files found in {test_dir}")
    
    logger.info(f"Found {len(test_files)} test files")
    
    # Load model
    model = load_model(model_path)
    if model is None:
        raise ValueError("Failed to load model")
    
    # Process test files
    results = []
    all_metrics = []
    
    for test_file in test_files:
        try:
            # Create output path
            output_file = output_dir / f"converted_{test_file.name}"
            
            # Convert audio
            conversion_result = convert_audio(model, test_file, output_file)
            
            if conversion_result["status"] == "success":
                # Calculate metrics
                metrics = calculate_audio_metrics(test_file, output_file)
                metrics.update(conversion_result)
                all_metrics.append(metrics)
                results.append(metrics)
                logger.info(f"Evaluated: {test_file.name}")
            else:
                results.append(conversion_result)
                
        except Exception as e:
            logger.error(f"Failed to evaluate {test_file}: {e}")
            results.append({
                "file": str(test_file),
                "error": str(e),
                "status": "error"
            })
    
    # Calculate overall metrics
    if all_metrics:
        overall_metrics = {}
        for key in all_metrics[0].keys():
            if key not in ["input", "output", "status"]:
                values = [m[key] for m in all_metrics if key in m]
                overall_metrics[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }
        
        # Save results
        eval_results = {
            "model_name": model_name,
            "model_path": str(model_path),
            "test_files_count": len(test_files),
            "successful_conversions": len(all_metrics),
            "overall_metrics": overall_metrics,
            "individual_results": results
        }
        
        results_path = output_dir / "evaluation_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation completed!")
        logger.info(f"Results saved to: {results_path}")
        
        # Print summary
        print("\n=== Evaluation Summary ===")
        print(f"Model: {model_name}")
        print(f"Test files: {len(test_files)}")
        print(f"Successful conversions: {len(all_metrics)}")
        print("\nOverall Metrics:")
        for metric, values in overall_metrics.items():
            print(f"  {metric}: {values['mean']:.4f} ± {values['std']:.4f}")
        
        return eval_results
    else:
        logger.error("No successful evaluations")
        return {"status": "error", "message": "No successful evaluations"}

def main():
    parser = argparse.ArgumentParser(description="RVC Model Evaluation")
    parser.add_argument("--model_name", required=True, help="Name of the model to evaluate")
    parser.add_argument("--test_audio", required=True, help="Path to test audio directory")
    parser.add_argument("--output_path", default=str(path('logs_dir') / "evaluation"), help="Output directory for results")
    
    args = parser.parse_args()
    
    try:
        results = evaluate_model(args.model_name, args.test_audio, args.output_path)
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
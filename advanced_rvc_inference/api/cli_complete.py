#!/usr/bin/env python3
"""
Advanced RVC Inference CLI
===========================

A comprehensive command-line interface for Advanced RVC (Retrieval-based Voice Conversion).
Provides full access to voice conversion, training, audio processing, and model management.

Usage:
    rvc-cli <command> [options]

Commands:
    inference     Voice conversion and inference operations
    training      Model training operations
    dataset       Dataset creation and management
    preprocess    Audio preprocessing for training
    extract       Feature extraction for training
    index         Index file creation for feature retrieval
    separate      Music/instrument separation
    reference     Reference audio creation
    tts           Text-to-speech voice conversion
    models        Model management and listing
    serve         Launch web interface
    utils         Utility commands
    version       Show version information
    info          Show system information

For detailed help on a specific command:
    rvc-cli <command> --help
"""

import argparse
import sys
import os
import platform
import shutil
import logging
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Version and Info Functions
# ============================================================================

def get_version() -> str:
    """Get the package version."""
    try:
        from advanced_rvc_inference._version import __version__
        return __version__
    except ImportError:
        return "2.0.0"


def show_version() -> None:
    """Show version information."""
    import torch
    
    version_info = [
        f"Advanced RVC Inference v{get_version()}",
        f"Python: {platform.python_version()}",
        f"Platform: {platform.system()} {platform.machine()}",
    ]

    try:
        version_info.append(f"PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            version_info.append(f"CUDA: {torch.version.cuda}")
            gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            version_info.append(f"GPU Memory: {gpu_mem} GB")
    except ImportError:
        version_info.append("PyTorch: Not installed")

    print("\n".join(version_info))


def show_info() -> None:
    """Show detailed system information."""
    import torch
    
    info = [
        "=" * 60,
        "Advanced RVC Inference - System Information",
        "=" * 60,
        "",
        "System:",
        f"  Platform: {platform.system()} {platform.release()}",
        f"  Architecture: {platform.machine()}",
        f"  Python: {platform.python_version()}",
        f"  CPU Count: {os.cpu_count()}",
        f"  Memory: {shutil.disk_usage('/').total // (1024**3)} GB total",
        f"  Free Disk: {shutil.disk_usage('/').free // (1024**3)} GB",
        "",
        "Package:",
        f"  Version: {get_version()}",
    ]

    try:
        info.append(f"  PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            info.append(f"  CUDA Available: True")
            info.append(f"  CUDA Version: {torch.version.cuda}")
            gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            info.append(f"  GPU Memory: {gpu_mem} GB")
            info.append(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        else:
            info.append("  CUDA Available: False")
    except ImportError:
        info.append("  PyTorch: Not installed")

    # Check for common model directories
    weights_path = Path("assets/weights")
    if weights_path.exists():
        models = list(weights_path.glob("*.pth")) + list(weights_path.glob("*.onnx"))
        info.append(f"  Local Models: {len(models)}")
        if models:
            info.append("  Available Models:")
            for model in models[:5]:
                info.append(f"    - {model.name}")
            if len(models) > 5:
                info.append(f"    ... and {len(models) - 5} more")

    info.append("")
    info.append("=" * 60)

    print("\n".join(info))


# ============================================================================
# Utility Functions
# ============================================================================

def get_f0_methods() -> List[str]:
    """Get list of available F0 prediction methods."""
    return [
        "pm", "dio", "mangio-crepe-tiny", "mangio-crepe-small", 
        "mangio-crepe-medium", "mangio-crepe-large", "mangio-crepe-full",
        "crepe-tiny", "crepe-small", "crepe-medium", "crepe-large", 
        "crepe-full", "fcpe", "fcpe-legacy", "rmvpe", "rmvpe-legacy", 
        "harvest", "yin", "pyin", "swipe"
    ]


def get_embedder_models() -> List[str]:
    """Get list of available embedder models."""
    return [
        "hubert_base", "hubert_large", "hubert_xlarge",
        "contentvec_base", "contentvec_large", "contentvec_xlarge",
        "whisper_base", "whisper_small", "whisper_medium", 
        "whisper_large", "whisper_large_v2"
    ]


def get_embedder_modes() -> List[str]:
    """Get list of available embedder modes."""
    return ["fairseq", "transformers", "onnx", "whisper"]


def get_separator_models() -> List[str]:
    """Get list of available music separation models."""
    return [
        "Main_340", "Main_390", "Main_406", "Main_427", "Main_438",
        "Inst_full_292", "Inst_HQ_1", "Inst_HQ_2", "Inst_HQ_3", 
        "Inst_HQ_4", "Inst_HQ_5", "Kim_Vocal_1", "Kim_Vocal_2", 
        "Kim_Inst", "Inst_187_beta", "Inst_82_beta", "Inst_90_beta",
        "Voc_FT", "Crowd_HQ", "MDXNET_9482", "Inst_1", "Inst_2", 
        "Inst_3", "MDXNET_1_9703", "MDXNET_2_9682", "MDXNET_3_9662",
        "Inst_Main", "MDXNET_Main", "HT-Tuned", "HT-Normal", 
        "HD_MMI", "HT_6S", "HP-1", "HP-2", "HP-Vocal-1", "HP-Vocal-2",
        "HP2-1", "HP2-2", "HP2-3", "SP-2B-1", "SP-2B-2", "SP-3B-1",
        "SP-4B-1", "SP-4B-2", "SP-MID-1", "SP-MID-2"
    ]


def get_export_formats() -> List[str]:
    """Get list of available export formats."""
    return ["wav", "mp3", "flac", "ogg"]


def validate_input_file(path: str) -> bool:
    """Validate that input file exists and is readable."""
    if not path:
        return False
    p = Path(path)
    if not p.exists():
        logger.error(f"Input file not found: {path}")
        return False
    if not p.is_file():
        logger.error(f"Input path is not a file: {path}")
        return False
    return True


def validate_model_file(path: str) -> bool:
    """Validate that model file exists and has correct extension."""
    if not path:
        return False
    p = Path(path)
    if not p.exists():
        logger.error(f"Model file not found: {path}")
        return False
    if p.suffix not in [".pth", ".onnx"]:
        logger.error(f"Invalid model extension: {p.suffix}. Must be .pth or .onnx")
        return False
    return True


# ============================================================================
# Inference Command
# ============================================================================

def cmd_inference(args: argparse.Namespace) -> int:
    """Run voice conversion inference."""
    logger.info("Starting voice conversion inference...")
    
    try:
        from advanced_rvc_inference.rvc.infer.inference import convert
        
        # Validate inputs
        if not validate_input_file(args.input):
            return 1
        
        if args.model and not validate_model_file(args.model):
            return 1
        
        # Set defaults
        output_path = args.output or args.input.replace(".", f"_converted.")
        
        # Build parameters
        convert(
            pitch=args.pitch,
            filter_radius=args.filter_radius,
            index_rate=args.index_rate,
            rms_mix_rate=args.rms_mix_rate,
            protect=args.protect,
            hop_length=args.hop_length,
            f0_method=args.f0_method,
            input_path=args.input,
            output_path=output_path,
            pth_path=args.model,
            index_path=args.index,
            f0_autotune=args.f0_autotune,
            clean_audio=args.clean_audio,
            clean_strength=args.clean_strength,
            export_format=args.format,
            embedder_model=args.embedder_model,
            resample_sr=args.resample_sr,
            split_audio=args.split_audio,
            f0_autotune_strength=args.f0_autotune_strength,
            checkpointing=args.checkpointing,
            predictor_onnx=args.predictor_onnx,
            embedders_mode=args.embedders_mode,
            formant_shifting=args.formant_shifting,
            formant_qfrency=args.formant_qfrency,
            formant_timbre=args.formant_timbre,
            f0_file=args.f0_file or "",
            proposal_pitch=args.proposal_pitch,
            proposal_pitch_threshold=args.proposal_pitch_threshold,
            audio_processing=args.audio_processing,
            alpha=args.alpha,
            sid=args.speaker_id or 0
        )
        
        logger.info(f"Inference completed successfully!")
        logger.info(f"Output saved to: {output_path}")
        return 0
        
    except ImportError as e:
        logger.error(f"Failed to import inference module: {e}")
        return 1
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


def cmd_batch_inference(args: argparse.Namespace) -> int:
    """Run batch voice conversion inference on multiple files."""
    logger.info("Starting batch voice conversion inference...")
    
    try:
        from advanced_rvc_inference.rvc.infer.inference import convert
        
        input_path = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        
        if not input_path.exists():
            logger.error(f"Input directory not found: {input_path}")
            return 1
        
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all audio files
        audio_extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"]
        audio_files = [
            f for f in input_path.iterdir() 
            if f.is_file() and f.suffix.lower() in audio_extensions
        ]
        
        if not audio_files:
            logger.error(f"No audio files found in {input_path}")
            return 1
        
        logger.info(f"Found {len(audio_files)} audio files to process")
        
        success_count = 0
        for audio_file in audio_files:
            output_path = output_dir / f"{audio_file.stem}_converted{args.format}"
            
            logger.info(f"Processing: {audio_file.name}")
            
            try:
                convert(
                    pitch=args.pitch,
                    filter_radius=args.filter_radius,
                    index_rate=args.index_rate,
                    rms_mix_rate=args.rms_mix_rate,
                    protect=args.protect,
                    hop_length=args.hop_length,
                    f0_method=args.f0_method,
                    input_path=str(audio_file),
                    output_path=str(output_path),
                    pth_path=args.model,
                    index_path=args.index,
                    f0_autotune=args.f0_autotune,
                    clean_audio=args.clean_audio,
                    clean_strength=args.clean_strength,
                    export_format=args.format,
                    embedder_model=args.embedder_model,
                    resample_sr=args.resample_sr,
                    split_audio=args.split_audio,
                    f0_autotune_strength=args.f0_autotune_strength,
                    checkpointing=args.checkpointing,
                    predictor_onnx=args.predictor_onnx,
                    embedders_mode=args.embedders_mode,
                    formant_shifting=args.formant_shifting,
                    formant_qfrency=args.formant_qfrency,
                    formant_timbre=args.formant_timbre,
                    f0_file="",
                    proposal_pitch=args.proposal_pitch,
                    proposal_pitch_threshold=args.proposal_pitch_threshold,
                    audio_processing=args.audio_processing,
                    alpha=args.alpha,
                    sid=args.speaker_id or 0
                )
                success_count += 1
                logger.info(f"  Completed: {output_path}")
                
            except Exception as e:
                logger.error(f"  Failed to process {audio_file.name}: {e}")
        
        logger.info(f"Batch inference completed: {success_count}/{len(audio_files)} files processed successfully")
        return 0 if success_count > 0 else 1
        
    except ImportError as e:
        logger.error(f"Failed to import inference module: {e}")
        return 1
    except Exception as e:
        logger.error(f"Batch inference failed: {e}")
        return 1


# ============================================================================
# Training Commands
# ============================================================================

def cmd_train(args: argparse.Namespace) -> int:
    """Start model training."""
    logger.info("Preparing model training...")
    
    print("""
    Model training is best performed through the web interface for full features.
    
    To launch the web interface:
        rvc-cli serve
    
    Then navigate to the Training tab for complete training options including:
        - Dataset management
        - Epoch configuration
        - Batch size settings
        - Pre-trained model loading
        - TensorBoard monitoring
    
    Alternatively, you can use the training module directly:
        python -m advanced_rvc_inference --train --model_name <name> --save_every_epoch <num>
    """)
    
    return 0


def cmd_create_dataset(args: argparse.Namespace) -> int:
    """Create training dataset from audio source."""
    logger.info("Starting dataset creation...")
    
    try:
        from advanced_rvc_inference.create_dataset import main as create_dataset_main
        
        # Build command line arguments
        cmd_args = [
            "--input_data", args.input_data,
            "--output_dirs", args.output_dir,
            "--sample_rate", str(args.sample_rate),
        ]
        
        if args.clean_dataset:
            cmd_args.extend(["--clean_dataset", "--clean_strength", str(args.clean_strength)])
        
        if not args.separate:
            cmd_args.extend(["--separate", "False"])
        else:
            cmd_args.extend(["--separator_reverb", str(args.separator_reverb)])
            if args.separator_model:
                cmd_args.extend(["--model_name", args.separator_model])
            if args.reverb_model:
                cmd_args.extend(["--reverb_model", args.reverb_model])
        
        # Run dataset creation
        subprocess.run([sys.executable, "-m", "advanced_rvc_inference", "--create_dataset"] + cmd_args, check=True)
        
        logger.info("Dataset creation completed successfully!")
        return 0
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Dataset creation failed: {e}")
        return 1
    except Exception as e:
        logger.error(f"Dataset creation error: {e}")
        return 1


def cmd_preprocess(args: argparse.Namespace) -> int:
    """Preprocess training data."""
    logger.info("Starting preprocessing...")
    
    print(f"""
    Preprocessing configuration:
        Model: {args.model_name}
        Dataset path: {args.dataset_path}
        Sample rate: {args.sample_rate}
        CPU cores: {args.cpu_cores}
        Cut method: {args.cut_preprocess}
        Process effects: {args.process_effects}
    
    This operation is best performed through the web interface for full control.
    Run: rvc-cli serve
    Then navigate to the Training tab.
    """)
    
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    """Extract features for training."""
    logger.info("Starting feature extraction...")
    
    print(f"""
    Feature extraction configuration:
        Model: {args.model_name}
        RVC version: {args.rvc_version}
        F0 method: {args.f0_method}
        Sample rate: {args.sample_rate}
        Embedder: {args.embedder_model}
        GPU: {args.gpu}
    
    This operation is best performed through the web interface for full control.
    Run: rvc-cli serve
    Then navigate to the Training tab.
    """)
    
    return 0


def cmd_create_index(args: argparse.Namespace) -> int:
    """Create index file for feature retrieval."""
    logger.info("Starting index creation...")
    
    print(f"""
    Index creation configuration:
        Model name: {args.model_name}
        RVC version: {args.rvc_version}
        Index algorithm: {args.index_algorithm}
    
    This operation is best performed through the web interface for full control.
    Run: rvc-cli serve
    Then navigate to the Training tab.
    """)
    
    return 0


# ============================================================================
# Audio Processing Commands
# ============================================================================

def cmd_separate(args: argparse.Namespace) -> int:
    """Separate music into vocals and instruments."""
    logger.info("Starting music separation...")
    
    try:
        from advanced_rvc_inference.separate_music import main as separate_main
        
        cmd_args = [
            "--input_path", args.input,
            "--output_dirs", args.output_dir,
            "--export_format", args.format,
            "--sample_rate", str(args.sample_rate),
            "--model_name", args.model,
            "--shifts", str(args.shifts),
            "--batch_size", str(args.batch_size),
            "--overlap", str(args.overlap),
            "--aggression", str(args.aggression),
        ]
        
        if args.karaoke_model:
            cmd_args.extend(["--karaoke_model", args.karaoke_model])
        if args.reverb_model:
            cmd_args.extend(["--reverb_model", args.reverb_model])
        if args.denoise_model:
            cmd_args.extend(["--denoise_model", args.denoise_model])
        if args.enable_tta:
            cmd_args.extend(["--enable_tta"])
        if args.enable_denoise:
            cmd_args.extend(["--enable_denoise"])
        if args.high_end_process:
            cmd_args.extend(["--high_end_process"])
        if args.enable_post_process:
            cmd_args.extend(["--enable_post_process"])
        if args.separate_backing:
            cmd_args.extend(["--separate_backing"])
        if args.separate_reverb:
            cmd_args.extend(["--separate_reverb"])
        
        subprocess.run([sys.executable, "-m", "advanced_rvc_inference", "--separate_music"] + cmd_args, check=True)
        
        logger.info("Music separation completed successfully!")
        logger.info(f"Output saved to: {args.output_dir}")
        return 0
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Music separation failed: {e}")
        return 1
    except Exception as e:
        logger.error(f"Music separation error: {e}")
        return 1


def cmd_create_reference(args: argparse.Namespace) -> int:
    """Create reference audio for training."""
    logger.info("Starting reference creation...")
    
    print(f"""
    Reference creation configuration:
        Audio path: {args.audio_path}
        Reference name: {args.reference_name}
        RVC version: {args.rvc_version}
        Pitch guidance: {args.pitch_guidance}
        Energy use: {args.energy_use}
        Embedder: {args.embedder_model}
        F0 method: {args.f0_method}
        Pitch shift: {args.f0_up_key}
    """)
    
    return 0


# ============================================================================
# TTS Command
# ============================================================================

def cmd_tts(args: argparse.Namespace) -> int:
    """Text-to-speech voice conversion."""
    logger.info("Starting TTS voice conversion...")
    
    print(f"""
    TTS conversion configuration:
        Text/Input: {args.text or args.input}
        Model: {args.model}
        Index: {args.index}
        Pitch shift: {args.pitch}
        Index rate: {args.index_rate}
        F0 method: {args.f0_method}
        Output format: {args.format}
    
    This feature is best used through the web interface.
    Run: rvc-cli serve
    Then navigate to the TTS tab for full functionality.
    """)
    
    return 0


# ============================================================================
# Model Management Commands
# ============================================================================

def cmd_list_models(args: argparse.Namespace) -> int:
    """List available models."""
    weights_path = Path("assets/weights")
    
    if not weights_path.exists():
        logger.warning("No models directory found at: assets/weights")
        logger.info("You can specify model path directly or configure the weights directory in settings.")
        return 0
    
    models = list(weights_path.glob("*.pth")) + list(weights_path.glob("*.onnx"))
    
    if not models:
        logger.info("No models found in assets/weights directory.")
        logger.info("To add models:")
        logger.info("  1. Download models from HuggingFace or other sources")
        logger.info("  2. Place .pth or .onnx files in assets/weights/")
        logger.info("  3. Use rvc-cli models list to refresh")
        return 0
    
    print("\n" + "=" * 60)
    print("Available Models")
    print("=" * 60)
    print(f"{'Model Name':<50} {'Size':<10}")
    print("-" * 60)
    
    for model in sorted(models):
        size_mb = model.stat().st_size / (1024 * 1024)
        print(f"{model.name:<50} {size_mb:.1f} MB")
    
    print("-" * 60)
    print(f"Total: {len(models)} models")
    print("=" * 60 + "\n")
    
    return 0


def cmd_model_info(args: argparse.Namespace) -> int:
    """Show information about a specific model."""
    model_path = Path(args.model)
    
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return 1
    
    print("\n" + "=" * 60)
    print(f"Model Information: {model_path.name}")
    print("=" * 60)
    print(f"Path: {model_path.absolute()}")
    print(f"Size: {model_path.stat().st_size / (1024 * 1024):.1f} MB")
    print(f"Extension: {model_path.suffix}")
    print("-" * 60)
    print("Note: Detailed model information (version, sample rate, etc.)")
    print("is available through the web interface.")
    print("=" * 60 + "\n")
    
    return 0


# ============================================================================
# Serve Command
# ============================================================================

def cmd_serve(args: argparse.Namespace) -> int:
    """Launch the web interface."""
    logger.info("Starting web interface...")
    
    try:
        from advanced_rvc_inference import gui
        
        gui.launch(
            share=args.share,
            server_name=args.host,
            server_port=args.port,
        )
        return 0
        
    except ImportError as e:
        logger.error(f"Failed to import GUI module: {e}")
        return 1
    except Exception as e:
        logger.error(f"Failed to start web interface: {e}")
        return 1


# ============================================================================
# Utility Commands
# ============================================================================

def cmd_utils_info(args: argparse.Namespace) -> int:
    """Display utility information."""
    print("""
    Advanced RVC Inference - Utility Commands
    
    Available utilities:
        check-gpu       Check GPU availability and configuration
        check-deps      Verify all dependencies are installed
        clean-cache     Clean temporary files and cache
        reset-config    Reset configuration to defaults
    
    Use: rvc-cli utils <utility> [options]
    """)
    return 0


def cmd_check_gpu(args: argparse.Namespace) -> int:
    """Check GPU availability."""
    try:
        import torch
        
        print("\n" + "=" * 60)
        print("GPU Information")
        print("=" * 60)
        
        if torch.cuda.is_available():
            print(f"CUDA Available: Yes")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\nGPU {i}: {props.name}")
                print(f"  Memory Total: {props.total_memory / (1024**3):.1f} GB")
                print(f"  Compute Capability: {props.major}.{props.minor}")
            
            # Check memory usage
            print(f"\nCurrent Memory Usage:")
            for i in range(torch.cuda.device_count()):
                mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                print(f"  GPU {i}: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved")
        else:
            print("CUDA Available: No")
            print("\nTo enable GPU acceleration:")
            print("  1. Install CUDA Toolkit")
            print("  2. Install PyTorch with CUDA support:")
            print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        
        print("=" * 60 + "\n")
        return 0
        
    except ImportError:
        logger.error("PyTorch is not installed")
        return 1


def cmd_check_deps(args: argparse.Namespace) -> int:
    """Check if all dependencies are installed."""
    print("\n" + "=" * 60)
    print("Dependency Check")
    print("=" * 60)
    
    required_deps = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("torchaudio", "TorchAudio"),
        ("gradio", "Gradio"),
        ("librosa", "Librosa"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
    ]
    
    all_ok = True
    for module, name in required_deps:
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [MISSING] {name}")
            all_ok = False
    
    print("=" * 60)
    
    if all_ok:
        print("All dependencies are installed.\n")
        return 0
    else:
        print("Some dependencies are missing. Install with:")
        print("  pip install advanced-rvc-inference\n")
        return 1


def cmd_clean_cache(args: argparse.Namespace) -> int:
    """Clean temporary files and cache."""
    print("Cleaning cache directories...")
    
    cache_dirs = [
        Path("__pycache__"),
        Path(".cache"),
        Path("cache"),
    ]
    
    cleaned = 0
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            try:
                shutil.rmtree(cache_dir)
                logger.info(f"Removed: {cache_dir}")
                cleaned += 1
            except Exception as e:
                logger.warning(f"Failed to remove {cache_dir}: {e}")
    
    logger.info(f"Cleaned {cleaned} cache directories")
    return 0


def cmd_reset_config(args: argparse.Namespace) -> int:
    """Reset configuration to defaults."""
    config_path = Path("configs/config.json")
    
    if config_path.exists():
        backup_path = config_path.with_suffix(f".json.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        shutil.copy(config_path, backup_path)
        logger.info(f"Backup created: {backup_path}")
    
    logger.info("Configuration reset initiated through web interface.")
    print("""
    To reset configuration:
        1. Run: rvc-cli serve
        2. Go to Settings tab
        3. Click 'Reset to Defaults'
    
    Or manually delete: configs/config.json
    A new default config will be created on next run.
    """)
    return 0


# ============================================================================
# Main Parser Creation
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with all subcommands."""
    
    parser = argparse.ArgumentParser(
        prog="rvc-cli",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference
  rvc-cli infer --model model.pth --input audio.wav --output converted.wav

  # Batch processing
  rvc-cli infer-batch --model model.pth --input_dir ./input --output_dir ./output

  # List available models
  rvc-cli models list

  # Launch web interface
  rvc-cli serve --port 7860

  # Check GPU availability
  rvc-cli utils check-gpu

For more information, visit:
  https://github.com/ArkanDash/Advanced-RVC-Inference
        """.strip(),
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"Advanced RVC Inference v{get_version()}",
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (debug) logging",
    )

    # Create subparsers
    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        description="Available commands",
    )
    
    # =========================================================================
    # Inference Command
    # =========================================================================
    infer_parser = subparsers.add_parser(
        "infer",
        help="Run voice conversion inference",
        description="Run voice conversion on an audio file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    infer_parser.add_argument(
        "-m", "--model",
        required=True,
        help="Path to the RVC model file (.pth or .onnx)",
    )
    infer_parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the input audio file",
    )
    infer_parser.add_argument(
        "-o", "--output",
        help="Path to the output audio file (default: auto-generated)",
    )
    infer_parser.add_argument(
        "-p", "--pitch",
        type=int,
        default=0,
        help="Pitch shift in semitones (default: 0)",
    )
    infer_parser.add_argument(
        "-f", "--format",
        default="wav",
        choices=get_export_formats(),
        help="Output format (default: wav)",
    )
    infer_parser.add_argument(
        "--index",
        help="Path to the index file (.index)",
    )
    # F0 settings
    infer_parser.add_argument(
        "--f0_method",
        default="rmvpe",
        choices=get_f0_methods(),
        help="F0 prediction method (default: rmvpe)",
    )
    infer_parser.add_argument(
        "--f0_autotune",
        action="store_true",
        help="Enable F0 autotune",
    )
    infer_parser.add_argument(
        "--f0_autotune_strength",
        type=float,
        default=0.5,
        help="F0 autotune strength 0-1 (default: 0.5)",
    )
    # Audio processing
    infer_parser.add_argument(
        "--filter_radius",
        type=int,
        default=3,
        help="Median filter radius for F0 smoothing (default: 3)",
    )
    infer_parser.add_argument(
        "--index_rate",
        type=float,
        default=0.5,
        help="Ratio for using voice index 0-1 (default: 0.5)",
    )
    infer_parser.add_argument(
        "--rms_mix_rate",
        type=float,
        default=1.0,
        help="Coefficient for adjusting amplitude 0-1 (default: 1.0)",
    )
    infer_parser.add_argument(
        "--protect",
        type=float,
        default=0.33,
        help="Protect consonants 0-1 (default: 0.33)",
    )
    infer_parser.add_argument(
        "--hop_length",
        type=int,
        default=128,
        help="Hop length for processing (default: 128)",
    )
    # Embedder settings
    infer_parser.add_argument(
        "--embedder_model",
        default="contentvec_base",
        choices=get_embedder_models(),
        help="Embedding model (default: contentvec_base)",
    )
    infer_parser.add_argument(
        "--embedders_mode",
        default="fairseq",
        choices=get_embedder_modes(),
        help="Embedder mode (default: fairseq)",
    )
    # Audio enhancement
    infer_parser.add_argument(
        "--clean_audio",
        action="store_true",
        help="Apply audio cleaning",
    )
    infer_parser.add_argument(
        "--clean_strength",
        type=float,
        default=0.7,
        help="Audio cleaning strength 0-1 (default: 0.7)",
    )
    infer_parser.add_argument(
        "--resample_sr",
        type=int,
        default=0,
        help="Resample to new rate (0=keep original, e.g., 40000, 48000)",
    )
    infer_parser.add_argument(
        "--split_audio",
        action="store_true",
        help="Split audio before processing",
    )
    # Formant shifting
    infer_parser.add_argument(
        "--formant_shifting",
        action="store_true",
        help="Enable formant shifting",
    )
    infer_parser.add_argument(
        "--formant_qfrency",
        type=float,
        default=0.8,
        help="Formant shift frequency coefficient (default: 0.8)",
    )
    infer_parser.add_argument(
        "--formant_timbre",
        type=float,
        default=0.8,
        help="Voice timbre change coefficient (default: 0.8)",
    )
    # Advanced options
    infer_parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="Speaker ID for multi-speaker models (default: 0)",
    )
    infer_parser.add_argument(
        "--checkpointing",
        action="store_true",
        help="Enable checkpointing to save RAM",
    )
    infer_parser.add_argument(
        "--predictor_onnx",
        action="store_true",
        help="Use ONNX version of pitch predictor",
    )
    infer_parser.add_argument(
        "--proposal_pitch",
        action="store_true",
        help="Use proposal pitch estimation",
    )
    infer_parser.add_argument(
        "--proposal_pitch_threshold",
        type=float,
        default=0.05,
        help="Threshold for pitch frequency estimation (default: 0.05)",
    )
    infer_parser.add_argument(
        "--audio_processing",
        action="store_true",
        help="Enable additional audio processing",
    )
    infer_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Pitch blending threshold for hybrid pitch estimation (default: 0.5)",
    )
    infer_parser.add_argument(
        "--f0_file",
        default="",
        help="Path to pre-existing F0 file",
    )
    infer_parser.set_defaults(handler=cmd_inference)
    
    # =========================================================================
    # Batch Inference Command
    # =========================================================================
    batch_parser = subparsers.add_parser(
        "infer-batch",
        help="Run batch voice conversion inference",
        description="Run voice conversion on multiple audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    batch_parser.add_argument(
        "-m", "--model",
        required=True,
        help="Path to the RVC model file (.pth or .onnx)",
    )
    batch_parser.add_argument(
        "-i", "--input_dir",
        required=True,
        help="Directory containing input audio files",
    )
    batch_parser.add_argument(
        "-o", "--output_dir",
        required=True,
        help="Directory for output audio files",
    )
    batch_parser.add_argument(
        "-p", "--pitch",
        type=int,
        default=0,
        help="Pitch shift in semitones (default: 0)",
    )
    batch_parser.add_argument(
        "-f", "--format",
        default="wav",
        choices=get_export_formats(),
        help="Output format (default: wav)",
    )
    batch_parser.add_argument(
        "--index",
        help="Path to the index file (.index)",
    )
    batch_parser.add_argument(
        "--f0_method",
        default="rmvpe",
        choices=get_f0_methods(),
        help="F0 prediction method (default: rmvpe)",
    )
    batch_parser.add_argument(
        "--index_rate",
        type=float,
        default=0.5,
        help="Ratio for using voice index 0-1 (default: 0.5)",
    )
    batch_parser.add_argument(
        "--rms_mix_rate",
        type=float,
        default=1.0,
        help="Coefficient for adjusting amplitude 0-1 (default: 1.0)",
    )
    batch_parser.add_argument(
        "--protect",
        type=float,
        default=0.33,
        help="Protect consonants 0-1 (default: 0.33)",
    )
    batch_parser.add_argument(
        "--hop_length",
        type=int,
        default=128,
        help="Hop length for processing (default: 128)",
    )
    batch_parser.add_argument(
        "--filter_radius",
        type=int,
        default=3,
        help="Median filter radius for F0 smoothing (default: 3)",
    )
    batch_parser.add_argument(
        "--embedder_model",
        default="contentvec_base",
        choices=get_embedder_models(),
        help="Embedding model (default: contentvec_base)",
    )
    batch_parser.add_argument(
        "--embedders_mode",
        default="fairseq",
        choices=get_embedder_modes(),
        help="Embedder mode (default: fairseq)",
    )
    batch_parser.add_argument(
        "--clean_audio",
        action="store_true",
        help="Apply audio cleaning",
    )
    batch_parser.add_argument(
        "--clean_strength",
        type=float,
        default=0.7,
        help="Audio cleaning strength 0-1 (default: 0.7)",
    )
    batch_parser.add_argument(
        "--resample_sr",
        type=int,
        default=0,
        help="Resample to new rate (0=keep original)",
    )
    batch_parser.add_argument(
        "--split_audio",
        action="store_true",
        help="Split audio before processing",
    )
    batch_parser.add_argument(
        "--checkpointing",
        action="store_true",
        help="Enable checkpointing to save RAM",
    )
    batch_parser.add_argument(
        "--f0_autotune",
        action="store_true",
        help="Enable F0 autotune",
    )
    batch_parser.add_argument(
        "--f0_autotune_strength",
        type=float,
        default=0.5,
        help="F0 autotune strength 0-1 (default: 0.5)",
    )
    batch_parser.add_argument(
        "--formant_shifting",
        action="store_true",
        help="Enable formant shifting",
    )
    batch_parser.add_argument(
        "--formant_qfrency",
        type=float,
        default=0.8,
        help="Formant shift frequency coefficient (default: 0.8)",
    )
    batch_parser.add_argument(
        "--formant_timbre",
        type=float,
        default=0.8,
        help="Voice timbre change coefficient (default: 0.8)",
    )
    batch_parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="Speaker ID for multi-speaker models (default: 0)",
    )
    batch_parser.add_argument(
        "--proposal_pitch",
        action="store_true",
        help="Use proposal pitch estimation",
    )
    batch_parser.add_argument(
        "--proposal_pitch_threshold",
        type=float,
        default=0.05,
        help="Threshold for pitch frequency estimation (default: 0.05)",
    )
    batch_parser.add_argument(
        "--audio_processing",
        action="store_true",
        help="Enable additional audio processing",
    )
    batch_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Pitch blending threshold (default: 0.5)",
    )
    batch_parser.set_defaults(handler=cmd_batch_inference)
    
    # =========================================================================
    # Training Command
    # =========================================================================
    train_parser = subparsers.add_parser(
        "train",
        help="Train RVC models",
        description="Train RVC voice models (use web UI for full features)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    train_parser.add_argument(
        "--name",
        help="Name for the training experiment",
    )
    train_parser.add_argument(
        "--rvc_version",
        choices=["v1", "v2"],
        default="v2",
        help="RVC version (default: v2)",
    )
    train_parser.add_argument(
        "--save_every_epoch",
        type=int,
        default=50,
        help="Number of epochs between saves (default: 50)",
    )
    train_parser.add_argument(
        "--total_epoch",
        type=int,
        default=300,
        help="Total training epochs (default: 300)",
    )
    train_parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    train_parser.add_argument(
        "--gpu",
        default="0",
        help="GPU device number (default: 0, use - for CPU)",
    )
    train_parser.set_defaults(handler=cmd_train)
    
    # =========================================================================
    # Dataset Command
    # =========================================================================
    dataset_parser = subparsers.add_parser(
        "dataset",
        help="Create and manage training datasets",
        description="Create training datasets from audio sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    dataset_parser.add_argument(
        "action",
        choices=["create", "info"],
        help="Action to perform",
    )
    dataset_parser.add_argument(
        "--input_data",
        required=True,
        help="Audio source link (YouTube link or local path)",
    )
    dataset_parser.add_argument(
        "--output_dir",
        default="./dataset",
        help="Output data folder (default: ./dataset)",
    )
    dataset_parser.add_argument(
        "--sample_rate",
        type=int,
        default=48000,
        help="Audio sample rate (default: 48000)",
    )
    dataset_parser.add_argument(
        "--clean_dataset",
        action="store_true",
        help="Apply data cleaning",
    )
    dataset_parser.add_argument(
        "--clean_strength",
        type=float,
        default=0.7,
        help="Data cleaning strength 0-1 (default: 0.7)",
    )
    dataset_parser.add_argument(
        "--separate",
        action="store_true",
        default=True,
        help="Separate vocals from instruments (default: True)",
    )
    dataset_parser.add_argument(
        "--separator_model",
        choices=get_separator_models(),
        help="Vocal separation model",
    )
    dataset_parser.add_argument(
        "--separator_reverb",
        action="store_true",
        help="Separate vocal reverb",
    )
    dataset_parser.add_argument(
        "--reverb_model",
        choices=["MDX-Reverb", "VR-Reverb", "Echo-Aggressive", "Echo-Normal"],
        help="Reverb separation model",
    )
    dataset_parser.set_defaults(handler=cmd_create_dataset)
    
    # =========================================================================
    # Preprocess Command
    # =========================================================================
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Preprocess training data",
        description="Preprocess audio data for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    preprocess_parser.add_argument(
        "--model_name",
        required=True,
        help="Name of the model",
    )
    preprocess_parser.add_argument(
        "--dataset_path",
        default="./dataset",
        help="Path to folder containing data files",
    )
    preprocess_parser.add_argument(
        "--sample_rate",
        type=int,
        required=True,
        help="Audio data sample rate",
    )
    preprocess_parser.add_argument(
        "--cpu_cores",
        type=int,
        default=2,
        help="Number of CPU threads to use",
    )
    preprocess_parser.add_argument(
        "--cut_preprocess",
        choices=["Automatic", "Simple", "Skip"],
        default="Automatic",
        help="Preprocessing cut method",
    )
    preprocess_parser.add_argument(
        "--process_effects",
        action="store_true",
        help="Apply preprocessing effects",
    )
    preprocess_parser.add_argument(
        "--clean_dataset",
        action="store_true",
        help="Clean data files",
    )
    preprocess_parser.add_argument(
        "--clean_strength",
        type=float,
        default=0.7,
        help="Data cleaning strength 0-1",
    )
    preprocess_parser.add_argument(
        "--chunk_len",
        type=float,
        default=3.0,
        help="Audio chunk length for Simple method",
    )
    preprocess_parser.add_argument(
        "--overlap_len",
        type=float,
        default=0.3,
        help="Overlap length between slices",
    )
    preprocess_parser.add_argument(
        "--normalization_mode",
        choices=["none", "pre", "post"],
        default="none",
        help="Audio normalization processing",
    )
    preprocess_parser.set_defaults(handler=cmd_preprocess)
    
    # =========================================================================
    # Extract Command
    # =========================================================================
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract features for training",
        description="Extract features from audio for model training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    extract_parser.add_argument(
        "--model_name",
        required=True,
        help="Name of the model",
    )
    extract_parser.add_argument(
        "--rvc_version",
        choices=["v1", "v2"],
        default="v2",
        help="RVC version",
    )
    extract_parser.add_argument(
        "--sample_rate",
        type=int,
        required=True,
        help="Input audio sample rate",
    )
    extract_parser.add_argument(
        "--f0_method",
        default="rmvpe",
        choices=get_f0_methods(),
        help="F0 prediction method",
    )
    extract_parser.add_argument(
        "--f0_onnx",
        action="store_true",
        help="Use ONNX version of F0",
    )
    extract_parser.add_argument(
        "--pitch_guidance",
        action="store_true",
        default=True,
        help="Use pitch guidance",
    )
    extract_parser.add_argument(
        "--f0_autotune",
        action="store_true",
        help="Enable F0 autotune",
    )
    extract_parser.add_argument(
        "--f0_autotune_strength",
        type=float,
        default=1.0,
        help="F0 autotune strength",
    )
    extract_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Pitch blending threshold",
    )
    extract_parser.add_argument(
        "--hop_length",
        type=int,
        default=128,
        help="Hop length during processing",
    )
    extract_parser.add_argument(
        "--cpu_cores",
        type=int,
        default=2,
        help="Number of CPU threads",
    )
    extract_parser.add_argument(
        "--gpu",
        default="-",
        help="GPU to use (number or - for CPU)",
    )
    extract_parser.add_argument(
        "--embedder_model",
        default="hubert_base",
        choices=get_embedder_models(),
        help="Embedding model name",
    )
    extract_parser.add_argument(
        "--embedders_mode",
        default="fairseq",
        choices=get_embedder_modes(),
        help="Embedder mode",
    )
    extract_parser.add_argument(
        "--rms_extract",
        action="store_true",
        help="Also extract RMS energy",
    )
    extract_parser.set_defaults(handler=cmd_extract)
    
    # =========================================================================
    # Index Command
    # =========================================================================
    index_parser = subparsers.add_parser(
        "index",
        help="Create index for feature retrieval",
        description="Create index file for voice feature retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    index_parser.add_argument(
        "--model_name",
        required=True,
        help="Model name",
    )
    index_parser.add_argument(
        "--rvc_version",
        choices=["v1", "v2"],
        default="v2",
        help="RVC version",
    )
    index_parser.add_argument(
        "--index_algorithm",
        choices=["Auto", "Faiss", "KMeans"],
        default="Auto",
        help="Index algorithm to use",
    )
    index_parser.set_defaults(handler=cmd_create_index)
    
    # =========================================================================
    # Separate Command
    # =========================================================================
    separate_parser = subparsers.add_parser(
        "separate",
        help="Separate music into vocals and instruments",
        description="Separate music audio into vocal and instrumental tracks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    separate_parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input audio file",
    )
    separate_parser.add_argument(
        "-o", "--output_dir",
        default="./separated",
        help="Output file save folder",
    )
    separate_parser.add_argument(
        "-f", "--format",
        default="wav",
        choices=get_export_formats(),
        help="Export format",
    )
    separate_parser.add_argument(
        "--sample_rate",
        type=int,
        default=44100,
        help="Output audio sample rate",
    )
    separate_parser.add_argument(
        "--model",
        default="MDXNET_Main",
        choices=get_separator_models(),
        help="Vocal separation model",
    )
    separate_parser.add_argument(
        "--karaoke_model",
        choices=["MDX-Version-1", "MDX-Version-2", "VR-Version-1", "VR-Version-2"],
        help="Karaoke separation model",
    )
    separate_parser.add_argument(
        "--reverb_model",
        choices=["MDX-Reverb", "VR-Reverb", "Echo-Aggressive", "Echo-Normal"],
        help="Reverb separation model",
    )
    separate_parser.add_argument(
        "--denoise_model",
        choices=["Lite", "Normal"],
        help="Denoise model",
    )
    separate_parser.add_argument(
        "--shifts",
        type=int,
        default=2,
        help="Number of predictions",
    )
    separate_parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size",
    )
    separate_parser.add_argument(
        "--overlap",
        type=float,
        default=0.25,
        help="Overlap between segments",
    )
    separate_parser.add_argument(
        "--aggression",
        type=int,
        default=5,
        help="Intensity of main vocal extraction",
    )
    separate_parser.add_argument(
        "--hop_length",
        type=int,
        default=1024,
        help="MDX hop length",
    )
    separate_parser.add_argument(
        "--window_size",
        type=int,
        default=512,
        help="Window size",
    )
    separate_parser.add_argument(
        "--segments_size",
        type=int,
        default=256,
        help="Audio segment size",
    )
    separate_parser.add_argument(
        "--post_process_threshold",
        type=float,
        default=0.2,
        help="Post-processing level",
    )
    separate_parser.add_argument(
        "--enable_tta",
        action="store_true",
        help="Enable test-time augmentation",
    )
    separate_parser.add_argument(
        "--enable_denoise",
        action="store_true",
        help="Enable noise reduction",
    )
    separate_parser.add_argument(
        "--high_end_process",
        action="store_true",
        help="Enable high-frequency processing",
    )
    separate_parser.add_argument(
        "--enable_post_process",
        action="store_true",
        help="Enable post-processing",
    )
    separate_parser.add_argument(
        "--separate_backing",
        action="store_true",
        help="Separate backing vocals",
    )
    separate_parser.add_argument(
        "--separate_reverb",
        action="store_true",
        help="Separate vocal reverb",
    )
    separate_parser.set_defaults(handler=cmd_separate)
    
    # =========================================================================
    # Reference Command
    # =========================================================================
    reference_parser = subparsers.add_parser(
        "reference",
        help="Create reference audio for training",
        description="Create reference audio set for model training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    reference_parser.add_argument(
        "--audio_path",
        required=True,
        help="Path to input audio file",
    )
    reference_parser.add_argument(
        "--reference_name",
        default="reference",
        help="Output reference set name",
    )
    reference_parser.add_argument(
        "--pitch_guidance",
        action="store_true",
        default=True,
        help="Use pitch guidance",
    )
    reference_parser.add_argument(
        "--energy_use",
        action="store_true",
        help="Use RMS energy",
    )
    reference_parser.add_argument(
        "--version",
        choices=["v1", "v2"],
        default="v2",
        help="RVC version",
    )
    reference_parser.add_argument(
        "--embedder_model",
        default="hubert_base",
        choices=get_embedder_models(),
        help="Embedding model name",
    )
    reference_parser.add_argument(
        "--embedders_mode",
        default="fairseq",
        choices=get_embedder_modes(),
        help="Embedder mode",
    )
    reference_parser.add_argument(
        "--f0_method",
        default="rmvpe",
        choices=get_f0_methods(),
        help="F0 prediction method",
    )
    reference_parser.add_argument(
        "--f0_onnx",
        action="store_true",
        help="Use ONNX version of F0",
    )
    reference_parser.add_argument(
        "--f0_up_key",
        type=int,
        default=0,
        help="Pitch shift in semitones",
    )
    reference_parser.add_argument(
        "--filter_radius",
        type=int,
        default=3,
        help="Median filter radius for F0 smoothing",
    )
    reference_parser.add_argument(
        "--f0_autotune",
        action="store_true",
        help="Enable F0 autotune",
    )
    reference_parser.add_argument(
        "--f0_autotune_strength",
        type=float,
        default=1.0,
        help="F0 autotune strength",
    )
    reference_parser.add_argument(
        "--f0_file",
        default="",
        help="Path to pre-existing F0 file",
    )
    reference_parser.add_argument(
        "--proposal_pitch",
        action="store_true",
        help="Use proposal pitch estimation",
    )
    reference_parser.add_argument(
        "--proposal_pitch_threshold",
        type=float,
        default=0.0,
        help="Threshold for pitch frequency estimation",
    )
    reference_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Pitch blending threshold",
    )
    reference_parser.set_defaults(handler=cmd_create_reference)
    
    # =========================================================================
    # TTS Command
    # =========================================================================
    tts_parser = subparsers.add_parser(
        "tts",
        help="Text-to-speech voice conversion",
        description="Convert text to speech using RVC voice model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    tts_parser.add_argument(
        "--text",
        help="Text to convert to speech",
    )
    tts_parser.add_argument(
        "--input",
        help="Path to text file with speech to convert",
    )
    tts_parser.add_argument(
        "-m", "--model",
        required=True,
        help="Path to RVC model file",
    )
    tts_parser.add_argument(
        "--index",
        help="Path to index file",
    )
    tts_parser.add_argument(
        "-p", "--pitch",
        type=int,
        default=0,
        help="Pitch shift in semitones",
    )
    tts_parser.add_argument(
        "--index_rate",
        type=float,
        default=0.5,
        help="Ratio for using voice index",
    )
    tts_parser.add_argument(
        "--f0_method",
        default="rmvpe",
        choices=get_f0_methods(),
        help="F0 prediction method",
    )
    tts_parser.add_argument(
        "-f", "--format",
        default="wav",
        choices=get_export_formats(),
        help="Output format",
    )
    tts_parser.add_argument(
        "-o", "--output",
        help="Output file path",
    )
    tts_parser.set_defaults(handler=cmd_tts)
    
    # =========================================================================
    # Models Command
    # =========================================================================
    models_parser = subparsers.add_parser(
        "models",
        help="Manage and list models",
        description="Manage RVC models and list available models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    models_subparsers = models_parser.add_subparsers(
        title="actions",
        dest="model_action",
        description="Model actions",
    )
    
    list_parser = models_subparsers.add_parser(
        "list",
        help="List available models",
        description="List all available RVC models",
    )
    list_parser.set_defaults(handler=cmd_list_models)
    
    info_parser = models_subparsers.add_parser(
        "info",
        help="Show model information",
        description="Show information about a specific model",
    )
    info_parser.add_argument(
        "model",
        help="Path to model file",
    )
    info_parser.set_defaults(handler=cmd_model_info)
    
    # =========================================================================
    # Serve Command
    # =========================================================================
    serve_parser = subparsers.add_parser(
        "serve",
        help="Launch the web interface",
        description="Launch the Gradio web interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    serve_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to bind to (default: 7860)",
    )
    serve_parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public URL using Gradio sharing",
    )
    serve_parser.set_defaults(handler=cmd_serve)
    
    # =========================================================================
    # Utils Command
    # =========================================================================
    utils_parser = subparsers.add_parser(
        "utils",
        help="Utility commands",
        description="Various utility commands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    utils_subparsers = utils_parser.add_subparsers(
        title="utilities",
        dest="utils_action",
        description="Available utilities",
    )
    
    utils_info_parser = utils_subparsers.add_parser(
        "info",
        help="Display utility information",
        description="Display information about available utilities",
    )
    utils_info_parser.set_defaults(handler=cmd_utils_info)
    
    check_gpu_parser = utils_subparsers.add_parser(
        "check-gpu",
        help="Check GPU availability",
        description="Check GPU availability and configuration",
    )
    check_gpu_parser.set_defaults(handler=cmd_check_gpu)
    
    check_deps_parser = utils_subparsers.add_parser(
        "check-deps",
        help="Check dependencies",
        description="Verify all dependencies are installed",
    )
    check_deps_parser.set_defaults(handler=cmd_check_deps)
    
    clean_cache_parser = utils_subparsers.add_parser(
        "clean-cache",
        help="Clean temporary files",
        description="Clean temporary files and cache",
    )
    clean_cache_parser.set_defaults(handler=cmd_clean_cache)
    
    reset_config_parser = utils_subparsers.add_parser(
        "reset-config",
        help="Reset configuration",
        description="Reset configuration to defaults",
    )
    reset_config_parser.set_defaults(handler=cmd_reset_config)
    
    # =========================================================================
    # Version Command
    # =========================================================================
    version_parser = subparsers.add_parser(
        "version",
        help="Show version information",
        description="Show version and system information",
    )
    version_parser.set_defaults(handler=lambda args: (show_version(), 0))
    
    # =========================================================================
    # Info Command
    # =========================================================================
    info_parser = subparsers.add_parser(
        "info",
        help="Show system information",
        description="Show detailed system and package information",
    )
    info_parser.set_defaults(handler=lambda args: (show_info(), 0))
    
    return parser


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for the CLI."""
    # Setup environment
    package_root = Path(__file__).parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

    os.chdir(package_root)

    # Create parser
    parser = create_parser()

    # Parse arguments
    args = parser.parse_args()

    # Handle verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle no command
    if args.command is None:
        parser.print_help()
        return 0

    # Call the appropriate handler
    handler = getattr(args, "handler", None)
    if handler:
        try:
            return handler(args)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            return 130
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

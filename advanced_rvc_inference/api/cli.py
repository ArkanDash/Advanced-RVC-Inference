#!/usr/bin/env python3
"""
Advanced RVC Inference CLI
===========================

A comprehensive command-line interface for the Advanced RVC (Retrieval-based Voice Conversion).
Provides full access to voice conversion, training, audio processing, and model management.

Usage:
    rvc-cli <command> [options]

Commands:
    infer           Voice conversion and inference operations
    uvr             Audio separation (vocals/instruments)
    create-dataset  Create training dataset from YouTube or local audio
    create-index    Create model index for feature retrieval
    extract         Feature extraction for training
    preprocess      Audio preprocessing for training
    train           Model training
    create-ref      Create reference audio for better inference
    download        Download models/audio from URLs
    serve           Launch the Gradio web interface
    info            Show system and environment information
    version         Show version information
    list-models     List available models
    list-f0-methods List F0 extraction methods

For detailed help on a specific command:
    rvc-cli <command> --help

Examples:
    # Basic voice conversion
    rvc-cli infer -m mymodel.pth -i input.wav -o output.wav

    # Convert with pitch shift
    rvc-cli infer -m mymodel.pth -i input.wav -p 12

    # Separate vocals from music
    rvc-cli uvr -i music.wav

    # Download a model
    rvc-cli download -l "https://huggingface.co/user/model/resolve/main/model.pth"

    # Launch web interface
    rvc-cli serve --port 7860 --share
"""

import argparse
import sys
import os
import platform
import shutil
import logging
import subprocess
from pathlib import Path
from typing import Optional, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Environment Setup
# ============================================================================

def setup_environment():
    """Setup the environment for RVC operations."""
    cwd = Path.cwd()
    if str(cwd) not in sys.path:
        sys.path.insert(0, str(cwd))


# ============================================================================
# Version and Info
# ============================================================================

def get_version() -> str:
    """Get the package version."""
    try:
        from advanced_rvc_inference._version import __version__
        return __version__
    except ImportError:
        return "2.1.0"


def show_version():
    """Show version information."""
    version_info = [
        f"Advanced RVC Inference v{get_version()}",
        f"Python: {platform.python_version()}",
        f"Platform: {platform.system()} {platform.machine()}",
    ]

    try:
        import torch
        version_info.append(f"PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            version_info.append(f"CUDA: {torch.version.cuda}")
            gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            version_info.append(f"GPU Memory: {gpu_mem} GB")
            # ZLUDA / T4 detection
            try:
                from advanced_rvc_inference.library.backends import zluda
                if zluda.is_available():
                    version_info.append(f"ZLUDA: Detected (AMD GPU via CUDA compatibility)")
            except ImportError:
                pass
            gpu_name = torch.cuda.get_device_name(0).lower()
            if "t4" in gpu_name or "tesla t4" in gpu_name:
                version_info.append(f"GPU: Tesla T4 (T4 optimizations active)")
    except ImportError:
        version_info.append("PyTorch: Not installed")

    print("\n".join(version_info))


def show_info():
    """Show system information and configuration."""
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
    ]

    try:
        import psutil
        mem = psutil.virtual_memory()
        info.append(f"  Memory: {mem.total // (1024**3)} GB total, {mem.available // (1024**3)} GB available")
    except ImportError:
        pass

    info.append(f"  Free Disk: {shutil.disk_usage('/').free // (1024**3)} GB")
    info.append("")
    info.append("Package:")
    info.append(f"  Version: {get_version()}")

    try:
        import torch
        info.append(f"  PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            info.append(f"  CUDA Available: True")
            info.append(f"  CUDA Version: {torch.version.cuda}")
            gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            gpu_name = torch.cuda.get_device_name(0)
            info.append(f"  GPU Memory: {gpu_mem} GB")
            info.append(f"  GPU Name: {gpu_name}")
            # ZLUDA detection
            try:
                from advanced_rvc_inference.library.backends import zluda
                if zluda.is_available():
                    info.append("  ZLUDA: Detected (AMD GPU via CUDA compatibility layer)")
                    info.append(f"  Backend: HIP/ROCm (via ZLUDA)")
            except ImportError:
                pass
            # T4 / low-VRAM detection
            gpu_name_lower = gpu_name.lower()
            if "t4" in gpu_name_lower or "tesla t4" in gpu_name_lower:
                info.append(f"  GPU Class: Tesla T4 (T4-optimized training defaults)")
            elif gpu_mem <= 16:
                info.append(f"  GPU Class: Low VRAM ({gpu_mem}GB, reduced memory defaults)")
            else:
                info.append(f"  GPU Class: High VRAM ({gpu_mem}GB, full optimizations)")
        else:
            info.append("  CUDA Available: False")
    except ImportError:
        info.append("  PyTorch: Not installed")

    # Check for common model directories
    try:
        from advanced_rvc_inference.utils.variables import WEIGHTS_PATH
        weights_path = Path(WEIGHTS_PATH)
    except ImportError:
        weights_path = Path("advanced_rvc_inference/assets/weights")

    if weights_path.exists():
        models = list(weights_path.glob("*.pth")) + list(weights_path.glob("*.onnx"))
        info.append(f"  Local Models: {len(models)}")
        if models:
            info.append("  Available Models:")
            for model in sorted(models)[:10]:
                info.append(f"    - {model.name}")
            if len(models) > 10:
                info.append(f"    ... and {len(models) - 10} more")

    info.append("")
    info.append("=" * 60)
    print("\n".join(info))


def list_models():
    """List available models."""
    try:
        from advanced_rvc_inference.utils.variables import WEIGHTS_PATH
        weights_path = Path(WEIGHTS_PATH)
    except ImportError:
        weights_path = Path("advanced_rvc_inference/assets/weights")

    if not weights_path.exists():
        logger.warning("No models directory found.")
        logger.info("Place .pth or .onnx files in: %s", weights_path)
        return 0

    models = sorted(weights_path.glob("*.pth")) + sorted(weights_path.glob("*.onnx"))

    if not models:
        logger.info("No models found in %s", weights_path)
        return 0

    print("\n" + "=" * 60)
    print("Available Models")
    print("=" * 60)
    print(f"{'Model Name':<50} {'Size':<10}")
    print("-" * 60)

    for model in models:
        size_mb = model.stat().st_size / (1024 * 1024)
        print(f"{model.name:<50} {size_mb:.1f} MB")

    print("-" * 60)
    print(f"Total: {len(models)} models")
    print("=" * 60 + "\n")
    return 0


def list_f0_methods():
    """List all available F0 extraction methods."""
    from advanced_rvc_inference.utils.variables import method_f0_full

    standard = [
        "rmvpe (recommended)",
        "crepe-full",
        "fcpe",
        "harvest",
        "pyin",
        "hybrid",
    ]

    print("\n" + "=" * 60)
    print("Available F0 Methods")
    print("=" * 60)
    print("\nStandard Methods:")
    for m in standard:
        print(f"  - {m}")

    print(f"\nExtended Methods ({len(method_f0_full)} total):")
    for m in method_f0_full:
        print(f"  - {m}")

    print("\nHybrid Methods (combine two methods):")
    from advanced_rvc_inference.utils.variables import hybrid_f0_method
    for m in hybrid_f0_method:
        print(f"  - {m}")

    print("=" * 60 + "\n")
    return 0


# ============================================================================
# Command Handlers
# ============================================================================

def cmd_infer(args):
    """Run voice conversion inference."""
    logger.info("Starting voice conversion inference...")

    try:
        from advanced_rvc_inference.rvc.infer.inference import convert

        # Validate inputs
        if not args.input or not Path(args.input).exists():
            logger.error("Input file not found: %s", args.input)
            return 1

        if not args.model:
            logger.error("Model path is required (-m)")
            return 1

        # Set output path
        output_path = args.output
        if not output_path:
            stem = Path(args.input).stem
            output_path = str(Path(args.input).parent / f"{stem}_converted.{args.format}")

        logger.info("Converting: %s -> %s", args.input, output_path)

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
            sid=args.speaker_id or 0,
        )

        logger.info("Inference completed successfully!")
        logger.info("Output saved to: %s", output_path)
        return 0

    except Exception as e:
        logger.error("Inference failed: %s", e)
        return 1


def cmd_uvr(args):
    """Separate vocals from instrumentals."""
    logger.info("Starting audio separation...")

    try:
        if not args.input or not Path(args.input).exists():
            logger.error("Input file not found: %s", args.input)
            return 1

        from advanced_rvc_inference.utils.variables import python, configs

        output_dirs = args.output or configs.get("uvr_path", "./audios/uvr")
        os.makedirs(output_dirs, exist_ok=True)

        cmd = [
            python, configs["separate_path"],
            "--input_path", args.input,
            "--output_dirs", output_dirs,
            "--export_format", args.format,
            "--model_name", args.model,
            "--karaoke_model", args.karaoke_model,
            "--reverb_model", args.reverb_model,
            "--denoise_model", args.denoise_model,
            "--sample_rate", str(args.sample_rate),
            "--shifts", str(args.shifts),
            "--batch_size", str(args.batch_size),
            "--overlap", str(args.overlap),
            "--aggression", str(args.aggression),
            "--hop_length", str(args.hop_length),
            "--window_size", str(args.window_size),
            "--segments_size", str(args.segments_size),
            "--post_process_threshold", str(args.post_process_threshold),
            "--enable_tta", str(args.enable_tta),
            "--enable_denoise", str(args.enable_denoise),
            "--high_end_process", str(args.high_end_process),
            "--enable_post_process", str(args.enable_post_process),
            "--separate_backing", str(args.separate_backing),
            "--separate_reverb", str(args.separate_reverb),
        ]

        result = subprocess.run(cmd, timeout=3600)
        if result.returncode != 0:
            logger.error("Separation failed with exit code %d", result.returncode)
            return 1

        logger.info("Audio separation completed!")
        logger.info("Output saved to: %s", output_dirs)
        return 0

    except Exception as e:
        logger.error("Audio separation failed: %s", e)
        return 1


def cmd_create_dataset(args):
    """Create training dataset."""
    logger.info("Creating training dataset...")

    try:
        from advanced_rvc_inference.utils.variables import python, configs

        cmd = [
            python, configs["create_dataset_path"],
            "--input_data", args.url or args.input,
            "--output_dirs", args.output,
            "--sample_rate", str(args.sample_rate),
        ]

        if args.clean_dataset:
            cmd.extend(["--clean_dataset", "--clean_strength", str(args.clean_strength)])

        if not args.separate:
            cmd.extend(["--separate", "False"])
        else:
            cmd.extend(["--separator_reverb", str(args.separate_reverb)])
            if args.separator_model:
                cmd.extend(["--model_name", args.separator_model])
            if args.reverb_model:
                cmd.extend(["--reverb_model", args.reverb_model])

        if args.skip_start > 0:
            cmd.extend(["--skip_start_audios", str(args.skip_start)])
        if args.skip_end > 0:
            cmd.extend(["--skip_end_audios", str(args.skip_end)])

        result = subprocess.run(cmd, timeout=7200)
        if result.returncode != 0:
            logger.error("Dataset creation failed with exit code %d", result.returncode)
            return 1

        logger.info("Dataset creation completed!")
        logger.info("Output saved to: %s", args.output)
        return 0

    except Exception as e:
        logger.error("Dataset creation failed: %s", e)
        return 1


def cmd_create_index(args):
    """Create model index."""
    logger.info("Creating model index for '%s'...", args.model_name)

    try:
        from advanced_rvc_inference.utils.variables import python, configs

        cmd = [
            python, configs["create_index_path"],
            "--model_name", args.model_name,
            "--rvc_version", args.version,
            "--index_algorithm", args.algorithm,
        ]

        result = subprocess.run(cmd, timeout=1800)
        if result.returncode != 0:
            logger.error("Index creation failed with exit code %d", result.returncode)
            return 1

        logger.info("Index creation completed!")
        return 0

    except Exception as e:
        logger.error("Index creation failed: %s", e)
        return 1


def cmd_extract(args):
    """Extract features for training."""
    logger.info("Extracting features for '%s'...", args.model_name)

    try:
        from advanced_rvc_inference.utils.variables import python, configs

        cmd = [
            python, configs["extract_path"],
            "--model_name", args.model_name,
            "--sample_rate", str(args.sample_rate),
            "--rvc_version", args.version,
            "--f0_method", args.f0_method,
            "--hop_length", str(args.hop_length),
            "--cpu_cores", str(args.cpu_cores),
            "--embedder_model", args.embedder_model,
        ]

        if args.f0_onnx:
            cmd.append("--f0_onnx")
        if not args.pitch_guidance:
            cmd.extend(["--pitch_guidance", "False"])
        if args.gpu is not None:
            cmd.extend(["--gpu", str(args.gpu)])
        else:
            cmd.extend(["--gpu", "-"])
        if args.rms_extract:
            cmd.append("--rms_extract")

        result = subprocess.run(cmd, timeout=3600)
        if result.returncode != 0:
            logger.error("Feature extraction failed with exit code %d", result.returncode)
            return 1

        logger.info("Feature extraction completed!")
        return 0

    except Exception as e:
        logger.error("Feature extraction failed: %s", e)
        return 1


def cmd_preprocess(args):
    """Preprocess training data."""
    logger.info("Preprocessing data for '%s'...", args.model_name)

    try:
        from advanced_rvc_inference.utils.variables import python, configs

        cmd = [
            python, configs["preprocess_path"],
            "--model_name", args.model_name,
            "--sample_rate", str(args.sample_rate),
            "--cpu_cores", str(args.cpu_cores),
            "--cut_preprocess", args.cut_method,
        ]

        if args.dataset_path:
            cmd.extend(["--dataset_path", args.dataset_path])
        if args.process_effects:
            cmd.append("--process_effects")
        if args.clean_dataset:
            cmd.extend(["--clean_dataset", "--clean_strength", str(args.clean_strength)])
        if args.chunk_len:
            cmd.extend(["--chunk_len", str(args.chunk_len)])
        if args.overlap_len:
            cmd.extend(["--overlap_len", str(args.overlap_len)])
        if args.normalization != "none":
            cmd.extend(["--normalization_mode", args.normalization])

        result = subprocess.run(cmd, timeout=3600)
        if result.returncode != 0:
            logger.error("Preprocessing failed with exit code %d", result.returncode)
            return 1

        logger.info("Preprocessing completed!")
        return 0

    except Exception as e:
        logger.error("Preprocessing failed: %s", e)
        return 1


def cmd_train(args):
    """Train a new RVC voice model."""
    logger.info("Starting model training for '%s'...", args.model_name)

    try:
        from advanced_rvc_inference.utils.variables import python, configs

        cmd = [
            python, configs["train_path"],
            "--model_name", args.model_name,
            "--rvc_version", args.version,
            "--save_every_epoch", str(args.save_every),
            "--total_epoch", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--gpu", str(args.gpu),
        ]

        if args.author:
            cmd.extend(["--model_author", args.author])
        if not args.save_latest:
            cmd.extend(["--save_only_latest", "False"])
        if not args.save_weights:
            cmd.extend(["--save_every_weights", "False"])
        if args.cache_gpu:
            cmd.append("--cache_data_in_gpu")
        if not args.pitch_guidance:
            cmd.extend(["--pitch_guidance", "False"])
        if args.pretrained_g:
            cmd.extend(["--g_pretrained_path", args.pretrained_g])
        if args.pretrained_d:
            cmd.extend(["--d_pretrained_path", args.pretrained_d])
        if args.vocoder != "HiFi-GAN":
            cmd.extend(["--vocoder", args.vocoder])
        if args.energy:
            cmd.append("--energy_use")
        if args.overtrain_detect:
            cmd.append("--overtraining_detector")
        if args.optimizer:
            cmd.extend(["--optimizer", args.optimizer])
        if args.multiscale_loss:
            cmd.append("--multiscale_mel_loss")
        if args.use_reference:
            cmd.extend(["--use_custom_reference", "--reference_path", args.reference_path])
        if args.checkpointing:
            cmd.append("--checkpointing")

        logger.info("Training started. This may take a while...")

        result = subprocess.run(cmd, timeout=None)
        if result.returncode != 0:
            logger.error("Training failed with exit code %d", result.returncode)
            return 1

        logger.info("Training completed!")
        return 0

    except Exception as e:
        logger.error("Training failed: %s", e)
        return 1


def cmd_create_ref(args):
    """Create reference audio."""
    logger.info("Creating reference from '%s'...", args.audio_file)

    try:
        from advanced_rvc_inference.utils.variables import python, configs

        cmd = [
            python, configs["create_reference_path"],
            "--audio_path", args.audio_file,
            "--reference_name", args.name,
            "--version", args.version,
        ]

        if not args.pitch_guidance:
            cmd.extend(["--pitch_guidance", "False"])
        if args.energy:
            cmd.append("--energy_use")
        if args.embedder_model:
            cmd.extend(["--embedder_model", args.embedder_model])
        if args.f0_method:
            cmd.extend(["--f0_method", args.f0_method])
        if args.pitch_shift != 0:
            cmd.extend(["--f0_up_key", str(args.pitch_shift)])
        if args.f0_autotune:
            cmd.append("--f0_autotune")
        if args.alpha != 0.5:
            cmd.extend(["--alpha", str(args.alpha)])

        result = subprocess.run(cmd, timeout=1800)
        if result.returncode != 0:
            logger.error("Reference creation failed with exit code %d", result.returncode)
            return 1

        logger.info("Reference creation completed!")
        return 0

    except Exception as e:
        logger.error("Reference creation failed: %s", e)
        return 1


def cmd_download(args):
    """Download models or audio from URLs."""
    if not args.link:
        logger.error("Download link is required (-l)")
        return 1

    logger.info("Downloading from: %s", args.link)

    try:
        from advanced_rvc_inference.core.downloads import download_model

        result = download_model(url=args.link, model=args.name)
        if result:
            logger.info("Download completed successfully!")
        else:
            logger.error("Download failed")
            return 1
        return 0

    except Exception as e:
        logger.error("Download failed: %s", e)
        return 1


def cmd_serve(args):
    """Launch the web interface."""
    easy_mode = args.easy is not None and args.easy.lower() in ("true", "1", "yes")
    mode_str = "Easy GUI" if easy_mode else "Full GUI"
    logger.info("Starting web interface (%s)...", mode_str)

    try:
        from advanced_rvc_inference.app.gui import launch

        launch(
            share=args.share,
            server_name=args.host,
            server_port=args.port,
            easy=easy_mode,
        )
        return 0

    except ImportError as e:
        logger.error("Failed to import GUI module: %s", e)
        return 1
    except Exception as e:
        logger.error("Failed to start web interface: %s", e)
        return 1


# ============================================================================
# Parser
# ============================================================================

def create_parser():
    """Create the argument parser matching the wiki CLI Guide."""

    parser = argparse.ArgumentParser(
        prog="rvc-cli",
        description="Advanced RVC Inference - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic voice conversion
  rvc-cli infer -m mymodel.pth -i input.wav -o output.wav

  # Convert with pitch shift (octave up)
  rvc-cli infer -m mymodel.pth -i input.wav -p 12

  # Convert with custom F0 method
  rvc-cli infer -m mymodel.pth -i input.wav --f0_method harvest

  # Separate vocals from music
  rvc-cli uvr -i music.wav --model HP-Vocal-2

  # Launch web interface
  rvc-cli serve --port 8080 --share

  # Launch simplified Easy GUI
  rvc-cli serve --easy true

  # Download a model
  rvc-cli download -l "https://huggingface.co/user/model/resolve/main/model.pth"

  # List available models
  rvc-cli list-models

  # Show system info
  rvc-cli info

For more information, visit:
  https://github.com/ArkanDash/Advanced-RVC-Inference/wiki/Cli-Guide
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

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        description="Available commands",
    )

    # ----- infer -----
    p = subparsers.add_parser("infer", help="Voice conversion",
        description="Convert voice in an audio file using an RVC model.")
    p.add_argument("-m", "--model", required=True, help="Path to RVC model (.pth or .onnx)")
    p.add_argument("-i", "--input", required=True, help="Path to input audio file")
    p.add_argument("-o", "--output", help="Output file path")
    p.add_argument("-p", "--pitch", type=int, default=0, help="Pitch shift in semitones (default: 0)")
    p.add_argument("-f", "--format", default="wav", choices=["wav", "mp3", "flac", "ogg"], help="Output format (default: wav)")
    p.add_argument("--index", help="Path to .index file")
    p.add_argument("--f0_method", default="rmvpe", help="F0 extraction method (default: rmvpe)")
    p.add_argument("--filter_radius", type=int, default=3, help="Filter radius for F0 smoothing (default: 3)")
    p.add_argument("--index_rate", type=float, default=0.5, help="Index strength 0.0-1.0 (default: 0.5)")
    p.add_argument("--rms_mix_rate", type=float, default=1.0, help="RMS mix rate (default: 1.0)")
    p.add_argument("--protect", type=float, default=0.33, help="Protect consonants 0.0-1.0 (default: 0.33)")
    p.add_argument("--hop_length", type=int, default=64, help="Hop length (default: 64)")
    p.add_argument("--embedder_model", default="hubert_base", help="Embedding model (default: hubert_base)")
    p.add_argument("--resample_sr", type=int, default=0, help="Resample rate (0=original)")
    p.add_argument("--split_audio", action="store_true", help="Split audio before processing")
    p.add_argument("--checkpointing", action="store_true", help="Enable memory checkpointing")
    p.add_argument("--f0_autotune", action="store_true", help="Enable F0 autotune")
    p.add_argument("--f0_autotune_strength", type=float, default=1.0, help="Autotune strength (default: 1.0)")
    p.add_argument("--formant_shifting", action="store_true", help="Enable formant shifting")
    p.add_argument("--formant_qfrency", type=float, default=0.8, help="Formant frequency (default: 0.8)")
    p.add_argument("--formant_timbre", type=float, default=0.8, help="Formant timbre (default: 0.8)")
    p.add_argument("--clean_audio", action="store_true", help="Apply audio cleaning")
    p.add_argument("--clean_strength", type=float, default=0.7, help="Cleaning strength (default: 0.7)")
    p.add_argument("--speaker_id", type=int, default=0, help="Speaker ID for multi-speaker models")
    p.add_argument("--predictor_onnx", action="store_true", help="Use ONNX F0 predictor")
    p.add_argument("--embedders_mode", default="fairseq", help="Embedder mode (default: fairseq)")
    p.add_argument("--f0_file", default="", help="Path to pre-existing F0 file")
    p.add_argument("--proposal_pitch", action="store_true", help="Use proposal pitch estimation")
    p.add_argument("--proposal_pitch_threshold", type=float, default=0.05, help="Pitch estimation threshold")
    p.add_argument("--audio_processing", action="store_true", help="Enable audio processing")
    p.add_argument("--alpha", type=float, default=0.5, help="Alpha blending (default: 0.5)")
    p.set_defaults(func=cmd_infer)

    # ----- uvr -----
    p = subparsers.add_parser("uvr", help="Audio separation",
        description="Separate vocals from instrumentals using UVR5.")
    p.add_argument("-i", "--input", required=True, help="Path to input audio file")
    p.add_argument("-o", "--output", help="Output directory (default: ./audios/uvr)")
    p.add_argument("-f", "--format", default="wav", help="Output format (default: wav)")
    p.add_argument("--model", default="MDXNET_Main", help="Separation model (default: MDXNET_Main)")
    p.add_argument("--karaoke_model", default="MDX-Version-1", help="Karaoke model")
    p.add_argument("--reverb_model", default="MDX-Reverb", help="Reverb model")
    p.add_argument("--denoise_model", default="Normal", help="Denoise model")
    p.add_argument("--sample_rate", type=int, default=44100, help="Output sample rate (default: 44100)")
    p.add_argument("--shifts", type=int, default=2, help="Number of predictions (default: 2)")
    p.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 1)")
    p.add_argument("--overlap", type=float, default=0.25, help="Overlap between segments (default: 0.25)")
    p.add_argument("--aggression", type=int, default=5, help="Extraction intensity (default: 5)")
    p.add_argument("--hop_length", type=int, default=1024, help="MDX hop length (default: 1024)")
    p.add_argument("--window_size", type=int, default=512, help="Window size (default: 512)")
    p.add_argument("--segments_size", type=int, default=256, help="Audio segment size (default: 256)")
    p.add_argument("--post_process_threshold", type=float, default=0.2, help="Post-process threshold")
    p.add_argument("--enable_tta", action="store_true", help="Enable test-time augmentation")
    p.add_argument("--enable_denoise", action="store_true", help="Enable denoising")
    p.add_argument("--high_end_process", action="store_true", help="High-frequency processing")
    p.add_argument("--enable_post_process", action="store_true", help="Enable post-processing")
    p.add_argument("--separate_backing", action="store_true", help="Separate backing vocals")
    p.add_argument("--separate_reverb", action="store_true", help="Separate reverb")
    p.set_defaults(func=cmd_uvr)

    # ----- create-dataset -----
    p = subparsers.add_parser("create-dataset", help="Create training dataset",
        description="Create training dataset from YouTube videos or local audio files.")
    p.add_argument("-u", "--url", help="YouTube URL (comma-separated for multiple)")
    p.add_argument("-i", "--input", help="Input directory with audio files")
    p.add_argument("-o", "--output", default="./dataset", help="Output directory (default: ./dataset)")
    p.add_argument("--sample_rate", type=int, default=48000, help="Sample rate (default: 48000)")
    p.add_argument("--clean_dataset", action="store_true", help="Apply data cleaning")
    p.add_argument("--clean_strength", type=float, default=0.7, help="Cleaning strength (default: 0.7)")
    p.add_argument("--separate", action="store_true", default=True, help="Separate vocals (default: True)")
    p.add_argument("--separator_model", default="MDXNET_Main", help="Separation model")
    p.add_argument("--separator_reverb", action="store_true", help="Separate reverb")
    p.add_argument("--reverb_model", default="MDX-Reverb", help="Reverb model")
    p.add_argument("--skip_start", type=int, default=0, help="Seconds to skip at start (default: 0)")
    p.add_argument("--skip_end", type=int, default=0, help="Seconds to skip at end (default: 0)")
    p.set_defaults(func=cmd_create_dataset)

    # ----- create-index -----
    p = subparsers.add_parser("create-index", help="Create model index",
        description="Create .index file for voice retrieval.")
    p.add_argument("model_name", help="Name of the model")
    p.add_argument("--version", choices=["v1", "v2"], default="v2", help="RVC version (default: v2)")
    p.add_argument("--algorithm", choices=["Auto", "Faiss", "KMeans"], default="Auto", help="Index algorithm (default: Auto)")
    p.set_defaults(func=cmd_create_index)

    # ----- extract -----
    p = subparsers.add_parser("extract", help="Feature extraction",
        description="Extract embeddings and F0 from training data.")
    p.add_argument("model_name", help="Name of the model")
    p.add_argument("--sample_rate", type=int, required=True, help="Sample rate of input audio")
    p.add_argument("--version", choices=["v1", "v2"], default="v2", help="RVC version (default: v2)")
    p.add_argument("--f0_method", default="rmvpe", help="F0 extraction method (default: rmvpe)")
    p.add_argument("--f0_onnx", action="store_true", help="Use ONNX F0 predictor")
    p.add_argument("--pitch_guidance", action="store_true", default=True, help="Use pitch guidance (default: True)")
    p.add_argument("--hop_length", type=int, default=128, help="Hop length (default: 128)")
    p.add_argument("--cpu_cores", type=int, default=2, help="CPU cores (default: 2)")
    p.add_argument("--gpu", help="GPU index (default: CPU)")
    p.add_argument("--embedder_model", default="hubert_base", help="Embedder model (default: hubert_base)")
    p.add_argument("--rms_extract", action="store_true", help="Extract RMS energy")
    p.set_defaults(func=cmd_extract)

    # ----- preprocess -----
    p = subparsers.add_parser("preprocess", help="Data preprocessing",
        description="Slice and normalize training audio.")
    p.add_argument("model_name", help="Name of the model")
    p.add_argument("--sample_rate", type=int, required=True, help="Sample rate")
    p.add_argument("--dataset_path", default="./dataset", help="Dataset path (default: ./dataset)")
    p.add_argument("--cpu_cores", type=int, default=2, help="CPU cores (default: 2)")
    p.add_argument("--cut_method", choices=["Automatic", "Simple", "Skip"], default="Automatic", help="Cutting method")
    p.add_argument("--process_effects", action="store_true", help="Apply preprocessing effects")
    p.add_argument("--clean_dataset", action="store_true", help="Clean dataset")
    p.add_argument("--clean_strength", type=float, default=0.7, help="Cleaning strength")
    p.add_argument("--chunk_len", type=float, default=3.0, help="Chunk length for Simple method")
    p.add_argument("--overlap_len", type=float, default=0.3, help="Overlap length")
    p.add_argument("--normalization", choices=["none", "pre", "post"], default="none", help="Normalization mode")
    p.set_defaults(func=cmd_preprocess)

    # ----- train -----
    p = subparsers.add_parser("train", help="Model training",
        description="Train a new RVC voice model.")
    p.add_argument("model_name", help="Name of the model")
    p.add_argument("--version", choices=["v1", "v2"], default="v2", help="RVC version (default: v2)")
    p.add_argument("--author", help="Model author name")
    p.add_argument("--epochs", type=int, default=300, help="Total training epochs (default: 300)")
    p.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8)")
    p.add_argument("--save_every", type=int, default=50, help="Save checkpoint every N epochs (default: 50)")
    p.add_argument("--save_latest", action="store_true", default=True, help="Save only latest checkpoint (default: True)")
    p.add_argument("--save_weights", action="store_true", default=True, help="Save all model weights (default: True)")
    p.add_argument("--gpu", default="0", help="GPU index (default: 0)")
    p.add_argument("--cache_gpu", action="store_true", help="Cache data in GPU")
    p.add_argument("--pitch_guidance", action="store_true", default=True, help="Use pitch guidance (default: True)")
    p.add_argument("--pretrained_g", help="Path to pre-trained G weights")
    p.add_argument("--pretrained_d", help="Path to pre-trained D weights")
    p.add_argument("--vocoder", default="HiFi-GAN", help="Vocoder (default: HiFi-GAN). Choices: HiFi-GAN, Default, MRF-HiFi-GAN, RefineGAN, BigVGAN, RingFormer, PCPH-GAN, Vocos, HiFi-GAN-v3, JVSF-HiFi-GAN, WaveGlow, NSF-APNet, FullBand-MRF")
    p.add_argument("--energy", action="store_true", help="Use RMS energy")
    p.add_argument("--overtrain_detect", action="store_true", help="Enable overtraining detection")
    p.add_argument("--optimizer", default="AdamW", help="Optimizer (default: AdamW)")
    p.add_argument("--multiscale_loss", action="store_true", help="Use multi-scale mel loss")
    p.add_argument("--use_reference", action="store_true", help="Use custom reference set")
    p.add_argument("--reference_path", help="Path to reference set")
    p.add_argument("--checkpointing", action="store_true", help="Enable checkpointing")
    p.set_defaults(func=cmd_train)

    # ----- create-ref -----
    p = subparsers.add_parser("create-ref", help="Create reference set",
        description="Create reference audio for better inference quality.")
    p.add_argument("audio_file", help="Path to audio file")
    p.add_argument("-n", "--name", default="reference", help="Reference name (default: reference)")
    p.add_argument("--version", choices=["v1", "v2"], default="v2", help="RVC version (default: v2)")
    p.add_argument("--pitch_guidance", action="store_true", default=True, help="Use pitch guidance (default: True)")
    p.add_argument("--energy", action="store_true", help="Use RMS energy")
    p.add_argument("--embedder_model", default="hubert_base", help="Embedder model")
    p.add_argument("--f0_method", default="rmvpe", help="F0 method (default: rmvpe)")
    p.add_argument("--pitch_shift", type=int, default=0, help="Pitch shift")
    p.add_argument("--filter_radius", type=int, default=3, help="Filter radius (default: 3)")
    p.add_argument("--f0_autotune", action="store_true", help="Enable F0 autotune")
    p.add_argument("--alpha", type=float, default=0.5, help="Alpha blending (default: 0.5)")
    p.set_defaults(func=cmd_create_ref)

    # ----- download -----
    p = subparsers.add_parser("download", help="Download models/audio",
        description="Download models from HuggingFace or audio from YouTube.")
    p.add_argument("-l", "--link", required=True, help="Download link (HuggingFace or YouTube URL)")
    p.add_argument("-n", "--name", help="Name to save as")
    p.set_defaults(func=cmd_download)

    # ----- serve -----
    p = subparsers.add_parser("serve", help="Launch web interface",
        description="Launch the Gradio web UI.")
    p.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=7860, help="Port to bind (default: 7860)")
    p.add_argument("--share", action="store_true", help="Create public share URL")
    p.add_argument("--easy", "-ez", type=str, default=None, help="Launch Easy GUI (simplified mode). Use 'true' to enable.")
    p.set_defaults(func=cmd_serve)

    # ----- info -----
    subparsers.add_parser("info", help="Show system information",
        description="Show system and environment information.").set_defaults(func=lambda a: (show_info(), 0))

    # ----- version -----
    subparsers.add_parser("version", help="Show version information",
        description="Show version and dependency information.").set_defaults(func=lambda a: (show_version(), 0))

    # ----- list-models -----
    subparsers.add_parser("list-models", help="List available models",
        description="List installed models in the weights folder.").set_defaults(func=lambda a: (list_models(), 0))

    # ----- list-f0-methods -----
    subparsers.add_parser("list-f0-methods", help="List F0 methods",
        description="Show all available pitch extraction methods.").set_defaults(func=lambda a: (list_f0_methods(), 0))

    return parser


def main():
    """Main entry point for the CLI."""
    setup_environment()

    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if getattr(args, "verbose", False):
        logging.getLogger().setLevel(logging.DEBUG)

    func = getattr(args, "func", None)
    if func:
        try:
            return func(args)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            return 130
        except Exception as e:
            logger.error("Error: %s", e)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

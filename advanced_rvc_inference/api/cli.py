#!/usr/bin/env python3
"""
Command Line Interface for Advanced RVC Inference.

Provides CLI commands for:
- inference: Run voice conversion
- train: Train RVC models
- download: Download pre-trained models
- serve: Launch the web interface
- version: Show version information
- info: Show system information
"""

import argparse
import sys
import os
import platform
import shutil
import logging
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Setup the environment for RVC operations."""
    # Add current directory to path
    cwd = Path.cwd()
    if str(cwd) not in sys.path:
        sys.path.insert(0, str(cwd))


def get_version():
    """Get the package version."""
    try:
        from advanced_rvc_inference._version import __version__

        return __version__
    except ImportError:
        return "2.0.0"


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
    except ImportError:
        version_info.append("PyTorch: Not installed")

    print("\n".join(version_info))


def show_info():
    """Show system information and configuration."""
    info = [
        "System Information:",
        f"  Platform: {platform.system()} {platform.release()}",
        f"  Architecture: {platform.machine()}",
        f"  Python: {platform.python_version()}",
        f"  Package Version: {get_version()}",
        f"  CPU Count: {os.cpu_count()}",
        f"  Memory: {shutil.disk_usage('/').total // (1024**3)} GB total",
        f"  Free Disk: {shutil.disk_usage('/').free // (1024**3)} GB",
    ]

    try:
        import torch

        info.append(f"PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            info.append(f"CUDA Available: True")
            info.append(f"CUDA Version: {torch.version.cuda}")
            gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            info.append(f"GPU Memory: {gpu_mem} GB")
        else:
            info.append("CUDA Available: False")
    except ImportError:
        info.append("PyTorch: Not installed")

    print("\n".join(info))


def cmd_inference(args):
    """Run voice conversion inference."""
    logger.info("Starting inference...")

    try:
        from advanced_rvc_inference.rvc.infer.inference import convert

        # Validate inputs
        if not args.input or not Path(args.input).exists():
            logger.error(f"Input file not found: {args.input}")
            return 1

        if not args.model:
            logger.error("Model path is required")
            return 1

        # Set defaults
        pitch = getattr(args, "pitch", 0)
        output = getattr(args, "output", None)
        format = getattr(args, "format", "wav")

        # Run conversion
        convert(
            pitch=pitch,
            filter_radius=3,
            index_rate=0.5,
            rms_mix_rate=1,
            protect=0.33,
            hop_length=128,
            f0_method="rmvpe",
            input_path=args.input,
            output_path=output or args.input.replace(".", f"_converted."),
            pth_path=args.model,
            index_path=getattr(args, "index", None),
            f0_autotune=False,
            clean_audio=False,
            clean_strength=0.5,
            export_format=format,
            embedder_model="contentvec_base",
            resample_sr=0,
            split_audio=False,
            f0_autotune_strength=0.5,
            checkpointing=False,
            f0_onnx=False,
            embedders_mode="fairseq",
            formant_shifting=False,
            formant_qfrency=0,
            formant_timbre=0,
            f0_file="",
            proposal_pitch=False,
            proposal_pitch_threshold=0.05,
            audio_processing=False,
            alpha=0.5,
        )

        logger.info("Inference completed successfully!")
        return 0

    except ImportError as e:
        logger.error(f"Failed to import inference module: {e}")
        return 1
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return 1


def cmd_train(args):
    """Start model training."""
    logger.info("Training module - Use web UI for full training features")

    if args.name:
        logger.info(f"Training configuration: name={args.name}")
        logger.info("Please use the web interface for complete training options")

    print("\nTraining is best performed through the web interface.")
    print("Run: rvc-gui")
    print("Then navigate to the Training tab.")

    return 0


def cmd_download(args):
    """Download pre-trained models."""
    logger.info("Download module - Use web UI for model downloads")

    print("\nModel downloads are best performed through the web interface.")
    print("Run: rvc-gui")
    print("Then navigate to the Downloads tab.")

    return 0


def cmd_serve(args):
    """Launch the web interface."""
    logger.info("Starting web interface...")

    try:
        from advanced_rvc_inference import gui

        gui.launch(
            share=args.share,
            server_name=getattr(args, "host", "0.0.0.0"),
            server_port=getattr(args, "port", 7860),
        )
        return 0
    except ImportError as e:
        logger.error(f"Failed to import GUI module: {e}")
        return 1
    except Exception as e:
        logger.error(f"Failed to start web interface: {e}")
        return 1


def cmd_realtime(args):
    """Start real-time voice conversion."""
    logger.info("Starting real-time mode...")

    try:
        from advanced_rvc_inference.core.realtime import realtime_tab

        print("Real-time voice conversion is best performed through the web interface.")
        print("Run: rvc-gui")
        print("Then navigate to the Realtime tab.")

        return 0
    except ImportError as e:
        logger.error(f"Failed to import realtime module: {e}")
        return 1


def create_parser():
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="rvc-cli",
        description="Advanced RVC Inference - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  rvc-cli infer --model model.pth --input audio.wav
  rvc-cli serve --share
  rvc-cli version
  rvc-cli info

For more information, visit:
  https://github.com/ArkanDash/Advanced-RVC-Inference
        """.strip(),
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"Advanced RVC Inference v{get_version()}",
    )

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        description="Available commands",
    )

    # Inference command
    infer_parser = subparsers.add_parser(
        "infer",
        help="Run voice conversion inference",
        description="Run voice conversion on an audio file",
    )
    infer_parser.add_argument(
        "-m", "--model", required=True, help="Path to the RVC model file (.pth or .onnx)"
    )
    infer_parser.add_argument(
        "-i", "--input", required=True, help="Path to the input audio file"
    )
    infer_parser.add_argument(
        "-o", "--output", help="Path to the output audio file"
    )
    infer_parser.add_argument(
        "-p", "--pitch", type=int, default=0, help="Pitch shift (semitones)"
    )
    infer_parser.add_argument(
        "-f",
        "--format",
        default="wav",
        choices=["wav", "mp3", "flac", "ogg"],
        help="Output format",
    )
    infer_parser.add_argument(
        "--index", help="Path to the index file (.index)"
    )
    infer_parser.set_defaults(handler=cmd_inference)

    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train RVC models",
        description="Train RVC voice models (use web UI for full features)",
    )
    train_parser.add_argument(
        "--name", help="Name for the training experiment"
    )
    train_parser.set_defaults(handler=cmd_train)

    # Download command
    download_parser = subparsers.add_parser(
        "download",
        help="Download pre-trained models",
        description="Download pre-trained models (use web UI for full features)",
    )
    download_parser.set_defaults(handler=cmd_download)

    # Serve command
    serve_parser = subparsers.add_parser(
        "serve",
        help="Launch the web interface",
        description="Launch the Gradio web interface",
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

    # Realtime command
    realtime_parser = subparsers.add_parser(
        "realtime",
        help="Start real-time voice conversion",
        description="Start real-time voice conversion (use web UI for full features)",
    )
    realtime_parser.set_defaults(handler=cmd_realtime)

    # Version command
    version_parser = subparsers.add_parser(
        "version",
        help="Show version information",
        description="Show version and system information",
    )
    version_parser.set_defaults(handler=lambda args: (show_version(), 0))

    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show system information",
        description="Show detailed system and package information",
    )
    info_parser.set_defaults(handler=lambda args: (show_info(), 0))

    return parser


def main():
    """Main entry point for the CLI."""
    import platform
    import os

    # Setup environment
    setup_environment()

    # Create parser
    parser = create_parser()

    # Parse arguments
    args = parser.parse_args()

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
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

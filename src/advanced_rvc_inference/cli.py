"""CLI interface for Advanced RVC Inference"""

import argparse
import sys
from pathlib import Path

from . import EnhancedF0Extractor, EnhancedAudioSeparator, RealtimeVoiceChanger, EnhancedModelManager, EnhancedUIComponents


def create_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Advanced RVC Inference - Enhanced voice conversion with Vietnamese-RVC integration"
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version="%(prog)s 3.4.0"
    )
    
    parser.add_argument(
        "--mode",
        choices=["web", "cli", "realtime"],
        default="web",
        help="Launch mode (default: web)"
    )
    
    parser.add_argument(
        "--theme",
        choices=["default", "dark", "light"],
        default="default",
        help="UI theme (default: default)"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public shareable link (web mode only)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for web interface (default: 7860)"
    )
    
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for web interface (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force CPU-only inference"
    )
    
    parser.add_argument(
        "--models-path",
        type=str,
        help="Custom path for model storage"
    )
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.debug:
        print("Debug mode enabled")
        print(f"Mode: {args.mode}")
        print(f"Theme: {args.theme}")
        print(f"Port: {args.port}")
        print(f"Host: {args.host}")
    
    if args.mode == "web":
        # Import the local app module
        import sys
        import os
        
        # Add the parent directory to Python path to access app.py
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, parent_dir)
        
        try:
            from app import run_web_interface
            run_web_interface(
                theme=args.theme,
                share=args.share,
                port=args.port,
                host=args.host,
                debug=args.debug,
                cpu_only=args.cpu_only,
                models_path=args.models_path
            )
        except ImportError as e:
            print(f"Error importing app module: {e}")
            print("Please ensure app.py is in the same directory as this package.")
            sys.exit(1)
    elif args.mode == "cli":
        print("CLI mode selected - Feature coming soon")
        sys.exit(1)
    elif args.mode == "realtime":
        print("Realtime mode selected - Feature coming soon")
        sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


def inference_cli():
    """CLI for inference operations."""
    parser = argparse.ArgumentParser(
        description="RVC Inference CLI - Batch voice conversion"
    )
    
    parser.add_argument(
        "--input",
        required=True,
        help="Input audio file or directory"
    )
    
    parser.add_argument(
        "--model",
        required=True,
        help="Path to RVC model file"
    )
    
    parser.add_argument(
        "--output",
        required=True,
        help="Output audio file or directory"
    )
    
    parser.add_argument(
        "--format",
        default="wav",
        choices=["wav", "mp3", "flac"],
        help="Output format (default: wav)"
    )
    
    parser.add_argument(
        "--quality",
        choices=["fast", "standard", "high"],
        default="standard",
        help="Processing quality (default: standard)"
    )
    
    parser.add_argument(
        "--f0-method",
        default="rmvpe",
        help="F0 extraction method (default: rmvpe)"
    )
    
    args = parser.parse_args()
    
    print(f"Converting {args.input} with model {args.model}")
    print(f"Output: {args.output}")
    print(f"Quality: {args.quality}")
    print("Feature coming soon - Use web interface for now")


def training_cli():
    """CLI for training operations."""
    parser = argparse.ArgumentParser(
        description="RVC Training CLI - Model training"
    )
    
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to training dataset"
    )
    
    parser.add_argument(
        "--output",
        required=True,
        help="Output model path"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of training epochs (default: 1000)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size (default: 4)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    
    args = parser.parse_args()
    
    print(f"Training model from dataset: {args.dataset}")
    print(f"Output model: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("Feature coming soon - Use training tab in web interface")


if __name__ == "__main__":
    main()
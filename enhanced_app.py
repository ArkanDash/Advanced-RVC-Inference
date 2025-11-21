"""
Enhanced Advanced RVC Inference Application

Main application entry point with comprehensive improvements based on
Vietnamese-RVC, Applio, and other reference projects.

Features:
- Enhanced project structure with proper Python packaging
- 40+ F0 extraction methods with 29 hybrid combinations
- Advanced audio separation with multiple backends
- Real-time voice changer with low latency
- Comprehensive model management
- Modern UI with multi-language support
- Docker support for multiple environments
- Enhanced performance and memory optimization
- Comprehensive testing framework
"""

import os
import sys
import logging
import argparse
import signal
import time
from pathlib import Path
from typing import Optional, Dict, Any
import threading
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Core imports
from src.advanced_rvc_inference.ui.components import get_ui_instance
from src.advanced_rvc_inference.core.f0_extractor import get_f0_extractor
from src.advanced_rvc_inference.audio.separation import get_audio_separator
from src.advanced_rvc_inference.audio.voice_changer import (
    get_voice_changer, 
    VoiceChangerConfig, 
    AudioDeviceConfig
)
from src.advanced_rvc_inference.models.manager import get_model_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_rvc_inference.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class EnhancedAdvancedRVCApp:
    """
    Enhanced Advanced RVC Inference Application.
    
    This is the main application class that orchestrates all components
    and provides a unified interface for voice conversion, audio separation,
    model management, and real-time voice changing.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the enhanced application.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.is_running = False
        self.shutdown_event = threading.Event()
        
        # Initialize core components
        self.f0_extractor = get_f0_extractor(
            device=self.config.get('compute', {}).get('device', 'auto'),
            enable_onnx=self.config.get('compute', {}).get('enable_onnx', True)
        )
        
        self.audio_separator = get_audio_separator(
            device=self.config.get('compute', {}).get('device', 'auto'),
            enable_onnx=self.config.get('compute', {}).get('enable_onnx', True),
            memory_efficient=self.config.get('performance', {}).get('memory_efficient', False)
        )
        
        self.model_manager = get_model_manager(
            models_dir=self.config.get('paths', {}).get('models_dir', 'models'),
            cache_dir=self.config.get('paths', {}).get('cache_dir', None)
        )
        
        # Initialize UI
        self.ui = get_ui_instance(
            title=self.config.get('ui', {}).get('title', 'Advanced RVC Inference V3.4'),
            theme=self.config.get('ui', {}).get('theme', 'gradio/default'),
            language=self.config.get('ui', {}).get('language', 'en-US')
        )
        
        logger.info("Enhanced Advanced RVC Application initialized")
    
    def _load_config(self, config_path: Optional[Path] = None) -> Dict[str, Any]:
        """Load application configuration."""
        if config_path is None:
            config_path = Path("config/enhanced_config.json")
        
        default_config = {
            "app": {
                "name": "Advanced RVC Inference",
                "version": "3.4.0",
                "description": "Enhanced Voice Conversion with Vietnamese-RVC Integration"
            },
            "compute": {
                "device": "auto",
                "enable_onnx": True,
                "memory_efficient": False,
                "max_workers": 4
            },
            "ui": {
                "title": "Advanced RVC Inference V3.4",
                "theme": "gradio/default",
                "language": "en-US",
                "share_mode": False,
                "show_warnings": True
            },
            "paths": {
                "models_dir": "models",
                "cache_dir": "cache",
                "logs_dir": "logs",
                "output_dir": "output"
            },
            "performance": {
                "memory_efficient": False,
                "cache_models": True,
                "preload_models": False,
                "optimize_memory": True
            },
            "audio": {
                "default_sample_rate": 44100,
                "chunk_size": 1024,
                "buffer_size": 2048,
                "enable_vad": True,
                "vad_sensitivity": 3
            },
            "f0_extraction": {
                "default_method": "rmvpe",
                "enable_hybrid": True,
                "enable_onnx": True,
                "f0_min": 50.0,
                "f0_max": 1200.0
            },
            "logging": {
                "level": "INFO",
                "file": "advanced_rvc_inference.log",
                "max_size_mb": 100,
                "backup_count": 5
            }
        }
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    
                # Deep merge configs
                def deep_merge(base, update):
                    for key, value in update.items():
                        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                            deep_merge(base[key], value)
                        else:
                            base[key] = value
                    return base
                
                config = deep_merge(default_config.copy(), user_config)
                
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}. Using defaults.")
                config = default_config
        else:
            logger.info(f"Config file not found at {config_path}. Creating default config.")
            config = default_config
            self._save_config(config, config_path)
        
        return config
    
    def _save_config(self, config: Dict[str, Any], config_path: Path):
        """Save configuration to file."""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Config saved to {config_path}")
    
    def run_web_ui(self, share: bool = False, debug: bool = False, port: int = 7860):
        """
        Run the web-based user interface.
        
        Args:
            share: Enable public sharing
            debug: Enable debug mode
            port: Port to run the server on
        """
        try:
            logger.info("Starting Enhanced RVC Inference Web UI...")
            
            # Create interface
            interface = self.ui.create_interface()
            
            # Launch interface
            interface.launch(
                server_name="0.0.0.0",
                server_port=port,
                share=share,
                debug=debug,
                show_error=True,
                quiet=False,
                inbrowser=True,
                prevent_thread_lock=True
            )
            
            logger.info(f"Web UI launched on http://localhost:{port}")
            
            return interface
            
        except Exception as e:
            logger.error(f"Failed to start Web UI: {e}")
            raise
    
    def run_cli(self, command: str, **kwargs):
        """
        Run CLI interface.
        
        Args:
            command: Command to execute
            **kwargs: Command-specific arguments
        """
        if command == "inference":
            self._run_cli_inference(**kwargs)
        elif command == "separate":
            self._run_cli_separation(**kwargs)
        elif command == "train":
            self._run_cli_training(**kwargs)
        elif command == "models":
            self._run_cli_models(**kwargs)
        else:
            logger.error(f"Unknown command: {command}")
    
    def _run_cli_inference(self, 
                          model_path: str,
                          input_audio: str,
                          output_audio: str = None,
                          **kwargs):
        """Run CLI voice conversion inference."""
        try:
            logger.info(f"Starting CLI inference: {model_path} -> {output_audio}")
            
            # Import inference module
            from src.advanced_rvc_inference.core.inference import RVCInference
            
            # Initialize inference
            inference = RVCInference(
                model_path=model_path,
                device=self.config.get('compute', {}).get('device', 'auto')
            )
            
            # Run inference
            result_audio = inference.infer_from_file(
                input_audio,
                output_path=output_audio,
                **kwargs
            )
            
            logger.info(f"Inference completed: {result_audio}")
            
        except Exception as e:
            logger.error(f"CLI inference failed: {e}")
            raise
    
    def _run_cli_separation(self,
                           input_audio: str,
                           output_dir: str = "separated",
                           model: str = "BS-Roformer-Viperx-1297",
                           **kwargs):
        """Run CLI audio separation."""
        try:
            logger.info(f"Starting CLI separation: {input_audio} -> {output_dir}")
            
            # Run separation
            results = self.audio_separator.separate_audio(
                audio_input=input_audio,
                model_name=model,
                output_dir=output_dir,
                **kwargs
            )
            
            logger.info(f"Separation completed. Output files: {list(results.keys())}")
            
        except Exception as e:
            logger.error(f"CLI separation failed: {e}")
            raise
    
    def _run_cli_training(self, dataset_path: str, **kwargs):
        """Run CLI model training."""
        try:
            logger.info(f"Starting CLI training: {dataset_path}")
            
            # Import training module
            from src.advanced_rvc_inference.training.trainer import RVCTrainer
            
            # Initialize trainer
            trainer = RVCTrainer(
                device=self.config.get('compute', {}).get('device', 'auto'),
                **kwargs
            )
            
            # Run training
            model_path = trainer.train(
                dataset_path=dataset_path,
                **kwargs
            )
            
            logger.info(f"Training completed: {model_path}")
            
        except Exception as e:
            logger.error(f"CLI training failed: {e}")
            raise
    
    def _run_cli_models(self, action: str, **kwargs):
        """Run CLI model management."""
        try:
            if action == "list":
                models = self.model_manager.search_models(**kwargs)
                print(f"Found {len(models)} models:")
                for model in models:
                    print(f"  - {model.name} ({model.size_mb:.1f} MB)")
            
            elif action == "download":
                source = kwargs.get('source')
                name = kwargs.get('name')
                if source and name:
                    success = self.model_manager.download_model(
                        source=source,
                        model_name=name,
                        category=kwargs.get('category', 'custom')
                    )
                    print(f"Download {'successful' if success else 'failed'}: {name}")
            
            elif action == "validate":
                model_name = kwargs.get('model_name')
                if model_name:
                    is_valid, message = self.model_manager.validate_model(model_name)
                    print(f"Validation: {'Valid' if is_valid else 'Invalid'} - {message}")
            
            else:
                logger.error(f"Unknown model action: {action}")
        
        except Exception as e:
            logger.error(f"CLI model management failed: {e}")
            raise
    
    def run_realtime_changer(self, model_path: str, **kwargs):
        """
        Run real-time voice changer.
        
        Args:
            model_path: Path to voice model
            **kwargs: Configuration parameters
        """
        try:
            logger.info("Starting Real-time Voice Changer...")
            
            # Create configurations
            voice_config = VoiceChangerConfig(
                model_path=model_path,
                **kwargs
            )
            
            device_config = AudioDeviceConfig(
                sample_rate=self.config.get('audio', {}).get('default_sample_rate', 44100),
                chunk_size=self.config.get('audio', {}).get('chunk_size', 1024),
                buffer_size=self.config.get('audio', {}).get('buffer_size', 2048)
            )
            
            # Initialize voice changer
            voice_changer = get_voice_changer(
                config=voice_config,
                device_config=device_config,
                backend=self.config.get('compute', {}).get('device', 'auto')
            )
            
            # Initialize and start
            if voice_changer.initialize():
                if voice_changer.start():
                    logger.info("Real-time Voice Changer started. Press Ctrl+C to stop.")
                    
                    # Keep running until interrupted
                    try:
                        while not self.shutdown_event.is_set():
                            time.sleep(0.1)
                    except KeyboardInterrupt:
                        logger.info("Stopping Real-time Voice Changer...")
                        voice_changer.stop()
                else:
                    logger.error("Failed to start Real-time Voice Changer")
            else:
                logger.error("Failed to initialize Real-time Voice Changer")
        
        except Exception as e:
            logger.error(f"Real-time voice changer failed: {e}")
            raise
    
    def benchmark(self, test_audio: str = None):
        """
        Run performance benchmarks.
        
        Args:
            test_audio: Path to test audio file
        """
        try:
            logger.info("Starting performance benchmarks...")
            
            if test_audio is None:
                test_audio = "test_audio.wav"
            
            # Benchmark F0 extraction methods
            logger.info("Benchmarking F0 extraction methods...")
            f0_methods = ["rmvpe", "crepe-tiny", "dio", "harvest", "fcpe"]
            f0_results = self.f0_extractor.benchmark_methods(
                test_audio, 
                sample_rate=44100,
                methods=f0_methods
            )
            
            # Benchmark audio separation models
            logger.info("Benchmarking audio separation models...")
            sep_models = ["BS-Roformer-Viperx-1297", "demucs_htdemucs", "mdx_original"]
            sep_results = self.audio_separator.benchmark_models(test_audio, sep_models)
            
            # Print results
            print("\n=== F0 Extraction Benchmark Results ===")
            for method, results in f0_results.items():
                print(f"{method:20s}: {results.get('processing_time', 0):.3f}s")
            
            print("\n=== Audio Separation Benchmark Results ===")
            for model, results in sep_results.items():
                print(f"{model:30s}: {results.get('processing_time', 0):.3f}s")
            
            logger.info("Benchmark completed")
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            raise
    
    def start(self):
        """Start the application."""
        self.is_running = True
        logger.info("Enhanced Advanced RVC Application started")
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def stop(self):
        """Stop the application."""
        self.is_running = False
        self.shutdown_event.set()
        logger.info("Enhanced Advanced RVC Application stopped")
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown."""
        logger.info(f"Received signal {signum}. Shutting down gracefully...")
        self.shutdown_event.set()
    
    def get_status(self) -> Dict[str, Any]:
        """Get application status."""
        return {
            "is_running": self.is_running,
            "config": self.config,
            "components": {
                "f0_extractor": self.f0_extractor is not None,
                "audio_separator": self.audio_separator is not None,
                "model_manager": self.model_manager is not None,
                "ui": self.ui is not None
            },
            "statistics": {
                "models_loaded": len(self.model_manager.models_metadata),
                "cache_size_mb": sum(m.size_mb for m in self.model_manager.models_metadata.values())
            }
        }


def create_app(config_path: Optional[Path] = None) -> EnhancedAdvancedRVCApp:
    """Create and return the enhanced application instance."""
    return EnhancedAdvancedRVCApp(config_path)


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced Advanced RVC Inference V3.4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run web UI
  python enhanced_app.py web --port 7860 --share
  
  # CLI inference
  python enhanced_app.py cli inference --model_path model.pth --input_audio input.wav --output_audio output.wav
  
  # Audio separation
  python enhanced_app.py cli separate --input_audio music.wav --model BS-Roformer-Viperx-1297
  
  # Real-time voice changer
  python enhanced_app.py realtime --model_path model.pth
  
  # Benchmark performance
  python enhanced_app.py benchmark --test_audio test.wav
  
  # Model management
  python enhanced_app.py cli models list
  python enhanced_app.py cli models download --source https://huggingface.co/model --name my_model
        """
    )
    
    parser.add_argument(
        "--config", 
        type=Path, 
        help="Path to configuration file"
    )
    parser.add_argument(
        "--version", 
        action="version", 
        version="Enhanced Advanced RVC Inference V3.4"
    )
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
        default="INFO",
        help="Set logging level"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Web UI command
    web_parser = subparsers.add_parser("web", help="Run web UI")
    web_parser.add_argument("--port", type=int, default=7860, help="Port to run server on")
    web_parser.add_argument("--share", action="store_true", help="Enable public sharing")
    web_parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    # CLI commands
    cli_parser = subparsers.add_parser("cli", help="Run CLI commands")
    cli_subparsers = cli_parser.add_subparsers(dest="cli_command")
    
    # CLI inference
    inference_parser = cli_subparsers.add_parser("inference", help="Voice conversion inference")
    inference_parser.add_argument("--model_path", required=True, help="Path to voice model")
    inference_parser.add_argument("--input_audio", required=True, help="Input audio file")
    inference_parser.add_argument("--output_audio", help="Output audio file")
    
    # CLI separation
    separation_parser = cli_subparsers.add_parser("separate", help="Audio separation")
    separation_parser.add_argument("--input_audio", required=True, help="Input audio file")
    separation_parser.add_argument("--model", default="BS-Roformer-Viperx-1297", help="Separation model")
    separation_parser.add_argument("--output_dir", default="separated", help="Output directory")
    
    # CLI models
    models_parser = cli_subparsers.add_parser("models", help="Model management")
    models_subparsers = models_parser.add_subparsers(dest="model_action")
    
    list_parser = models_subparsers.add_parser("list", help="List available models")
    download_parser = models_subparsers.add_parser("download", help="Download a model")
    download_parser.add_argument("--source", required=True, help="Model source URL")
    download_parser.add_argument("--name", required=True, help="Model name")
    download_parser.add_argument("--category", default="custom", help="Model category")
    
    validate_parser = models_subparsers.add_parser("validate", help="Validate a model")
    validate_parser.add_argument("--model_name", required=True, help="Model name to validate")
    
    # Training command
    training_parser = subparsers.add_parser("training", help="Model training")
    training_parser.add_argument("--dataset_path", required=True, help="Path to training dataset")
    
    # Real-time voice changer command
    realtime_parser = subparsers.add_parser("realtime", help="Real-time voice changer")
    realtime_parser.add_argument("--model_path", required=True, help="Path to voice model")
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run performance benchmarks")
    benchmark_parser.add_argument("--test_audio", help="Test audio file path")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Set up logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        # Create application
        app = create_app(args.config)
        
        # Start application
        app.start()
        
        # Execute command
        if args.command == "web":
            app.run_web_ui(
                share=args.share,
                debug=args.debug,
                port=args.port
            )
        
        elif args.command == "cli":
            if args.cli_command == "inference":
                app.run_cli("inference", **vars(args))
            elif args.cli_command == "separate":
                app.run_cli("separate", **vars(args))
            elif args.cli_command == "models":
                app.run_cli("models", **vars(args))
            else:
                cli_parser.print_help()
        
        elif args.command == "training":
            app.run_cli("train", **vars(args))
        
        elif args.command == "realtime":
            app.run_realtime_changer(args.model_path, **vars(args))
        
        elif args.command == "benchmark":
            app.benchmark(args.test_audio)
        
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise
    finally:
        if 'app' in locals():
            app.stop()


if __name__ == "__main__":
    main()
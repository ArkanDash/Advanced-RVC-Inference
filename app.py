"""
Simplified Application Launcher
Professional Entry Point for Advanced RVC Inference
"""

import os
import sys
import logging
import argparse
import traceback
from pathlib import Path

# Add src to Python path for imports
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

try:
    from advanced_rvc_inference.config import get_config, get_device
    from advanced_rvc_inference.core.memory_manager import monitor_memory, cleanup_memory
    from advanced_rvc_inference.core.app_launcher import AdvancedRVCApp
    KADVC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import advanced_rvc_inference modules: {e}")
    print("Falling back to basic launcher mode...")
    KADVC_AVAILABLE = False


def setup_logging(debug: bool = False, log_level: str = "INFO") -> logging.Logger:
    """Setup application logging."""
    log_level = "DEBUG" if debug else log_level
    
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # File handler
    log_file = Path("logs") / "app_launcher.log"
    log_file.parent.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def validate_environment() -> bool:
    """Validate environment and dependencies."""
    errors = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        errors.append("Python 3.8+ is required")
    
    # Check PyTorch
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                logger.info(f"GPU {i}: {gpu_name}")
    except ImportError:
        errors.append("PyTorch is not installed")
    
    # Check Gradio
    try:
        import gradio as gr
        logger.info(f"Gradio version: {gr.__version__}")
    except ImportError:
        errors.append("Gradio is not installed")
    
    # Check FFmpeg
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            logger.info("FFmpeg is available")
        else:
            errors.append("FFmpeg is not working properly")
    except (ImportError, subprocess.TimeoutExpired, FileNotFoundError):
        errors.append("FFmpeg is not installed or not in PATH")
    
    if errors:
        logger.error("Environment validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    logger.info("Environment validation passed")
    return True


def create_directories() -> None:
    """Create required directories."""
    directories = [
        "weights",
        "indexes",
        "logs", 
        "cache",
        "temp",
        "audio_files",
        "outputs"
    ]
    
    for dir_name in directories:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
    
    logger.info("Required directories created/verified")


def main():
    """Main application entry point."""
    global logger
    
    # Setup logging
    logger = setup_logging()
    logger.info("🚀 Advanced RVC Inference Launcher Starting...")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Advanced RVC Inference - Professional Voice Conversion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py                                    # Run with default settings
  python app.py --share --port 8080               # Public sharing on port 8080
  python app.py --debug --log-level DEBUG         # Debug mode
  python app.py --cpu                              # Force CPU mode
  python app.py --config custom_config.json       # Use custom config
        """
    )
    
    parser.add_argument(
        "--share", 
        action="store_true", 
        help="Enable public sharing of the application"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the application on (default: 7860)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode (disable GPU)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom configuration file"
    )
    
    parser.add_argument(
        "--no-monitor",
        action="store_true",
        help="Disable automatic memory monitoring"
    )
    
    try:
        args = parser.parse_args()
        
        # Update logging if debug specified
        if args.debug:
            logger = setup_logging(debug=True)
        
        logger.info(f"Command line arguments: {args}")
        
        # Validate environment
        if not validate_environment():
            print("\n❌ Environment validation failed. Please install required dependencies.")
            print("\nTo install dependencies, run:")
            print("pip install -r requirements.txt")
            sys.exit(1)
        
        # Create directories
        create_directories()
        
        # Load configuration
        if args.config:
            config = get_config()
            config._config_file = Path(args.config)
            config._load_config()
            config._validate_config()
            logger.info(f"Loaded custom configuration: {args.config}")
        
        config = get_config()
        
        # Override configuration with command line arguments
        if args.cpu:
            config.performance_config.enable_mixed_precision = False
            logger.info("CPU mode enabled")
        
        config.server_config.update({
            'share': args.share,
            'port': args.port,
            'host': args.host,
            'debug': args.debug
        })
        
        # Start memory monitoring if enabled
        if not args.no_monitor:
            try:
                monitor_memory(interval=30)
                logger.info("Memory monitoring started")
            except Exception as e:
                logger.warning(f"Could not start memory monitoring: {e}")
        
        # Report system status
        device = get_device()
        perf_report = config.get_performance_report()
        logger.info(f"System Status: Device={device}, Batch Size={config.training_config.batch_size}")
        logger.info(f"Configuration: {config}")
        
        # Create and launch application
        if KADVC_AVAILABLE:
            logger.info("🚀 Starting Advanced RVC Application with KADVC support...")
            app_launcher = AdvancedRVCApp(config)
            app = app_launcher.create_app()
        else:
            logger.warning("⚠️ Running in basic mode without advanced features")
            app = create_basic_app()
        
        # Launch application
        logger.info(f"🌐 Launching on {config.server_config['host']}:{config.server_config['port']}")
        app.launch(
            share=config.server_config['share'],
            server_port=config.server_config['port'],
            server_name=config.server_config['host'],
            show_error=config.server_config['show_error'],
            inbrowser=config.server_config['inbrowser'],
            debug=args.debug
        )
        
    except KeyboardInterrupt:
        logger.info("👋 Application interrupted by user")
    except Exception as e:
        logger.error(f"💥 Critical error: {e}")
        logger.error(traceback.format_exc())
        print(f"""
❌ Critical Error: {e}

Please check:
1. All dependencies are installed correctly
2. FFmpeg is available in your system PATH
3. You have sufficient disk space
4. Your system meets the minimum requirements

For help, please check the documentation or create an issue on GitHub.
        """)
        
        # Emergency cleanup
        try:
            cleanup_memory(aggressive=True)
        except:
            pass
        
        sys.exit(1)
    
    finally:
        # Cleanup
        try:
            cleanup_memory()
        except:
            pass


def create_basic_app():
    """Create a basic Gradio app as fallback."""
    try:
        import gradio as gr
        
        with gr.Blocks(title="Advanced RVC Inference - Basic Mode") as app:
            gr.Markdown("# Advanced RVC Inference - Basic Mode")
            gr.Markdown("⚠️ Running in basic mode without advanced features")
            gr.Markdown("Please install all dependencies for full functionality")
            
            with gr.Tab("Information"):
                gr.Markdown("""
                ## System Information
                
                To get full functionality, please ensure:
                
                1. **PyTorch** is installed with CUDA support
                2. **Gradio** is updated to the latest version
                3. **FFmpeg** is available in your system PATH
                4. **All Python dependencies** are installed
                
                Run: `pip install -r requirements.txt`
                """)
        
        return app
        
    except Exception as e:
        logger.error(f"Could not create basic app: {e}")
        raise


if __name__ == "__main__":
    main()
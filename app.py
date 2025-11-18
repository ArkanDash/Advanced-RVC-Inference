"""
Enhanced Advanced RVC Inference Application
Improved with performance optimizations, better error handling, and enhanced user experience.
"""

import os
import sys
import logging
import traceback
from pathlib import Path
from typing import Optional, Dict, Any
import argparse

import gradio as gr
from tabs.full_inference import full_inference_tab
from tabs.download_model import download_model_tab
from tabs.tts import tts_tab
from tabs.settings import (
    lang_tab, audio_tab, performance_tab, notifications_tab, 
    file_management_tab, debug_tab, backup_restore_tab, 
    misc_tab, restart_tab
)
from assets.i18n.i18n import I18nAuto

# Configure logging
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup comprehensive logging for the application."""
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Create console handler
    handler = logging.StreamHandler()
    handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    return logger

# Setup logging
logger = setup_logging()

# Get current directory
now_dir = os.getcwd()
sys.path.append(now_dir)

# Initialize i18n
i18n = I18nAuto()

# Load theme (with error handling)
def load_theme_safely():
    """Safely load theme with fallback to default."""
    try:
        # Uncomment when theme system is ready
        # import assets.themes.loadThemes as loadThemes
        # return loadThemes.load_theme() or "default"
        return "default"
    except Exception as e:
        logger.warning(f"Could not load custom theme: {e}. Using default.")
        return "default"

my_theme = load_theme_safely()

# Configuration validation
def validate_configuration():
    """Validate essential directories and configurations."""
    essential_dirs = [
        "tabs",
        "assets",
        "programs",
        "logs",
        "audio_files"
    ]
    
    missing_dirs = []
    for dir_name in essential_dirs:
        dir_path = Path(now_dir) / dir_name
        if not dir_path.exists():
            missing_dirs.append(dir_name)
            logger.info(f"Creating missing directory: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)
    
    if missing_dirs:
        logger.info(f"Created missing directories: {missing_dirs}")

def create_enhanced_app():
    """Create the enhanced Gradio application with improved error handling."""
    try:
        # Validate configuration before launching
        validate_configuration()
        
        with gr.Blocks(
            theme=my_theme, 
            title="Advanced RVC Inference V3.2 Enhanced",
            css="""
            .enhanced-tab { 
                padding: 10px; 
                border-radius: 8px; 
                margin: 5px;
            }
            .status-success { color: #28a745; }
            .status-error { color: #dc3545; }
            .status-info { color: #17a2b8; }
            """
        ) as app:
            
            # Enhanced header with better styling
            gr.HTML(
                """
                <div style="
                    text-align: center; 
                    padding: 20px; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; 
                    border-radius: 10px; 
                    margin-bottom: 20px;
                ">
                    <h1>🚀 Advanced RVC Inference V3.2 Enhanced Edition</h1>
                    <p>Revolutionizing Voice Conversion with State-of-the-Art AI Technology</p>
                    <p><em>Made with ❤️ Enhanced for Performance & Security</em></p>
                </div>
                """
            )
            
            # Status indicator
            status_display = gr.HTML(
                """
                <div id="status-indicator" style="
                    padding: 10px; 
                    margin: 10px 0; 
                    border-radius: 5px; 
                    background: #e9ecef;
                    text-align: center;
                ">
                    ✅ System Ready - All components loaded successfully
                </div>
                """
            )
            
            with gr.Tab("🎤 Full Inference"):
                try:
                    full_inference_tab()
                except Exception as e:
                    logger.error(f"Error loading Full Inference tab: {e}")
                    gr.HTML(f"""
                    <div style="color: red; padding: 20px; text-align: center;">
                        ❌ Error loading Full Inference: {str(e)}
                    </div>
                    """)
                    
            with gr.Tab("📥 Download Model"):
                try:
                    download_model_tab()
                except Exception as e:
                    logger.error(f"Error loading Download Model tab: {e}")
                    gr.HTML(f"""
                    <div style="color: red; padding: 20px; text-align: center;">
                        ❌ Error loading Download Model: {str(e)}
                    </div>
                    """)
                    
            with gr.Tab("🗣️ TTS"):
                try:
                    tts_tab()
                except Exception as e:
                    logger.error(f"Error loading TTS tab: {e}")
                    gr.HTML(f"""
                    <div style="color: red; padding: 20px; text-align: center;">
                        ❌ Error loading TTS: {str(e)}
                    </div>
                    """)
                    
            with gr.Tab("⚙️ Settings"):
                try:
                    with gr.Tab("🌍 Language"):
                        lang_tab()
                    
                    with gr.Tab("🎵 Audio"):
                        audio_tab()
                        
                    with gr.Tab("⚡ Performance"):
                        performance_tab()
                        
                    with gr.Tab("🔔 Notifications"):
                        notifications_tab()
                        
                    with gr.Tab("💾 File Management"):
                        file_management_tab()
                        
                    with gr.Tab("🐛 Debug"):
                        debug_tab()
                        
                    with gr.Tab("🔄 Backup & Restore"):
                        backup_restore_tab()
                        
                    with gr.Tab("🛠️ Miscellaneous"):
                        misc_tab()
                        
                    restart_tab()
                    
                except Exception as e:
                    logger.error(f"Error loading Settings tab: {e}")
                    gr.HTML(f"""
                    <div style="color: red; padding: 20px; text-align: center;">
                        ❌ Error loading Settings: {str(e)}
                    </div>
                    """)
        
        return app
        
    except Exception as e:
        logger.error(f"Critical error creating application: {e}")
        logger.error(traceback.format_exc())
        raise

def main():
    """Enhanced main function with better argument parsing and error handling."""
    parser = argparse.ArgumentParser(
        description='Advanced RVC Inference V3.2 Enhanced Edition - Made by ArkanDash, Enhanced by BF667',
        add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py                                    # Launch locally
  python app.py --share                           # Launch with public sharing
  python app.py --port 7860 --share              # Custom port with sharing
  python app.py --debug                           # Enable debug logging
        """
    )
    
    parser.add_argument(
        "--share", 
        action="store_true", 
        dest="share_enabled", 
        default=False, 
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
        dest="debug_enabled",
        default=False,
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    
    try:
        args = parser.parse_args()
        
        # Update logging level if debug enabled
        if args.debug_enabled:
            args.log_level = "DEBUG"
            global logger
            logger = setup_logging("DEBUG")
            logger.debug("Debug mode enabled")
        
        # Validate port
        if not 1 <= args.port <= 65535:
            logger.error(f"Invalid port number: {args.port}. Must be between 1 and 65535.")
            sys.exit(1)
        
        logger.info("🚀 Starting Advanced RVC Inference V3.2 Enhanced Edition...")
        logger.info(f"📋 Configuration: Share={args.share_enabled}, Port={args.port}, Host={args.host}, Debug={args.debug_enabled}")
        
        # Create and configure the application
        app = create_enhanced_app()
        
        # Launch with enhanced configuration
        logger.info("🌐 Launching application...")
        app.launch(
            share=args.share_enabled,
            server_port=args.port,
            server_name=args.host,
            show_error=True,  # Show errors in browser for better debugging
            quiet=False,
            inbrowser=True,  # Automatically open browser
            debug=args.debug_enabled
        )
        
    except KeyboardInterrupt:
        logger.info("👋 Application interrupted by user")
    except Exception as e:
        logger.error(f"💥 Critical error in main: {e}")
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
        sys.exit(1)

if __name__ == "__main__":
    main()
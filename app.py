"""
Enhanced Advanced RVC Inference Application V3.2
Improved with Vietnamese-RVC architecture and performance optimizations.
Based on improvements from https://github.com/PhamHuynhAnh16/Vietnamese-RVC
"""

import os
import sys
import json
import logging
import traceback
import warnings
import ssl
from pathlib import Path
from typing import Optional, Dict, Any
import argparse

import gradio as gr
from tabs.full_inference import full_inference_tab
from tabs.download_model import download_model_tab
from tabs.tts import tts_tab
from tabs.training_tab import training_tab
from tabs.settings import (
    lang_tab, audio_tab, performance_tab, notifications_tab, 
    file_management_tab, debug_tab, backup_restore_tab, 
    misc_tab, restart_tab
)
from assets.i18n.i18n import I18nAuto

# Enhanced SSL handling (from Vietnamese RVC)
ssl._create_default_https_context = ssl._create_unverified_context

# Suppress warnings (from Vietnamese RVC)
warnings.filterwarnings("ignore")
for logger_name in ["httpx", "gradio", "uvicorn", "httpcore", "urllib3", "faiss"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# Enhanced configuration system (from Vietnamese RVC)
def load_configuration():
    """Load application configuration from JSON file."""
    config_path = Path("config_enhanced.json")
    default_config = {
        "application": {
            "title": "Advanced RVC Inference V3.2 Enhanced",
            "version": "3.2.1",
            "description": "Enhanced Voice Conversion with Performance Optimizations"
        },
        "server": {
            "host": "0.0.0.0",
            "port": 7860,
            "share_mode": False,
            "debug_mode": False,
            "log_level": "INFO"
        },
        "language": {
            "default": "en-US",
            "supported": ["en-US", "de-DE", "es-ES", "fr-FR"]
        },
        "theme": {
            "default": "gradio/default",
            "dark_mode": True
        }
    }
    
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load config: {e}. Using defaults.")
            return default_config
    else:
        logging.info("Config file not found. Using default configuration.")
        return default_config

# Load configuration
CONFIG = load_configuration()

# Enhanced logging system (improved from Vietnamese RVC)
def setup_logging(log_level: str = None) -> logging.Logger:
    """Setup comprehensive logging for the application."""
    if log_level is None:
        log_level = CONFIG.get("server", {}).get("log_level", "INFO")
    
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # File handler (if enabled)
    if CONFIG.get("logging", {}).get("file_logging", True):
        file_handler = logging.FileHandler("rvc_inference.log")
        file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        logger.addHandler(file_handler)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
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
    """Create the enhanced Gradio application with Vietnamese RVC-inspired improvements."""
    try:
        # Validate configuration before launching
        validate_configuration()
        
        # Get configuration values
        app_title = CONFIG.get("application", {}).get("title", "Advanced RVC Inference")
        app_version = CONFIG.get("application", {}).get("version", "3.2")
        dark_mode = CONFIG.get("theme", {}).get("dark_mode", True)
        font_url = "https://fonts.googleapis.com/css2?family=Roboto&display=swap"
        
        # Enhanced CSS with Vietnamese RVC-inspired styling
        enhanced_css = f"""
        @import url('{font_url}');
        * {{font-family: 'Roboto', sans-serif !important;}}
        body, html {{font-family: 'Roboto', sans-serif !important;}}
        .enhanced-tab {{ 
            padding: 15px; 
            border-radius: 10px; 
            margin: 8px;
            border: 1px solid #e0e0e0;
        }}
        .enhanced-header {{
            text-align: center; 
            padding: 25px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; 
            border-radius: 15px; 
            margin-bottom: 25px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .status-indicator {{
            padding: 12px; 
            margin: 15px 0; 
            border-radius: 8px; 
            text-align: center;
            font-weight: bold;
        }}
        .status-success {{ background-color: #d4edda; color: #155724; }}
        .status-error {{ background-color: #f8d7da; color: #721c24; }}
        .status-info {{ background-color: #cce7ff; color: #004085; }}
        .config-panel {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }}
        """
        
        with gr.Blocks(
            theme=my_theme, 
            title=app_title,
            css=enhanced_css
        ) as app:
            
            # Enhanced header with configuration-based styling
            gr.HTML(
                f"""
                <div class="enhanced-header">
                    <h1>🚀 {app_title}</h1>
                    <p>Enhanced with Vietnamese-RVC Architecture & Performance Optimizations</p>
                    <p><strong>Version:</strong> {app_version} | <strong>Language:</strong> {CONFIG.get('language', {}).get('default', 'en-US')}</p>
                    <p><em>✨ Revolutionary Voice Conversion with State-of-the-Art AI Technology ✨</em></p>
                    <p><small>Based on improvements from Vietnamese-RVC by PhamHuynhAnh16</small></p>
                </div>
                """
            )
            
            # Enhanced status indicator with system information
            status_html = f"""
            <div class="config-panel">
                <h3>🔧 System Status</h3>
                <div class="status-indicator status-success">
                    ✅ System Ready - All components loaded successfully
                </div>
                <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
                    <div><strong>🖥️ Platform:</strong> {os.name}</div>
                    <div><strong>🐍 Python:</strong> {sys.version.split()[0]}</div>
                    <div><strong>📁 Working Dir:</strong> {now_dir}</div>
                    <div><strong>⚙️ Config:</strong> Loaded</div>
                </div>
            </div>
            """
            
            status_display = gr.HTML(status_html)
            
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
                    
            with gr.Tab("🎓 Training"):
                try:
                    training_tab.create_training_interface()
                except Exception as e:
                    logger.error(f"Error loading Training tab: {e}")
                    gr.HTML(f"""
                    <div style="color: red; padding: 20px; text-align: center;">
                        ❌ Error loading Training: {str(e)}
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
    """Enhanced main function with Vietnamese RVC-inspired configuration."""
    # Get default values from configuration
    config_server = CONFIG.get("server", {})
    default_host = config_server.get("host", "0.0.0.0")
    default_port = config_server.get("port", 7860)
    default_share = config_server.get("share_mode", False)
    default_debug = config_server.get("debug_mode", False)
    default_log_level = config_server.get("log_level", "INFO")
    
    parser = argparse.ArgumentParser(
        description=f"🚀 {CONFIG.get('application', {}).get('title', 'Advanced RVC Inference')} V{CONFIG.get('application', {}).get('version', '3.2')}",
        add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Enhanced Configuration System (from Vietnamese RVC):
  • Configuration file: config_enhanced.json
  • Default language: {CONFIG.get('language', {}).get('default', 'en-US')}
  • Theme: {CONFIG.get('theme', {}).get('default', 'gradio/default')}

Examples:
  python app.py                                    # Run with config defaults
  python app.py --share --port 8080               # Public sharing on custom port
  python app.py --debug --log-level DEBUG         # Enhanced debug mode
  python app.py --host 127.0.0.1                  # Localhost only
        """
    )
    
    parser.add_argument(
        "--share", 
        action="store_true", 
        dest="share_enabled", 
        default=default_share, 
        help=f"Enable public sharing of the application (default: {default_share})"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=default_port,
        help=f"Port to run the application on (default: {default_port})"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default=default_host,
        help=f"Host to bind to (default: {default_host})"
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
        default=default_log_level,
        help=f"Set logging level (default: {default_log_level})"
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
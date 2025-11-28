"""
Advanced RVC Inference V4.0.0 - CLI Enhanced Version
Optimized for command line execution
"""

import os
import sys
import time
import socket
import argparse
from typing import Optional, Tuple

# Enhanced path handling
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import Gradio after path setup
import gradio as gr

# Configuration Constants
DEFAULT_PORT = 7755
MAX_PORT_ATTEMPTS = 20

def detect_environment() -> Tuple[bool, bool]:
    """Comprehensive environment detection with fallbacks"""
    colab = False
    kaggle = False
    
    try:
        import google.colab
        colab = True
    except ImportError:
        pass
    
    try:
        if 'COLAB_RELEASE_TAG' in os.environ:
            colab = True
    except:
        pass
        
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None:
        kaggle = True
    
    try:
        from IPython import get_ipython
        if 'google.colab' in str(get_ipython()):
            colab = True
    except:
        pass
    
    return colab, kaggle

# Detect environment early
COLAB_ENVIRONMENT, KAGGLE_ENVIRONMENT = detect_environment()

# Add current directory to path
now_dir = os.getcwd()
sys.path.append(now_dir)

# Optional imports with better error handling
def safe_import(module_path: str, item_name: str) -> tuple[bool, Optional[object]]:
    """Safely import optional modules with detailed logging"""
    try:
        module = __import__(module_path, fromlist=[item_name])
        component = getattr(module, item_name)
        return True, component
    except ImportError as e:
        return False, None
    except Exception as e:
        return False, None

# Import core components with error handling
try:
    import assets.themes.loadThemes as loadThemes
    rvc_theme = loadThemes.load_json() or gr.themes.Default()
except Exception as e:
    rvc_theme = gr.themes.Default()

# Internationalization with fallback
try:
    from assets.i18n.i18n import I18nAuto
    i18n = I18nAuto()
except ImportError as e:
    class I18nAuto:
        def __call__(self, key: str) -> str:
            return key
        def __getattr__(self, key: str) -> str:
            return key
    i18n = I18nAuto()

# Import tabs with progress indication
core_tabs = {}
tab_modules = [
    ('training', 'training_tab'),
    ('settings', 'select_themes_tab'), 
    ('real_time', 'real_time_inference_tab'),
    ('model_manager', 'model_manager_tab'),
    ('full_inference', 'full_inference_tab'),
    ('enhancement', 'enhancement_tab'),
    ('download_music', 'download_music_tab'),
    ('download_model', 'download_model_tab'),
    ('config_options', 'extra_options_tab')
]

for module_name, tab_name in tab_modules:
    available, tab_func = safe_import(f'advanced_rvc_inference.tabs.{module_name}', tab_name)
    if available:
        core_tabs[tab_name] = tab_func

# Optional advanced features
optional_features = {
    'tts_tab': ('advanced_rvc_inference.tabs.tts.tts', 'tts_tab'),
    'voice_blender_tab': ('advanced_rvc_inference.tabs.voice_blender.voice_blender', 'voice_blender_tab'),
    'plugins_tab': ('advanced_rvc_inference.tabs.plugins.plugins', 'plugins_tab'),
    'extra_tab': ('advanced_rvc_inference.tabs.extra.extra', 'extra_tab'),
    'f0_extractor_tab': ('advanced_rvc_inference.tabs.f0_extractor', 'f0_extractor_tab'),
    'embedders_tab': ('advanced_rvc_inference.tabs.embedders', 'embedders_tab')
}

optional_tabs = {}
for feature_name, (module_path, tab_name) in optional_features.items():
    available, tab_func = safe_import(module_path, tab_name)
    if available:
        optional_tabs[feature_name] = tab_func

def find_available_port(start_port: int, max_attempts: int = MAX_PORT_ATTEMPTS) -> int:
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    return start_port

def create_app() -> gr.Blocks:
    """Create and configure the main Gradio application"""
    app = gr.Blocks(title="Advanced RVC Inference v4.0 - CLI Enhanced")
    
    with app:
        # Header Section
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("# 🎤 Advanced RVC Inference v4.0")
                gr.Markdown("### *CLI Enhanced Version*")
            with gr.Column(scale=2):
                env_status = "🏢 Local" 
                if COLAB_ENVIRONMENT:
                    env_status = "☁️ Google Colab"
                elif KAGGLE_ENVIRONMENT:
                    env_status = "📊 Kaggle"
                    
                gr.Markdown(f"""
                **Environment**: {env_status}  
                **Version**: 4.0.0 CLI Enhanced  
                **Gradio**: {gr.__version__}
                """)

        # Status indicator
        with gr.Row():
            with gr.Column():
                status_text = gr.Markdown(
                    f"✅ **System Ready** | Loaded {len(core_tabs)} core tabs + {len(optional_tabs)} optional features"
                )

        # Main Tabs
        with gr.Tabs():
            # Inference Section
            with gr.Tab("🎵 Inference"):
                with gr.Tabs():
                    with gr.Tab("🎛️ Full Inference"):
                        if 'full_inference_tab' in core_tabs:
                            core_tabs['full_inference_tab']()
                        else:
                            gr.Markdown("⚠️ Full Inference tab not available")
                    
                    with gr.Tab("🎤 Real-Time Voice"):
                        if 'real_time_inference_tab' in core_tabs:
                            core_tabs['real_time_inference_tab']()
                        else:
                            gr.Markdown("⚠️ Real-Time Inference tab not available")
                    
                    if 'tts_tab' in optional_tabs:
                        with gr.Tab("📢 Text-to-Speech"):
                            optional_tabs['tts_tab']()

            # Download Section
            with gr.Tab("📥 Downloader"):
                with gr.Tabs():
                    with gr.Tab("🎵 Music Downloader"):
                        if 'download_music_tab' in core_tabs:
                            core_tabs['download_music_tab']()
                        else:
                            gr.Markdown("⚠️ Music Downloader tab not available")
                    
                    with gr.Tab("📦 Model Downloader"):
                        if 'download_model_tab' in core_tabs:
                            core_tabs['download_model_tab']()
                        else:
                            gr.Markdown("⚠️ Model Downloader tab not available")

            # Training Section
            with gr.Tab("🎓 Training"):
                if 'training_tab' in core_tabs:
                    core_tabs['training_tab']()
                else:
                    gr.Markdown("⚠️ Training tab not available")

            # Audio Tools Section
            with gr.Tab("🔧 Audio Tools"):
                with gr.Tabs():
                    with gr.Tab("✨ Enhancement"):
                        if 'enhancement_tab' in core_tabs:
                            core_tabs['enhancement_tab']()
                        else:
                            gr.Markdown("⚠️ Enhancement tab not available")
                    
                    if 'f0_extractor_tab' in optional_tabs:
                        with gr.Tab("🎵 F0 Extractor"):
                            optional_tabs['f0_extractor_tab']()

            # Advanced Features Section
            if any(key in optional_tabs for key in ['voice_blender_tab', 'plugins_tab', 'extra_tab']):
                with gr.Tab("🚀 Advanced"):
                    with gr.Tabs():
                        if 'voice_blender_tab' in optional_tabs:
                            with gr.Tab("🎭 Voice Blender"):
                                optional_tabs['voice_blender_tab']()
                        
                        if 'plugins_tab' in optional_tabs:
                            with gr.Tab("🧩 Plugins"):
                                optional_tabs['plugins_tab']()
                        
                        if 'extra_tab' in optional_tabs:
                            with gr.Tab("⚡ Extra Features"):
                                optional_tabs['extra_tab']()

            # Settings Section
            with gr.Tab("⚙️ Settings"):
                if 'select_themes_tab' in core_tabs:
                    core_tabs['select_themes_tab']()
                else:
                    gr.Markdown("⚠️ Settings tab not available")
                
                # System Info
                with gr.Accordion("📊 System Information", open=False):
                    gr.Markdown(f"""
                    **Environment Details:**
                    - Platform: {sys.platform}
                    - Python: {sys.version.split()[0]}
                    - Gradio: {gr.__version__}
                    - Working Directory: {now_dir}
                    - Colab: {COLAB_ENVIRONMENT}
                    - Kaggle: {KAGGLE_ENVIRONMENT}
                    """)

    return app

def launch_app(port: int, share: bool = False, open_browser: bool = False):
    """Launch the Gradio application"""
    actual_port = find_available_port(port)
    if actual_port != port:
        print(f"⚠️  Port {port} busy, using port {actual_port}")
    
    app = create_app()
    
    # Determine if we should share based on environment
    should_share = share or COLAB_ENVIRONMENT or KAGGLE_ENVIRONMENT
    
    print("\n" + "="*60)
    print("🚀 ADVANCED RVC INFERENCE - CLI ENHANCED")
    print("="*60)
    print(f"🌍 Environment: {'Google Colab' if COLAB_ENVIRONMENT else 'Kaggle' if KAGGLE_ENVIRONMENT else 'Local'}")
    print(f"🔗 Port: {actual_port}")
    print(f"🌐 Public URL: {'Enabled' if should_share else 'Disabled'}")
    print(f"🖥️  Browser Auto-open: {'Enabled' if open_browser else 'Disabled'}")
    print(f"📊 Loaded Tabs: {len(core_tabs)} core + {len(optional_tabs)} optional")
    print("-"*60)
    print("💡 Usage Tips:")
    print("   • Keep this terminal running to maintain the connection")
    print("   • Press Ctrl+C to stop the server")
    print("   • Access the interface at http://localhost:" + str(actual_port))
    if should_share:
        print("   • Public URL will be displayed once started")
    print("="*60 + "\n")
    
    app.launch(
        share=should_share,
        inbrowser=open_browser,
        server_port=actual_port,
        show_error=True,
        theme=rvc_theme
    )

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description='Advanced RVC Inference - CLI Enhanced')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, help=f'Port to run on (default: {DEFAULT_PORT})')
    parser.add_argument('--share', action='store_true', help='Create a public shareable link')
    parser.add_argument('--open', action='store_true', help='Open browser automatically')
    parser.add_argument('--no-browser', action='store_true', help='Disable browser auto-open')
    parser.add_argument('--colab', action='store_true', help='Force Colab environment mode')
    
    args = parser.parse_args()
    
    # Override environment detection if specified
    if args.colab:
        global COLAB_ENVIRONMENT
        COLAB_ENVIRONMENT = True
    
    # Determine browser behavior
    open_browser = args.open
    if args.no_browser:
        open_browser = False
    elif (COLAB_ENVIRONMENT or KAGGLE_ENVIRONMENT):
        open_browser = False
    
    launch_app(args.port, args.share, open_browser)

if __name__ == "__main__":
    main()

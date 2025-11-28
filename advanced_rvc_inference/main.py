"""
Advanced RVC Inference V4.0.0 - Enhanced for Colab/Local Deployment
Optimized for Gradio 6.x with Better Performance and Reliability
"""

import os
import sys
import time
import socket
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
MAX_PORT_ATTEMPTS = 20  # Increased for better port finding
COLAB_ENVIRONMENT = False
KAGGLE_ENVIRONMENT = False

# Enhanced Environment Detection
def detect_environment() -> Tuple[bool, bool]:
    """Comprehensive environment detection with fallbacks"""
    colab = False
    kaggle = False
    
    try:
        # Method 1: Check for Colab
        import google.colab
        colab = True
        print("✅ Google Colab environment detected")
    except ImportError:
        pass
    
    try:
        # Method 2: Check environment variables
        if 'COLAB_RELEASE_TAG' in os.environ:
            colab = True
            print("✅ Google Colab environment detected (env var)")
    except:
        pass
        
    # Method 3: Check for Kaggle
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None:
        kaggle = True
        print("✅ Kaggle environment detected")
    
    # Method 4: Check runtime type (alternative Colab detection)
    try:
        from IPython import get_ipython
        if 'google.colab' in str(get_ipython()):
            colab = True
            print("✅ Google Colab environment detected (IPython)")
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
        print(f"✅ Successfully imported {item_name} from {module_path}")
        return True, component
    except ImportError as e:
        print(f"⚠️  Optional feature not available: {module_path}.{item_name} - {str(e)}")
        return False, None
    except Exception as e:
        print(f"❌ Error importing {module_path}.{item_name}: {str(e)}")
        return False, None

# Import core components with error handling
try:
    import assets.themes.loadThemes as loadThemes
    rvc_theme = loadThemes.load_json() or gr.themes.Default()
    print("✅ Theme system loaded successfully")
except Exception as e:
    print(f"⚠️  Using default theme: {str(e)}")
    rvc_theme = gr.themes.Default()

# Internationalization with fallback
try:
    from assets.i18n.i18n import I18nAuto
    i18n = I18nAuto()
    print("✅ Internationalization loaded")
except ImportError as e:
    print(f"⚠️  Using fallback i18n: {str(e)}")
    class I18nAuto:
        def __call__(self, key: str) -> str:
            return key
        def __getattr__(self, key: str) -> str:
            return key
    i18n = I18nAuto()

# Import tabs with progress indication
print("🔄 Loading application tabs...")

# Core tabs with error handling
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
        print(f"✅ Loaded {tab_name}")

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

print(f"✅ Loaded {len(optional_tabs)} optional features")

# Enhanced CSS for better UI
ENHANCED_CSS = """
/* Modern UI Variables with Dark Mode Support */
:root {
    --primary-color: #4f46e5;
    --secondary-color: #7c3aed;
    --accent-color: #06b6d4;
    --background-color: #f8fafc;
    --card-background: #ffffff;
    --border-color: #e2e8f0;
    --text-primary: #1e293b;
    --text-secondary: #64748b;
    --success-color: #10b981;
    --warning-color: #f59e0b;
    --error-color: #ef4444;
}

/* Dark mode variables */
.dark {
    --background-color: #0f172a;
    --card-background: #1e293b;
    --border-color: #334155;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
}

/* Global styles */
.gradio-container {
    background: var(--background-color) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Header styling */
.header-row {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)) !important;
    border-radius: 12px !important;
    margin: 1rem 0 !important;
    padding: 1.5rem !important;
    color: white !important;
}

.header-row .markdown {
    color: white !important;
}

/* Tab improvements */
.tab-nav {
    background: var(--card-background) !important;
    border-radius: 12px !important;
    margin: 0.5rem 0 !important;
    padding: 0.5rem !important;
}

.tabitem {
    border-radius: 12px !important;
    margin: 0.75rem !important;
    padding: 1.5rem !important;
    background: var(--card-background) !important;
    border: 1px solid var(--border-color) !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
}

/* Button enhancements */
.gradio-button.primary {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.gradio-button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 20px rgba(79, 70, 229, 0.3) !important;
}

/* Card components */
.card {
    background: var(--card-background) !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    border: 1px solid var(--border-color) !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    margin: 1rem 0 !important;
}

/* Status indicators */
.status-success { color: var(--success-color) !important; font-weight: 600; }
.status-warning { color: var(--warning-color) !important; font-weight: 600; }
.status-error { color: var(--error-color) !important; font-weight: 600; }

/* Hide footer and improve layout */
footer { display: none !important; }
.gradio-container .main { min-height: calc(100vh - 2rem) !important; }

/* Responsive design */
@media (max-width: 768px) {
    .gradio-container { 
        padding: 0.5rem !important; 
        max-width: 100% !important;
    }
    .tabitem { 
        margin: 0.25rem !important; 
        padding: 1rem !important; 
    }
    .card { 
        padding: 1rem !important; 
        margin: 0.5rem 0 !important; 
    }
    .header-row {
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
    }
}

/* Loading animations */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}

.loading-pulse {
    animation: pulse 2s infinite;
}

/* Colab-specific optimizations */
.colab-optimized .gradio-container {
    max-width: 100% !important;
    padding: 0.5rem !important;
}
"""

def find_available_port(start_port: int, max_attempts: int = MAX_PORT_ATTEMPTS) -> int:
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port - max_attempts, -1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    # If no port found, return the original
    return start_port

def create_app() -> gr.Blocks:
    """Create and configure the main Gradio application with enhanced features"""
    
    # Add colab-optimized class if in Colab
    additional_css = ENHANCED_CSS
    if COLAB_ENVIRONMENT:
        additional_css += "\n.gradio-container { margin: 0 auto !important; }"
    
    app = gr.Blocks(
        title="Advanced RVC Inference v4.0 - Enhanced",
        fill_width=True,
        analytics_enabled=False,
        theme=rvc_theme,
        css=additional_css
    )
    
    with app:
        # Enhanced Header Section
        with gr.Row(elem_classes=["header-row", "colab-optimized"] if COLAB_ENVIRONMENT else ["header-row"]):
            with gr.Column(scale=1, min_width=200):
                gr.Markdown("# 🎤 Advanced RVC Inference v4.0")
                gr.Markdown("### *Enhanced for Colab & Local Deployment*")
            with gr.Column(scale=2, min_width=300):
                env_status = "🏢 Local" 
                if COLAB_ENVIRONMENT:
                    env_status = "☁️ Google Colab"
                elif KAGGLE_ENVIRONMENT:
                    env_status = "📊 Kaggle"
                    
                gr.Markdown(f"""
                **Environment**: {env_status}  
                **Version**: 4.0.0 Enhanced  
                **Gradio**: {gr.__version__}  
                *Optimized for performance and reliability*
                """)

        # Enhanced status indicator
        with gr.Row():
            with gr.Column():
                status_text = gr.Markdown(
                    f"✅ **System Ready** | Loaded {len(core_tabs)} core tabs + {len(optional_tabs)} optional features",
                    elem_classes="status-success"
                )

        # Main Tabs with enhanced organization
        with gr.Tabs(elem_classes="tab-nav"):
            # Inference Section
            with gr.Tab("🎵 Inference", elem_classes="tabitem"):
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
            with gr.Tab("📥 Downloader", elem_classes="tabitem"):
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
            with gr.Tab("🎓 Training", elem_classes="tabitem"):
                if 'training_tab' in core_tabs:
                    core_tabs['training_tab']()
                else:
                    gr.Markdown("⚠️ Training tab not available")

            # Audio Tools Section
            with gr.Tab("🔧 Audio Tools", elem_classes="tabitem"):
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
                with gr.Tab("🚀 Advanced", elem_classes="tabitem"):
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
            with gr.Tab("⚙️ Settings", elem_classes="tabitem"):
                if 'select_themes_tab' in core_tabs:
                    core_tabs['select_themes_tab']()
                else:
                    gr.Markdown("⚠️ Settings tab not available")
                
                # System Info Card
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

def get_server_urls(port: int, share: bool = False) -> Tuple[str, Optional[str]]:
    """Get local and share URLs for the server"""
    local_url = f"http://localhost:{port}"
    share_url = None
    
    if share:
        try:
            # Try to get the public URL from Gradio
            from gradio import networking
            share_url = networking.get_share_url(port)
        except Exception as e:
            print(f"⚠️  Could not get share URL: {e}")
    
    return local_url, share_url

def display_urls(port: int, share: bool = False):
    """Display URLs in a user-friendly format"""
    local_url, share_url = get_server_urls(port, share)
    
    print("\n" + "="*70)
    print("🎉 ADVANCED RVC INFERENCE - SUCCESSFULLY LAUNCHED!")
    print("="*70)
    
    if share_url:
        print(f"🌐 PUBLIC URL (Share with anyone):")
        print(f"   🔗 {share_url}")
        print()
    
    print(f"💻 LOCAL URL (Your machine only):")
    print(f"   🔗 {local_url}")
    print()
    
    if COLAB_ENVIRONMENT and share_url:
        print(f"📱 COLAB MOBILE URL:")
        print(f"   🔗 {share_url}")
        print()
    
    print("⚡ ENHANCED FEATURES:")
    print("   ✅ Optimized for Colab performance")
    print("   ✅ Better error handling and recovery") 
    print("   ✅ Enhanced UI with dark mode support")
    print("   ✅ Improved port conflict resolution")
    print()
    print("⚠️  IMPORTANT: Keep this cell running to maintain the connection!")
    print("="*70)
    
    # Create clickable links for Colab
    if COLAB_ENVIRONMENT:
        try:
            from IPython.display import HTML, display
            html_content = """
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 25px; border-radius: 15px; margin: 20px 0; 
                        color: white; font-family: Arial, sans-serif;">
                <h2 style="margin: 0 0 15px 0; text-align: center;">🎤 Advanced RVC Inference - READY!</h2>
            """
            
            if share_url:
                html_content += f"""
                <div style="text-align: center; margin: 15px 0;">
                    <a href="{share_url}" target="_blank" 
                       style="background: rgba(255,255,255,0.2); color: white; 
                              padding: 12px 24px; border-radius: 8px; text-decoration: none;
                              font-weight: bold; font-size: 16px; display: inline-block;
                              border: 2px solid white; margin: 5px;">
                       🌐 Public URL (Share This)
                    </a>
                </div>
                """
            
            html_content += f"""
            <div style="text-align: center; margin: 15px 0;">
                <a href="{local_url}" target="_blank"
                   style="background: rgba(255,255,255,0.1); color: white;
                          padding: 10px 20px; border-radius: 6px; text-decoration: none;
                          font-weight: normal; font-size: 14px; display: inline-block;
                          border: 1px solid rgba(255,255,255,0.3); margin: 5px;">
                   💻 Local URL
                </a>
            </div>
            """
                
            html_content += """
                <p style="text-align: center; margin: 15px 0 0 0; font-size: 14px; opacity: 0.9;">
                    ⚠️ Keep this cell running to maintain the connection!
                </p>
            </div>
            """
            display(HTML(html_content))
        except ImportError:
            pass  # Not in IPython environment

def launch_app(port: int):
    """Enhanced app launcher with better error handling"""
    
    # Find available port
    actual_port = find_available_port(port)
    if actual_port != port:
        print(f"⚠️  Port {port} busy, using port {actual_port}")
    
    # Configure launch parameters
    share = any(arg in sys.argv for arg in ['--share', '--listen']) or COLAB_ENVIRONMENT or KAGGLE_ENVIRONMENT
    inbrowser = '--open' in sys.argv and not (COLAB_ENVIRONMENT or KAGGLE_ENVIRONMENT)
    server_name = "0.0.0.0" if any(arg in sys.argv for arg in ['--listen', '--share']) or COLAB_ENVIRONMENT or KAGGLE_ENVIRONMENT else "127.0.0.1"
    
    print(f"🚀 Launching Advanced RVC Inference...")
    print(f"   📍 Port: {actual_port}")
    print(f"   🌐 Server: {server_name}")
    print(f"   🔗 Sharing: {share}")
    print(f"   🖥️  Browser: {inbrowser}")
    
    # Create and launch app
    app = create_app()
    
    try:
        # Launch the app
        app.launch(
            share=share,
            inbrowser=inbrowser,
            server_port=actual_port,
            server_name=server_name,
            show_error=True,
            show_api=False,
            quiet=False,
            prevent_thread_lock=True  # Important for Colab
        )
        
        # Display URLs immediately after launch
        display_urls(actual_port, share)
        
        return actual_port
        
    except Exception as e:
        print(f"❌ Failed to launch app: {str(e)}")
        raise

def main():
    """Enhanced main function with better error handling and recovery"""
    
    # Parse command line arguments
    port = DEFAULT_PORT
    if "--port" in sys.argv:
        try:
            port_index = sys.argv.index("--port") + 1
            if port_index < len(sys.argv):
                port = int(sys.argv[port_index])
        except (ValueError, IndexError):
            print("⚠️  Invalid port specified, using default")
    
    print("="*70)
    print("🎤 ADVANCED RVC INFERENCE v4.0 - ENHANCED DEPLOYMENT")
    print("="*70)
    print(f"📍 Initial port: {port}")
    print(f"🏢 Environment: {'Google Colab' if COLAB_ENVIRONMENT else 'Kaggle' if KAGGLE_ENVIRONMENT else 'Local'}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"🎨 Gradio: {gr.__version__}")
    print("="*70)
    
    try:
        actual_port = launch_app(port)
        
        # Environment-specific handling
        if COLAB_ENVIRONMENT or KAGGLE_ENVIRONMENT:
            print("🔄 Cloud environment detected - keeping server alive...")
            try:
                # Keep the server running
                import time
                while True:
                    time.sleep(10)
                    # Heartbeat indicator
                    print("💓 Server heartbeat...", end='\r')
            except KeyboardInterrupt:
                print("\n👋 Shutting down server...")
        else:
            # Local environment - wait for interrupt
            print("🌐 Server running. Press Ctrl+C to stop.")
            try:
                # Block the main thread to keep server running
                import time
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n👋 Shutting down server...")
                
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

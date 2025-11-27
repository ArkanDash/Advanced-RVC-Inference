"""
Advanced RVC Inference V4.0.0 - Main Application Entry Point

Ultimate Voice Conversion Platform with Advanced Performance Optimization
"""

import os
import sys
import time
from typing import Optional

# Add the parent directory to Python path to access assets module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import gradio as gr
import assets.themes.loadThemes as loadThemes

# Core tab imports
from advanced_rvc_inference.tabs.training import training_tab
from advanced_rvc_inference.tabs.settings import select_themes_tab
from advanced_rvc_inference.tabs.real_time import real_time_inference_tab
from advanced_rvc_inference.tabs.model_manager import model_manager_tab
from advanced_rvc_inference.tabs.full_inference import full_inference_tab
from advanced_rvc_inference.tabs.enhancement import enhancement_tab
from advanced_rvc_inference.tabs.download_music import download_music_tab
from advanced_rvc_inference.tabs.download_model import download_model_tab
from advanced_rvc_inference.tabs.config_options import extra_options_tab


# Environment Detection
def is_colab() -> bool:
    """Check if running in Google Colab environment"""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def is_kaggle() -> bool:
    """Check if running in Kaggle environment"""
    return os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None


# Configuration Constants
DEFAULT_PORT = 7755
MAX_PORT_ATTEMPTS = 10
COLAB_ENVIRONMENT = is_colab()
KAGGLE_ENVIRONMENT = is_kaggle()

# Add current directory to path
now_dir = os.getcwd()
sys.path.append(now_dir)


# Optional Feature Imports
def safe_import(module_path: str, item_name: str) -> tuple[bool, Optional[object]]:
    """Safely import optional modules and return availability status"""
    try:
        module = __import__(module_path, fromlist=[item_name])
        return True, getattr(module, item_name)
    except ImportError:
        return False, None


# Advanced feature availability
TTS_AVAILABLE, tts_tab = safe_import('advanced_rvc_inference.tabs.tts.tts', 'tts_tab')
VOICE_BLENDER_AVAILABLE, voice_blender_tab = safe_import('advanced_rvc_inference.tabs.voice_blender.voice_blender', 'voice_blender_tab')
PLUGINS_AVAILABLE, plugins_tab = safe_import('advanced_rvc_inference.tabs.plugins.plugins', 'plugins_tab')
EXTRA_AVAILABLE, extra_tab = safe_import('advanced_rvc_inference.tabs.extra.extra', 'extra_tab')
F0_EXTRACTOR_AVAILABLE, f0_extractor_tab = safe_import('advanced_rvc_inference.tabs.f0_extractor', 'f0_extractor_tab')
EMBEDDER_AVAILABLE, embedders_tab = safe_import('advanced_rvc_inference.tabs.embedders', 'embedders_tab')

# Internationalization
try:
    from assets.i18n.i18n import I18nAuto
    i18n = I18nAuto()
except ImportError:
    class I18nAuto:
        def __call__(self, key: str) -> str:
            return key
    i18n = I18nAuto()

# Theme and Styling
rvc_theme = loadThemes.load_json() or gr.themes.Default()

# Modern UI CSS
CUSTOM_CSS = """
/* Modern UI Variables */
:root {
    --primary-color: #4f46e5;
    --secondary-color: #7c3aed;
    --accent-color: #06b6d4;
    --background-color: #f8fafc;
    --card-background: #ffffff;
    --border-color: #e2e8f0;
    --text-primary: #1e293b;
    --text-secondary: #64748b;
}

/* Dark mode support */
.dark .dark\\:bg-gray-900 { background-color: #0f172a !important; }

/* Component styling */
.tabitem {
    border-radius: 0.5rem !important;
    margin: 0.5rem !important;
    padding: 1rem !important;
    background-color: var(--card-background) !important;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1) !important;
}

.gradio-button {
    border-radius: 0.5rem !important;
    padding: 0.5rem 1rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.card {
    background: var(--card-background) !important;
    border-radius: 0.5rem !important;
    padding: 1.5rem !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    border: 1px solid var(--border-color) !important;
    margin: 0.5rem 0 !important;
}

/* Hide footer */
footer { display: none !important; }

/* Responsive design */
@media (max-width: 768px) {
    .gradio-container { padding: 0.5rem !important; }
    .tabitem { margin: 0.25rem !important; padding: 0.5rem !important; }
    .card { padding: 1rem !important; margin: 0.25rem 0 !important; }
}
"""

# Gradio Application
def create_app() -> gr.Blocks:
    """Create and configure the main Gradio application"""
    
    app = gr.Blocks(
        title="Advanced RVC Inference v4.0",
        fill_width=True,
        analytics_enabled=False,
        theme=rvc_theme,
        css=CUSTOM_CSS
    )
    
    with app:
        # Header Section
        with gr.Row(elem_classes="header-row", equal_height=True):
            with gr.Column(scale=1, min_width=200):
                gr.Markdown("# 🎤 Advanced RVC Inference v4.0")
            with gr.Column(scale=2, min_width=300, elem_classes="header-info"):
                gr.Markdown(
                    "> *Kernel Advanced RVC - 2x Faster Training & Inference*  \n"
                    "> *Ultimate Voice Conversion Platform with Advanced Performance Optimization*"
                )

        # Main Tabs
        with gr.Tab("🎵 Inference"):
            with gr.Tab("Full Inference"):
                full_inference_tab()
            with gr.Tab("🎤 Real-Time"):
                real_time_inference_tab()
            if TTS_AVAILABLE:
                with gr.Tab("📢 Text-to-Speech"):
                    tts_tab()

        with gr.Tab("📥 Downloader"):
            with gr.Tab("🎵 Download Music"):
                download_music_tab()
            with gr.Tab("📦 Download Model"):
                download_model_tab()

        with gr.Tab("🎓 Training"):
            training_tab()

        with gr.Tab("🔧 Audio Tools"):
            enhancement_tab()
            if F0_EXTRACTOR_AVAILABLE:
                f0_extractor_tab()

        # Optional Advanced Features
        if VOICE_BLENDER_AVAILABLE:
            with gr.Tab("🎭 Voice Blender"):
                voice_blender_tab()

        if PLUGINS_AVAILABLE:
            with gr.Tab("🧩 Plugins"):
                plugins_tab()

        if EXTRA_AVAILABLE:
            with gr.Tab("⚡ Extra"):
                extra_tab()

        with gr.Tab("⚙️ Settings"):
            select_themes_tab()
    
    return app


# Create the application instance
app = create_app()


def launch(port):
    """Launch the Gradio interface with proper environment handling"""
    # Configure launch parameters based on environment
    share = "--share" in sys.argv or "--listen" in sys.argv or COLAB_ENVIRONMENT or KAGGLE_ENVIRONMENT
    inbrowser = "--open" in sys.argv and not (COLAB_ENVIRONMENT or KAGGLE_ENVIRONMENT)

    # Special handling for Colab/Kaggle environments
    if COLAB_ENVIRONMENT:
        print("🚀 Detected Google Colab environment")
        print("📱 Setting up public URL sharing...")
        share = True
        server_name = "0.0.0.0"
        prevent_thread_lock = True  # Don't block in Colab
    elif KAGGLE_ENVIRONMENT:
        print("🚀 Detected Kaggle environment")
        share = True
        server_name = "0.0.0.0"
        prevent_thread_lock = True
    else:
        server_name = "127.0.0.1" if not ("--listen" in sys.argv) else "0.0.0.0"
        prevent_thread_lock = False

    # Launch the app
    print(f"🌐 Starting server on {server_name}:{port}")
    demo = app.launch(
        share=share,
        inbrowser=inbrowser,
        server_port=port,
        server_name=server_name,
        show_error=True,
        prevent_thread_lock=prevent_thread_lock,
        show_api=True,
        quiet=False
    )

    # Enhanced URL display for Colab users
    if COLAB_ENVIRONMENT:
        import time
        time.sleep(2)  # Wait for Gradio to fully initialize
        
        if hasattr(demo, 'share_url') and demo.share_url:
            print(f"\n🎉 SUCCESS! Advanced RVC Inference is now running!")
            print(f"🌐 Public URL: {demo.share_url}")
            print("📋 Copy this URL to access the interface from anywhere!")
            print("⚠️  Keep this cell running to maintain the connection!")
            
            # Display clickable link in Colab
            try:
                from IPython.display import HTML, display
                display(HTML(f'''
                <div style="background: linear-gradient(90deg, #4285f4, #34a853); padding: 20px; border-radius: 10px; margin: 10px 0;">
                    <h3 style="color: white; margin: 0 0 10px 0;">🎤 Advanced RVC Inference is Ready!</h3>
                    <a href="{demo.share_url}" target="_blank" 
                       style="color: white; font-size: 18px; font-weight: bold; text-decoration: none; 
                              background: rgba(255,255,255,0.2); padding: 10px 20px; border-radius: 5px; display: inline-block;">
                        🔗 Click Here to Open Interface
                    </a>
                    <p style="color: white; margin: 10px 0 0 0; font-size: 14px;">
                        ⚠️ Keep this cell running to maintain the connection!
                    </p>
                </div>
                '''))
            except ImportError:
                pass
        else:
            print("⚠️  Gradio share URL not available yet. Retrying...")
            time.sleep(3)
            if hasattr(demo, 'share_url') and demo.share_url:
                print(f"🌐 Public URL: {demo.share_url}")

    return demo


def get_port_from_args():
    if "--port" in sys.argv:
        port_index = sys.argv.index("--port") + 1
        if port_index < len(sys.argv):
            return int(sys.argv[port_index])
    return DEFAULT_PORT


def main():
    """Main function to launch Advanced RVC Inference"""
    port = get_port_from_args()

    for _ in range(MAX_PORT_ATTEMPTS):
        try:
            print(f"🚀 Launching Advanced RVC Inference on port {port}...")
            demo = launch(port)

            # Handle different environments
            if COLAB_ENVIRONMENT or KAGGLE_ENVIRONMENT:
                print("🔄 Running in cloud environment - keeping server alive...")
                # In Colab/Kaggle, we want to keep the server running but not block the cell
                # The prevent_thread_lock=True in launch() handles this
                try:
                    # Keep the demo object alive and wait indefinitely
                    import time
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\n👋 Shutting down Advanced RVC Inference...")
                    if hasattr(demo, 'close'):
                        demo.close()
            else:
                # Local environment - block the thread normally
                if demo:
                    try:
                        print("🌐 Server is running. Press Ctrl+C to stop.")
                        demo.block_thread()
                    except KeyboardInterrupt:
                        print("\n👋 Shutting down Advanced RVC Inference...")
                    except Exception as e:
                        print(f"❌ Error during execution: {e}")

            break
        except OSError as e:
            if "Address already in use" in str(e):
                print(f"⚠️  Port {port} is busy, trying port {port - 1}...")
                port -= 1
            else:
                print(f"❌ Network error: {e}")
                break
        except Exception as error:
            print(f"❌ An error occurred launching Gradio: {error}")
            import traceback
            traceback.print_exc()
            break


if __name__ == "__main__":
    main()

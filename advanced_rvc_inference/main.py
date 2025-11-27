import sys, os
import gradio as gr

now_dir = os.getcwd()
sys.path.append(now_dir)
DEFAULT_PORT = 7755
MAX_PORT_ATTEMPTS = 10

from advanced_rvc_inference.tabs.full_inference import full_inference_tab
from advanced_rvc_inference.tabs.download_model import download_model_tab
from advanced_rvc_inference.tabs.download_music import download_music_tab
from advanced_rvc_inference.tabs.settings import select_themes_tab
from advanced_rvc_inference.tabs.training import training_tab
from advanced_rvc_inference.tabs.model_manager import model_manager_tab
from advanced_rvc_inference.tabs.enhancement import enhancement_tab
from advanced_rvc_inference.tabs.real_time import real_time_inference_tab
from advanced_rvc_inference.tabs.config_options import extra_options_tab

# Attempt to import additional advanced features from Applio and Vietnamese-RVC
try:
    from .tabs.tts.tts import tts_tab
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

try:
    from .tabs.voice_blender.voice_blender import voice_blender_tab
    VOICE_BLENDER_AVAILABLE = True
except ImportError:
    VOICE_BLENDER_AVAILABLE = False

try:
    from .tabs.plugins.plugins import plugins_tab
    PLUGINS_AVAILABLE = True
except ImportError:
    PLUGINS_AVAILABLE = False

try:
    from .tabs.extra.extra import extra_tab
    EXTRA_AVAILABLE = True
except ImportError:
    EXTRA_AVAILABLE = False

try:
    from .tabs.f0_extractor import f0_extractor_tab
    F0_EXTRACTOR_AVAILABLE = True
except ImportError:
    F0_EXTRACTOR_AVAILABLE = False

try:
    from .tabs.embedders import embedders_tab
    EMBEDDER_AVAILABLE = True
except ImportError:
    EMBEDDER_AVAILABLE = False

import assets.themes.loadThemes as loadThemes

try:
    from assets.i18n.i18n import I18nAuto
except ImportError:
    class I18nAuto:
        def __init__(self):
            pass
        def __call__(self, key):
            return key

i18n = I18nAuto()

rvc_theme = loadThemes.load_json() or gr.themes.Default()

# Custom CSS for modern UI
custom_css = """
/* Modern UI enhancements */
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

/* Main container styling */
.dark .dark\:bg-gray-900 {
    background-color: #0f172a !important;
}

/* Tab styling */
.tabitem {
    border-radius: 0.5rem !important;
    margin: 0.5rem !important;
    padding: 1rem !important;
    background-color: var(--card-background) !important;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px -1px rgba(0, 0, 0, 0.1) !important;
}

/* Button styling */
.gradio-button {
    border-radius: 0.5rem !important;
    padding: 0.5rem 1rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

/* Slider styling */
.gradio-slider input[type=range] {
    height: 0.5rem !important;
    border-radius: 9999px !important;
}

/* Accordion styling */
.accordion {
    border: 1px solid var(--border-color) !important;
    border-radius: 0.5rem !important;
    margin: 0.5rem 0 !important;
    overflow: hidden !important;
}

.accordion-header {
    background-color: #f1f5f9 !important;
    padding: 0.75rem 1rem !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
}

.accordion-content {
    padding: 1rem !important;
}

/* Card styling */
.card {
    background: var(--card-background) !important;
    border-radius: 0.5rem !important;
    padding: 1.5rem !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1) !important;
    border: 1px solid var(--border-color) !important;
    margin: 0.5rem 0 !important;
}

/* Audio player styling */
audio {
    width: 100% !important;
    margin: 0.5rem 0 !important;
}

/* Markdown styling */
.markdown h1 {
    font-size: 2rem !important;
    font-weight: 700 !important;
    margin-bottom: 1rem !important;
    color: var(--text-primary) !important;
}

.markdown h2 {
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    margin-top: 1.5rem !important;
    margin-bottom: 1rem !important;
    color: var(--text-primary) !important;
}

/* Footer styling */
footer {
    display: none !important;
}

/* Responsive design */
@media (max-width: 768px) {
    .gradio-container {
        padding: 0.5rem !important;
    }

    .tabitem {
        margin: 0.25rem !important;
        padding: 0.5rem !important;
    }

    .card {
        padding: 1rem !important;
        margin: 0.25rem 0 !important;
    }
}
"""

with gr.Blocks(
    title="Advanced RVC Inference ",
    fill_width=True,
    analytics_enabled=False  # Disable analytics for privacy
) as app:
    # Improved header with better layout
    with gr.Row(elem_classes="header-row", equal_height=True):
        with gr.Column(scale=1, min_width=200):
            gr.Markdown("# 🎤 Advanced RVC Inference v4.0")
        with gr.Column(scale=2, min_width=300, elem_classes="header-info"):
            gr.Markdown(
                "> *Kernel Advanced RVC - 2x Faster Training & Inference*  \n"
                "> *Ultimate Voice Conversion Platform with Advanced Performance Optimization*"
            )

    # Main content area with organized tabs
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
        from .tabs.training import training_tab
        training_tab()

    with gr.Tab("🔧 Audio Tools"):
        enhancement_tab()
        if F0_EXTRACTOR_AVAILABLE:
            f0_extractor_tab()

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


def launch(port):
    app.launch(
        share="--share" in sys.argv,
        inbrowser="--open" in sys.argv,
        server_port=port,
        show_error=True,
        prevent_thread_lock=False,
        theme=rvc_theme,
        css=custom_css,
        footer_links=["api", "gradio", "settings"]  # Gradio 6: footer links instead of show_api
    )


def get_port_from_args():
    if "--port" in sys.argv:
        port_index = sys.argv.index("--port") + 1
        if port_index < len(sys.argv):
            return int(sys.argv[port_index])
    return DEFAULT_PORT


if __name__ == "__main__":
    port = get_port_from_args()
    for _ in range(MAX_PORT_ATTEMPTS):
        try:
            launch(port)
            break
        except OSError:
            print(
                f"Failed to launch on port {port}, trying again on port {port - 1}..."
            )
            port -= 1
        except Exception as error:
            print(f"An error occurred launching Gradio: {error}")
            import traceback
            traceback.print_exc()
            break

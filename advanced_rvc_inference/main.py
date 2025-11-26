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
    from advanced_rvc_inference.tabs.tts.tts import tts_tab
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

try:
    from advanced_rvc_inference.tabs.voice_blender.voice_blender import voice_blender_tab
    VOICE_BLENDER_AVAILABLE = True
except ImportError:
    VOICE_BLENDER_AVAILABLE = False

try:
    from advanced_rvc_inference.tabs.plugins.plugins import plugins_tab
    PLUGINS_AVAILABLE = True
except ImportError:
    PLUGINS_AVAILABLE = False

try:
    from advanced_rvc_inference.tabs.extra.extra import extra_tab
    EXTRA_AVAILABLE = True
except ImportError:
    EXTRA_AVAILABLE = False

try:
    from advanced_rvc_inference.tabs.f0_extractor import f0_extractor_tab
    F0_EXTRACTOR_AVAILABLE = True
except ImportError:
    F0_EXTRACTOR_AVAILABLE = False

try:
    from advanced_rvc_inference.tabs.embedders import embedders_tab
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

with gr.Blocks(
    theme=rvc_theme,
    title="Advanced RVC Inference ",
    css="footer{display:none !important}",
    fill_width=True
) as app:
    gr.Markdown("# 🎤 Advanced RVC Inference v4.0\n> *Kernel Advanced RVC - 2x Faster Training & Inference*")

    with gr.Tab(" Inference"):
        with gr.Tab("🎵 Full Inference"):
            full_inference_tab()
        with gr.Tab("🎤 Real-Time"):
            real_time_inference_tab()
        if TTS_AVAILABLE:
            with gr.Tab("📢 Text-to-Speech"):
                tts_tab() 
    with gr.Tab("Downloader"):
        with gr.Tab("🎵 Download Music"):
            download_music_tab()
        
        with gr.Tab("📦 Download Model"):
            download_model_tab()

    with gr.Tab("Train"):
        with gr.Tab("🎙️ Training"):
            from advanced_rvc_inference.tabs.training import training_tab
            training_tab()
        if F0_EXTRACTOR_AVAILABLE:
            with gr.Tab("🔍 F0 Extractor"):
                f0_extractor_tab()
        if EMBEDDER_AVAILABLE:
            with gr.Tab("🧠 Embedders"):
                embedders_tab() 

    with gr.Tab("📚 Model Manager"):
        model_manager_tab()

    with gr.Tab("🎧 Enhancement"):
        enhancement_tab()


    with gr.Tab("🔧 Extra Options"):
        extra_options_tab()



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
            break

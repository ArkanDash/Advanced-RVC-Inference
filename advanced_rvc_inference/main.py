"""
Advanced RVC Inference - KRVC Kernel
Kernel Advanced RVC - 2x Faster Training & Inference
Version 3.5.2
"""

import gradio as gr
import sys, os
from tabs.full_inference import full_inference_tab
from tabs.download_model import download_model_tab
from tabs.download_music import download_music_tab
from tabs.settings import select_themes_tab
from tabs.training import training_tab
from tabs.model_manager import model_manager_tab
from tabs.enhancement import enhancement_tab
from tabs.real_time import real_time_inference_tab
from tabs.config_options import extra_options_tab
import assets.themes.loadThemes as loadThemes

now_dir = os.getcwd()
sys.path.append(now_dir)
DEFAULT_PORT = 7755
MAX_PORT_ATTEMPTS = 10

from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

import assets.themes.loadThemes as loadThemes

rvc_theme = loadThemes.load_json() or gr.themes.Default()

with gr.Blocks(
    theme=rvc_theme,
    title="Advanced RVC Inference - KRVC Kernel",
    css="footer{display:none !important}",
    head="<style>.footer { visibility: hidden; }</style>",
    fill_width=True
) as RVCAICoverMaker:
    gr.Markdown("# 🎤 Advanced RVC Inference - KRVC Kernel v3.5.2\n> *Kernel Advanced RVC - 2x Faster Training & Inference*")

    with gr.Tab("🎵 Full Inference"):
        full_inference_tab()

    with gr.Tab("🎙️ Training"):
        training_tab()

    with gr.Tab("📚 Model Manager"):
        model_manager_tab()

    with gr.Tab("🎧 Enhancement"):
        enhancement_tab()

    with gr.Tab("🎤 Real-Time"):
        real_time_inference_tab()

    with gr.Tab("🔧 Extra Options"):
        extra_options_tab()

    with gr.Tab("🎵 Download Music"):
        download_music_tab()

    with gr.Tab("📦 Download Model"):
        download_model_tab()

    with gr.Tab("⚙️ Settings"):
        select_themes_tab()


def launch(port):
    RVCAICoverMaker.launch(
        favicon_path=os.path.join(now_dir, "assets", "logo.ico"),
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

import gradio as gr
import sys
import os
import logging

# Constants
DEFAULT_PORT = 7897
MAX_PORT_ATTEMPTS = 10

# Set up logging
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Add current directory to sys.path
now_dir = os.getcwd()
sys.path.append(now_dir)

# Import Tabs
from tabs.inference.inference import inference_tab
from tabs.train.train import train_tab
from tabs.utilities.utilities import utilities_tab
from tabs.download.download import download_tab
from tabs.tts.tts import tts_tab
from tabs.voice_blender.voice_blender import voice_blender_tab
from tabs.settings.settings import settings_tab

# Run prerequisites
from core import run_prerequisites_script

run_prerequisites_script(
    pretraineds_hifigan=True,
    models=True,
    exe=True,
)

# Check installation
import assets.installation_checker as installation_checker

installation_checker.check_installation()

# Load theme
import assets.themes.loadThemes as loadThemes

CodenameViolet = loadThemes.load_theme() or "ParityError/Interstellar"

# Define Gradio interface
with gr.Blocks(
    theme=CodenameViolet, title="Codename-RVC-Fork 🍇", css="footer{display:none !important}"
) as Applio:
    gr.Markdown("# Codename-RVC-Fork 🍇 v4.0.8")
    gr.Markdown(
        "ㅤㅤBased on Applioㅤㅤ"
    )
    gr.Markdown(
        "ㅤㅤㅤ[Support - Community Discord](https://discord.gg/Qcfsuk4qN5) ㅤ/ ㅤ[GitHub](https://github.com/codename0og/codename-rvc-fork-4)"
    )
    with gr.Tab("Inference"):
        inference_tab()

    with gr.Tab("Training"):
        train_tab()

    with gr.Tab("TTS"):
        tts_tab()

    with gr.Tab("Voice Blender"):
        voice_blender_tab()

    with gr.Tab("Download"):
        download_tab()

    with gr.Tab("Utilities"):
        utilities_tab()

    with gr.Tab("Settings"):
        settings_tab()


def launch_gradio(port):
    Applio.launch(
        favicon_path="assets/ICON.ico",
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
            launch_gradio(port)
            break
        except OSError:
            print(
                f"Failed to launch on port {port}, trying again on port {port - 1}..."
            )
            port -= 1
        except Exception as error:
            print(f"An error occurred launching Gradio: {error}")
            break

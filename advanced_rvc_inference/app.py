import gradio as gr
import sys
import os
import logging
from pathlib import Path
from typing import Any

DEFAULT_SERVER_NAME = "127.0.0.1"
DEFAULT_PORT = 6969
MAX_PORT_ATTEMPTS = 10

# Set up logging
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Get the absolute path of the project root directory
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

# Zluda hijack
from advanced_rvc_inference.rvc.lib.zluda import *

# Import Tabs
from advanced_rvc_inference.tabs.inference.inference import inference_tab
from advanced_rvc_inference.tabs.train.train import train_tab
from advanced_rvc_inference.tabs.extra.extra import extra_tab
from advanced_rvc_inference.tabs.report.report import report_tab
from advanced_rvc_inference.tabs.download.download import download_tab
from advanced_rvc_inference.tabs.tts.tts import tts_tab
from advanced_rvc_inference.tabs.voice_blender.voice_blender import voice_blender_tab
from advanced_rvc_inference.tabs.plugins.plugins import plugins_tab
from advanced_rvc_inference.tabs.settings.settings import settings_tab
from advanced_rvc_inference.tabs.realtime.realtime import realtime_tab
from advanced_rvc_inference.tabs.separation.separation import separation_tab
from advanced_rvc_inference.tabs.full_inference.full_inference import full_inference_tab

# Run prerequisites
from advanced_rvc_inference.core import run_prerequisites_script

run_prerequisites_script(
    pretraineds_hifigan=True,
    models=True,
    exe=True,
)

# Initialize i18n
from advanced_rvc_inference.assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

# Start Discord presence if enabled
from advanced_rvc_inference.tabs.settings.sections.presence import load_config_presence

if load_config_presence():
    from advanced_rvc_inference.assets.discord_presence import RPCManager

    RPCManager.start_presence()

# Check installation
import advanced_rvc_inference.assets.installation_checker as installation_checker

installation_checker.check_installation()

# Load theme
import advanced_rvc_inference.assets.themes.loadThemes as loadThemes

my_applio = loadThemes.load_theme() or "terastudio/yellow"

# Define Gradio interface
with gr.Blocks(
    theme=my_applio, title="Advanced RVC Inference", css="footer{display:none !important}"
) as Applio:
    gr.Markdown("# Advanced RVC inference")
    
    
    with gr.Tab(i18n("Inference")):
        inference_tab()

    with gr.Tab(i18n("Training")):
        train_tab()

    with gr.Tab(i18n("TTS")):
        tts_tab()

    with gr.Tab(i18n("Voice Blender")):
        voice_blender_tab()

    with gr.Tab(i18n("Realtime")):
        realtime_tab()

    with gr.Tab(i18n("Separation")):
        separation_tab()

    with gr.Tab(i18n("Full Inference (RVC x UVR)")):
        full_inference_tab()

    with gr.Tab(i18n("Plugins")):
        plugins_tab()

    with gr.Tab(i18n("Download")):
        download_tab()

    with gr.Tab(i18n("Report a Bug")):
        report_tab()

    with gr.Tab(i18n("Extra")):
        extra_tab()

    with gr.Tab(i18n("Settings")):
        settings_tab()

    gr.Markdown(
        """
    <div style="text-align: center; font-size: 0.9em; text-color: a3a3a3;">
    By using Applio, you agree to comply with ethical and legal standards, respect intellectual property and privacy rights, avoid harmful or prohibited uses, and accept full responsibility for any outcomes, while Applio disclaims liability and reserves the right to amend these terms.
    </div>
    """
    )


def launch_gradio(server_name: str, server_port: int) -> None:
    favicon_path = os.path.join(str(project_root), "assets", "ICON.ico")
    Applio.launch(
        favicon_path=favicon_path,
        share="--share" in sys.argv,
        inbrowser="--open" in sys.argv,
        server_name=server_name,
        server_port=server_port,
    )


def get_value_from_args(key: str, default: Any = None) -> Any:
    if key in sys.argv:
        index = sys.argv.index(key) + 1
        if index < len(sys.argv):
            return sys.argv[index]
    return default


if __name__ == "__main__":
    port = int(get_value_from_args("--port", DEFAULT_PORT))
    server = get_value_from_args("--server-name", DEFAULT_SERVER_NAME)

    for _ in range(MAX_PORT_ATTEMPTS):
        try:
            launch_gradio(server, port)
            break
        except OSError:
            print(
                f"Failed to launch on port {port}, trying again on port {port - 1}..."
            )
            port -= 1
        except Exception as error:
            print(f"An error occurred launching Gradio: {error}")
            break

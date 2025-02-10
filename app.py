import gradio as gr
import sys
import os
import logging
from typing import Any
DEFAULT_SERVER_NAME = "127.0.0.1"
DEFAULT_PORT = 6969
MAX_PORT_ATTEMPTS = 10
import assets.themes.loadThemes as loadThemes
# Set up logging
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Add current directory to sys.path
now_dir = os.getcwd()
sys.path.append(now_dir)

# Zluda hijack
import rvc.lib.zluda

# Import Tabs
from tabs.inference.inference import inference_tab
from tabs.download.download import download_tab

# Run prerequisites
from rvc.lib.tools.prerequisites_download import prerequisites_download_pipeline
prerequisites_download_pipeline(models=True, exe=True)





# Define Gradio interface
with gr.Blocks(
    theme = loadThemes.load_json() or "NoCrypt/miku", title="Advanced-RVC-Inference", css="footer{display:none !important}"
) as adrvc:
    gr.Markdown("# Advanced-RVC-Inference")
    gr.Markdown("Advanced RVC Inference for quicker and effortless model downloads.")
    gr.Markdown("[Support](https://discord.gg/hvmsukmBHE) — [GitHub](https://github.com/ArkanDash/Advanced-RVC-Inference.git)")
    gr.Makrdown("Thanks to [NeoDev](https://github.com/TheNeodev) for improve this project")
    
    
    with gr.Tab("Inference"):
        inference_tab()
    with gr.Tab("Download"):
        download_tab()

    with gr.Tab("Settings"):
        gr.Markdown("On Progress...")
        #settings_tab()

    gr.Markdown(
    """
    <div style="text-align: center; font-size: 0.9em; color: #a3a3a3;">
        <strong>Disclaimer:</strong> By accessing <span style="color: #555;">Advanced-RVC-Inference</span>, you acknowledge your responsibility to follow all ethical and legal standards, respect intellectual property and privacy rights, and accept full accountability for your actions. <br><br>
        Please note that Advanced-RVC-Inference disclaims all liability and reserves the right to update these terms at any time.
    </div>
    """
)


def launch_gradio(server_name: str, server_port: int) -> None:
    adrvc.launch(
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

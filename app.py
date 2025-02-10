import gradio as gr
import sys
import os
import logging
from typing import Any

DEFAULT_SERVER_NAME = "127.0.0.1"
DEFAULT_PORT = 6969
MAX_PORT_ATTEMPTS = 10

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

# Run prerequisites
from rvc.lib.tools.prerequisites_download import prerequisites_download_pipeline


print("downloading models...")
prerequisites_download_pipeline(models=True, exe=True)



# Load theme (demo)
#import assets.themes.loadThemes as loadThemes
# loadThemes.load_theme() 
my_theme =  "ParityError/Interstellar"

# Define Gradio interface
with gr.Blocks(
    theme=my_applio, title="Applio", css="footer{display:none !important}"
) as adrvc:
    gr.Markdown("# Advanced-RVC-Inference")
    gr.Markdown("A simple, high-quality voice conversion tool focused on ease of use and performance.")
    gr.Markdown("[Support](https://discord.gg/hvmsukmBHE) — [GitHub](https://github.com/ArkanDash/Advanced-RVC-Inference.git)")
    )
    
    with gr.Tab(i18n("Inference")):
        inference_tab()



    with gr.Tab(i18n("Download")):
        download_tab()


    with gr.Tab("Settings"):
        gr.Markdown("On Progress...")
        #settings_tab()

    gr.Markdown(
        """
    <div style="text-align: center; font-size: 0.9em; text-color: a3a3a3;">
    By using Advanced-RVC-Inference, you agree to comply with ethical and legal standards, respect intellectual property and privacy rights, avoid harmful or prohibited uses, and accept full responsibility for any outcomes, while Advanced-RVC-Inference disclaims liability and reserves the right to amend these terms.
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

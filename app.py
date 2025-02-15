import gradio as gr
import sys, os
from tabs.full_inference import full_inference_tab
from tabs.download_model import download_model_tab
from tabs.settings import select_themes_tab, lang_tab, restart_tab
from pypresence import Presence
import datetime as dt

class RichPresenceManager:
    def __init__(self):
        self.client_id = "1340358329771364430"
        self.rpc = None
        self.running = False

    def start_presence(self):
        if not self.running:
            self.running = True
            self.rpc = Presence(self.client_id)
            try:
                self.rpc.connect()
                self.update_presence()
            except KeyboardInterrupt as error:
                print(error)
                self.rpc = None
                self.running = False
            except Exception as error:
                print(f"An error occurred connecting to Discord: {error}")
                self.rpc = None
                self.running = False

    def update_presence(self):
        if self.rpc:
            self.rpc.update(
                details="Advanced RVC Inference",
                buttons=[{"label": "Download", "url": "https://github.com/ArkanDash/Advanced-RVC-Inference.git"}],
                large_image="nfvs4co",
                large_text="Advanced RVC Inference for quicker and effortless model downloads",
                start=dt.datetime.now().timestamp(),
            )

    def stop_presence(self):
        self.running = False
        if self.rpc:
            self.rpc.close()
            self.rpc = None

RPCManager = RichPresenceManager()

now_dir = os.getcwd()
sys.path.append(now_dir)
DEFAULT_PORT = 7755
MAX_PORT_ATTEMPTS = 10

from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

import assets.themes.loadThemes as loadThemes

rvc_theme = loadThemes.load_json() or "NoCrypt/miku"

with gr.Blocks(
    theme=rvc_theme, title="Advanced-RVC-Inference"
) as rvc:
    gr.Markdown("# Advanced-RVC-Inference")
    with gr.Tab(i18n("Full Inference")):
        full_inference_tab()
    with gr.Tab(i18n("Download Model")):
        download_model_tab()
    with gr.Tab(i18n("Settings")):
        with gr.Tab("Theme Selection"):
            select_themes_tab()
        with gr.Tab("Language Changer"):
            lang_tab()
        restart_tab()
    gr.Markdown('<div align="center">this project Maintained by <a href="https://discord.com/1314204512814235689">NeoDev</a></div>')



def launch(port):
    rvc.launch(
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

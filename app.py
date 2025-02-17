import gradio as gr
import sys, os
from tabs.full_inference import full_inference_tab
from tabs.download_model import download_model_tab
from tabs.settings import select_themes_tab, lang_tab, restart_tab
from programs.applio_code.rvc.lib.tools.prerequisites_download import prequisites_download_pipeline

now_dir = os.getcwd()
sys.path.append(now_dir)
DEFAULT_PORT = 7755
MAX_PORT_ATTEMPTS = 10


prequisites_download_pipeline(
    False,
    False,
    True, 
    False,
)


from assets.i18n.i18n import I18nAuto
import assets.themes.loadThemes as loadThemes


i18n = I18nAuto()




rvc_theme = loadThemes.load_json() or "NoCrypt/miku"

with gr.Blocks(
    theme=rvc_theme, title="Advanced RVC Inference"
) as rvc:
    gr.Markdown('<div align="center"><h1>Advanced RVC Inference</h1></a></div>')
    gr.Markdown('<div align="center">this project Maintained by <a href="https://discord.com/1314204512814235689">NeoDev</a></div>')
  
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
    gr.Markdown(
        """
    <div style="text-align: center; font-size: 0.9em; text-color: a3a3a3;">
    By using Advanced RVC Inference, you agree to comply with ethical and legal standards, respect intellectual property and privacy rights, avoid harmful or prohibited uses, and accept full responsibility for any outcomes, while Applio disclaims liability and reserves the right to amend these terms.
    </div>
    """
    )


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

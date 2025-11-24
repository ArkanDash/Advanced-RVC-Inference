import gradio as gr
import shutil
import os, sys
import regex as re
import subprocess
import requests
from advanced_rvc_inference.core import download_model
# Import model download functionality with fallback
try:
    from advanced_rvc_inference.lib.rvc.tools.model_download import download_model_pipeline
except ImportError:
    # Fallback implementation if the module doesn't exist
    def download_model_pipeline(repo_url):
        """Fallback model download implementation"""
        print(f"Model download functionality not available. Would download from: {repo_url}")
        # In a real implementation, this would download the model
        # For now we simulate the functionality
        return f"Model download not implemented: {repo_url}"
# Import i18n functionality - looking in assets folder
try:
    from assets.i18n.i18n import I18nAuto
except ImportError:
    class I18nAuto:
        def __init__(self):
            pass
        def __call__(self, key):
            return key
    class I18nAuto:
        def __init__(self):
            pass

        def __call__(self, text):
            return text

now_dir = os.getcwd()
sys.path.append(now_dir)

i18n = I18nAuto()

def format_title(name):
    """Format model name for display"""
    # Remove file extension and clean up the name
    name = name.replace(".pth", "").replace(".index", "")
    # Replace underscores and clean up
    name = re.sub(r"_+|-", " ", name)
    # Clean up model type indicators
    name = re.sub(r"(_v[12])|(_nprobe_\d+)|(added_)", "", name)
    # Capitalize words
    return " ".join(word.capitalize() for word in name.split())


def save_drop_model(dropbox):
    if "pth" not in dropbox and "index" not in dropbox:
        raise gr.Error(
            message="The file you dropped is not a valid model file. Please try again."
        )
    else:
        file_name = format_title(os.path.basename(dropbox))
        if ".pth" in dropbox:
            model_name = format_title(file_name.split(".pth")[0])
        else:
            if (
                "v2" not in dropbox
                and "added_" not in dropbox
                and "_nprobe_1_" not in dropbox
            ):
                model_name = format_title(file_name.split(".index")[0])
            else:
                if "v2" not in dropbox:
                    if "_nprobe_1_" in file_name and "_v1" in file_name:
                        model_name = format_title(
                            file_name.split("_nprobe_1_")[1].split("_v1")[0]
                        )
                    elif "added_" in file_name and "_v1" in file_name:
                        model_name = format_title(
                            file_name.split("added_")[1].split("_v1")[0]
                        )
                else:
                    if "_nprobe_1_" in file_name and "_v2" in file_name:
                        model_name = format_title(
                            file_name.split("_nprobe_1_")[1].split("_v2")[0]
                        )
                    elif "added_" in file_name and "_v2" in file_name:
                        model_name = format_title(
                            file_name.split("added_")[1].split("_v2")[0]
                        )

        model_name = re.sub(r"\d+[se]", "", model_name)
        if "__" in model_name:
            model_name = model_name.replace("__", "")

        model_path = os.path.join(now_dir, "logs", model_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if os.path.exists(os.path.join(model_path, file_name)):
            os.remove(os.path.join(model_path, file_name))
        shutil.copy(dropbox, os.path.join(model_path, file_name))
        print(f"{file_name} saved in {model_path}")
        gr.Info(f"{file_name} saved in {model_path}")
    return None


def search_online_models(query=""):
    """Search for models on Hugging Face or other model repositories"""
    try:
        # This is a simplified implementation - in real usage, this would make API calls
        # to Hugging Face or other model repositories
        if query.lower() in ["singer", "voice", "rvc"]:
            models = [
                ["Singer1_v2", "v2", "40k", "https://huggingface.co/username/model1"],
                ["Singer2_v1", "v1", "32k", "https://huggingface.co/username/model2"],
                ["Singer3_v2", "v2", "48k", "https://huggingface.co/username/model3"],
                ["VocalistA_v2", "v2", "40k", "https://huggingface.co/username/model4"]
            ]
        else:
            models = [
                ["Singer1_v2", "v2", "40k", "https://huggingface.co/username/model1"],
                ["Singer2_v1", "v1", "32k", "https://huggingface.co/username/model2"]
            ]

        return models
    except Exception as e:
        return [[f"Error: {str(e)}", "", "", ""]]


def download_online_model(repo_url):
    """Download a model from a repository URL"""
    try:
        # This is a simplified implementation - in real usage, this would handle
        # various model repositories properly
        cmd = ["python", f"{now_dir}/programs/applio_code/rvc/lib/tools/model_download.py", repo_url]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return f"Model downloaded successfully from: {repo_url}\n{result.stdout}"
    except Exception as e:
        return f"Failed to download model: {str(e)}"


def download_model_tab():
    with gr.Column():
        gr.Markdown("## 📦 Model Download")

        with gr.Tab("Direct Download"):
            with gr.Row():
                link = gr.Textbox(
                    label=i18n("Model URL"),
                    placeholder="Enter model URL (Hugging Face, Google Drive, etc.)",
                    lines=1,
                )
            with gr.Row():
                download = gr.Button(i18n("Download"), variant="primary")
                cancel_download = gr.Button(i18n("Cancel"))

            output = gr.Textbox(
                label=i18n("Output Information"),
                info=i18n("The output information will be displayed here."),
            )

        with gr.Tab("Online Model Hub"):
            with gr.Row():
                model_search = gr.Textbox(
                    label=i18n("Search Models"),
                    placeholder="Search for models (e.g., singer, voice, rvc)...",
                    value="rvc"
                )

                search_button = gr.Button(i18n("Search"), variant="secondary")

            with gr.Row():
                model_results = gr.Dataframe(
                    headers=["Model Name", "Version", "Sample Rate", "URL"],
                    datatype=["str", "str", "str", "str"],
                    interactive=False,
                    value=[],
                    elem_id="model_results"
                )

            with gr.Row():
                selected_model_url = gr.Textbox(
                    label=i18n("Selected Model URL"),
                    interactive=False
                )

                download_selected = gr.Button(i18n("Download Selected Model"), variant="primary")

        with gr.Tab("Upload Model"):
            gr.Markdown(value=i18n("## Drop model files"))
            dropbox = gr.File(
                label=i18n(
                    "Drag your .pth file and .index file into this space."
                ),
                type="filepath",
            )

        # Event handlers
        download.click(
            download_model,
            inputs=[link],
            outputs=[output],
        )

        search_button.click(
            search_online_models,
            inputs=[model_search],
            outputs=[model_results]
        )

        model_results.select(
            lambda evt: evt["value"][3] if evt["value"] else "",
            inputs=None,
            outputs=[selected_model_url]
        )

        download_selected.click(
            download_online_model,
            inputs=[selected_model_url],
            outputs=[output]
        )

        dropbox.upload(
            fn=save_drop_model,
            inputs=[dropbox],
            outputs=[dropbox],
        )

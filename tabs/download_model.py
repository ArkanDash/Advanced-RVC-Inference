import gradio as gr
import shutil
import os, sys
import regex as re

from core import download_model
from programs.applio_code.rvc.lib.utils import format_title
from assets.i18n.i18n import I18nAuto

now_dir = os.getcwd()
sys.path.append(now_dir)

i18n = I18nAuto()


def save_drop_model(dropbox):
    if dropbox is None or ("pth" not in dropbox and "index" not in dropbox):
        gr.Error(
            message="The file you dropped is not a valid model file. Please try again."
        )
        return None
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


def download_model_tab():
    with gr.Row():
        link = gr.Textbox(
            label=i18n("Model URL"),
            lines=1,
        )
    output = gr.Textbox(
        label=i18n("Output Information"),
        info=i18n("The output information will be displayed here."),
    )
    download = gr.Button(i18n("Download"))

    download.click(
        download_model,
        inputs=[link],
        outputs=[output],
    )
    gr.Markdown(value=i18n("## Drop files"))
    dropbox = gr.File(
        label=i18n(
            "Drag your .pth file and .index file into this space. Drag one and then the other."
        ),
        type="filepath",
    )
    dropbox.upload(
        fn=save_drop_model,
        inputs=[dropbox],
        outputs=[dropbox],
    )

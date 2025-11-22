import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from main.app.variables import translations, configs
from main.app.tabs.training.child.training import training_model_tab
from main.app.tabs.training.child.create_dataset import create_dataset_tab
from main.app.tabs.training.child.create_reference import create_reference_tab

def training_tab():
    with gr.TabItem(translations["training_model"], visible=configs.get("create_and_training_tab", True)):
        with gr.TabItem(translations["createdataset"], visible=configs.get("create_dataset_tab", True)):
            gr.Markdown(translations["create_dataset_markdown"])
            create_dataset_tab()

        with gr.TabItem(translations["create_reference"], visible=configs.get("create_reference_tab", True)):
            gr.Markdown(translations["create_reference_markdown"])
            create_reference_tab()

        with gr.TabItem(translations["training_model"], visible=configs.get("training_tab", True)):
            gr.Markdown(f"## {translations['training_model']}")
            training_model_tab()
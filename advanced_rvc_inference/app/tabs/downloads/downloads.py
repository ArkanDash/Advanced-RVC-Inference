"""
Downloads tab for Advanced RVC Inference.

Provides model download, pretrained model download, and search functionality.
"""

import gradio as gr
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from advanced_rvc_inference.utils.variables import translations, configs, model_options
from advanced_rvc_inference.core.downloads import download_model, download_pretrained_model, search_models


def download_tab():
    """Create the downloads tab UI."""
    with gr.TabItem(translations.get("downloads", "Downloads"), visible=configs.get("downloads_tab", True)):
        gr.Markdown(f"## {translations.get('downloads', 'Downloads')}")

        with gr.TabItem(translations.get("download_models", "Download Models")):
            with gr.Row():
                download_url_input = gr.Textbox(
                    label=translations.get("model_url", "Model URL"),
                    placeholder="https://huggingface.co/...",
                    interactive=True,
                )
                model_name_input = gr.Textbox(
                    label=translations.get("model_name", "Model Name (optional)"),
                    placeholder="model_name",
                    interactive=True,
                )
            download_btn = gr.Button(translations.get("download", "Download"), variant="primary")
            download_result = gr.Textbox(label=translations.get("result", "Result"), interactive=False)

            download_btn.click(
                fn=download_model,
                inputs=[download_url_input, model_name_input],
                outputs=[download_result],
            )

        with gr.TabItem(translations.get("search_models", "Search Models")):
            with gr.Row():
                search_input = gr.Textbox(
                    label=translations.get("search_name", "Model Name"),
                    placeholder=translations.get("search_placeholder", "Enter model name to search..."),
                    interactive=True,
                )
                search_btn = gr.Button(translations.get("search", "Search"), variant="primary")
            with gr.Row():
                search_results = gr.Dropdown(
                    label=translations.get("search_results", "Results"),
                    interactive=True,
                )
                download_selected_btn = gr.Button(
                    translations.get("download", "Download"),
                    variant="primary",
                )
            download_selected_result = gr.Textbox(
                label=translations.get("result", "Result"), interactive=False
            )

            search_btn.click(
                fn=lambda name: search_models(name) if name else [None, None],
                inputs=[search_input],
                outputs=[search_results, download_selected_result],
            )
            download_selected_btn.click(
                fn=lambda name: download_model(url=model_options.get(name, "")) if name else None,
                inputs=[search_results],
                outputs=[download_selected_result],
            )

        with gr.TabItem(translations.get("download_pretrained", "Download Pretrained Models")):
            with gr.Row():
                pretrained_choice = gr.Dropdown(
                    choices=[
                        translations.get("list_model", "List Models"),
                        translations.get("download_url", "Download from URL"),
                    ],
                    label=translations.get("download_type", "Download Type"),
                    value=translations.get("list_model", "List Models"),
                    interactive=True,
                )
            with gr.Row():
                pretrained_model = gr.Dropdown(
                    label=translations.get("pretrained_model", "Pretrained Model"),
                    interactive=True,
                )
                pretrained_sr = gr.Dropdown(
                    label=translations.get("sample_rate", "Sample Rate"),
                    interactive=True,
                )
            with gr.Row():
                pretrained_url_d = gr.Textbox(
                    label=translations.get("d_model_url", "D Model URL"),
                    interactive=True,
                    visible=False,
                )
                pretrained_url_g = gr.Textbox(
                    label=translations.get("g_model_url", "G Model URL"),
                    interactive=True,
                    visible=False,
                )
            pretrained_btn = gr.Button(
                translations.get("download_pretrain", "Download Pretrained"), variant="primary"
            )
            pretrained_result = gr.Textbox(
                label=translations.get("result", "Result"), interactive=False
            )

            pretrained_btn.click(
                fn=download_pretrained_model,
                inputs=[pretrained_choice, pretrained_url_d, pretrained_url_g],
                outputs=[pretrained_result, pretrained_result],
            )

"""
Downloads tab for the Advanced RVC Inference GUI.

Provides UI for downloading RVC voice models, pretrained models,
and audio from URLs.
"""

import os
import sys

import gradio as gr

from arvc.utils.variables import translations, configs, model_options, model_name
from arvc.ui.feedback import (
    gr_info, gr_warning, gr_error,
    change_models_choices, change_pretrained_choices,
    change_download_choices, change_download_pretrained_choices,
    replace_url, replace_modelname,
)
from arvc.services.downloads import (
    download_url,
    download_model,
    download_pretrained_model,
    search_models,
)


def download_tab():
    """Create the Downloads tab UI."""
    with gr.TabItem(translations["downloads"], visible=configs.get("downloads_tab", True)):
        # ── Download Voice Models ──
        with gr.TabItem(translations.get("download_model", "Download Model")):
            gr.Markdown(f"## {translations.get('download_model', 'Download Model')}")

            with gr.Row():
                download_select = gr.Dropdown(
                    choices=[
                        translations.get("download_url", "Download from URL"),
                        translations.get("download_from_csv", "Download from CSV"),
                        translations.get("search_models", "Search Models"),
                        translations.get("upload", "Upload"),
                    ],
                    value=translations.get("download_url", "Download from URL"),
                    label=translations.get("select_option", "Select Option"),
                    interactive=True,
                )

            # URL download section
            with gr.Row(visible=True) as url_row:
                download_model_url = gr.Textbox(
                    label=translations.get("model_url", "Model URL"),
                    placeholder="https://huggingface.co/...",
                    interactive=True,
                )
                download_model_name = gr.Textbox(
                    label=translations.get("model_name", "Model Name"),
                    placeholder=translations.get("model_name_placeholder", "Leave empty to use original name"),
                    interactive=True,
                )
                download_url_button = gr.Button(
                    translations.get("download", "Download"),
                    variant="primary",
                )

            # CSV download section
            with gr.Row(visible=False) as csv_row:
                download_csv_model = gr.Dropdown(
                    choices=list(model_name.keys()) if model_name else [],
                    label=translations.get("select_model", "Select Model"),
                    interactive=True,
                )
                download_csv_button = gr.Button(
                    translations.get("download", "Download"),
                    variant="primary",
                )

            # Search section
            with gr.Row(visible=False) as search_row:
                search_model_name = gr.Textbox(
                    label=translations.get("search_model_name", "Search Model Name"),
                    placeholder=translations.get("search_placeholder", "Enter model name..."),
                    interactive=True,
                )
                search_button = gr.Button(
                    translations.get("search", "Search"),
                    variant="primary",
                )
            with gr.Row(visible=False) as search_result_row:
                search_dropdown = gr.Dropdown(
                    choices=[],
                    label=translations.get("search_results", "Search Results"),
                    interactive=True,
                )
                search_download_model = gr.Button(
                    translations.get("download", "Download"),
                )

            # Upload section
            with gr.Row(visible=False) as upload_row:
                upload_model_files = gr.File(
                    label=translations.get("upload_model", "Upload Model Files"),
                    file_count="multiple",
                )

            # Status output
            download_status = gr.Textbox(
                label=translations.get("status", "Status"),
                interactive=False,
            )

            # ── Download Audio from URL ──
            with gr.Row():
                audio_url = gr.Textbox(
                    label=translations.get("audio_url", "Audio URL (YouTube, etc.)"),
                    placeholder="https://www.youtube.com/watch?v=...",
                    interactive=True,
                )
                audio_download_button = gr.Button(
                    translations.get("download_audio", "Download Audio"),
                    variant="secondary",
                )

            audio_output_path = gr.Textbox(
                label=translations.get("output_path", "Output Path"),
                interactive=False,
            )
            audio_player = gr.Audio(
                label=translations.get("audio_output", "Audio Output"),
                type="filepath",
            )

            # ── Event Handlers ──
            download_select.change(
                fn=change_download_choices,
                inputs=[download_select],
                outputs=[
                    url_row,
                    csv_row,
                    search_row,
                    search_result_row,
                    upload_row,
                ],
            )

            download_url_button.click(
                fn=lambda url, name: download_model(url=url, model=name) if url else gr_warning(translations.get("provide_url", "Please provide a URL")),
                inputs=[download_model_url, download_model_name],
                outputs=[download_status],
            )

            search_button.click(
                fn=search_models,
                inputs=[search_model_name],
                outputs=[search_dropdown, search_download_model],
            )

            search_download_model.click(
                fn=lambda selection: download_model(url=model_options.get(selection, ""), model=selection) if selection else None,
                inputs=[search_dropdown],
                outputs=[download_status],
            )

            audio_download_button.click(
                fn=download_url,
                inputs=[audio_url],
                outputs=[audio_output_path, audio_player, download_status],
            )

        # ── Download Pretrained Models ──
        with gr.TabItem(translations.get("download_pretrained", "Download Pretrained")):
            gr.Markdown(f"## {translations.get('download_pretrained', 'Download Pretrained Models')}")

            with gr.Row():
                pretrained_select = gr.Dropdown(
                    choices=[
                        translations.get("download_url", "Download from URL"),
                        translations.get("list_model", "List Model"),
                        translations.get("upload", "Upload"),
                    ],
                    value=translations.get("download_url", "Download from URL"),
                    label=translations.get("select_option", "Select Option"),
                    interactive=True,
                )

            # URL section for pretrained
            with gr.Row(visible=True) as pretrained_url_row:
                pretrained_d_url = gr.Textbox(
                    label=translations.get("d_model_url", "D Model URL"),
                    placeholder="https://huggingface.co/...",
                    interactive=True,
                )
                pretrained_g_url = gr.Textbox(
                    label=translations.get("g_model_url", "G Model URL"),
                    placeholder="https://huggingface.co/...",
                    interactive=True,
                )
                pretrained_url_button = gr.Button(
                    translations.get("download", "Download"),
                    variant="primary",
                )

            # List model section
            with gr.Row(visible=False) as pretrained_list_row:
                pretrained_list_model = gr.Dropdown(
                    choices=[],
                    label=translations.get("select_model", "Select Model"),
                    interactive=True,
                )
                pretrained_sample_rate = gr.Dropdown(
                    choices=["32000", "40000", "48000"],
                    value="40000",
                    label=translations.get("sample_rate", "Sample Rate"),
                    interactive=True,
                )
                pretrained_list_button = gr.Button(
                    translations.get("download", "Download"),
                    variant="primary",
                )

            # Upload section for pretrained
            with gr.Row(visible=False) as pretrained_upload_row:
                pretrained_upload_files = gr.File(
                    label=translations.get("upload_pretrained", "Upload Pretrained Files"),
                    file_count="multiple",
                )

            pretrained_status = gr.Textbox(
                label=translations.get("status", "Status"),
                interactive=False,
            )

            # Event handlers
            pretrained_select.change(
                fn=change_download_pretrained_choices,
                inputs=[pretrained_select],
                outputs=[
                    pretrained_url_row,
                    pretrained_list_row,
                    pretrained_upload_row,
                ],
            )

            pretrained_url_button.click(
                fn=download_pretrained_model,
                inputs=[
                    pretrained_select,
                    pretrained_d_url,
                    pretrained_g_url,
                ],
                outputs=[pretrained_status],
            )

            pretrained_list_button.click(
                fn=download_pretrained_model,
                inputs=[
                    pretrained_select,
                    pretrained_list_model,
                    pretrained_sample_rate,
                ],
                outputs=[pretrained_status],
            )

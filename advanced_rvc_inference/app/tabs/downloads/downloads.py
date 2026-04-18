"""
Downloads Tab for Advanced RVC Inference.

Provides UI for downloading voice models, pre-trained models,
searching model repositories, and uploading models.
"""

import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from advanced_rvc_inference.utils.variables import (
    translations,
    configs,
    models,
    model_options,
    method_f0,
    export_format_choices,
)
from advanced_rvc_inference.core.downloads import (
    download_model,
    download_pretrained_model,
    search_models,
)
from advanced_rvc_inference.core.ui import (
    change_download_choices,
    change_download_pretrained_choices,
    change_models_choices,
)


def download_tab():
    """Build the Downloads tab UI."""
    with gr.Column():
        gr.Markdown(translations["download_markdown"])
        gr.Markdown(translations["download_markdown_2"])

        # --- Voice Model Download ---
        with gr.Group():
            gr.Markdown(f"### {translations['model_download']}")

            with gr.Row():
                model_download_method = gr.Dropdown(
                    label=translations["model_download_select"],
                    choices=[
                        translations["download_url"],
                        translations["download_from_csv"],
                        translations["search_models"],
                        translations["upload"],
                    ],
                    value=translations["download_url"],
                    interactive=True,
                )

            # Download from URL
            with gr.Row(visible=True) as url_row:
                model_url_input = gr.Textbox(
                    label=translations["model_url"],
                    placeholder="https://huggingface.co/...",
                    interactive=True,
                    show_label=True,
                )

            with gr.Row(visible=True) as url_name_row:
                model_name_input = gr.Textbox(
                    label=translations["modelname"],
                    placeholder=translations["provide_name_is_save"],
                    interactive=True,
                    show_label=True,
                )

            with gr.Row(visible=True) as url_btn_row:
                download_url_btn = gr.Button(
                    translations["download"], variant="primary"
                )

            # Download from CSV
            with gr.Row(visible=False) as csv_row:
                csv_model_select = gr.Dropdown(
                    label=translations["model_warehouse"],
                    choices=list(models.keys()),
                    interactive=True,
                    show_label=True,
                )

            with gr.Row(visible=False) as csv_btn_row:
                download_csv_btn = gr.Button(
                    translations["get_model"], variant="primary"
                )

            # Search models
            with gr.Row(visible=False) as search_row:
                search_name_input = gr.Textbox(
                    label=translations["name_to_search"],
                    placeholder="e.g. Gumi, Hatsune",
                    interactive=True,
                    show_label=True,
                )

            with gr.Row(visible=False) as search_btn_row:
                search_btn = gr.Button(
                    f"🔍 {translations['search_2']}", variant="secondary"
                )

            with gr.Row(visible=False) as search_result_row:
                search_result_select = gr.Dropdown(
                    label=translations["select_download_model"],
                    choices=[],
                    interactive=True,
                    show_label=True,
                )

            with gr.Row(visible=False) as search_dl_row:
                download_search_btn = gr.Button(
                    translations["download"], variant="primary"
                )

            # Upload model
            with gr.Row(visible=False) as upload_row:
                model_upload = gr.File(
                    label=translations["upload"],
                    file_types=[".pth", ".onnx", ".index", ".zip"],
                    interactive=True,
                    show_label=True,
                )

            model_download_method.change(
                fn=change_download_choices,
                inputs=[model_download_method],
                outputs=[
                    url_row,
                    url_name_row,
                    url_btn_row,
                    csv_row,
                    csv_btn_row,
                    search_row,
                    search_result_row,
                    search_dl_row,
                    upload_row,
                ],
            )

            download_url_btn.click(
                fn=download_model,
                inputs=[model_url_input, model_name_input],
                outputs=[download_url_btn],
            )

            download_csv_btn.click(
                fn=download_model,
                inputs=[csv_model_select, gr.Textbox(value="", visible=False)],
                outputs=[download_csv_btn],
            )

            search_btn.click(
                fn=search_models,
                inputs=[search_name_input],
                outputs=[search_result_select, search_dl_row],
            )

            download_search_btn.click(
                fn=download_model,
                inputs=[
                    search_result_select,
                    gr.Textbox(value="", visible=False),
                ],
                outputs=[download_search_btn],
            )

        # --- Pre-trained Model Download ---
        with gr.Group():
            gr.Markdown(f"### {translations['download_pretrained_2']}")

            with gr.Row():
                pretrained_download_method = gr.Dropdown(
                    label=translations["select_pretrain"],
                    choices=[
                        translations["download_url"],
                        translations["list_model"],
                        translations["upload"],
                    ],
                    value=translations["list_model"],
                    interactive=True,
                )

            # Pretrained: list model
            with gr.Row(visible=True) as pretrained_list_row:
                pretrained_model_select = gr.Dropdown(
                    label=translations["select_pretrain_info"],
                    choices=["D", "G"],
                    value="D",
                    interactive=True,
                )

            with gr.Row(visible=True) as pretrained_sr_row:
                pretrained_sr_select = gr.Dropdown(
                    label=translations["pretrain_sr"],
                    choices=["32000", "40000", "48000"],
                    value="40000",
                    interactive=True,
                )

            with gr.Row(visible=True) as pretrained_list_btn_row:
                pretrained_list_btn = gr.Button(
                    f"⬇️ {translations['download_pretrained']}", variant="primary"
                )

            # Pretrained: URL
            with gr.Row(visible=False) as pretrained_url_d_row:
                pretrained_d_url = gr.Textbox(
                    label=translations["pretrained_url"].format(dg="D"),
                    placeholder="https://huggingface.co/...",
                    interactive=True,
                    show_label=True,
                )

            with gr.Row(visible=False) as pretrained_url_g_row:
                pretrained_g_url = gr.Textbox(
                    label=translations["pretrained_url"].format(dg="G"),
                    placeholder="https://huggingface.co/...",
                    interactive=True,
                    show_label=True,
                )

            with gr.Row(visible=False) as pretrained_url_btn_row:
                pretrained_url_btn = gr.Button(
                    f"⬇️ {translations['download_pretrained']}", variant="primary"
                )

            # Pretrained: Upload
            with gr.Row(visible=False) as pretrained_upload_d_row:
                pretrained_d_upload = gr.File(
                    label=translations["drop_pretrain"].format(dg="D"),
                    file_types=[".pth"],
                    interactive=True,
                    show_label=True,
                )

            with gr.Row(visible=False) as pretrained_upload_g_row:
                pretrained_g_upload = gr.File(
                    label=translations["drop_pretrain"].format(dg="G"),
                    file_types=[".pth"],
                    interactive=True,
                    show_label=True,
                )

            pretrained_download_method.change(
                fn=change_download_pretrained_choices,
                inputs=[pretrained_download_method],
                outputs=[
                    pretrained_url_d_row,
                    pretrained_url_g_row,
                    pretrained_url_btn_row,
                    pretrained_list_row,
                    pretrained_sr_row,
                    pretrained_list_btn_row,
                    pretrained_upload_d_row,
                ],
            )

            pretrained_list_btn.click(
                fn=download_pretrained_model,
                inputs=[
                    pretrained_download_method,
                    pretrained_model_select,
                    pretrained_sr_select,
                ],
                outputs=[pretrained_list_btn],
            )

            pretrained_url_btn.click(
                fn=download_pretrained_model,
                inputs=[
                    pretrained_download_method,
                    pretrained_d_url,
                    pretrained_g_url,
                ],
                outputs=[pretrained_url_btn],
            )

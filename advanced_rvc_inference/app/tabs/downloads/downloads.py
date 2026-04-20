import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from advanced_rvc_inference.utils.variables import (
    models, 
    configs, 
    translations, 
    model_options
)

from advanced_rvc_inference.core.downloads import (
    search_models, 
    download_model, 
    download_pretrained_model
)

from advanced_rvc_inference.core.ui import (
    shutil_move,
    change_download_choices, 
    change_download_pretrained_choices 
)

from advanced_rvc_inference.core.process import (
    save_drop_model, 
    fetch_pretrained_data, 
    update_sample_rate_dropdown
)

def download_tab():
    """Downloads tab for managing RVC models and pretrained weights.

    Provides functionality to:
    - Download RVC voice models from URLs, CSV lists, or by searching
    - Upload model files directly (zip, pth, onnx, index)
    - Download pretrained base models (D/G) from URLs or a curated list
    - Upload pretrained weight files directly

    Based on Vietnamese-RVC download tab implementation.
    """
    with gr.TabItem(
        translations.get("downloads", "Downloads")
    ):
        gr.Markdown(translations.get("download_markdown", "## Download Models"))
        with gr.Row():
            gr.Markdown(translations.get("download_markdown_2", ""))

        # ── Model Download Section ──
        with gr.Row():
            with gr.Accordion(
                translations.get("model_download", "Model Download"), 
                open=True
            ):
                with gr.Row():
                    download_method = gr.Radio(
                        label=translations.get("model_download_select", "Download Method"), 
                        choices=[
                            translations.get("download_url", "Download URL"), 
                            translations.get("download_from_csv", "Download from CSV"), 
                            translations.get("search_models", "Search Models"), 
                            translations.get("upload", "Upload")
                        ], 
                        interactive=True, 
                        value=translations.get("download_url", "Download URL")
                    )

                with gr.Row():
                    gr.Markdown("___")

                # URL download
                with gr.Column():
                    with gr.Row():
                        download_model_url = gr.Textbox(
                            label=translations.get("model_url", "Model URL"), 
                            value="", 
                            placeholder="https://...", 
                            scale=6
                        )
                        download_model_name = gr.Textbox(
                            label=translations.get("modelname", "Model Name"), 
                            value="", 
                            placeholder=translations.get("modelname", "Model Name"), 
                            scale=2
                        )
                    download_url_button = gr.Button(
                        value=translations.get("downloads", "Download"), 
                        scale=2
                    )

                # CSV download
                with gr.Column():
                    download_csv_model = gr.Dropdown(
                        choices=models.keys(), 
                        label=translations.get("model_warehouse", "Model Warehouse"), 
                        scale=8, 
                        allow_custom_value=True, 
                        visible=False
                    )
                    download_csv_button = gr.Button(
                        value=translations.get("get_model", "Get Model"), 
                        scale=2, 
                        variant="primary", 
                        visible=False
                    )

                # Search models
                with gr.Column():
                    search_model_name = gr.Textbox(
                        label=translations.get("name_to_search", "Name to Search"), 
                        placeholder=translations.get("modelname", "Model Name"), 
                        interactive=True, 
                        scale=8, 
                        visible=False
                    )
                    search_button = gr.Button(
                        translations.get("search_2", "Search"), 
                        scale=2, 
                        visible=False
                    )
                    search_dropdown = gr.Dropdown(
                        label=translations.get("select_download_model", "Select Model to Download"), 
                        value="", 
                        choices=[], 
                        allow_custom_value=True, 
                        interactive=False, 
                        visible=False
                    )
                    search_download_model = gr.Button(
                        translations.get("downloads", "Download"), 
                        variant="primary", 
                        visible=False
                    )

                # Upload
                with gr.Column():
                    upload_model_files = gr.Files(
                        label=translations.get("drop_model", "Drop Model Files"), 
                        file_types=[".pth", ".onnx", ".index", ".zip"], 
                        visible=False
                    )

        # ── Pretrained Model Download Section ──
        with gr.Row():
            with gr.Accordion(
                translations.get("download_pretrained_2", "Download Pretrained Model"), 
                open=False
            ):
                with gr.Row():
                    download_pretrain_method = gr.Radio(
                        label=translations.get("model_download_select", "Download Method"), 
                        choices=[
                            translations.get("download_url", "Download URL"), 
                            translations.get("list_model", "List Model"), 
                            translations.get("upload", "Upload")
                        ], 
                        value=translations.get("download_url", "Download URL"), 
                        interactive=True
                    )

                with gr.Row():
                    gr.Markdown("___")

                # URL-based pretrained download
                with gr.Column():
                    with gr.Row():
                        download_pretrain_disc_url = gr.Textbox(
                            label=translations.get("pretrained_url", "Pretrained URL (D)").format(dg="D"), 
                            value="", 
                            placeholder="https://...", 
                            interactive=True, 
                            scale=4
                        )
                        download_pretrain_gen_url = gr.Textbox(
                            label=translations.get("pretrained_url", "Pretrained URL (G)").format(dg="G"), 
                            value="", 
                            placeholder="https://...", 
                            interactive=True, 
                            scale=4
                        )
                    download_pretrain_button = gr.Button(
                        translations.get("downloads", "Download"), 
                        scale=2,
                        visible=True
                    )

                # List-based pretrained download
                with gr.Column():
                    with gr.Row():
                        pretrained_data = fetch_pretrained_data()
                        download_pretrain_choices = gr.Dropdown(
                            label=translations.get("select_pretrain", "Select Pretrained Model"), 
                            info=translations.get("select_pretrain_info", "Select a pretrained base model"), 
                            choices=list(pretrained_data.keys()), 
                            value="Titan_Medium", 
                            allow_custom_value=True, 
                            interactive=True, 
                            scale=6, 
                            visible=False
                        )
                        download_pretrain_sample_rate = gr.Dropdown(
                            label=translations.get("pretrain_sr", "Sample Rate"), 
                            info=translations.get("pretrain_sr_info", "Select sample rate"), 
                            choices=["48k", "40k", "32k"], 
                            value="48k", 
                            interactive=True, 
                            visible=False
                        )
                    download_pretrain_choices_button = gr.Button(
                        translations.get("downloads", "Download"), 
                        scale=2, 
                        variant="primary", 
                        visible=False
                    )

                # Upload pretrained
                with gr.Row():
                    upload_pretrains = gr.Files(
                        label=translations.get("drop_pretrain", "Drop Pretrained Files (D, G)").format(dg="G, D"), 
                        file_types=[".pth"], 
                        visible=False
                    )

        # ── Event Handlers: Model Download ──
        with gr.Row():
            download_url_button.click(
                fn=download_model, 
                inputs=[
                    download_model_url, 
                    download_model_name
                ], 
                outputs=[download_model_url],
                api_name="download_model"
            )

            download_csv_button.click(
                fn=lambda model: download_model(
                    models[model], 
                    model
                ), 
                inputs=[
                    download_csv_model
                ], 
                outputs=[
                    download_csv_model
                ],
                api_name="download_browser"
            )

        with gr.Row():
            download_method.change(
                fn=change_download_choices, 
                inputs=[
                    download_method
                ], 
                outputs=[
                    download_model_url, 
                    download_model_name, 
                    download_url_button, 
                    download_csv_model, 
                    download_csv_button, 
                    search_model_name, 
                    search_button, 
                    search_dropdown, 
                    search_download_model, 
                    upload_model_files
                ]
            )

            search_button.click(
                fn=search_models, 
                inputs=[
                    search_model_name
                ], 
                outputs=[
                    search_dropdown, 
                    search_download_model
                ]
            )

        with gr.Row():
            upload_model_files.upload(
                fn=save_drop_model, 
                inputs=[
                    upload_model_files
                ], 
                outputs=[
                    upload_model_files
                ]
            )

            search_download_model.click(
                fn=lambda model: download_model(
                    model_options[model], 
                    model
                ), 
                inputs=[
                    search_dropdown
                ], 
                outputs=[
                    search_dropdown
                ],
                api_name="search_models"
            )

        # ── Event Handlers: Pretrained Model Download ──
        with gr.Row():
            download_pretrain_method.change(
                fn=change_download_pretrained_choices, 
                inputs=[
                    download_pretrain_method
                ], 
                outputs=[
                    download_pretrain_disc_url, 
                    download_pretrain_gen_url, 
                    download_pretrain_button, 
                    download_pretrain_choices, 
                    download_pretrain_sample_rate, 
                    download_pretrain_choices_button, 
                    upload_pretrains
                ]
            )

            download_pretrain_choices.change(
                fn=update_sample_rate_dropdown, 
                inputs=[
                    download_pretrain_choices
                ], 
                outputs=[
                    download_pretrain_sample_rate
                ]
            )

        with gr.Row():
            download_pretrain_button.click(
                fn=download_pretrained_model,
                inputs=[
                    download_pretrain_method, 
                    download_pretrain_disc_url, 
                    download_pretrain_gen_url
                ],
                outputs=[
                    download_pretrain_disc_url, 
                    download_pretrain_gen_url
                ],
                api_name="download_pretrain_link"
            )

            download_pretrain_choices_button.click(
                fn=download_pretrained_model,
                inputs=[
                    download_pretrain_method, 
                    download_pretrain_choices, 
                    download_pretrain_sample_rate
                ],
                outputs=[
                    download_pretrain_choices
                ],
                api_name="download_pretrain_choices"
            )

            upload_pretrains.upload(
                fn=lambda upload_pretrains: [
                    shutil_move(pretrain.name, configs["pretrained_custom_path"]) 
                    for pretrain in upload_pretrains
                ], 
                inputs=[
                    upload_pretrains
                ], 
                outputs=[],
                api_name="upload_pretrain"
            )

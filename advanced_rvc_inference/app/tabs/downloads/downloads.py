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
    with gr.TabItem(
        translations["downloads"], 
        visible=configs.get("downloads_tab", True)
    ):
        gr.Markdown(translations["download_markdown"])
        with gr.Row():
            gr.Markdown(translations["download_markdown_2"])
        with gr.Row():
            with gr.Accordion(
                translations["model_download"], 
                open=True
            ):
                with gr.Row():
                    download_method = gr.Radio(
                        label=translations["model_download_select"], 
                        choices=[
                            translations["download_url"], 
                            translations["download_from_csv"], 
                            translations["search_models"], 
                            translations["upload"]
                        ], 
                        interactive=True, 
                        value=translations["download_url"]
                    )
                with gr.Row():
                    gr.Markdown("___")
                with gr.Column():
                    with gr.Row():
                        download_model_url = gr.Textbox(
                            label=translations["model_url"], 
                            value="", 
                            placeholder="https://...", 
                            scale=6
                        )
                        download_model_name = gr.Textbox(
                            label=translations["modelname"], 
                            value="", 
                            placeholder=translations["modelname"], 
                            scale=2
                        )
                    download_url_button = gr.Button(
                        value=translations["downloads"], 
                        scale=2
                    )
                with gr.Column():
                    download_csv_model = gr.Dropdown(
                        choices=models.keys(), 
                        label=translations["model_warehouse"], 
                        scale=8, 
                        allow_custom_value=True, 
                        visible=False
                    )
                    download_csv_button = gr.Button(
                        value=translations["get_model"], 
                        scale=2, 
                        variant="primary", 
                        visible=False
                    )
                with gr.Column():
                    search_model_name = gr.Textbox(
                        label=translations["name_to_search"], 
                        placeholder=translations["modelname"], 
                        interactive=True, 
                        scale=8, 
                        visible=False
                    )
                    search_button = gr.Button(
                        translations["search_2"], 
                        scale=2, 
                        visible=False
                    )
                    search_dropdown = gr.Dropdown(
                        label=translations["select_download_model"], 
                        value="", 
                        choices=[], 
                        allow_custom_value=True, 
                        interactive=False, 
                        visible=False
                    )
                    search_download_model = gr.Button(
                        translations["downloads"], 
                        variant="primary", 
                        visible=False
                    )
                with gr.Column():
                    upload_model_files = gr.Files(
                        label=translations["drop_model"], 
                        file_types=[".pth", ".onnx", ".index", ".zip"], 
                        visible=False
                    )
        with gr.Row():
            with gr.Accordion(
                translations["download_pretrained_2"], 
                open=False
            ):
                with gr.Row():
                    download_pretrain_method = gr.Radio(
                        label=translations["model_download_select"], 
                        choices=[
                            translations["download_url"], 
                            translations["list_model"], 
                            translations["upload"]
                        ], 
                        value=translations["download_url"], 
                        interactive=True
                    )  
                with gr.Row():
                    gr.Markdown("___")
                with gr.Column():
                    with gr.Row():
                        download_pretrain_disc_url = gr.Textbox(
                            label=translations["pretrained_url"].format(dg="D"), 
                            value="", 
                            placeholder="https://...", 
                            interactive=True, 
                            scale=4
                        )
                        download_pretrain_gen_url = gr.Textbox(
                            label=translations["pretrained_url"].format(dg="G"), 
                            value="", 
                            placeholder="https://...", 
                            interactive=True, 
                            scale=4
                        )
                    download_pretrain_button = gr.Button(
                        translations["downloads"], 
                        scale=2,
                        visible=True
                    )
                with gr.Column():
                    with gr.Row():
                        download_pretrain_choices = gr.Dropdown(
                            label=translations["select_pretrain"], 
                            info=translations["select_pretrain_info"], 
                            choices=list(fetch_pretrained_data().keys()), 
                            value="Titan_Medium", 
                            allow_custom_value=True, 
                            interactive=True, 
                            scale=6, 
                            visible=False
                        )
                        download_pretrain_sample_rate = gr.Dropdown(
                            label=translations["pretrain_sr"], 
                            info=translations["pretrain_sr"], 
                            choices=["48k", "40k", "32k"], 
                            value="48k", 
                            interactive=True, 
                            visible=False
                        )
                    download_pretrain_choices_button = gr.Button(
                        translations["downloads"], 
                        scale=2, 
                        variant="primary", 
                        visible=False
                    )
                with gr.Row():
                    upload_pretrains = gr.Files(
                        label=translations["drop_pretrain"].format(dg="G, D"), 
                        file_types=[".pth"], 
                        visible=False
                    )
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

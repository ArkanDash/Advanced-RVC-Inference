"""
Downloads tab for the Advanced RVC Inference Gradio interface.

Provides model download, audio download, and model search functionality.
Pretrained model download UI ported from Vietnamese-RVC.
"""

import os

import gradio as gr

from arvc.utils.variables import translations, configs, model_options
from arvc.services.downloads import (
    download_model,
    download_url,
    download_pretrained_model,
    search_models,
)
from arvc.services.process import fetch_pretrained_data, update_sample_rate_dropdown
from arvc.ui.feedback import (
    change_models_choices,
    change_download_pretrained_choices,
    gr_info,
    gr_warning,
)


def download_tab():
    """Build the Downloads tab UI for the full Gradio interface."""
    with gr.TabItem(
        translations.get("downloads", "Downloads"),
        visible=configs.get("downloads_tab", True),
    ):
        # ── Download Model Section ──
        with gr.TabItem(
            translations.get("download", "Download Model"),
            visible=configs.get("download_tab", True),
        ):
            gr.Markdown(translations.get("download_markdown", ""))

            with gr.Row():
                link = gr.Textbox(
                    label=translations.get("download_model_link", "Model URL"),
                    placeholder="https://huggingface.co/...",
                    interactive=True,
                    scale=4,
                )
                model_name = gr.Textbox(
                    label=translations.get("model_name", "Model Name"),
                    placeholder=translations.get("model_name_placeholder", "(optional)"),
                    interactive=True,
                    scale=2,
                )

            with gr.Row():
                download_model_btn = gr.Button(
                    translations.get("download", "Download"),
                    variant="primary",
                )
                refresh_models_btn = gr.Button(
                    translations.get("refresh", "Refresh Models"),
                )

            download_model_output = gr.Textbox(
                label=translations.get("status", "Status"),
                interactive=False,
                lines=3,
            )

            # ── Search Models Section ──
            with gr.Accordion(
                translations.get("search_models", "Search Models"),
                open=False,
            ):
                with gr.Row():
                    search_name = gr.Textbox(
                        label=translations.get("search_name", "Model Name"),
                        placeholder=translations.get("search_placeholder", "Enter model name to search..."),
                        interactive=True,
                        scale=4,
                    )
                    search_btn = gr.Button(
                        translations.get("search", "Search"),
                        variant="primary",
                    )

                search_results = gr.Dropdown(
                    label=translations.get("search_results", "Search Results"),
                    choices=[],
                    interactive=True,
                    allow_custom_value=True,
                )
                download_search_btn = gr.Button(
                    translations.get("download", "Download"),
                )
                download_search_output = gr.Textbox(
                    label=translations.get("status", "Status"),
                    interactive=False,
                    lines=3,
                )

        # ── Download Audio Section ──
        with gr.TabItem(
            translations.get("download_music", "Download Audio"),
            visible=configs.get("downloads_tab", True),
        ):
            gr.Markdown(translations.get("download_music_markdown", ""))

            with gr.Row():
                audio_link = gr.Textbox(
                    label=translations.get("audio_url", "Audio URL (YouTube, etc.)"),
                    placeholder="https://www.youtube.com/watch?v=...",
                    interactive=True,
                    scale=5,
                )
                download_audio_btn = gr.Button(
                    translations.get("download_music", "Download Audio"),
                    variant="primary",
                )

            download_audio_output = gr.Audio(
                label=translations.get("output_audio", "Output Audio"),
                type="filepath",
                interactive=False,
            )
            download_audio_path = gr.Textbox(
                label=translations.get("output_path", "Output Path"),
                interactive=False,
            )
            download_audio_status = gr.Textbox(
                label=translations.get("status", "Status"),
                interactive=False,
            )

        # ── Download Pretrained Section (Vietnamese-RVC style) ──
        with gr.TabItem(
            translations.get("download_pretrain", "Download Pretrained"),
            visible=configs.get("downloads_tab", True),
        ):
            gr.Markdown(translations.get("download_pretrain_markdown", ""))

            with gr.Row():
                pretrained_choice = gr.Radio(
                    label=translations.get("model_download_select", "Source"),
                    choices=[
                        translations.get("download_url", "URL"),
                        translations.get("list_model", "List Model"),
                        translations.get("ultimate_rvc_models", "Ultimate RVC Models"),
                        translations.get("upload", "Upload"),
                    ],
                    value=translations.get("download_url", "URL"),
                    interactive=True,
                )

            # ── URL mode: D URL + G URL textboxes ──
            with gr.Column():
                pretrained_d_url = gr.Textbox(
                    label=translations.get("pretrained_url", "D Model URL").format(dg="D") if isinstance(translations.get("pretrained_url", ""), str) and "{dg}" in translations.get("pretrained_url", "") else translations.get("pretrained_url", "D Model URL"),
                    placeholder="https://huggingface.co/.../D_48k.pth",
                    interactive=True,
                    visible=True,
                )
                pretrained_g_url = gr.Textbox(
                    label=translations.get("pretrained_url", "G Model URL").format(dg="G") if isinstance(translations.get("pretrained_url", ""), str) and "{dg}" in translations.get("pretrained_url", "") else translations.get("pretrained_url", "G Model URL"),
                    placeholder="https://huggingface.co/.../G_48k.pth",
                    interactive=True,
                    visible=True,
                )
                pretrained_url_btn = gr.Button(
                    translations.get("download_pretrain", "Download Pretrained"),
                    variant="primary",
                    visible=True,
                )

            # ── List Model mode: Dropdown from HF JSON + dynamic sample rate ──
            with gr.Column():
                pretrained_list = gr.Dropdown(
                    label=translations.get("select_pretrain", "Select Pretrained Model"),
                    choices=list(fetch_pretrained_data().keys()),
                    value="Titan_Medium" if "Titan_Medium" in fetch_pretrained_data() else None,
                    allow_custom_value=True,
                    interactive=True,
                    visible=False,
                )
                pretrained_sr = gr.Dropdown(
                    label=translations.get("pretrain_sr", "Sample Rate"),
                    choices=["48k", "40k", "32k"],
                    value="48k",
                    interactive=True,
                    visible=False,
                )
                pretrained_list_btn = gr.Button(
                    translations.get("download_pretrain", "Download Pretrained"),
                    variant="primary",
                    visible=False,
                )

            # ── Ultimate RVC Models mode: D/G filename from R-Kentaren repo ──
            with gr.Column():
                alt_base_url = configs.get("alternative_pretrained_url", "https://huggingface.co/buckets/R-Kentaren/Ultimate-RVC-Models/resolve/main/")
                gr.Markdown(
                    f"**Source:** [{alt_base_url}]({alt_base_url})\n\n"
                    "Enter the D and G model filenames or relative paths from this repo "
                    "(e.g. `D_48k.pth` and `G_48k.pth`). Full URLs are also accepted."
                )
                ultimate_d_file = gr.Textbox(
                    label=translations.get("pretrained_url", "D Model Filename/URL").format(dg="D") if isinstance(translations.get("pretrained_url", ""), str) and "{dg}" in translations.get("pretrained_url", "") else "D Model Filename/URL",
                    placeholder="D_48k.pth or full URL",
                    interactive=True,
                    visible=False,
                )
                ultimate_g_file = gr.Textbox(
                    label=translations.get("pretrained_url", "G Model Filename/URL").format(dg="G") if isinstance(translations.get("pretrained_url", ""), str) and "{dg}" in translations.get("pretrained_url", "") else "G Model Filename/URL",
                    placeholder="G_48k.pth or full URL",
                    interactive=True,
                    visible=False,
                )
                ultimate_btn = gr.Button(
                    translations.get("download_pretrain", "Download Pretrained"),
                    variant="primary",
                    visible=False,
                )

            # ── Upload mode: .pth file upload ──
            with gr.Row():
                upload_pretrains = gr.Files(
                    label=translations.get("drop_pretrain", "Drop G & D pretrained files (.pth)").format(dg="G, D") if isinstance(translations.get("drop_pretrain", ""), str) and "{dg}" in translations.get("drop_pretrain", "") else translations.get("drop_pretrain", "Drop G & D pretrained files (.pth)"),
                    file_types=[".pth"],
                    visible=False,
                )

            pretrained_output = gr.Textbox(
                label=translations.get("status", "Status"),
                interactive=False,
                lines=3,
            )

        # ── Event bindings ──

        # Download model from URL
        download_model_btn.click(
            fn=download_model,
            inputs=[link, model_name],
            outputs=[download_model_output],
        )

        # Refresh models list
        refresh_models_btn.click(
            fn=change_models_choices,
            inputs=[],
            outputs=[download_model_output],
        )

        # Search models
        search_btn.click(
            fn=search_models,
            inputs=[search_name],
            outputs=[search_results, download_search_btn],
        )

        # Download from search results
        def _download_from_search(selected_model):
            if not selected_model or selected_model not in model_options:
                gr_warning(translations.get("select_model", "Please select a model."))
                return translations.get("select_model", "Please select a model.")
            url = model_options[selected_model]
            return download_model(url=url, model=None)

        download_search_btn.click(
            fn=_download_from_search,
            inputs=[search_results],
            outputs=[download_search_output],
        )

        # Download audio from URL
        download_audio_btn.click(
            fn=download_url,
            inputs=[audio_link],
            outputs=[download_audio_output, download_audio_path, download_audio_status],
        )

        # ── Pretrained: visibility toggle when source radio changes ──
        pretrained_choice.change(
            fn=change_download_pretrained_choices,
            inputs=[pretrained_choice],
            outputs=[
                pretrained_d_url,       # [0] D URL textbox
                pretrained_g_url,       # [1] G URL textbox
                pretrained_url_btn,     # [2] URL download button
                pretrained_list,        # [3] List model dropdown
                pretrained_sr,          # [4] Sample rate dropdown
                pretrained_list_btn,    # [5] List download button
                upload_pretrains,       # [6] Upload files
                ultimate_d_file,        # [7] Ultimate D file textbox
                ultimate_g_file,        # [8] Ultimate G file textbox
                ultimate_btn,           # [9] Ultimate download button
            ],
        )

        # ── Pretrained: update sample rate dropdown when model changes ──
        pretrained_list.change(
            fn=update_sample_rate_dropdown,
            inputs=[pretrained_list],
            outputs=[pretrained_sr],
        )

        # ── Pretrained: download from URL ──
        pretrained_url_btn.click(
            fn=download_pretrained_model,
            inputs=[pretrained_choice, pretrained_d_url, pretrained_g_url],
            outputs=[pretrained_output],
        )

        # ── Pretrained: download from list selection ──
        pretrained_list_btn.click(
            fn=download_pretrained_model,
            inputs=[pretrained_choice, pretrained_list, pretrained_sr],
            outputs=[pretrained_output],
        )

        # ── Pretrained: download from Ultimate RVC Models ──
        ultimate_btn.click(
            fn=download_pretrained_model,
            inputs=[pretrained_choice, ultimate_d_file, ultimate_g_file],
            outputs=[pretrained_output],
        )

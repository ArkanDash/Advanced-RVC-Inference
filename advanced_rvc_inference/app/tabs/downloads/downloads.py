"""
Downloads tab for the Advanced RVC Inference Gradio interface.

Provides model download, audio download, and model search functionality.
"""

import os

import gradio as gr

from advanced_rvc_inference.utils.variables import translations, configs, model_options
from advanced_rvc_inference.services.downloads import (
    download_model,
    download_url,
    download_pretrained_model,
    search_models,
)
from advanced_rvc_inference.ui.feedback import (
    change_models_choices,
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

        # ── Download Pretrained Section ──
        with gr.TabItem(
            translations.get("download_pretrain", "Download Pretrained"),
            visible=configs.get("downloads_tab", True),
        ):
            gr.Markdown(translations.get("download_pretrain_markdown", ""))

            with gr.Row():
                pretrained_choice = gr.Radio(
                    label=translations.get("pretrained_source", "Source"),
                    choices=[
                        translations.get("list_model", "List"),
                        translations.get("download_url", "URL"),
                    ],
                    value=translations.get("list_model", "List"),
                    interactive=True,
                )

            with gr.Row():
                pretrained_model = gr.Textbox(
                    label=translations.get("pretrained_model", "Model Name / D URL"),
                    interactive=True,
                )
                pretrained_sr = gr.Textbox(
                    label=translations.get("pretrained_sr", "Sample Rate / G URL"),
                    interactive=True,
                )

            pretrained_btn = gr.Button(
                translations.get("download_pretrain", "Download Pretrained"),
                variant="primary",
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

        # Download pretrained model
        pretrained_btn.click(
            fn=download_pretrained_model,
            inputs=[pretrained_choice, pretrained_model, pretrained_sr],
            outputs=[pretrained_output],
        )

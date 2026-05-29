"""
Downloads tab for the Advanced RVC Inference GUI.

Provides UI for downloading RVC voice models, pretrained models,
and audio from URLs. Model download and audio download are in
separate tabs.
"""

import os

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
from arvc.services.process import fetch_pretrained_data, update_sample_rate_dropdown


def download_tab():
    """Create the Downloads tab UI."""
    with gr.TabItem(translations["downloads"], visible=configs.get("downloads_tab", True)):
        # ════════════════════════════════════════════════════════════
        # Tab 1: Download Model
        # ════════════════════════════════════════════════════════════
        with gr.TabItem(translations.get("download_model", "Download Model")):
            gr.Markdown(translations.get("download_markdown", "## Download Model"))

            # ── Download mode selector ──
            download_select = gr.Dropdown(
                choices=[
                    translations.get("download_url", "Download from URL"),
                    translations.get("download_from_csv", "Download from CSV"),
                    translations.get("search_models", "Search Models"),
                    translations.get("upload", "Upload"),
                ],
                value=translations.get("download_url", "Download from URL"),
                label=translations.get("model_download_select", "Choose a model download method"),
                interactive=True,
            )

            # ── URL download section ──
            with gr.Row(visible=True) as url_row:
                link = gr.Textbox(
                    label=translations.get("model_url", "Model URL"),
                    placeholder="https://huggingface.co/...",
                    interactive=True,
                    scale=4,
                )
                model_name_input = gr.Textbox(
                    label=translations.get("model_name", "Model Name"),
                    placeholder=translations.get("model_name_placeholder", "(optional)"),
                    interactive=True,
                    scale=2,
                )

            with gr.Row(visible=True) as url_btn_row:
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

            # ── CSV / Model List section ──
            with gr.Row(visible=False) as csv_row:
                csv_dropdown = gr.Dropdown(
                    choices=model_name if isinstance(model_name, list) else list(model_name.keys()) if model_name else [],
                    label=translations.get("select_model", "Select Model"),
                    interactive=True,
                    allow_custom_value=True,
                    scale=4,
                )
                csv_download_btn = gr.Button(
                    translations.get("download", "Download"),
                    variant="primary",
                )

            csv_download_output = gr.Textbox(
                label=translations.get("status", "Status"),
                interactive=False,
                visible=False,
            )

            # ── Search Models section ──
            with gr.Accordion(
                translations.get("search_models", "Search Models"),
                open=False,
                visible=False,
            ) as search_accordion:
                with gr.Row():
                    search_name = gr.Textbox(
                        label=translations.get("search_model_name", "Model Name"),
                        placeholder=translations.get("search_placeholder", "Enter model name to search..."),
                        interactive=True,
                        scale=4,
                    )
                    search_btn = gr.Button(
                        translations.get("search", "Search"),
                        variant="primary",
                    )

                search_results = gr.Dropdown(
                    label=translations.get("select_download_model", "Choose a searched model (Click to select)"),
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

            # ── Upload section ──
            with gr.Row(visible=False) as upload_row:
                upload_model_files = gr.Files(
                    label=translations.get("upload_model", "Upload Model Files"),
                )

            # ══════════════════════════════════════════════════════════
            # Event Handlers — Download Model
            # ══════════════════════════════════════════════════════════

            # Mode selector: toggle visibility of sub-sections
            download_select.change(
                fn=change_download_choices,
                inputs=[download_select],
                outputs=[
                    link, model_name_input, download_model_btn,
                    refresh_models_btn,
                    csv_dropdown, csv_download_btn,
                    search_name, search_btn,
                    search_results, download_search_btn,
                    upload_model_files,
                ],
            )

            # Download model from URL
            download_model_btn.click(
                fn=download_model,
                inputs=[link, model_name_input],
                outputs=[download_model_output],
            )

            # Refresh models list — wrapper to handle 2-return into 1 output
            def _refresh_models():
                result = change_models_choices()
                # change_models_choices returns [model_update, index_update]
                # We only display the first as a status-like message
                status_msg = translations.get("success", "Success") if result and result[0].get("choices") else translations.get("not_found", "No models found")
                return status_msg

            refresh_models_btn.click(
                fn=_refresh_models,
                inputs=[],
                outputs=[download_model_output],
            )

            # Download from CSV / model list
            def _download_from_csv(selected_name):
                from arvc.utils.variables import models
                if not selected_name:
                    gr_warning(translations.get("select_model", "Please select a model."))
                    return translations.get("select_model", "Please select a model.")
                url = models.get(selected_name, "")
                if not url:
                    gr_warning(translations.get("not_found", "Not found"))
                    return translations.get("not_found", "Not found")
                return download_model(url=url, model=None)

            csv_download_btn.click(
                fn=_download_from_csv,
                inputs=[csv_dropdown],
                outputs=[csv_download_output],
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

        # ════════════════════════════════════════════════════════════
        # Tab 2: Download Audio (separate tab)
        # ════════════════════════════════════════════════════════════
        with gr.TabItem(translations.get("download_music", "Download Audio")):
            gr.Markdown(translations.get("download_music", "## Download Audio"))

            with gr.Row():
                audio_link = gr.Textbox(
                    label=translations.get("url_audio", "Audio URL"),
                    placeholder="https://www.youtube.com/watch?v=...",
                    interactive=True,
                    scale=5,
                )
                download_audio_btn = gr.Button(
                    translations.get("download_music", "Download Audio"),
                    variant="secondary",
                )

            download_audio_output = gr.Audio(
                label=translations.get("output_audio", "Output Audio"),
                type="filepath",
                interactive=False,
            )
            download_audio_path = gr.Textbox(
                label=translations.get("output_path", "Audio output path"),
                interactive=False,
            )
            download_audio_status = gr.Textbox(
                label=translations.get("status", "Status"),
                interactive=False,
            )

            # Download audio from URL
            download_audio_btn.click(
                fn=download_url,
                inputs=[audio_link],
                outputs=[download_audio_output, download_audio_path, download_audio_status],
            )

        # ════════════════════════════════════════════════════════════
        # Tab 3: Download Pretrained Models
        # ════════════════════════════════════════════════════════════
        with gr.TabItem(translations.get("download_pretrained", "Download Pre-trained")):
            gr.Markdown(translations.get("download_pretrained_2", "Download pre-trained model"))

            # ── Pretrained mode selector ──
            pretrained_choice = gr.Radio(
                label=translations.get("model_download_select", "Choose a model download method"),
                choices=[
                    translations.get("download_url", "Download from URL"),
                    translations.get("list_model", "Model list"),
                    translations.get("upload", "Upload"),
                ],
                value=translations.get("download_url", "Download from URL"),
                interactive=True,
            )

            # ── URL mode: D URL + G URL ──
            with gr.Column(visible=True) as pretrained_url_col:
                pretrained_d_url = gr.Textbox(
                    label=translations.get("provide_pretrain", "Please provide a pre-trained model url D").format(dg="D") if "{dg}" in translations.get("provide_pretrain", "") else "D Model URL",
                    placeholder="https://huggingface.co/...",
                    interactive=True,
                )
                pretrained_g_url = gr.Textbox(
                    label=translations.get("provide_pretrain", "Please provide a pre-trained model url G").format(dg="G") if "{dg}" in translations.get("provide_pretrain", "") else "G Model URL",
                    placeholder="https://huggingface.co/...",
                    interactive=True,
                )
                pretrained_url_btn = gr.Button(
                    translations.get("download_pretrained", "Download Pre-trained"),
                    variant="primary",
                )

            # ── List Model mode ──
            with gr.Column(visible=False) as pretrained_list_col:
                pretrained_list = gr.Dropdown(
                    label=translations.get("select_model", "Select Model"),
                    choices=list(fetch_pretrained_data().keys()),
                    value=None,
                    allow_custom_value=True,
                    interactive=True,
                )
                pretrained_sr = gr.Dropdown(
                    label=translations.get("sample_rate", "Sample rate"),
                    choices=["32000", "40000", "48000"],
                    value="40000",
                    interactive=True,
                )
                pretrained_list_btn = gr.Button(
                    translations.get("download_pretrained", "Download Pre-trained"),
                    variant="primary",
                )

            # ── Upload mode ──
            with gr.Row(visible=False) as pretrained_upload_col:
                upload_pretrains = gr.Files(
                    label=translations.get("drop_pretrain", "Drop G & D pretrained files (.pth)"),
                    file_types=[".pth"],
                )

            pretrained_output = gr.Textbox(
                label=translations.get("status", "Status"),
                interactive=False,
                lines=3,
            )

            # ══════════════════════════════════════════════════════════
            # Event Handlers — Pretrained
            # ══════════════════════════════════════════════════════════

            # Toggle visibility based on mode
            def _toggle_pretrained_mode(choice):
                url_vis = choice == translations.get("download_url", "Download from URL")
                list_vis = choice == translations.get("list_model", "Model list")
                upload_vis = choice == translations.get("upload", "Upload")
                return (
                    gr.update(visible=url_vis),   # pretrained_url_col
                    gr.update(visible=list_vis),   # pretrained_list_col
                    gr.update(visible=upload_vis), # pretrained_upload_col
                )

            pretrained_choice.change(
                fn=_toggle_pretrained_mode,
                inputs=[pretrained_choice],
                outputs=[pretrained_url_col, pretrained_list_col, pretrained_upload_col],
            )

            # Update sample rate dropdown when model changes
            pretrained_list.change(
                fn=update_sample_rate_dropdown,
                inputs=[pretrained_list],
                outputs=[pretrained_sr],
            )

            # Download pretrained from URL
            pretrained_url_btn.click(
                fn=download_pretrained_model,
                inputs=[pretrained_choice, pretrained_d_url, pretrained_g_url],
                outputs=[pretrained_output],
            )

            # Download pretrained from list
            pretrained_list_btn.click(
                fn=download_pretrained_model,
                inputs=[pretrained_choice, pretrained_list, pretrained_sr],
                outputs=[pretrained_output],
            )

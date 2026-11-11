"""
Downloads tab for the Advanced RVC Inference GUI.

Provides UI for downloading RVC voice models, pretrained models,
and audio from URLs.
"""

import os

import gradio as gr

from arvc.utils.variables import translations, configs, model_options, model_name
from arvc.utils.feedback import gr_warning, shutil_move
from arvc.uploader.huggingface import (
    list_local_training_models,
    list_model_files,
    hf_upload_training,
    hf_list_repo_files,
    hf_load_training,
)
from arvc.downloader.downloads import (
    download_url,
    download_model,
    download_pretrained_model,
    search_models,
)
from arvc.utils.process import update_sample_rate_dropdown


def download_tab():
    """Create the Downloads tab UI."""

    def change_download_mode(option):
        """Show only the download section matching the selected option."""
        return [
            gr.update(visible=option == translations.get("download_url", "Download from URL")),
            gr.update(visible=option == translations.get("download_from_csv", "Download from CSV")),
            gr.update(visible=option == translations.get("search_models", "Search Models")),
            gr.update(visible=False),
            gr.update(visible=option == translations.get("upload", "Upload")),
        ]

    def change_pretrained_mode(option):
        """Show only the pretrained download section matching the selected option."""
        return [
            gr.update(visible=option == translations.get("download_url", "Download from URL")),
            gr.update(visible=option == translations.get("list_model", "List Model")),
            gr.update(visible=option == translations.get("upload", "Upload")),
        ]

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
                    choices=list(model_options.keys()) if isinstance(model_options, dict) and model_options else (model_name if model_name else []),
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

            def upload_model_files_handler(files):
                """Move uploaded model files into the weights directory."""
                if not files:
                    return None
                moved = []
                for f in files:
                    try:
                        moved.append(shutil_move(f.name, configs["weights_path"]))
                    except Exception as e:
                        gr_warning(f"Could not move {f.name}: {e}")
                return "\n".join(moved) if moved else None

            # ── Event Handlers ──
            download_select.change(
                fn=change_download_mode,
                inputs=[download_select],
                outputs=[
                    url_row,
                    csv_row,
                    search_row,
                    search_result_row,
                    upload_row,
                ],
            )

            download_csv_button.click(
                fn=lambda selection: download_model(url=model_options.get(selection, ""), model=selection) if selection else None,
                inputs=[download_csv_model],
                outputs=[download_status],
            )

            upload_model_files.upload(
                fn=upload_model_files_handler,
                inputs=[upload_model_files],
                outputs=[download_status],
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
                    choices=[],
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

            def upload_pretrained_files_handler(files):
                """Move uploaded pretrained model files into the pretrained directory."""
                if not files:
                    return None
                moved = []
                for f in files:
                    try:
                        moved.append(shutil_move(f.name, configs["pretrained_custom_path"]))
                    except Exception as e:
                        gr_warning(f"Could not move {f.name}: {e}")
                return "\n".join(moved) if moved else None

            # Event handlers
            pretrained_select.change(
                fn=change_pretrained_mode,
                inputs=[pretrained_select],
                outputs=[
                    pretrained_url_row,
                    pretrained_list_row,
                    pretrained_upload_row,
                ],
            )

            # When a pretrained model is selected from list, update sample rate choices
            pretrained_list_model.change(
                fn=update_sample_rate_dropdown,
                inputs=[pretrained_list_model],
                outputs=[pretrained_sample_rate],
            )

            pretrained_upload_files.upload(
                fn=upload_pretrained_files_handler,
                inputs=[pretrained_upload_files],
                outputs=[pretrained_status],
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

        # ── Download Audio ──
        with gr.TabItem(translations.get("download_audio", "Download Audio")):
            gr.Markdown(f"## {translations.get('download_audio', 'Download Audio')}")

            with gr.Row():
                audio_url = gr.Textbox(
                    label=translations.get("audio_url", "Audio URL (YouTube, etc.)"),
                    placeholder="https://www.youtube.com/watch?v=...",
                    interactive=True,
                )
                audio_download_button = gr.Button(
                    translations.get("download", "Download"),
                    variant="primary",
                )

            audio_output_path = gr.Textbox(
                label=translations.get("output_path", "Output Path"),
                interactive=False,
            )
            audio_player = gr.Audio(
                label=translations.get("audio_output", "Audio Output"),
                type="filepath",
            )
            audio_status = gr.Textbox(
                label=translations.get("status", "Status"),
                interactive=False,
            )

            audio_download_button.click(
                fn=download_url,
                inputs=[audio_url],
                outputs=[audio_output_path, audio_player, audio_status],
            )

        # ── HuggingFace Hub (Upload & Load Training Results) ──
        with gr.TabItem(translations.get("hf_hub_tab", "HuggingFace Hub")):
            gr.Markdown(f"## {translations.get('hf_hub_markdown', 'Upload & Load Training Results')}")

            with gr.Row():
                hf_token = gr.Textbox(
                    label=translations.get("hf_token", "HuggingFace Token"),
                    info=translations.get(
                        "hf_token_info",
                        "Access token with write permission. Leave empty to use the locally logged-in account (huggingface-cli login).",
                    ),
                    type="password",
                    interactive=True,
                )

            # ── Upload panel ──
            with gr.TabItem(translations.get("hf_upload", "Upload")):
                gr.Markdown(
                    f"### {translations.get('hf_upload_markdown', 'Upload training results to HuggingFace')}\n"
                    f"{translations.get('hf_upload_structure_info', 'Files are uploaded under a folder named after the model: [Model Name] -> training files.')}"
                )
                with gr.Row():
                    hf_model_select = gr.Dropdown(
                        choices=list_local_training_models(),
                        label=translations.get("hf_select_model", "Select Trained Model"),
                        interactive=True,
                    )
                    hf_scan_button = gr.Button(
                        translations.get("hf_scan_files", "Scan Files"),
                        variant="secondary",
                    )

                hf_files_select = gr.Dropdown(
                    choices=[],
                    multiselect=True,
                    label=translations.get("hf_files", "Training Files"),
                    info=translations.get(
                        "hf_files_info",
                        "Click 'Scan Files' to list the model's training files, then pick what to upload.",
                    ),
                    interactive=True,
                )

                with gr.Row():
                    hf_repo_upload = gr.Textbox(
                        label=translations.get("hf_repo", "HuggingFace Repository"),
                        placeholder="username/my-model",
                        info=translations.get(
                            "hf_repo_info",
                            "Leave empty to upload to <your-username>/<model-name>.",
                        ),
                        interactive=True,
                    )
                    hf_private = gr.Checkbox(
                        label=translations.get("hf_private", "Private repository"),
                        value=False,
                        interactive=True,
                    )

                hf_upload_button = gr.Button(
                    translations.get("hf_upload", "Upload"),
                    variant="primary",
                )
                hf_upload_status = gr.Textbox(
                    label=translations.get("status", "Status"),
                    interactive=False,
                )

            # ── Load panel ──
            with gr.TabItem(translations.get("hf_load", "Load")):
                gr.Markdown(
                    f"### {translations.get('hf_load_markdown', 'Load training results from HuggingFace')}\n"
                    f"{translations.get('hf_load_structure_info', 'Files are restored into your local logs folder under the model name, ready to use or resume training.')}"
                )
                with gr.Row():
                    hf_repo_load = gr.Textbox(
                        label=translations.get("hf_repo", "HuggingFace Repository"),
                        placeholder="username/my-model",
                        interactive=True,
                    )
                    hf_list_button = gr.Button(
                        translations.get("hf_list_files", "List Repository Files"),
                        variant="secondary",
                    )

                hf_repo_files = gr.Dropdown(
                    choices=[],
                    multiselect=True,
                    label=translations.get("hf_files", "Training Files"),
                    info=translations.get(
                        "hf_files_info",
                        "Click 'List Repository Files' to browse the repository, then pick what to load.",
                    ),
                    interactive=True,
                )

                hf_dest_model = gr.Textbox(
                    label=translations.get("hf_dest_model", "Destination Model Name"),
                    placeholder="my-model",
                    info=translations.get(
                        "hf_dest_model_info",
                        "Local folder name for the loaded files. Defaults to the repository name.",
                    ),
                    interactive=True,
                )

                hf_load_button = gr.Button(
                    translations.get("hf_load", "Load"),
                    variant="primary",
                )
                hf_load_status = gr.Textbox(
                    label=translations.get("status", "Status"),
                    interactive=False,
                )

            # Event handlers
            def refresh_model_files(model):
                """List a local model's training files and preselect all of them."""
                files = list_model_files(model) if model else []
                return gr.update(choices=files, value=files)

            def refresh_repo_files(repo, token):
                """List a repository's files and preselect all downloadable ones."""
                files, err = hf_list_repo_files(repo, token)
                if err is not None:
                    return gr.update(choices=[], value=[]), gr.update()
                downloadable = [f for f in files if os.path.basename(f) != "README.md" and not f.startswith(".")]
                dest = repo.strip().strip("/").split("/")[-1] if repo and "/" in repo.strip().strip("/") else ""
                return gr.update(choices=downloadable, value=downloadable), gr.update(value=dest)

            hf_scan_button.click(
                fn=refresh_model_files,
                inputs=[hf_model_select],
                outputs=[hf_files_select],
            )

            hf_upload_button.click(
                fn=hf_upload_training,
                inputs=[hf_model_select, hf_files_select, hf_token, hf_repo_upload, hf_private],
                outputs=[hf_upload_status],
            )

            hf_list_button.click(
                fn=refresh_repo_files,
                inputs=[hf_repo_load, hf_token],
                outputs=[hf_repo_files, hf_dest_model],
            )

            hf_load_button.click(
                fn=hf_load_training,
                inputs=[hf_repo_load, hf_token, hf_dest_model, hf_repo_files],
                outputs=[hf_load_status],
            )

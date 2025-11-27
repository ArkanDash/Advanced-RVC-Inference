import gradio as gr
import os, sys
from ...lib.i18n import I18nAuto
from ...lib.path_manager import path

i18n = I18nAuto()

now_dir = os.getcwd()
sys.path.append(now_dir)

# Model root directory - RVC models are stored in logs directory with subdirectories per model
model_root = str(path('logs_dir'))

def get_models_list():
    """Get list of available models"""
    models = []
    if os.path.exists(model_root):
        # RVC stores models in subdirectories under logs directory
        for model_dir_name in os.listdir(model_root):
            model_dir_path = os.path.join(model_root, model_dir_name)
            if os.path.isdir(model_dir_path):
                # Look for .pth or .onnx files in the model subdirectory
                for file in os.listdir(model_dir_path):
                    if file.endswith(('.pth', '.onnx')):
                        # Extract model name from the file
                        model_name = os.path.splitext(file)[0]  # Remove extension
                        # Check if there's a corresponding index file in the same directory
                        index_file = None
                        for idx_file in os.listdir(model_dir_path):
                            if idx_file.endswith('.index') and model_name in idx_file:
                                index_file = idx_file
                                break
                        models.append([model_name, os.path.join(model_dir_name, file), index_file or "No index", "Available"])
    return models or [["No models found", "", "", ""]]

def model_manager_tab():
    with gr.Column():
        gr.Markdown("## 🗂️ Model Manager")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Available Models")
                model_list = gr.Dataframe(
                    headers=["Model Name", "Model File", "Index File", "Status"],
                    datatype=["str", "str", "str", "str"],
                    value=get_models_list(),
                    interactive=False,
                    elem_id="model_list"
                )

            with gr.Column():
                gr.Markdown("### Model Operations")

                with gr.Row():
                    refresh_models = gr.Button(i18n("Refresh Models"), variant="secondary")
                    delete_model = gr.Button(i18n("Delete Selected"), variant="stop")

                with gr.Row():
                    model_search = gr.Textbox(
                        label=i18n("Search Models"),
                        placeholder=i18n("Enter model name...")
                    )

                with gr.Row():
                    model_fusion = gr.Button(i18n("Fuse Models"), variant="primary")
                    model_conversion = gr.Button(i18n("Convert Format"), variant="primary")

                with gr.Accordion("Model Details", open=False):
                    model_info = gr.Textbox(
                        label=i18n("Model Information"),
                        interactive=False,
                        lines=8
                    )

        with gr.Row():
            gr.Markdown("### Model Fusion")
            with gr.Column():
                model_1 = gr.Dropdown(
                    label="First Model",
                    choices=[],  # Will be populated dynamically
                    info="Select the first model for fusion"
                )
                model_2 = gr.Dropdown(
                    label="Second Model",
                    choices=[],  # Will be populated dynamically
                    info="Select the second model for fusion"
                )
                fusion_ratio = gr.Slider(
                    label="Fusion Ratio",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    value=0.5,
                    info="Ratio of first model to second model (0.0 = all first, 1.0 = all second)"
                )
                fuse_btn = gr.Button("Fuse Models", variant="primary")

        with gr.Row():
            gr.Markdown("### Online Model Hub")
            with gr.Column():
                hub_search = gr.Textbox(
                    label=i18n("Search Online"),
                    placeholder=i18n("Search for models online...")
                )

                hub_search_btn = gr.Button("Search", variant="secondary")

                hub_results = gr.Dataframe(
                    headers=["Model Name", "Author", "Version", "Sample Rate", "Rating", "Download"],
                    datatype=["str", "str", "str", "str", "number", "str"],
                    value=[],
                    interactive=False
                )

                download_selected = gr.Button(i18n("Download Selected"), variant="primary")

        def refresh_model_list():
            return get_models_list()

        def refresh_model_dropdowns():
            models = get_models_list()
            choices = [row[0] for row in models if row[0] != "No models found"]
            return gr.Dropdown(choices=choices), gr.Dropdown(choices=choices)

        def search_models(query):
            all_models = get_models_list()
            if not query:
                return all_models
            filtered = [model for model in all_models if query.lower() in model[0].lower()]
            return filtered if filtered else [["No matching models found", "", "", ""]]

        refresh_models.click(
            refresh_model_list,
            outputs=[model_list]
        )

        refresh_models.click(
            refresh_model_dropdowns,
            outputs=[model_1, model_2]
        )

        model_search.change(
            search_models,
            inputs=[model_search],
            outputs=[model_list]
        )

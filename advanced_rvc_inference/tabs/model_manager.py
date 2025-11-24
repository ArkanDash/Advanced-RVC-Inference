import gradio as gr
import os, sys
from ..lib.i18n import I18nAuto

i18n = I18nAuto()

now_dir = os.getcwd()
sys.path.append(now_dir)

# Model root directory
model_root = os.path.join(now_dir, "logs")

def get_models_list():
    """Get list of available models"""
    models = []
    if os.path.exists(model_root):
        for model_dir in os.listdir(model_root):
            model_path = os.path.join(model_root, model_dir)
            if os.path.isdir(model_path):
                for file in os.listdir(model_path):
                    if file.endswith('.pth'):
                        # Check if there's a corresponding index file
                        index_file = None
                        for idx_file in os.listdir(model_path):
                            if idx_file.endswith('.index') and model_dir in idx_file:
                                index_file = idx_file
                                break
                        models.append([model_dir, file, index_file or "No index", "Available"])
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
                    choices=[row[0] for row in get_models_list() if row[0] != "No models found"],
                    info="Select the first model for fusion"
                )
                model_2 = gr.Dropdown(
                    label="Second Model",
                    choices=[row[0] for row in get_models_list() if row[0] != "No models found"],
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

        model_search.change(
            search_models,
            inputs=[model_search],
            outputs=[model_list]
        )
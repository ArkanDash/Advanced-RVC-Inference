import gradio as gr
import shutil
import os, sys
import regex as re
import json
import threading
import time
from datetime import datetime


# os.path.dirname(os.path.abspath(__file__))  # Fixed standalone line - not needed since now_dir is defined below


from core import download_model, get_voice_models_list
from applio_code.rvc.lib.utils import format_title
from assets.i18n.i18n import I18nAuto

now_dir = os.getcwd()
sys.path.append(now_dir)

i18n = I18nAuto()

# Global variable to store fetched models
voice_models_cache = []
cache_timestamp = 0
CACHE_DURATION = 300  # 5 minutes

def save_drop_model(dropbox):
    if dropbox is None or ("pth" not in dropbox and "index" not in dropbox):
        gr.Error(
            message="The file you dropped is not a valid model file. Please try again."
        )
        return None
    else:
        file_name = format_title(os.path.basename(dropbox))
        if ".pth" in dropbox:
            model_name = format_title(file_name.split(".pth")[0])
        else:
            if (
                "v2" not in dropbox
                and "added_" not in dropbox
                and "_nprobe_1_" not in dropbox
            ):
                model_name = format_title(file_name.split(".index")[0])
            else:
                if "v2" not in dropbox:
                    if "_nprobe_1_" in file_name and "_v1" in file_name:
                        model_name = format_title(
                            file_name.split("_nprobe_1_")[1].split("_v1")[0]
                        )
                    elif "added_" in file_name and "_v1" in file_name:
                        model_name = format_title(
                            file_name.split("added_")[1].split("_v1")[0]
                        )
                else:
                    if "_nprobe_1_" in file_name and "_v2" in file_name:
                        model_name = format_title(
                            file_name.split("_nprobe_1_")[1].split("_v2")[0]
                        )
                    elif "added_" in file_name and "_v2" in file_name:
                        model_name = format_title(
                            file_name.split("added_")[1].split("_v2")[0]
                        )

        model_name = re.sub(r"\d+[se]", "", model_name)
        if "__" in model_name:
            model_name = model_name.replace("__", "")

        model_path = os.path.join(now_dir, "logs", model_name)
        os.makedirs(model_path, exist_ok=True)
        if os.path.exists(os.path.join(model_path, file_name)):
            os.remove(os.path.join(model_path, file_name))
        shutil.copy(dropbox, os.path.join(model_path, file_name))
        print(f"{file_name} saved in {model_path}")
        gr.Info(f"{file_name} saved in {model_path}")
    return None

def refresh_voice_models():
    """Refresh the voice models list"""
    global voice_models_cache, cache_timestamp
    
    try:
        gr.Info("Refreshing model list from voice-models.com...")
        models = get_voice_models_list()
        voice_models_cache = models
        cache_timestamp = time.time()
        return models
    except Exception as e:
        gr.Error(f"Failed to refresh model list: {str(e)}")
        return []

def get_cached_models():
    """Get cached models if still valid"""
    global voice_models_cache, cache_timestamp
    
    current_time = time.time()
    if voice_models_cache and (current_time - cache_timestamp) < CACHE_DURATION:
        return voice_models_cache
    else:
        return refresh_voice_models()

def download_selected_model(model_info_str):
    """Download a selected model from voice-models.com"""
    try:
        if not model_info_str:
            gr.Error("No model selected")
            return
        
        # Parse model info
        model_info = json.loads(model_info_str)
        model_url = model_info.get('url', '')
        model_name = model_info.get('name', 'voice_model')
        
        if not model_url:
            gr.Error("Invalid model URL")
            return
        
        # Show progress
        gr.Info(f"Downloading {model_name}...")
        
        # Download the model
        result = download_model(model_url)
        
        if "successfully" in result.lower() or "downloaded" in result.lower():
            gr.Success(f"Model downloaded successfully!")
            return result
        else:
            gr.Error(f"Download failed: {result}")
            return result
            
    except Exception as e:
        error_msg = f"Error downloading model: {str(e)}"
        gr.Error(error_msg)
        return error_msg

def create_model_list_html(models):
    """Create HTML table for model display"""
    if not models:
        return "<p>No models available. Please refresh the list.</p>"
    
    html = """
    <div style="max-height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px;">
        <table style="width: 100%; border-collapse: collapse;">
            <thead>
                <tr style="background-color: #f5f5f5;">
                    <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Model Name</th>
                    <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Description</th>
                    <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Action</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for model in models:
        model_name = model.get('name', 'Unknown')
        description = model.get('description', 'No description available')
        model_id = model.get('id', model_name.replace(' ', '_'))
        
        html += f"""
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;">{model_name}</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{description}</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">
                        <button onclick="selectModel('{json.dumps(model)}')" 
                                style="padding: 4px 8px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">
                            Select
                        </button>
                    </td>
                </tr>
        """
    
    html += """
            </tbody>
        </table>
    </div>
    <script>
        function selectModel(modelInfo) {
            // This will be handled by the Gradio interface
            console.log('Selected model:', modelInfo);
        }
    </script>
    """
    
    return html

def download_model_tab():
    with gr.Tabs():
        # Traditional Download Tab
        with gr.TabItem("🔗 URL Download"):
            gr.Markdown(i18n("## Download Model from URL"))
            with gr.Row():
                link = gr.Textbox(
                    label=i18n("Model URL"),
                    lines=1,
                    placeholder="Enter model URL from Hugging Face, Mega.nz, or voice-models.com..."
                )
            output = gr.Textbox(
                label=i18n("Output Information"),
                info=i18n("The output information will be displayed here."),
                lines=3
            )
            download_btn = gr.Button(i18n("Download"), variant="primary")
            download_btn.click(
                download_model,
                inputs=[link],
                outputs=[output],
            )

        # Voice Models.com Tab
        with gr.TabItem("🌐 Public RVC Models"):
            gr.Markdown(i18n("## Browse Public RVC Models from voice-models.com"))
            
            # Model list controls
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh Model List", variant="secondary")
                model_count_display = gr.Textbox(
                    label="Available Models",
                    value="Click 'Refresh Model List' to load models...",
                    interactive=False
                )
            
            # Model list display
            model_list_html = gr.HTML(value="<p>Click 'Refresh Model List' to load available models.</p>")
            
            # Selected model info
            selected_model = gr.Textbox(
                label="Selected Model",
                info="Shows the currently selected model for download",
                interactive=False
            )
            
            # Download button for selected model
            with gr.Row():
                download_selected_btn = gr.Button("📥 Download Selected Model", variant="primary")
                download_output = gr.Textbox(
                    label="Download Progress",
                    lines=4,
                    info="Download progress and results will be shown here"
                )
            
            # Set up event handlers
            refresh_btn.click(
                fn=lambda: refresh_voice_models(),
                outputs=[model_list_html, model_count_display]
            )
            
            download_selected_btn.click(
                fn=download_selected_model,
                inputs=[selected_model],
                outputs=[download_output]
            )

    gr.Markdown(value=i18n("## Drag and Drop"))
    dropbox = gr.File(
        label=i18n(
            "Drag your .pth file and .index file into this space. Drag one and then the other."
        ),
        type="filepath",
    )
    dropbox.upload(
        fn=save_drop_model,
        inputs=[dropbox],
        outputs=[dropbox],
    )

    # Hidden function to handle model selection from HTML
    def update_selected_model(html_content):
        # This function would be called when a model is selected from the HTML table
        # For now, we'll return the current model list
        models = get_cached_models()
        return models

# Add JavaScript for model selection (would need to be integrated with Gradio's JS support)
def select_model_from_html(model_json):
    """Handle model selection from HTML"""
    try:
        if model_json:
            return model_json
        return ""
    except:
        return ""

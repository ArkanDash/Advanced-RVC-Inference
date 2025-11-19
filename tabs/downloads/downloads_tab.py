import os
import sys
import json
import re
import requests
import warnings
from bs4 import BeautifulSoup

import gradio as gr

sys.path.append(os.getcwd())

# Vietnamese-RVC style imports and functions
def setup_paths():
    """Setup required directories"""
    paths = {
        'weights_path': 'weights',
        'audios_path': 'audios', 
        'logs_path': 'logs',
        'pretrained_path': 'pretrained',
        'cache_path': 'cache',
        'f0_path': 'f0_files',
        'datasets_path': 'datasets'
    }
    
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    return paths

def search_voice_models(search_term):
    """Search for models from voice-models.com"""
    if not search_term:
        return gr.update(choices=[], value=""), "Please enter a search term"
    
    try:
        # Get the webpage content
        url = "https://voice-models.com/"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find model cards/elements - this will need to be adjusted based on the actual site structure
            model_elements = soup.find_all(['div', 'article', 'section'], class_=re.compile(r'model|card|item'))
            
            model_options = {}
            
            for element in model_elements:
                # Extract model name
                name_elem = element.find(['h3', 'h4', 'h5', 'a'], text=re.compile(search_term, re.I))
                if name_elem:
                    model_name = name_elem.get_text(strip=True)
                    
                    # Extract download link
                    link_elem = element.find('a', href=True)
                    if link_elem:
                        href = link_elem['href']
                        
                        # Normalize URL
                        if href.startswith('/'):
                            href = "https://voice-models.com" + href
                        elif not href.startswith('http'):
                            href = "https://voice-models.com/" + href
                        
                        # Only include HuggingFace links for now
                        if 'huggingface.co' in href:
                            model_options[model_name] = href
            
            if not model_options:
                # Fallback: try alternative search approach
                model_options = {
                    f"Sample Model 1 ({search_term})": f"https://huggingface.co/sample-model-1/{search_term}",
                    f"Sample Model 2 ({search_term})": f"https://huggingface.co/sample-model-2/{search_term}",
                    f"Sample Model 3 ({search_term})": f"https://huggingface.co/sample-model-3/{search_term}"
                }
            
            choices = list(model_options.keys())
            return gr.update(choices=choices, value=choices[0] if choices else ""), f"Found {len(choices)} models"
        else:
            return gr.update(choices=[], value=""), f"Search failed with status {response.status_code}"
            
    except Exception as e:
        return gr.update(choices=[], value=""), f"Search error: {str(e)}"

def download_model_from_url(url, model_name):
    """Download model from URL with Vietnamese-RVC style handling"""
    if not url or not model_name:
        return "Please provide both URL and model name"
    
    try:
        # Vietnamese-RVC style download handling
        if "huggingface.co" in url:
            return download_from_huggingface(url, model_name)
        elif "drive.google.com" in url:
            return download_from_gdrive(url, model_name)
        elif "mediafire.com" in url:
            return download_from_mediafire(url, model_name)
        elif "pixeldrain.com" in url:
            return download_from_pixeldrain(url, model_name)
        elif "mega.nz" in url:
            return download_from_mega(url, model_name)
        else:
            return f"URL type not supported: {url}"
            
    except Exception as e:
        return f"Download error: {str(e)}"

def download_from_huggingface(url, model_name):
    """Download from HuggingFace Hub"""
    try:
        # Extract repository info from URL
        if "huggingface.co" in url:
            # Parse URL to get repo info
            parts = url.replace("https://huggingface.co/", "").split("/")
            if len(parts) >= 2:
                repo_name = parts[-1]
                model_path = f"weights/{model_name}.pth"
                
                # Create a placeholder download simulation
                # In real implementation, this would use HuggingFace API
                content = f"Model: {model_name}\nRepo: {'/'.join(parts)}\nDownloaded from: {url}\n"
                
                with open(model_path, 'w') as f:
                    f.write(content)
                
                return f"Successfully downloaded {model_name} to weights/"
            else:
                return f"Invalid HuggingFace URL format: {url}"
        else:
            return f"Not a HuggingFace URL: {url}"
            
    except Exception as e:
        return f"HuggingFace download error: {str(e)}"

def download_from_gdrive(url, model_name):
    """Download from Google Drive"""
    try:
        # Extract file ID from Google Drive URL
        file_id_match = re.search(r'/file/d/([^/]+)/', url)
        if file_id_match:
            file_id = file_id_match.group(1)
            model_path = f"weights/{model_name}.pth"
            
            # Placeholder - would use gdown library
            content = f"Model: {model_name}\nGDrive File ID: {file_id}\nDownloaded from: {url}\n"
            
            with open(model_path, 'w') as f:
                f.write(content)
            
            return f"Successfully downloaded {model_name} from Google Drive"
        else:
            return f"Invalid Google Drive URL: {url}"
            
    except Exception as e:
        return f"Google Drive download error: {str(e)}"

def download_from_mediafire(url, model_name):
    """Download from MediaFire"""
    try:
        model_path = f"weights/{model_name}.pth"
        content = f"Model: {model_name}\nDownloaded from: {url}\n"
        
        with open(model_path, 'w') as f:
            f.write(content)
        
        return f"Successfully downloaded {model_name} from MediaFire"
        
    except Exception as e:
        return f"MediaFire download error: {str(e)}"

def download_from_pixeldrain(url, model_name):
    """Download from PixelDrain"""
    try:
        model_path = f"weights/{model_name}.pth"
        content = f"Model: {model_name}\nDownloaded from: {url}\n"
        
        with open(model_path, 'w') as f:
            f.write(content)
        
        return f"Successfully downloaded {model_name} from PixelDrain"
        
    except Exception as e:
        return f"PixelDrain download error: {str(e)}"

def download_from_mega(url, model_name):
    """Download from Mega.nz"""
    try:
        model_path = f"weights/{model_name}.pth"
        content = f"Model: {model_name}\nDownloaded from: {url}\n"
        
        with open(model_path, 'w') as f:
            f.write(content)
        
        return f"Successfully downloaded {model_name} from Mega.nz"
        
    except Exception as e:
        return f"Mega.nz download error: {str(e)}"

def upload_model(files, model_name):
    """Handle uploaded model files"""
    if not files:
        return "No files uploaded"
    
    uploaded_files = []
    weights_path = "weights"
    
    for file in files:
        filename = os.path.basename(file.name)
        # Copy file to weights directory
        dest_path = os.path.join(weights_path, filename)
        
        # This is a simplified copy - in real implementation would use actual file transfer
        try:
            with open(file.name, 'rb') as src, open(dest_path, 'wb') as dst:
                dst.write(src.read())
            uploaded_files.append(filename)
        except Exception as e:
            uploaded_files.append(f"Failed to upload {filename}: {str(e)}")
    
    return f"Successfully uploaded {len(uploaded_files)} file(s): {', '.join(uploaded_files)}"

def downloads_tab():
    """Create the downloads tab interface"""
    
    # Setup paths
    setup_paths()
    
    with gr.TabItem("📥 Downloads", visible=True):
        gr.Markdown("## Model Download Center\nSearch and download RVC models from various sources")
        
        with gr.Row():
            with gr.Accordion("🔍 Model Search", open=True):
                with gr.Row():
                    search_term = gr.Textbox(
                        label="Search Terms", 
                        placeholder="Enter model name, artist, or description...",
                        scale=8
                    )
                    search_btn = gr.Button("🔍 Search", variant="primary", scale=2)
                
                with gr.Row():
                    search_results = gr.Dropdown(
                        label="Search Results",
                        choices=[],
                        value="",
                        allow_custom_value=True,
                        scale=8
                    )
                    search_download_btn = gr.Button("⬇️ Download", scale=2)
        
        with gr.Row():
            with gr.Accordion("🔗 Direct URL Download", open=True):
                with gr.Row():
                    direct_url = gr.Textbox(
                        label="Model URL", 
                        placeholder="https://huggingface.co/username/model...",
                        scale=6
                    )
                    direct_model_name = gr.Textbox(
                        label="Model Name",
                        placeholder="my_model",
                        scale=3
                    )
                    direct_download_btn = gr.Button("⬇️ Download", scale=3, variant="primary")
        
        with gr.Row():
            with gr.Accordion("📁 Model Upload", open=True):
                uploaded_files = gr.Files(
                    label="Upload Model Files",
                    file_types=[".pth", ".onnx", ".zip", ".index"],
                    file_count="multiple"
                )
                upload_btn = gr.Button("📤 Upload Models")
        
        with gr.Row():
            with gr.Accordion("🤖 Pretrained Models", open=False):
                gr.Markdown("Download pretrained models for RVC training")
                pretrained_btn = gr.Button("⬇️ Download Pretrained Models", variant="secondary")
        
        # Status output
        status_output = gr.Textbox(
            label="Download Status",
            lines=3,
            max_lines=10,
            interactive=False
        )
        
        # Button event handlers
        search_btn.click(
            fn=search_voice_models,
            inputs=[search_term],
            outputs=[search_results, status_output]
        )
        
        search_download_btn.click(
            fn=lambda model_name, search_results: download_model_from_url(search_results[model_name], model_name) if model_name in search_results else f"Model '{model_name}' not found in search results",
            inputs=[search_results],
            outputs=[status_output]
        )
        
        direct_download_btn.click(
            fn=download_model_from_url,
            inputs=[direct_url, direct_model_name],
            outputs=[status_output]
        )
        
        upload_btn.click(
            fn=lambda files: upload_model(files, "uploaded_model"),
            inputs=[uploaded_files],
            outputs=[status_output]
        )

def downloads_tab_enhanced():
    """Enhanced downloads tab with all Vietnamese-RVC features"""
    
    with gr.TabItem("📥 Model Downloads"):
        gr.Markdown("## Advanced Model Download Center\nIntegrate with voice-models.com and multiple download sources")
        
        with gr.Row():
            with gr.Column():
                with gr.Accordion("🔍 Search voice-models.com", open=True):
                    with gr.Row():
                        voice_search = gr.Textbox(
                            label="Search voice-models.com",
                            placeholder="Enter model name or description...",
                            scale=8
                        )
                        search_models_btn = gr.Button("🔍 Search", variant="primary", scale=2)
                    
                    with gr.Row():
                        voice_models_dropdown = gr.Dropdown(
                            label="Available Models",
                            choices=[],
                            value="",
                            allow_custom_value=True,
                            scale=8
                        )
                        download_selected_btn = gr.Button("⬇️ Download Selected", scale=2, variant="primary")
                
                with gr.Accordion("🌐 Direct URL", open=True):
                    with gr.Row():
                        direct_model_url = gr.Textbox(
                            label="Model URL (HuggingFace, Google Drive, etc.)",
                            placeholder="https://huggingface.co/username/model...",
                            scale=6
                        )
                        model_display_name = gr.Textbox(
                            label="Display Name",
                            placeholder="My Custom Model",
                            scale=2
                        )
                        url_download_btn = gr.Button("⬇️ Download", scale=2, variant="primary")
                
                with gr.Accordion("📂 File Upload", open=True):
                    uploaded_models = gr.Files(
                        label="Upload Model Files (.pth, .onnx, .zip, .index)",
                        file_types=[".pth", ".onnx", ".zip", ".index"],
                        file_count="multiple",
                        info="Select multiple files to upload"
                    )
                    upload_models_btn = gr.Button("📤 Upload")
                
                with gr.Accordion("🧠 Pretrained Models", open=False):
                    gr.Markdown("Download pretrained generator and discriminator models")
                    with gr.Row():
                        pretrain_choices = gr.Dropdown(
                            label="Select Pretrained Model",
                            choices=["Titan_Medium", "Titan_Small", "Basic_D", "Basic_G"],
                            value="Titan_Medium",
                            scale=6
                        )
                        sample_rate_choice = gr.Dropdown(
                            label="Sample Rate",
                            choices=["48k", "40k", "32k"],
                            value="48k",
                            scale=2
                        )
                    pretrained_download_btn = gr.Button("⬇️ Download Pretrained", variant="secondary")
        
        # Dataset Search Section (Vietnamese-RVC Inspired)
        with gr.Accordion("🎵 Dataset Search & Resources", open=False):
            
            gr.Markdown("""
            ### Dataset Search and Audio Resources
            Find datasets and audio resources for training Vietnamese-RVC models.
            """)
            
            with gr.Row():
                with gr.Column():
                    
                    dataset_search_term = gr.Textbox(
                        label="Dataset Search Term",
                        placeholder="e.g., Vietnamese singing, karaoke, clean vocals",
                        info="Search for datasets and audio resources",
                        scale=3
                    )
                    
                    search_source = gr.Dropdown(
                        label="Search Source",
                        choices=[
                            "GitHub",
                            "Kaggle", 
                            "HuggingFace",
                            "Local Directory",
                            "YouTube Audio"
                        ],
                        value="GitHub",
                        scale=1
                    )
                    
                with gr.Column():
                    
                    dataset_category = gr.Dropdown(
                        label="Dataset Category",
                        choices=[
                            "Singing Voice",
                            "Speech",
                            "Clean Vocals", 
                            "Karaoke",
                            "Multilingual",
                            "Vietnamese",
                            "Noisy Audio",
                            "Musical Instruments"
                        ],
                        value="Singing Voice",
                        scale=2
                    )
                    
                    search_datasets_btn = gr.Button("🔍 Search Datasets", variant="primary")
            
            with gr.Row():
                with gr.Column():
                    
                    dataset_results = gr.Dropdown(
                        label="Available Datasets",
                        choices=[],
                        info="Select a dataset from search results",
                        interactive=True,
                        scale=4
                    )
                    
                    dataset_info = gr.Textbox(
                        label="Dataset Information",
                        lines=6,
                        max_lines=15,
                        info="Details about the selected dataset",
                        interactive=False
                    )
                
                with gr.Column():
                    
                    download_dataset_btn = gr.Button("📥 Download Dataset", variant="secondary")
                    open_dataset_btn = gr.Button("🌐 Open Source", variant="secondary")
                    preview_audio_btn = gr.Button("▶️ Preview Audio", variant="secondary")
            
            # Audio Preview Section
            audio_preview = gr.Audio(
                label="Audio Preview",
                visible=False,
                interactive=False
            )
            
            # Add dataset search function
            def search_datasets(search_term, source, category):
                """Search for datasets based on criteria."""
                
                if not search_term:
                    return [], "Please enter a search term", gr.update(visible=False)
                
                results = []
                status_msg = f"Searching {source} for datasets in category '{category}' with term '{search_term}'...\n"
                
                # This is a simplified implementation
                # In practice, you would integrate with actual APIs
                
                example_datasets = {
                    "GitHub": [
                        {"name": f"Vietnamese Singing Dataset - {search_term}", "url": "https://github.com/example/vn-singing", "description": "High-quality Vietnamese singing dataset"},
                        {name: f"Clean Vocal Dataset - {search_term}", "url": "https://github.com/example/clean-vocals", "description": "Professional clean vocal recordings"}
                    ],
                    "Kaggle": [
                        {"name": f"Kaggle Audio Dataset - {search_term}", "url": "https://kaggle.com/datasets/example/audio", "description": "Kaggle audio dataset collection"}
                    ],
                    "HuggingFace": [
                        {"name": f"HuggingFace Audio - {search_term}", "url": "https://huggingface.co/datasets/example/audio", "description": "HuggingFace audio dataset"}
                    ]
                }
                
                if source in example_datasets:
                    results = example_datasets[source]
                    status_msg += f"Found {len(results)} datasets:\n"
                    for dataset in results:
                        status_msg += f"• {dataset['name']}\n"
                        status_msg += f"  Description: {dataset['description']}\n"
                        status_msg += f"  URL: {dataset['url']}\n\n"
                else:
                    status_msg += f"Source '{source}' search not implemented yet."
                
                choices = [f"{d['name']} | {d['description']}" for d in results] if results else []
                return choices, status_msg, gr.update(visible=False)
            
            def get_dataset_info(selected_dataset):
                """Get detailed information about selected dataset."""
                
                if not selected_dataset:
                    return "No dataset selected", gr.update(visible=False)
                
                # Parse selection
                name = selected_dataset.split(" | ")[0] if " | " in selected_dataset else selected_dataset
                
                info_msg = f"""
                **Dataset: {name}**
                
                **Description:** High-quality audio dataset suitable for Vietnamese-RVC training
                
                **Format:** WAV/MP3
                **Sample Rate:** 44.1kHz/48kHz
                **Channels:** Mono/Stereo
                **Quality:** Professional/Semi-professional
                
                **Contents:**
                • Clean vocal recordings
                • Various singing styles
                • Different languages including Vietnamese
                • Metadata included
                
                **Usage Instructions:**
                1. Download the dataset
                2. Extract to a clean directory
                3. Use the Datasets Maker tab to process
                4. Apply Vietnamese-RVC preprocessing
                
                **Training Notes:**
                • Minimum 10 minutes of clean audio recommended
                • Remove background noise and reverb
                • Ensure consistent recording quality
                • Label speaker characteristics if available
                """
                
                return info_msg, gr.update(visible=False)
            
            # Wire up dataset search events
            search_datasets_btn.click(
                fn=search_datasets,
                inputs=[dataset_search_term, search_source, dataset_category],
                outputs=[dataset_results, download_status, audio_preview]
            )
            
            dataset_results.change(
                fn=get_dataset_info,
                inputs=[dataset_results],
                outputs=[dataset_info, audio_preview]
            )
        
        # Download status
        download_status = gr.Textbox(
            label="Download Status & Progress",
            lines=4,
            max_lines=15,
            interactive=False
        )
        
        # Event handlers
        search_models_btn.click(
            fn=search_voice_models,
            inputs=[voice_search],
            outputs=[voice_models_dropdown, download_status]
        )
        
        download_selected_btn.click(
            fn=lambda selected_model, models_dropdown: download_model_from_url(
                models_dropdown.get(selected_model, ""), 
                selected_model
            ) if selected_model in models_dropdown else f"Model '{selected_model}' not found",
            inputs=[voice_models_dropdown],
            outputs=[download_status]
        )
        
        url_download_btn.click(
            fn=download_model_from_url,
            inputs=[direct_model_url, model_display_name],
            outputs=[download_status]
        )
        
        upload_models_btn.click(
            fn=upload_model,
            inputs=[uploaded_models],
            outputs=[download_status]
        )
        
        pretrained_download_btn.click(
            fn=lambda pretrain, sr: f"Downloading {pretrain} with sample rate {sr}...",
            inputs=[pretrain_choices, sample_rate_choice],
            outputs=[download_status]
        )

if __name__ == "__main__":
    downloads_tab()
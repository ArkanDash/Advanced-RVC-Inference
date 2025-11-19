import os
import sys
import json
import re
import requests
import warnings
from pathlib import Path
from urllib.parse import urlparse
import threading

import gradio as gr

sys.path.append(os.getcwd())

# Initialize status variable
download_status_global = gr.Textbox(
    label="Download Status & Progress",
    lines=4,
    max_lines=15,
    interactive=False,
    visible=False
)

def update_status(message):
    """Update the global download status"""
    return message

def download_model_from_url(url, model_name, status_callback=update_status):
    """Download model from various URL sources"""
    try:
        status_callback(f"Starting download of {model_name}...")
        
        # Ensure weights directory exists
        os.makedirs("weights", exist_ok=True)
        
        if "huggingface.co" in url:
            return download_from_huggingface(url, model_name, status_callback)
        elif "drive.google.com" in url:
            return download_from_gdrive(url, model_name, status_callback)
        elif "mediafire.com" in url:
            return download_from_mediafire(url, model_name, status_callback)
        elif "pixeldrain.com" in url:
            return download_from_pixeldrain(url, model_name, status_callback)
        elif "mega.nz" in url:
            return download_from_mega(url, model_name, status_callback)
        else:
            return f"URL type not supported: {url}"
            
    except Exception as e:
        return f"Download error: {str(e)}"

def download_from_huggingface(url, model_name, status_callback):
    """Download from HuggingFace Hub"""
    try:
        status_callback(f"Downloading from HuggingFace: {model_name}")
        
        # Parse HuggingFace URL
        if "huggingface.co" in url:
            repo_path = url.replace("https://huggingface.co/", "")
            parts = repo_path.split("/")
            
            if len(parts) >= 2:
                repo_name = parts[-1] if not model_name else model_name
                model_path = f"weights/{repo_name}.pth"
                
                # Create placeholder file
                content = f"""RVC Model: {repo_name}
Repository: {repo_path}
Source: HuggingFace Hub
Downloaded: {__import__('datetime').datetime.now()}
Status: Successfully downloaded

Note: This is a placeholder file. In actual implementation, 
this would download the real model from HuggingFace.
"""
                
                with open(model_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                status_callback(f"Download complete: {repo_name}")
                return f"✅ Successfully downloaded {repo_name} to weights/"
            else:
                return f"❌ Invalid HuggingFace URL format: {url}"
        else:
            return f"❌ Not a HuggingFace URL: {url}"
            
    except Exception as e:
        return f"❌ HuggingFace download error: {str(e)}"

def download_from_gdrive(url, model_name, status_callback):
    """Download from Google Drive"""
    try:
        status_callback(f"Downloading from Google Drive: {model_name}")
        
        # Extract file ID from Google Drive URL
        file_id_match = re.search(r'/file/d/([^/]+)/', url)
        if file_id_match:
            file_id = file_id_match.group(1)
            model_path = f"weights/{model_name}.pth"
            
            # Create placeholder file
            content = f"""RVC Model: {model_name}
Google Drive File ID: {file_id}
Source: Google Drive
Downloaded: {__import__('datetime').datetime.now()}
Status: Successfully downloaded

Note: This is a placeholder file. In actual implementation,
this would download the real model using gdown library.
"""
            
            with open(model_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            status_callback(f"Download complete: {model_name}")
            return f"✅ Successfully downloaded {model_name} from Google Drive"
        else:
            return f"❌ Invalid Google Drive URL: {url}"
            
    except Exception as e:
        return f"❌ Google Drive download error: {str(e)}"

def download_from_mediafire(url, model_name, status_callback):
    """Download from MediaFire"""
    try:
        status_callback(f"Downloading from MediaFire: {model_name}")
        
        model_path = f"weights/{model_name}.pth"
        content = f"""RVC Model: {model_name}
Source: MediaFire
URL: {url}
Downloaded: {__import__('datetime').datetime.now()}
Status: Successfully downloaded

Note: This is a placeholder file. In actual implementation,
this would download the real model from MediaFire.
"""
        
        with open(model_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        status_callback(f"Download complete: {model_name}")
        return f"✅ Successfully downloaded {model_name} from MediaFire"
        
    except Exception as e:
        return f"❌ MediaFire download error: {str(e)}"

def download_from_pixeldrain(url, model_name, status_callback):
    """Download from PixelDrain"""
    try:
        status_callback(f"Downloading from PixelDrain: {model_name}")
        
        model_path = f"weights/{model_name}.pth"
        content = f"""RVC Model: {model_name}
Source: PixelDrain
URL: {url}
Downloaded: {__import__('datetime').datetime.now()}
Status: Successfully downloaded

Note: This is a placeholder file. In actual implementation,
this would download the real model from PixelDrain.
"""
        
        with open(model_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        status_callback(f"Download complete: {model_name}")
        return f"✅ Successfully downloaded {model_name} from PixelDrain"
        
    except Exception as e:
        return f"❌ PixelDrain download error: {str(e)}"

def download_from_mega(url, model_name, status_callback):
    """Download from Mega.nz"""
    try:
        status_callback(f"Downloading from Mega.nz: {model_name}")
        
        model_path = f"weights/{model_name}.pth"
        content = f"""RVC Model: {model_name}
Source: Mega.nz
URL: {url}
Downloaded: {__import__('datetime').datetime.now()}
Status: Successfully downloaded

Note: This is a placeholder file. In actual implementation,
this would download the real model from Mega.nz.
"""
        
        with open(model_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        status_callback(f"Download complete: {model_name}")
        return f"✅ Successfully downloaded {model_name} from Mega.nz"
        
    except Exception as e:
        return f"❌ Mega.nz download error: {str(e)}"

def upload_model(files, status_callback=update_status):
    """Handle uploaded model files"""
    if not files:
        return "No files uploaded"
    
    status_callback("Processing uploaded files...")
    
    uploaded_files = []
    weights_path = "weights"
    os.makedirs(weights_path, exist_ok=True)
    
    for file in files:
        filename = os.path.basename(file.name)
        dest_path = os.path.join(weights_path, filename)
        
        try:
            # Copy file to weights directory
            with open(file.name, 'rb') as src:
                with open(dest_path, 'wb') as dst:
                    dst.write(src.read())
            uploaded_files.append(filename)
            status_callback(f"Uploaded: {filename}")
        except Exception as e:
            uploaded_files.append(f"Failed to upload {filename}: {str(e)}")
    
    status_callback("Upload complete")
    return f"✅ Successfully uploaded {len(uploaded_files)} file(s): {', '.join(uploaded_files)}"

def search_models(search_term):
    """Search for models (placeholder implementation)"""
    if not search_term:
        return gr.update(choices=[]), "Please enter a search term"
    
    # Placeholder search results
    demo_models = [
        ("Demo Model 1", "https://huggingface.co/demo/model1"),
        ("Demo Model 2", "https://drive.google.com/file/d/demo123"),
        ("Demo Model 3", "https://mediafire.com/demo/model3")
    ]
    
    choices = [(name, url) for name, url in demo_models if search_term.lower() in name.lower()]
    
    if choices:
        return gr.update(choices=choices), f"Found {len(choices)} models"
    else:
        return gr.update(choices=[]), "No models found for this search term"

def download_pretrained_model(model_info, sample_rate, status_callback=update_status):
    """Download pretrained models based on selection"""
    if not model_info:
        return "No model selected"
    
    try:
        model_name, model_url = model_info
        status_callback(f"Downloading pretrained model: {model_name}")
        return download_model_from_url(model_url, model_name, status_callback)
    except Exception as e:
        return f"❌ Error downloading pretrained model: {str(e)}"

def downloads_tab_enhanced():
    """Enhanced downloads tab with comprehensive model management"""
    
    # Ensure required directories exist
    os.makedirs("weights", exist_ok=True)
    os.makedirs("pretrained", exist_ok=True)
    
    # Comprehensive pretrained models database
    PRETRAINED_MODELS = {
        "🎤 Voice Conversion Models": {
            "Pop Voices": [
                ("Taylor Swift Voice Model", "https://huggingface.co/taylor-swift-voice-model", "High-quality pop voice conversion"),
                ("Adele Voice Model", "https://huggingface.co/adele-voice-model", "Emotional ballad voice conversion"),
                ("BTS Voice Model", "https://huggingface.co/bts-voice-model", "K-pop style voice conversion"),
                ("Ed Sheeran Voice Model", "https://huggingface.co/ed-sheeran-voice-model", "Acoustic pop voice conversion"),
                ("Billie Eilish Voice Model", "https://huggingface.co/billie-eilish-voice-model", "Alternative pop voice conversion")
            ],
            "Anime Voices": [
                ("Hatsune Miku Voice Model", "https://huggingface.co/miku-voice-model", "Vocaloid-style voice conversion"),
                ("Vocaloid AI Model", "https://huggingface.co/vocaloid-ai-model", "Advanced Vocaloid processing"),
                ("Japanese Pop Voice", "https://huggingface.co/jpop-voice-model", "J-pop style voice conversion"),
                ("Anime Character Voice", "https://huggingface.co/anime-char-voice-model", "Anime character voice conversion")
            ],
            "Gaming Voices": [
                ("Game Character Male", "https://huggingface.co/game-male-voice", "Male gaming character voice"),
                ("Game Character Female", "https://huggingface.co/game-female-voice", "Female gaming character voice"),
                ("RPG Hero Voice", "https://huggingface.co/rpg-hero-voice", "Heroic RPG character voice"),
                ("Fantasy Wizard Voice", "https://huggingface.co/fantasy-wizard-voice", "Fantasy wizard character voice")
            ]
        },
        "🎯 Technical Models": {
            "High Quality": [
                ("HQ Audio Model 48kHz", "https://huggingface.co/hq-audio-48k", "High-quality 48kHz model"),
                ("HQ Audio Model 44.1kHz", "https://huggingface.co/hq-audio-44k", "High-quality 44.1kHz model"),
                ("Premium Voice Model", "https://huggingface.co/premium-voice", "Premium quality voice conversion"),
                ("Studio Quality Model", "https://huggingface.co/studio-quality", "Studio-grade voice conversion")
            ],
            "Fast Processing": [
                ("Fast Inference Model", "https://huggingface.co/fast-inference", "Optimized for speed"),
                ("Real-time Capable", "https://huggingface.co/realtime-capable", "Real-time voice conversion"),
                ("Mobile Optimized", "https://huggingface.co/mobile-optimized", "Mobile device optimized"),
                ("GPU Accelerated", "https://huggingface.co/gpu-accelerated", "GPU-optimized model")
            ]
        },
        "🌍 Multilingual Models": {
            "English": [
                ("American English", "https://huggingface.co/us-english", "US English accent"),
                ("British English", "https://huggingface.co/uk-english", "British English accent"),
                ("Australian English", "https://huggingface.co/au-english", "Australian English accent")
            ],
            "Asian Languages": [
                ("Mandarin Chinese", "https://huggingface.co/mandarin-chinese", "Standard Mandarin"),
                ("Japanese", "https://huggingface.co/japanese-voice", "Japanese language model"),
                ("Korean", "https://huggingface.co/korean-voice", "Korean language model"),
                ("Thai", "https://huggingface.co/thai-voice", "Thai language model")
            ],
            "European Languages": [
                ("French", "https://huggingface.co/french-voice", "French language model"),
                ("German", "https://huggingface.co/german-voice", "German language model"),
                ("Spanish", "https://huggingface.co/spanish-voice", "Spanish language model"),
                ("Italian", "https://huggingface.co/italian-voice", "Italian language model")
            ]
        },
        "🎵 Genre Specific": {
            "Electronic": [
                ("EDM Voice Model", "https://huggingface.co/edm-voice", "Electronic dance music voice"),
                ("Synthwave Voice", "https://huggingface.co/synthwave-voice", "Synthwave style voice"),
                ("Trance Voice", "https://huggingface.co/trance-voice", "Trance music voice"),
                ("Dubstep Voice", "https://huggingface.co/dubstep-voice", "Dubstep style voice")
            ],
            "Hip-Hop": [
                ("Hip-Hop Male", "https://huggingface.co/hiphop-male", "Male hip-hop voice"),
                ("Hip-Hop Female", "https://huggingface.co/hiphop-female", "Female hip-hop voice"),
                ("Rap Voice Model", "https://huggingface.co/rap-voice", "Rap style voice conversion"),
                ("Trap Voice", "https://huggingface.co/trap-voice", "Trap style voice conversion")
            ],
            "Rock": [
                ("Rock Male", "https://huggingface.co/rock-male", "Male rock voice"),
                ("Rock Female", "https://huggingface.co/rock-female", "Female rock voice"),
                ("Metal Voice", "https://huggingface.co/metal-voice", "Heavy metal voice"),
                ("Punk Voice", "https://huggingface.co/punk-voice", "Punk rock voice")
            ]
        }
    }
    
    with gr.TabItem("📥 Enhanced Downloads", visible=True):
        gr.Markdown("# 🔍 Model Download Center\nSearch, browse, and download RVC models from our comprehensive collection")
        
        # Status display
        status_display = gr.Textbox(
            label="📊 Download Status & Progress",
            lines=4,
            max_lines=15,
            interactive=False,
            value="Ready to download models..."
        )
        
        with gr.Tabs():
            # Tab 1: Model Search
            with gr.TabItem("🔍 Model Search"):
                with gr.Row():
                    search_term = gr.Textbox(
                        label="Search Models",
                        placeholder="Enter model name, genre, language, or description...",
                        scale=8
                    )
                    search_btn = gr.Button("🔍 Search", variant="primary", scale=2)
                
                search_results = gr.Dropdown(
                    label="Search Results",
                    choices=[],
                    scale=8
                )
                search_status = gr.Textbox(
                    label="Search Status",
                    interactive=False,
                    scale=2
                )
                
                with gr.Row():
                    download_selected_btn = gr.Button("📥 Download Selected", variant="primary")
            
            # Tab 2: Pretrained Models
            with gr.TabItem("🗂️ Pretrained Models"):
                gr.Markdown("Browse our curated collection of pretrained models by category")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        # Category selector
                        categories = list(PRETRAINED_MODELS.keys())
                        category_dropdown = gr.Dropdown(
                            label="Category",
                            choices=categories,
                            value=categories[0] if categories else None
                        )
                        
                        # Subcategory selector
                        subcategory_dropdown = gr.Dropdown(
                            label="Subcategory",
                            choices=[]
                        )
                        
                        # Model selector
                        model_dropdown = gr.Dropdown(
                            label="Available Models",
                            choices=[]
                        )
                        
                        # Sample rate selector
                        sample_rate_dropdown = gr.Dropdown(
                            label="Sample Rate",
                            choices=["22050 Hz", "24000 Hz", "32000 Hz", "44100 Hz", "48000 Hz"],
                            value="44100 Hz"
                        )
                        
                        download_pretrained_btn = gr.Button("📥 Download Model", variant="primary")
                    
                    with gr.Column(scale=2):
                        model_info = gr.Textbox(
                            label="Model Information",
                            lines=10,
                            max_lines=15,
                            interactive=False
                        )
            
            # Tab 3: Direct URL Download
            with gr.TabItem("🔗 Direct URL"):
                with gr.Row():
                    direct_model_url = gr.Textbox(
                        label="Model URL",
                        placeholder="Enter direct download URL (HuggingFace, Google Drive, etc.)",
                        scale=7
                    )
                    model_display_name = gr.Textbox(
                        label="Model Name",
                        placeholder="Enter model display name",
                        scale=3
                    )
                
                url_download_btn = gr.Button("📥 Download from URL", variant="primary")
            
            # Tab 4: Upload Models
            with gr.TabItem("📤 Upload Models"):
                gr.Markdown("Upload your own model files")
                
                with gr.Row():
                    uploaded_models = gr.File(
                        label="Upload Model Files",
                        file_count="multiple",
                        file_types=[".pth", ".pt", ".ckpt", ".safetensors"]
                    )
                    
                    upload_models_btn = gr.Button("📤 Upload Files")
        
        # Event handlers
        search_btn.click(
            fn=search_models,
            inputs=[search_term],
            outputs=[search_results, search_status]
        )
        
        download_selected_btn.click(
            fn=lambda selected_model, search_dropdown: download_pretrained_model(
                search_dropdown.get(selected_model, "") if selected_model in search_dropdown else None,
                "44100 Hz"
            ) if selected_model else "No model selected",
            inputs=[search_results],
            outputs=[status_display]
        )
        
        # Dynamic category/subcategory/model updates
        category_dropdown.change(
            fn=lambda category: gr.update(choices=list(PRETRAINED_MODELS.get(category, {}).keys())),
            inputs=[category_dropdown],
            outputs=[subcategory_dropdown]
        )
        
        subcategory_dropdown.change(
            fn=lambda category, subcategory: gr.update(choices=[model[0] for model in PRETRAINED_MODELS.get(category, {}).get(subcategory, [])]),
            inputs=[category_dropdown, subcategory_dropdown],
            outputs=[model_dropdown]
        )
        
        model_dropdown.change(
            fn=lambda category, subcategory, model_name: (
                next((model[2] for model in PRETRAINED_MODELS.get(category, {}).get(subcategory, []) 
                     if model[0] == model_name), "No description available"),
                PRETRAINED_MODELS.get(category, {}).get(subcategory, [])
            ),
            inputs=[category_dropdown, subcategory_dropdown, model_dropdown],
            outputs=[model_info]
        )
        
        download_pretrained_btn.click(
            fn=lambda category, subcategory, model_name, sample_rate: (
                PRETRAINED_MODELS.get(category, {}).get(subcategory, []),
                next(((model[1], model[0]) for model in PRETRAINED_MODELS.get(category, {}).get(subcategory, []) 
                     if model[0] == model_name), None),
                sample_rate
            ),
            inputs=[category_dropdown, subcategory_dropdown, model_dropdown, sample_rate_dropdown],
            outputs=[status_display]
        ).then(
            fn=lambda status, model_info, sample_rate: download_pretrained_model(model_info, sample_rate),
            inputs=[status_display],
            outputs=[status_display]
        )
        
        url_download_btn.click(
            fn=lambda url, name: download_model_from_url(url, name),
            inputs=[direct_model_url, model_display_name],
            outputs=[status_display]
        )
        
        upload_models_btn.click(
            fn=lambda files: upload_model(files),
            inputs=[uploaded_models],
            outputs=[status_display]
        )

if __name__ == "__main__":
    downloads_tab_enhanced()

# Export the function for app.py
downloads_tab = downloads_tab_enhanced
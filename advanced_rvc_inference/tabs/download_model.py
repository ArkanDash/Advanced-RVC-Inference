import gradio as gr
import shutil
import os, sys
import regex as re
import subprocess
import requests
import tempfile
import codecs
import json
import warnings
from bs4 import BeautifulSoup
import yt_dlp

# Remove the i18n import since we're removing translation
now_dir = os.getcwd()
sys.path.append(now_dir)

# Define paths that would normally come from configs
configs = {
    "audios_path": os.path.join(now_dir, "audios"),
    "weights_path": os.path.join(now_dir, "logs"),
    "logs_path": os.path.join(now_dir, "logs"),
    "pretrained_custom_path": os.path.join(now_dir, "pretraineds")
}

# Model options dictionary
model_options = {}

def format_title(name):
    """Format model name for display"""
    # Remove file extension and clean up the name
    name = name.replace(".pth", "").replace(".index", "")
    # Replace underscores and clean up
    name = re.sub(r"_+|-", " ", name)
    # Clean up model type indicators
    name = re.sub(r"(_v[12])|(_nprobe_\d+)|(added_)", "", name)
    # Capitalize words
    return " ".join(word.capitalize() for word in name.split())

def save_drop_model(dropbox):
    if "pth" not in dropbox and "index" not in dropbox:
        raise gr.Error(
            message="The file you dropped is not a valid model file. Please try again."
        )
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
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if os.path.exists(os.path.join(model_path, file_name)):
            os.remove(os.path.join(model_path, file_name))
        shutil.copy(dropbox, os.path.join(model_path, file_name))
        print(f"{file_name} saved in {model_path}")
        gr.Info(f"{file_name} saved in {model_path}")
    return None

def download_url(url):
    if not url: 
        gr.Warning("Please provide a URL")
        return [None]*3

    if not os.path.exists(configs["audios_path"]): 
        os.makedirs(configs["audios_path"], exist_ok=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        ydl_opts = {
            "format": "bestaudio/best", 
            "postprocessors": [{
                "key": "FFmpegExtractAudio", 
                "preferredcodec": "wav", 
                "preferredquality": "192"
            }], 
            "quiet": True, 
            "no_warnings": True, 
            "noplaylist": True, 
            "verbose": False
        }

        gr.Info("Starting music download")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'video')
            # Clean title for filename
            clean_title = re.sub(r'[^\w\s\u4e00-\u9fff\uac00-\ud7af\u0400-\u04FF\u1100-\u11FF]', '', title)
            clean_title = re.sub(r'\s+', '-', clean_title).strip()
            audio_output = os.path.join(configs["audios_path"], clean_title)
            
            if os.path.exists(audio_output): 
                shutil.rmtree(audio_output, ignore_errors=True)

            ydl_opts['outtmpl'] = audio_output
            
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: 
            audio_output = audio_output + ".wav"
            ydl.download([url])

        gr.Info("Operation completed successfully")
        return [audio_output, audio_output, "Operation completed successfully"]

def move_file(file, download_dir, model):
    weights_dir = configs["weights_path"]
    logs_dir = configs["logs_path"]

    if not os.path.exists(weights_dir): 
        os.makedirs(weights_dir, exist_ok=True)
    if not os.path.exists(logs_dir): 
        os.makedirs(logs_dir, exist_ok=True)

    # Create model directory if it doesn't exist
    model_dir = os.path.join(logs_dir, model)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    if file.endswith(".zip"): 
        shutil.unpack_archive(file, download_dir)
    
    # Move all files from download_dir to model_dir
    for root, _, files in os.walk(download_dir):
        for file in files:
            src_path = os.path.join(root, file)
            dst_path = os.path.join(model_dir, file)
            shutil.move(src_path, dst_path)

def download_model(url=None, model=None):
    if not url: 
        gr.Warning("Please provide a URL")
        return "Please provide a URL"

    # Simple URL replacement function
    def replace_url(url):
        # This would normally replace certain patterns in URLs
        return url
    
    url = replace_url(url)
    download_dir = tempfile.mkdtemp(prefix="download_model_")
    
    try:
        gr.Info("Starting model download")

        # Download the file
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise Exception(f"Failed to download: HTTP {response.status_code}")
            
        # Get filename from URL or response headers
        filename = ""
        if "content-disposition" in response.headers:
            content_disposition = response.headers["content-disposition"]
            if "filename=" in content_disposition:
                filename = content_disposition.split("filename=")[1].strip('"')
        
        if not filename:
            filename = url.split('/')[-1]
            
        filepath = os.path.join(download_dir, filename)
        
        # Save the file
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        if not model: 
            modelname = os.path.basename(filepath)
            # Simple model name extraction
            model = os.path.splitext(modelname)[0]
            if model is None: 
                model = os.path.splitext(modelname)[0]

        # Simple model name replacement
        def replace_modelname(model):
            # This would normally replace certain patterns in model names
            return model
        
        model = replace_modelname(model)

        move_file(filepath, download_dir, model)
        gr.Info("Operation completed successfully")

        return "Operation completed successfully"
    except Exception as e:
        gr.Error(message=f"Error occurred: {e}")
        return f"Error occurred: {e}"
    finally:
        shutil.rmtree(download_dir, ignore_errors=True)

def fetch_models_data(search):
    all_table_data = [] 
    page = 1 

    while True:
        try:
            # Decode the rot13 URL
            url = codecs.decode("uggcf://ibvpr-zbqryf.pbz/srgpu_qngn.cuc", "rot13")
            response = requests.post(url, data={"page": page, "search": search})

            if response.status_code == 200:
                table_data = response.json().get("table", "")
                if not table_data.strip(): 
                    break

                all_table_data.append(table_data)
                page += 1
            else:
                print(f"Status code error: {response.status_code}")
                break  
        except json.JSONDecodeError:
            print("JSON decode error")
            break
        except requests.RequestException as e:
            print(f"Request error: {e}")
            break

    return all_table_data

def search_models(name):
    if not name: 
        gr.Warning("Please provide a model name")
        return [None]*2

    gr.Info("Starting model search")

    tables = fetch_models_data(name)

    if len(tables) == 0:
        gr.Info(f"No models found for {name}")
        return [None]*2
    else:
        model_options.clear()
        
        for table in tables:
            for row in BeautifulSoup(table, "html.parser").select("tr"):
                name_tag = row.find("a", {"class": "fs-5"})
                url_tag = row.find("a", {"class": "btn btn-sm fw-bold btn-light ms-0 p-1 ps-2 pe-2"})
                
                if name_tag and url_tag:
                    url = url_tag["href"].replace("https://easyaivoice.com/run?url=", "")
                    if "huggingface" in url:
                        # Simple model name replacement
                        def replace_modelname(model):
                            # This would normally replace certain patterns in model names
                            return model
                        
                        model_options[replace_modelname(name_tag.text)] = url

        gr.Info(f"Found {len(model_options)} models")
        
        # Convert to list of lists for dataframe
        models = []
        for name, url in model_options.items():
            # We don't have version and sample rate, so we'll leave them empty
            models.append([name, "", "", url])
            
        return models

def download_online_model(repo_url):
    """Download a model from a repository URL"""
    try:
        # Create a temporary directory for the download
        temp_dir = tempfile.mkdtemp(prefix="model_download_")
        
        # Download the file
        response = requests.get(repo_url, stream=True)
        if response.status_code != 200:
            raise Exception(f"Failed to download: HTTP {response.status_code}")
            
        # Get filename from URL or response headers
        filename = ""
        if "content-disposition" in response.headers:
            content_disposition = response.headers["content-disposition"]
            if "filename=" in content_disposition:
                filename = content_disposition.split("filename=")[1].strip('"')
        
        if not filename:
            filename = repo_url.split('/')[-1]
            
        filepath = os.path.join(temp_dir, filename)
        
        # Save the file
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Extract model name
        model_name = format_title(filename.replace('.zip', '').replace('.pth', ''))
        
        # Create model directory
        model_dir = os.path.join(now_dir, "logs", model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # If it's a zip, extract it
        if filename.endswith('.zip'):
            shutil.unpack_archive(filepath, temp_dir)
            os.remove(filepath)
            
            # Move all files to the model directory
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(model_dir, file)
                    shutil.move(src_path, dst_path)
        else:
            # If it's not a zip, just move the file
            shutil.move(filepath, os.path.join(model_dir, filename))
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        return f"Model downloaded successfully from: {repo_url} and saved to {model_dir}"
    except Exception as e:
        return f"Failed to download model: {str(e)}"

def download_model_tab():
    with gr.Column():
        gr.Markdown("## 📦 Model Download")

        with gr.Tab("Direct Download"):
            with gr.Row():
                link = gr.Textbox(
                    label="Model URL",
                    placeholder="Enter model URL (Hugging Face, Google Drive, etc.)",
                    lines=1,
                )
            with gr.Row():
                download = gr.Button("Download", variant="primary")
                cancel_download = gr.Button("Cancel")

            output = gr.Textbox(
                label="Output Information",
                info="The output information will be displayed here.",
            )

        with gr.Tab("Online Model Hub"):
            with gr.Row():
                model_search = gr.Textbox(
                    label="Search Models",
                    placeholder="Search for models (e.g., singer, voice, rvc)...",
                    value="rvc"
                )

                search_button = gr.Button("Search", variant="secondary")

            with gr.Row():
                model_results = gr.Dataframe(
                    headers=["Model Name", "Version", "Sample Rate", "URL"],
                    datatype=["str", "str", "str", "str"],
                    interactive=False,
                    value=[],
                    elem_id="model_results"
                )

            with gr.Row():
                selected_model_url = gr.Textbox(
                    label="Selected Model URL",
                    interactive=False
                )

                download_selected = gr.Button("Download Selected Model", variant="primary")

        with gr.Tab("Upload Model"):
            gr.Markdown(value="## Drop model files")
            dropbox = gr.File(
                label="Drag your .pth file and .index file into this space.",
                type="filepath",
            )

        with gr.Tab("Download Audio"):
            with gr.Row():
                audio_url = gr.Textbox(
                    label="Audio URL",
                    placeholder="Enter audio URL (YouTube, etc.)",
                    lines=1,
                )
            with gr.Row():
                download_audio = gr.Button("Download", variant="primary")

            audio_output = gr.Textbox(
                label="Audio Path",
                info="The downloaded audio path will be displayed here.",
            )

        # Event handlers
        download.click(
            download_model,
            inputs=[link],
            outputs=[output],
        )

        search_button.click(
            search_models,
            inputs=[model_search],
            outputs=[model_results]
        )

        model_results.select(
            lambda evt: evt[3] if evt else "",
            inputs=None,
            outputs=[selected_model_url]
        )

        download_selected.click(
            download_online_model,
            inputs=[selected_model_url],
            outputs=[output]
        )

        dropbox.upload(
            fn=save_drop_model,
            inputs=[dropbox],
            outputs=[dropbox],
        )
        
        download_audio.click(
            download_url,
            inputs=[audio_url],
            outputs=[audio_output, audio_output, output],
        )

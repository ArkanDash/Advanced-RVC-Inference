import os
import requests
from pathlib import Path

# Function to download file
def download_file(url, dest_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check if the request was successful

        with open(dest_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        print(f"Successfully downloaded {dest_path}")
    
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")

# Directory structure
base_dir = "assets"
directories = ["fcpe", "hubert", "rmvpe"]
urls = [
    "https://huggingface.co/datasets/ylzz1997/rmvpe_pretrain_model/resolve/main/fcpe.pt",
    "https://huggingface.co/Kit-Lemonfoot/RVC_DidntAsk/resolve/main/hubert_base.pt",
    "https://huggingface.co/Kit-Lemonfoot/RVC_DidntAsk/resolve/main/rmvpe.pt"
]

# Ensure directories exist
for directory in directories:
    os.makedirs(Path(base_dir) / directory, exist_ok=True)

# Download the files
for url, directory in zip(urls, directories):
    file_name = url.split("/")[-1]
    dest_path = Path(base_dir) / directory / file_name
    download_file(url, dest_path)

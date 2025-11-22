import urllib.request
from urllib.parse import urlparse

import gdown
import gradio as gr
import requests
from mega import Mega


# Universal function for downloading a file from different sources
def download_file(url, zip_name, progress):
    try:
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname
        if hostname == "drive.google.com":
            download_from_google_drive(url, zip_name, progress)
        elif hostname == "huggingface.co":
            download_from_huggingface(url, zip_name, progress)
        elif hostname == "pixeldrain.com":
            download_from_pixeldrain(url, zip_name, progress)
        elif hostname == "mega.nz":
            download_from_mega(url, zip_name, progress)
        elif hostname in {"disk.yandex.ru", "yadi.sk"}:
            download_from_yandex(url, zip_name, progress)
        else:
            raise ValueError(f"Unsupported source: {url}")  # Handling unsupported links
    except Exception as e:
        # Handle any errors that occurred during downloading
        raise gr.Error(f"Error during download: {str(e)}")


# Downloading a file from Google Drive using the gdown library
def download_from_google_drive(url, zip_name, progress):
    progress(0.5, desc="[~] Downloading model from Google Drive...")
    file_id = url.split("file/d/")[1].split("/")[0] if "file/d/" in url else url.split("id=")[1].split("&")[0]  # Extract file ID
    gdown.download(id=file_id, output=str(zip_name), quiet=False)


# Downloading a file from HuggingFace using urllib
def download_from_huggingface(url, zip_name, progress):
    progress(0.5, desc="[~] Downloading model from HuggingFace...")
    urllib.request.urlretrieve(url, zip_name)


# Downloading a file from Pixeldrain via API
def download_from_pixeldrain(url, zip_name, progress):
    progress(0.5, desc="[~] Downloading model from Pixeldrain...")
    file_id = url.split("pixeldrain.com/u/")[1]  # Extract file ID
    response = requests.get(f"https://pixeldrain.com/api/file/{file_id}")
    with open(zip_name, "wb") as f:
        f.write(response.content)


# Downloading a file from Mega using the Mega library
def download_from_mega(url, zip_name, progress):
    progress(0.5, desc="[~] Downloading model from Mega...")
    m = Mega()
    m.download_url(url, dest_filename=str(zip_name))


# Downloading a file from Yandex Disk via public API
def download_from_yandex(url, zip_name, progress):
    progress(0.5, desc="[~] Downloading model from Yandex Disk...")
    yandex_public_key = f"download?public_key={url}"  # Form the public key
    yandex_api_url = f"https://cloud-api.yandex.net/v1/disk/public/resources/{yandex_public_key}"
    response = requests.get(yandex_api_url)
    if response.status_code == 200:
        download_link = response.json().get("href")  # Obtain the download link
        urllib.request.urlretrieve(download_link, zip_name)
    else:
        # Handle error when retrieving the link from Yandex Disk
        raise gr.Error(f"Error retrieving link from Yandex Disk: {response.status_code}")

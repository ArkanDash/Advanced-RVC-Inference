import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import requests

url_base = "https://huggingface.co/IAHispano/Applio/resolve/main/Resources"

# Define the file lists
models_list = [("predictors/", ["rmvpe.pt", "fcpe.pt"])]
embedders_list = [("embedders/contentvec/", ["pytorch_model.bin", "config.json"])]
executables_list = [
    ("", ["ffmpeg.exe", "ffprobe.exe"]),
]

folder_mapping_list = {
    "embedders/contentvec/": "rvc/models/embedders/contentvec/",
    "predictors/": "rvc/models/predictors/",
    "formant/": "rvc/models/formant/",
}


def get_file_size_all(file_list):
    """
    Calculate the total size of files to be downloaded, regardless of local existence.
    """
    total_size = 0
    for remote_folder, files in file_list:
        # Use the mapping if available; otherwise, use an empty local folder
        local_folder = folder_mapping_list.get(remote_folder, "")
        for file in files:
            url = f"{url_base}/{remote_folder}{file}"
            response = requests.head(url)
            total_size += int(response.headers.get("content-length", 0))
    return total_size


def download_file(url, destination_path, global_bar):
    """
    Download a file from the given URL to the specified destination path,
    updating the global progress bar as data is downloaded.
    """
    dir_name = os.path.dirname(destination_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    response = requests.get(url, stream=True)
    block_size = 1024
    with open(destination_path, "wb") as file:
        for data in response.iter_content(block_size):
            file.write(data)
            global_bar.update(len(data))


def download_mapping_files(file_mapping_list, global_bar):
    """
    Download all files in the provided file mapping list using a thread pool executor,
    and update the global progress bar as downloads progress.
    This version downloads all files regardless of whether they already exist.
    """
    with ThreadPoolExecutor() as executor:
        futures = []
        for remote_folder, file_list in file_mapping_list:
            local_folder = folder_mapping_list.get(remote_folder, "")
            for file in file_list:
                destination_path = os.path.join(local_folder, file)
                url = f"{url_base}/{remote_folder}{file}"
                futures.append(
                    executor.submit(download_file, url, destination_path, global_bar)
                )
        for future in futures:
            future.result()


def calculate_total_size(models, exe):
    """
    Calculate the total size of all files to be downloaded based on selected categories.
    """
    total_size = 0
    if models:
        total_size += get_file_size_all(models_list)
        total_size += get_file_size_all(embedders_list)
    if exe and os.name == "nt":
        total_size += get_file_size_all(executables_list)
    return total_size


def prerequisites_download_pipeline(models, exe):
    """
    Manage the download pipeline for different categories of files.
    """
    total_size = calculate_total_size(models, exe)
    if total_size > 0:
        with tqdm(
            total=total_size, unit="iB", unit_scale=True, desc="Downloading all files"
        ) as global_bar:
            if models:
                download_mapping_files(models_list, global_bar)
                download_mapping_files(embedders_list, global_bar)
            if exe:
                if os.name == "nt":
                    download_mapping_files(executables_list, global_bar)
                else:
                    print("No executables needed for non-Windows systems.")
    else:
        print("No files to download.")

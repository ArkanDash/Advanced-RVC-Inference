import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import requests
import codecs

url_base = "https://huggingface.co/IAHispano/Applio/resolve/main/Resources"
embedders_url = "https://huggingface.co/AnhP/Vietnamese-RVC-Project/resolve/main/embedders/"
predictors_url = "https://huggingface.co/AnhP/Vietnamese-RVC-Project/resolve/main/predictors/"
whisper_url = "https://huggingface.co/AnhP/Vietnamese-RVC-Project/resolve/main/speaker_diarization/"

pretraineds_hifigan_list = [
    (
        "pretrained_v2/",
        [
            "f0D32k.pth",
            "f0D40k.pth",
            "f0D48k.pth",
            "f0G32k.pth",
            "f0G40k.pth",
            "f0G48k.pth",
        ],
    )
]

# Vietnamese-RVC F0 models list (only models that actually exist)
models_list = [
    ("predictors/", [
        "rmvpe.pth",
        "rmvpe.onnx",
        "fcpe.pth", 
        "fcpe.onnx",
        "fcn.pth",
        "fcn.onnx",
        "djcm.pth",
        "djcm.onnx",
        "pesto.pth",
        "pesto.onnx",
        "swift.onnx"
    ])
]

# Enhanced embedders list with multiple modes (only models that exist)
embedders_list = [
    ("embedders/contentvec/", ["pytorch_model.bin", "config.json"]),
    ("embedders/fairseq/", ["hubert_base.pt"]),
    ("embedders/onnx/", ["hubert_base.onnx"])
]
executables_list = [
    ("", ["ffmpeg.exe", "ffprobe.exe"]),
]

folder_mapping_list = {
    "pretrained_v2/": "advanced_rvc_inference/rvc/models/pretraineds/hifi-gan/",
    "embedders/contentvec/": "advanced_rvc_inference/rvc/models/embedders/contentvec/",
    "embedders/fairseq/": "advanced_rvc_inference/rvc/models/embedders/fairseq/",
    "embedders/onnx/": "advanced_rvc_inference/rvc/models/embedders/onnx/",
    "predictors/": "advanced_rvc_inference/rvc/models/predictors/",
    "formant/": "advanced_rvc_inference/rvc/models/formant/",
}


def get_file_size_if_missing(file_list):
    """
    Calculate the total size of files to be downloaded only if they do not exist locally.
    """
    total_size = 0
    for remote_folder, files in file_list:
        local_folder = folder_mapping_list.get(remote_folder, "")
        for file in files:
            destination_path = os.path.join(local_folder, file)
            if not os.path.exists(destination_path):
                url = get_download_url(remote_folder, file)
                try:
                    response = requests.head(url)
                    total_size += int(response.headers.get("content-length", 0))
                except Exception as e:
                    print(f"Could not get size for {url}: {e}")
    return total_size


def get_download_url(remote_folder, file_name):
    """Get the appropriate download URL based on the file type and folder."""
    if remote_folder.startswith("predictors/"):
        return f"{predictors_url}{file_name}"
    elif remote_folder.startswith("embedders/"):
        if "whisper" in remote_folder:
            return f"{whisper_url}{file_name}"
        elif "fairseq" in remote_folder or "onnx" in remote_folder:
            return f"{embedders_url}fairseq/{file_name}" if "fairseq" in remote_folder else f"{embedders_url}onnx/{file_name}"
        else:
            return f"{embedders_url}{remote_folder}{file_name}"
    else:
        return f"{url_base}/{remote_folder}{file_name}"


def download_file(url, destination_path, global_bar):
    """
    Download a file from the given URL to the specified destination path,
    updating the global progress bar as data is downloaded.
    """

    dir_name = os.path.dirname(destination_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        block_size = 1024
        with open(destination_path, "wb") as file:
            for data in response.iter_content(block_size):
                file.write(data)
                global_bar.update(len(data))
    except Exception as e:
        print(f"Failed to download {url}: {str(e)}")
        # Create empty file to prevent repeated failed downloads
        with open(destination_path, "wb") as file:
            file.write(b"")


def download_mapping_files(file_mapping_list, global_bar):
    """
    Download all files in the provided file mapping list using a thread pool executor,
    and update the global progress bar as downloads progress.
    """
    with ThreadPoolExecutor() as executor:
        futures = []
        for remote_folder, file_list in file_mapping_list:
            local_folder = folder_mapping_list.get(remote_folder, "")
            for file in file_list:
                destination_path = os.path.join(local_folder, file)
                if not os.path.exists(destination_path):
                    url = get_download_url(remote_folder, file)
                    futures.append(
                        executor.submit(
                            download_file, url, destination_path, global_bar
                        )
                    )
        for future in futures:
            future.result()


def split_pretraineds(pretrained_list):
    f0_list = []
    non_f0_list = []
    for folder, files in pretrained_list:
        f0_files = [f for f in files if f.startswith("f0")]
        non_f0_files = [f for f in files if not f.startswith("f0")]
        if f0_files:
            f0_list.append((folder, f0_files))
        if non_f0_files:
            non_f0_list.append((folder, non_f0_files))
    return f0_list, non_f0_list


pretraineds_hifigan_list, _ = split_pretraineds(pretraineds_hifigan_list)


def calculate_total_size(
    pretraineds_hifigan,
    models,
    exe,
):
    """
    Calculate the total size of all files to be downloaded based on selected categories.
    """
    total_size = 0
    if models:
        total_size += get_file_size_if_missing(models_list)
        total_size += get_file_size_if_missing(embedders_list)
    if exe and os.name == "nt":
        total_size += get_file_size_if_missing(executables_list)
    total_size += get_file_size_if_missing(pretraineds_hifigan)
    return total_size


def prequisites_download_pipeline(
    pretraineds_hifigan,
    models,
    exe,
):
    """
    Manage the download pipeline for different categories of files.
    """
    total_size = calculate_total_size(
        pretraineds_hifigan_list if pretraineds_hifigan else [],
        models,
        exe,
    )

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
                    print("No executables needed")
            if pretraineds_hifigan:
                download_mapping_files(pretraineds_hifigan_list, global_bar)
    else:
        pass

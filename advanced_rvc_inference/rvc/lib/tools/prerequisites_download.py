import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import requests
from urllib.parse import urlparse

# Base URLs
url_base = "https://huggingface.co/IAHispano/Applio/resolve/main/Resources"
vietnamese_base = "https://huggingface.co/AnhP/Vietnamese-RVC-Project/resolve/main"
ffmpeg_base = "https://huggingface.co/AnhP/Vietnamese-RVC-Project/resolve/main/ffmpeg/"

# Files that actually exist (updated based on available files)
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

# Fixed models list with correct filenames and paths
models_list = [
    ("predictors/", [
        "crepe_full.onnx",
        "crepe_full.pth",
        "crepe_large.pth",
        "crepe_large.onnx",
        "crepe_medium.pth",  # Fixed: Added missing comma
        "crepe_medium.onnx",
        "crepe_small.pth",
        "crepe_small.onnx",  # Fixed: Corrected typo from "onxx" to "onnx"
        "crepe_tiny.onnx",
        "crepe_tiny.pth",
        "fcpe_legacy.pt",
        "fcpe_legacy.onnx",
        "rmvpe.pt",  
        "rmvpe.onnx",
        "fcpe.pt",  
        "fcpe.onnx",
        "swift.onnx",  
        "djcm.pt",
        "djcm.onnx",
        "fcn.pt",
        "fcn.onnx",
    ])
]

# Updated embedders list
embedders_list = [
    ("embedders/contentvec/", ["pytorch_model.bin", "config.json"]),
    ("embedders/fairseq/", ["hubert_base.pt"]),
    ("embedders/onnx/", ["hubert_base.onnx"])
]

# FFmpeg executables for Windows
executables_list = [
    ("ffmpeg/", ["ffmpeg.exe", "ffprobe.exe"]),
]

folder_mapping_list = {
    "pretrained_v2/": "advanced_rvc_inference/rvc/models/pretraineds/hifi-gan/",
    "embedders/contentvec/": "advanced_rvc_inference/rvc/models/embedders/contentvec/",
    "embedders/fairseq/": "advanced_rvc_inference/rvc/models/embedders/fairseq/",
    "embedders/onnx/": "advanced_rvc_inference/rvc/models/embedders/onnx/",
    "predictors/": "advanced_rvc_inference/rvc/models/predictors/",
    "ffmpeg/": "advanced_rvc_inference/ffmpeg/",
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
                    # Use GET with stream=True to handle redirects properly
                    response = requests.get(url, stream=True, timeout=10)
                    if response.status_code == 200:
                        total_size = int(response.headers.get("content-length", 0))
                        response.close()  # Close the connection immediately
                    else:
                        print(f"Warning: File not found {url} (Status: {response.status_code})")
                        response.close()
                except Exception as e:
                    print(f"Could not get size for {url}: {e}")
    return total_size


def get_download_url(remote_folder, file_name):
    """Get the appropriate download URL based on the file type and folder."""
    # FFmpeg executables
    if remote_folder == "ffmpeg/":
        return f"{ffmpeg_base}{file_name}"
    # Files from Vietnamese-RVC-Project repository
    elif remote_folder.startswith("predictors/") or remote_folder.startswith("embedders/"):
        return f"{vietnamese_base}/{remote_folder}{file_name}"
    # Files from Applio repository
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
    
    # Skip if already exists
    if os.path.exists(destination_path):
        return
    
    try:
        # Handle redirects properly by allowing them
        response = requests.get(url, stream=True, timeout=30, allow_redirects=True)
        response.raise_for_status()
        
        # Get total size for better progress tracking
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        with open(destination_path, "wb") as file:
            for data in response.iter_content(block_size):
                file.write(data)
                global_bar.update(len(data))
        
        print(f"✓ Downloaded: {os.path.basename(destination_path)}")
        response.close()
                
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"✗ File not found (404): {url}")
        else:
            print(f"✗ Failed to download {url}: {str(e)}")
        # Don't create empty file - let it fail
        if 'response' in locals():
            response.close()
    except Exception as e:
        print(f"✗ Failed to download {url}: {str(e)}")
        if 'response' in locals():
            response.close()


def download_mapping_files(file_mapping_list, global_bar):
    """
    Download all files in the provided file mapping list using a thread pool executor,
    and update the global progress bar as downloads progress.
    """
    # Filter files that don't exist locally
    download_tasks = []
    for remote_folder, file_list in file_mapping_list:
        local_folder = folder_mapping_list.get(remote_folder, "")
        for file in file_list:
            destination_path = os.path.join(local_folder, file)
            if not os.path.exists(destination_path):
                url = get_download_url(remote_folder, file)
                download_tasks.append((url, destination_path))
    
    if not download_tasks:
        print("All files already exist locally.")
        return
    
    # Download with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for url, destination_path in download_tasks:
            futures.append(
                executor.submit(download_file, url, destination_path, global_bar)
            )
        
        # Wait for all downloads to complete
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Download error: {e}")


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
    if pretraineds_hifigan:
        total_size += get_file_size_if_missing(pretraineds_hifigan_list)
    return total_size


def prerequisites_download_pipeline(
    pretraineds_hifigan=False,
    models=False,
    exe=False,
):
    """
    Manage the download pipeline for different categories of files.
    """
    print("=" * 60)
    print("Checking for missing files...")
    print("=" * 60)
    
    total_size = calculate_total_size(
        pretraineds_hifigan,
        models,
        exe,
    )

    if total_size > 0:
        print(f"Total download size: {total_size / (1024*1024):.2f} MB")
        print("-" * 60)
        with tqdm(
            total=total_size, unit="B", unit_scale=True, unit_divisor=1024, 
            desc="Overall Progress", ncols=80
        ) as global_bar:
            if models:
                print("\nDownloading models...")
                download_mapping_files(models_list, global_bar)
                download_mapping_files(embedders_list, global_bar)
            
            if exe:
                if os.name == "nt":
                    print("\nDownloading FFmpeg executables...")
                    download_mapping_files(executables_list, global_bar)
                else:
                    print("\nNote: FFmpeg executables are only available for Windows")
            
            if pretraineds_hifigan:
                print("\nDownloading pretrained models...")
                download_mapping_files(pretraineds_hifigan_list, global_bar)
        
        print("\n" + "=" * 60)
        print("Download completed!")
        print("=" * 60)
    else:
        print("\nAll required files are already downloaded.")
        print("=" * 60)


# Utility function to verify file existence
def check_file_existence():
    """Check if files exist at their URLs before attempting download."""
    print("Verifying file URLs...")
    print("-" * 60)
    
    # Check predictors
    print("Checking predictors:")
    for remote_folder, files in models_list:
        for file in files:
            url = get_download_url(remote_folder, file)
            try:
                response = requests.head(url, timeout=5, allow_redirects=True)
                status = "✓" if response.status_code == 200 else "✗"
                print(f"  {status} {file}: {response.status_code}")
                response.close()
            except:
                print(f"  ✗ {file}: Connection failed")
    
    print("\nChecking embedders:")
    for remote_folder, files in embedders_list:
        for file in files:
            url = get_download_url(remote_folder, file)
            try:
                response = requests.head(url, timeout=5, allow_redirects=True)
                status = "✓" if response.status_code == 200 else "✗"
                print(f"  {status} {file}: {response.status_code}")
                response.close()
            except:
                print(f"  ✗ {file}: Connection failed")
    
    print("\nChecking FFmpeg executables:")
    for remote_folder, files in executables_list:
        for file in files:
            url = get_download_url(remote_folder, file)
            try:
                response = requests.head(url, timeout=5, allow_redirects=True)
                status = "✓" if response.status_code == 200 else "✗"
                print(f"  {status} {file}: {response.status_code}")
                response.close()
            except:
                print(f"  ✗ {file}: Connection failed")
    
    print("-" * 60)


# Example usage:
if __name__ == "__main__":
    # First verify files exist
    check_file_existence()
    
    # Then download everything
    prerequisites_download_pipeline(
        pretraineds_hifigan=True,
        models=True,
        exe=True  # Download FFmpeg executables
    )

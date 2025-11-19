import os
import re
import sys
import gc
import torch
import faiss
import codecs
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import requests

# Decoded URL from Vietnamese-RVC (rot13 encoded)
url_base = "https://huggingface.co/AnhP/Vietnamese-RVC-Project/resolve/main"
predictors_url = "https://huggingface.co/AnhP/Vietnamese-RVC-Project/resolve/main/predictors/"
embedders_url = "https://huggingface.co/AnhP/Vietnamese-RVC-Project/resolve/main/embedders/"

# Vietnamese-RVC style comprehensive model lists
pretraineds_v1_list = [
    (
        "pretrained_v1/",
        [
            "D32k.pth",
            "D40k.pth", 
            "D48k.pth",
            "G32k.pth",
            "G40k.pth",
            "G48k.pth",
            "f0D32k.pth",
            "f0D40k.pth",
            "f0D48k.pth",
            "f0G32k.pth",
            "f0G40k.pth",
            "f0G48k.pth",
        ],
    )
]
pretraineds_v2_list = [
    (
        "pretrained_v2/",
        [
            "D32k.pth",
            "D40k.pth",
            "D48k.pth", 
            "G32k.pth",
            "G40k.pth",
            "G48k.pth",
            "f0D32k.pth",
            "f0D40k.pth",
            "f0D48k.pth",
            "f0G32k.pth",
            "f0G40k.pth",
            "f0G48k.pth",
        ],
    )
]

# Comprehensive predictors list from Vietnamese-RVC
predictors_list = [
    ("predictors/", [
        # RMVPE variants
        "rmvpe.pt", "rmvpe.onnx",
        # FCPE variants  
        "fcpe.pt", "fcpe.onnx", "fcpe_legacy.pt", "fcpe_legacy.onnx",
        "ddsp_200k.pt", "ddsp_200k.onnx",
        # CREPE variants
        "crepe_tiny.pth", "crepe_tiny.onnx",
        "crepe_small.pth", "crepe_small.onnx", 
        "crepe_medium.pth", "crepe_medium.onnx",
        "crepe_large.pth", "crepe_large.onnx",
        "crepe_full.pth", "crepe_full.onnx",
        # PENN
        "fcn.pt", "fcn.onnx",
        # DJCM
        "djcm.pt", "djcm.onnx",
        # SWIFT (ONNX only)
        "swift.onnx",
        # PESTO
        "pesto.pt", "pesto.onnx",
    ])
]

# Enhanced embedders list (Vietnamese-RVC style)
embedders_list = [
    ("embedders/fairseq/", [
        "contentvec_base.pt",
        "hubert_base.pt",
        "vietnamese_hubert_base.pt",
        "japanese_hubert_base.pt",
        "korean_hubert_base.pt", 
        "chinese_hubert_base.pt",
        "portuguese_hubert_base.pt",
    ]),
    ("embedders/onnx/", [
        "contentvec_base.onnx",
        "hubert_base.onnx",
    ]),
    ("embedders/spin/", [
        "spin-v1/model.safetensors",
        "spin-v1/config.json",
        "spin-v2/model.safetensors", 
        "spin-v2/config.json",
    ]),
    ("embedders/whisper/", [
        "tiny.pt", "base.pt", "small.pt", "medium.pt", "large-v1.pt", "large-v2.pt", "large-v3.pt"
    ]),
]

linux_executables_list = [("formant/", ["stftpitchshift"])]
executables_list = [
    ("", ["ffmpeg.exe", "ffprobe.exe"]),
    ("formant/", ["stftpitchshift.exe"]),
]

# Updated folder mapping to match Vietnamese-RVC structure
folder_mapping_list = {
    "pretrained_v1/": "programs/applio_code/rvc/models/pretraineds/pretrained_v1/",
    "pretrained_v2/": "programs/applio_code/rvc/models/pretraineds/pretrained_v2/",
    "embedders/fairseq/": "programs/applio_code/rvc/models/embedders/fairseq/",
    "embedders/onnx/": "programs/applio_code/rvc/models/embedders/onnx/",
    "embedders/spin/": "programs/applio_code/rvc/models/embedders/spin/",
    "embedders/whisper/": "programs/applio_code/rvc/models/embedders/whisper/",
    "predictors/": "programs/applio_code/rvc/models/predictors/",
    "formant/": "programs/applio_code/rvc/models/formant/",
}

def get_modelname_from_f0_method(f0_method, f0_onnx=False):
    """
    Vietnamese-RVC style model name resolution from F0 method.
    Matches the logic in Vietnamese-RVC's utils.py check_assets function.
    """
    suffix = ".onnx" if f0_onnx else (".pt" if "crepe" not in f0_method else ".pth")
    
    if "rmvpe" in f0_method:
        modelname = "rmvpe"
    elif "fcpe" in f0_method:
        if "legacy" in f0_method and "previous" not in f0_method:
            modelname = "fcpe_legacy"
        elif "previous" in f0_method:
            modelname = "fcpe_legacy" 
        else:
            modelname = "ddsp_200k"
    elif "crepe" in f0_method:
        modelname = "crepe_" + f0_method.replace("mangio-", "").split("-")[1]
    elif "penn" in f0_method:
        modelname = "fcn"
    elif "djcm" in f0_method:
        modelname = "djcm"
    elif "pesto" in f0_method:
        modelname = "pesto"
    elif "swift" in f0_method:
        return "swift.onnx"
    else:
        return None
    
    return modelname + suffix

def download_predictor_files(f0_methods, f0_onnx=False):
    """
    Download predictor files based on F0 methods (supports hybrid methods).
    Vietnamese-RVC style implementation.
    """
    models_to_download = []
    
    for f0_method in f0_methods:
        if "hybrid" in f0_method:
            # Handle hybrid methods by extracting individual methods
            methods_str = re.search(r"hybrid\[(.+)\]", f0_method)
            if methods_str:
                methods = [method.strip() for method in methods_str.group(1).split("+")]
                for method in methods:
                    modelname = get_modelname_from_f0_method(method, f0_onnx)
                    if modelname:
                        models_to_download.append(modelname)
        else:
            # Handle single methods
            modelname = get_modelname_from_f0_method(f0_method, f0_onnx)
            if modelname:
                models_to_download.append(modelname)
    
    return list(set(models_to_download))  # Remove duplicates

def check_predictors_downloaded(f0_methods, f0_onnx=False):
    """
    Check if required predictors are already downloaded.
    Vietnamese-RVC style implementation.
    """
    predictors_path = "programs/applio_code/rvc/models/predictors"
    models_to_download = download_predictor_files(f0_methods, f0_onnx)
    
    missing_models = []
    for modelname in models_to_download:
        model_path = os.path.join(predictors_path, modelname)
        if not os.path.exists(model_path):
            missing_models.append(modelname)
    
    return missing_models


def get_file_size_if_missing(file_list):
    """
    Calculate the total size of files to be downloaded only if they do not exist locally.
    Vietnamese-RVC enhanced version.
    """
    total_size = 0
    for remote_folder, files in file_list:
        local_folder = folder_mapping_list.get(remote_folder, "")
        for file in files:
            destination_path = os.path.join(local_folder, file)
            if not os.path.exists(destination_path):
                try:
                    # Use appropriate URL based on file type
                    if remote_folder == "predictors/":
                        url = f"{predictors_url}{file}"
                    else:
                        url = f"{url_base}/{remote_folder}{file}"
                    response = requests.head(url, timeout=10)
                    if response.status_code == 200:
                        total_size += int(response.headers.get("content-length", 0))
                except Exception as e:
                    print(f"Warning: Could not check size for {file}: {e}")
    return total_size


def download_file(url, destination_path, global_bar):
    """
    Download a file from the given URL to the specified destination path,
    updating the global progress bar as data is downloaded.
    Vietnamese-RVC enhanced version with better error handling.
    """
    try:
        dir_name = os.path.dirname(destination_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        block_size = 1024
        with open(destination_path, "wb") as file:
            for data in response.iter_content(block_size):
                file.write(data)
                if global_bar:
                    global_bar.update(len(data))
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        # Create a placeholder file to avoid future download attempts
        if not os.path.exists(destination_path):
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            with open(destination_path, "w") as f:
                f.write(f"# Download failed: {e}")
        raise


def download_mapping_files(file_mapping_list, global_bar=None):
    """
    Download all files in the provided file mapping list using a thread pool executor,
    and update the global progress bar as downloads progress.
    Vietnamese-RVC enhanced version with better error handling.
    """
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for remote_folder, file_list in file_mapping_list:
            local_folder = folder_mapping_list.get(remote_folder, "")
            for file in file_list:
                destination_path = os.path.join(local_folder, file)
                if not os.path.exists(destination_path):
                    # Use appropriate URL based on file type
                    if remote_folder == "predictors/":
                        url = f"{predictors_url}{file}"
                    else:
                        url = f"{url_base}/{remote_folder}{file}"
                    
                    futures.append(
                        executor.submit(
                            download_file, url, destination_path, global_bar
                        )
                    )
        
        # Wait for all downloads to complete
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Download failed: {e}")
                # Continue with other downloads even if one fails


def calculate_total_size(pretraineds_v1, pretraineds_v2, models, exe, f0_methods=None):
    """
    Calculate the total size of all files to be downloaded based on selected categories.
    Vietnamese-RVC enhanced version with F0 method support.
    """
    total_size = 0
    
    if models:
        # Download all predictors list (comprehensive)
        total_size += get_file_size_if_missing(predictors_list)
        total_size += get_file_size_if_missing(embedders_list)
    
    if exe:
        total_size += get_file_size_if_missing(
            executables_list if os.name == "nt" else linux_executables_list
        )
    
    if pretraineds_v1:
        total_size += get_file_size_if_missing(pretraineds_v1_list)
    
    if pretraineds_v2:
        total_size += get_file_size_if_missing(pretraineds_v2_list)
    
    return total_size


def prerequisites_download_pipeline(pretraineds_v1, pretraineds_v2, models, exe, f0_methods=None):
    """
    Manage the download pipeline for different categories of files.
    Vietnamese-RVC enhanced version with comprehensive predictor support.
    
    Args:
        pretraineds_v1: Download pretrained v1 models
        pretraineds_v2: Download pretrained v2 models  
        models: Download predictors and embedders
        exe: Download executable files
        f0_methods: List of F0 methods to download predictors for (supports hybrid methods)
    """
    total_size = calculate_total_size(pretraineds_v1, pretraineds_v2, models, exe, f0_methods)

    if total_size > 0:
        print(f"Starting download of {total_size / (1024*1024):.2f} MB of files...")
        with tqdm(
            total=total_size, unit="iB", unit_scale=True, desc="Downloading all files"
        ) as global_bar:
            if models:
                print("Downloading comprehensive predictor models...")
                download_mapping_files(predictors_list, global_bar)
                print("Downloading embedder models...")
                download_mapping_files(embedders_list, global_bar)
            if exe:
                print("Downloading executables...")
                download_mapping_files(
                    executables_list if os.name == "nt" else linux_executables_list,
                    global_bar,
                )
            if pretraineds_v1:
                print("Downloading pretrained v1 models...")
                download_mapping_files(pretraineds_v1_list, global_bar)
            if pretraineds_v2:
                print("Downloading pretrained v2 models...")
                download_mapping_files(pretraineds_v2_list, global_bar)
        print("Download completed successfully!")
    else:
        print("All files are already downloaded.")

def check_and_download_predictors(f0_methods):
    """
    Check and download predictors for specific F0 methods (Vietnamese-RVC style).
    This function is used to ensure required predictors are available for inference.
    """
    missing_models = check_predictors_downloaded(f0_methods)
    
    if missing_models:
        print(f"Downloading missing predictors: {missing_models}")
        for model in missing_models:
            destination_path = os.path.join(folder_mapping_list["predictors/"], model)
            if not os.path.exists(destination_path):
                try:
                    if "predictors/" == "predictors/":
                        url = f"{predictors_url}{model}"
                    else:
                        url = f"{url_base}/predictors/{model}"
                    
                    print(f"Downloading {model}...")
                    download_file(url, destination_path)
                    print(f"✓ Downloaded {model}")
                except Exception as e:
                    print(f"✗ Failed to download {model}: {e}")
    else:
        print("All required predictors are already available.")

# Backward compatibility function
prequisites_download_pipeline = prerequisites_download_pipeline


def download_comprehensive_predictors():
    """
    Download all comprehensive predictors available in Vietnamese-RVC system.
    This includes all F0 extraction methods and their variants.
    """
    # Vietnamese-RVC comprehensive F0 method list
    vietnamese_rvc_f0_methods = [
        # Basic methods
        "rmvpe", "fcpe", "harvest", "yin", "pyin", "swipe", "piptrack",
        # PM variants
        "pm-ac", "pm-cc", "pm-shs", 
        # DIO
        "dio",
        # Mangio-CREPE variants
        "mangio-crepe-tiny", "mangio-crepe-small", "mangio-crepe-medium", 
        "mangio-crepe-large", "mangio-crepe-full",
        # CREPE variants
        "crepe-tiny", "crepe-small", "crepe-medium", "crepe-large", "crepe-full",
        # FCPE variants
        "fcpe-legacy", "fcpe-previous",
        # RMVPE variants
        "rmvpe-clipping", "rmvpe-medfilt", "rmvpe-clipping-medfilt",
        # Traditional methods
        "penn", "mangio-penn",
        # DJCM variants
        "djcm", "djcm-clipping", "djcm-medfilt", "djcm-clipping-medfilt",
        # Modern methods
        "swift", "pesto",
        # Hybrid methods (will download all required predictors)
        "hybrid[pm+dio]", "hybrid[pm+crepe-tiny]", "hybrid[pm+crepe]", "hybrid[pm+fcpe]",
        "hybrid[pm+rmvpe]", "hybrid[pm+harvest]", "hybrid[pm+yin]",
        "hybrid[dio+crepe-tiny]", "hybrid[dio+crepe]", "hybrid[dio+fcpe]",
        "hybrid[dio+rmvpe]", "hybrid[dio+harvest]", "hybrid[dio+yin]",
        "hybrid[crepe-tiny+crepe]", "hybrid[crepe-tiny+fcpe]", "hybrid[crepe-tiny+rmvpe]",
        "hybrid[crepe-tiny+harvest]", "hybrid[crepe+fcpe]", "hybrid[crepe+rmvpe]",
        "hybrid[crepe+harvest]", "hybrid[crepe+yin]", "hybrid[fcpe+rmvpe]",
        "hybrid[fcpe+harvest]", "hybrid[fcpe+yin]", "hybrid[rmvpe+harvest]",
        "hybrid[rmvpe+yin]", "hybrid[harvest+yin]",
    ]
    
    print("Downloading comprehensive Vietnamese-RVC predictor set...")
    check_and_download_predictors(vietnamese_rvc_f0_methods)

def download_essential_predictors():
    """
    Download only the essential predictors for basic RVC functionality.
    """
    essential_f0_methods = ["rmvpe", "fcpe", "harvest"]
    print("Downloading essential predictors...")
    check_and_download_predictors(essential_f0_methods)

def list_available_predictors():
    """
    List all available predictors that can be downloaded.
    """
    predictors_info = {
        "F0 Extraction Methods": {
            "Basic": ["rmvpe", "fcpe", "harvest", "yin", "pyin", "swipe", "piptrack"],
            "PM Variants": ["pm-ac", "pm-cc", "pm-shs"],
            "DIO": ["dio"],
            "CREPE Variants": [
                "mangio-crepe-tiny", "mangio-crepe-small", "mangio-crepe-medium", 
                "mangio-crepe-large", "mangio-crepe-full",
                "crepe-tiny", "crepe-small", "crepe-medium", "crepe-large", "crepe-full"
            ],
            "FCPE Variants": ["fcpe-legacy", "fcpe-previous"],
            "RMVPE Variants": ["rmvpe-clipping", "rmvpe-medfilt", "rmvpe-clipping-medfilt"],
            "Advanced": ["penn", "mangio-penn", "djcm", "djcm-clipping", "djcm-medfilt", 
                        "djcm-clipping-medfilt", "swift", "pesto"],
            "Hybrid Methods": [
                "hybrid[pm+dio]", "hybrid[pm+crepe]", "hybrid[pm+fcpe]", "hybrid[pm+rmvpe]",
                "hybrid[crepe+rmvpe]", "hybrid[crepe+harvest]", "hybrid[fcpe+rmvpe]",
                "hybrid[rmvpe+harvest]", "hybrid[harvest+yin]"
            ]
        }
    }
    
    print("Available F0 Extraction Methods:")
    for category, methods in predictors_info["F0 Extraction Methods"].items():
        print(f"\n{category}:")
        for method in methods:
            print(f"  - {method}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "comprehensive":
            download_comprehensive_predictors()
        elif command == "essential":
            download_essential_predictors() 
        elif command == "list":
            list_available_predictors()
        elif command == "pipeline":
            # Run full pipeline (backward compatibility)
            prequisites_download_pipeline(False, False, True, False)
        else:
            print("Usage: python prerequisites_download.py [comprehensive|essential|list|pipeline]")
            print("  comprehensive: Download all Vietnamese-RVC predictors")
            print("  essential: Download only essential predictors")
            print("  list: List all available predictors")
            print("  pipeline: Run backward compatible download pipeline")
    else:
        # Default: download essential predictors
        download_essential_predictors()

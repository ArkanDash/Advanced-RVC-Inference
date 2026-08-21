import os
import re
import sys
import shutil
import zipfile
import requests


from arvc.utils.variables import logger, translations, configs
from arvc.utils.feedback import gr_info, gr_warning, gr_error, process_output, replace_punctuation

def read_docx_text(path):
    import xml.etree.ElementTree

    with zipfile.ZipFile(path) as docx:
        with docx.open("word/document.xml") as document_xml:
            xml_content = document_xml.read()

    WORD_NAMESPACE = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'

    paragraphs = []
    for paragraph in xml.etree.ElementTree.XML(xml_content).iter(WORD_NAMESPACE + 'p'):
        texts = [node.text for node in paragraph.iter(WORD_NAMESPACE + 't') if node.text]
        if texts: paragraphs.append(''.join(texts))

    return '\n'.join(paragraphs)

def process_input(file_path):
    if file_path.endswith(".srt"): file_contents = ""
    elif file_path.endswith(".docx"): file_contents = read_docx_text(file_path)
    else:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                file_contents = file.read()
        except Exception as e:
            gr_warning(translations["read_error"])
            logger.debug(e)
            file_contents = ""

    gr_info(translations["upload_success"].format(name=translations["text"]))
    return file_contents

def move_files_from_directory(src_dir, dest_weights, dest_logs, model_name, use_orig_weight_name=False):
    """Move model files from a source directory to their proper destinations.

    Args:
        src_dir: Source directory containing downloaded/unpacked files.
        dest_weights: Destination directory for model weights (.pth, .onnx).
        dest_logs: Destination directory for index files and model logs.
        model_name: Name to assign to the model.
        use_orig_weight_name: If True, keep the original filename for weights
            instead of renaming to model_name.pth/model_name.onnx.
            Ported from Vietnamese-RVC.
    """
    for root, _, files in os.walk(src_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".index"):
                model_log_dir = os.path.join(dest_logs, model_name)
                os.makedirs(model_log_dir, exist_ok=True)

                filepath = process_output(os.path.join(model_log_dir, replace_punctuation(file)))

                shutil.move(file_path, filepath)
            elif file.endswith(".pth") and not file.startswith("D_") and not file.startswith("G_"):
                pth_path = process_output(os.path.join(dest_weights, model_name + ".pth"))

                shutil.move(file_path, pth_path if not use_orig_weight_name else dest_weights)
            elif file.endswith(".onnx") and not file.startswith("D_") and not file.startswith("G_"):
                pth_path = process_output(os.path.join(dest_weights, model_name + ".onnx"))

                shutil.move(file_path, pth_path if not use_orig_weight_name else dest_weights)

def extract_name_model(filename):
    match = re.search(r"_([A-Za-z0-9]+)(?=_v\d*)", replace_punctuation(filename))
    return match.group(1) if match else None

def save_drop_model(dropboxs):
    weight_folder = configs["weights_path"]
    logs_folder = configs["logs_path"]
    save_model_temp = "save_model_temp"

    if not os.path.exists(weight_folder): os.makedirs(weight_folder, exist_ok=True)
    if not os.path.exists(logs_folder): os.makedirs(logs_folder, exist_ok=True)
    if not os.path.exists(save_model_temp): os.makedirs(save_model_temp, exist_ok=True)

    try:
        for dropbox in dropboxs:
            shutil.move(dropbox, save_model_temp)
            file_name = os.path.basename(dropbox)

            if file_name.endswith(".zip"):
                shutil.unpack_archive(os.path.join(save_model_temp, file_name), save_model_temp)
                move_files_from_directory(save_model_temp, weight_folder, logs_folder, file_name.replace(".zip", ""))
            elif file_name.endswith((".pth", ".onnx")): 
                output_file = process_output(os.path.join(weight_folder, file_name))
                
                shutil.move(os.path.join(save_model_temp, file_name), output_file)
            elif file_name.endswith(".index"):
                modelname = extract_name_model(file_name)
                if modelname is None: modelname = os.path.splitext(os.path.basename(file_name))[0]

                model_logs = os.path.join(logs_folder, modelname)
                if not os.path.exists(model_logs): os.makedirs(model_logs, exist_ok=True)

                shutil.move(os.path.join(save_model_temp, file_name), model_logs)
            else: 
                gr_warning(translations["unable_analyze_model"])
                return None
        
        gr_info(translations["upload_success"].format(name=translations["model"]))
        return None
    except Exception as e:
        gr_error(message=translations["error_occurred"].format(e=e))
        return None
    finally:
        shutil.rmtree(save_model_temp, ignore_errors=True)

def zip_file(name, pth, index):
    pth_path = os.path.join(configs["weights_path"], pth)
    if not pth or not os.path.exists(pth_path) or not pth.endswith((".pth", ".onnx")): return gr_warning(translations["provide_file"].format(filename=translations["model"]))

    zip_file_path = os.path.join(configs["logs_path"], name, name + ".zip")
    zip_dir = os.path.dirname(zip_file_path)
    if not os.path.exists(zip_dir): os.makedirs(zip_dir, exist_ok=True)

    gr_info(translations["start"].format(start=translations["zip"]))

    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        zipf.write(pth_path, os.path.basename(pth_path))
        if index: zipf.write(index, os.path.basename(index))

    gr_info(translations["success"])
    return {"visible": True, "value": zip_file_path, "__type__": "update"}

def fetch_pretrained_data():
    try:
        url = configs.get("pretrained_json_url", "https://huggingface.co/buckets/R-Kentaren/Ultimate-RVC-Models/resolve/json/custom_pretrained.json")
        if not url:
            url = "https://huggingface.co/buckets/R-Kentaren/Ultimate-RVC-Models/resolve/json/custom_pretrained.json"
        url = url.replace("/tree/", "/resolve/")
        response = requests.get(url, timeout=15)
        response.raise_for_status()

        return response.json()
    except Exception as e:
        logger.debug(f"Failed to fetch pretrained data: {e}")
        return {}

def push_to_hub(model_file, index_file, hf_token, hf_repo):
    """Push a trained RVC model and its index file to HuggingFace Hub.

    Args:
        model_file: Filename of the .pth model in the weights directory.
        index_file: Path to the .index file.
        hf_token: HuggingFace API token with write access.
        hf_repo: Target HuggingFace repository (e.g. "username/model-name").

    Returns:
        Status message string.
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        return gr_error("huggingface_hub is not installed. Run: pip install huggingface_hub")

    # Validate inputs
    if not model_file:
        return gr_warning("Please select a model file.")
    if not hf_token:
        return gr_warning("Please provide a HuggingFace token.")
    if not hf_repo:
        return gr_warning("Please provide a HuggingFace repository name (e.g. username/model-name).")

    # Resolve model path
    pth_path = os.path.join(configs["weights_path"], model_file)
    if not os.path.exists(pth_path) or not model_file.endswith((".pth", ".onnx")):
        return gr_warning(f"Model file not found: {pth_path}")

    gr_info(f"Pushing model to HuggingFace Hub: {hf_repo}")

    try:
        api = HfApi()

        # Create repo if it doesn't exist
        create_repo(
            repo_id=hf_repo,
            token=hf_token,
            repo_type="model",
            exist_ok=True,
            private=False,
        )

        # Upload model file
        api.upload_file(
            path_or_fileobj=pth_path,
            path_in_repo=os.path.basename(pth_path),
            repo_id=hf_repo,
            token=hf_token,
            repo_type="model",
        )

        # Upload index file if provided
        if index_file and os.path.exists(index_file):
            api.upload_file(
                path_or_fileobj=index_file,
                path_in_repo=os.path.basename(index_file),
                repo_id=hf_repo,
                token=hf_token,
                repo_type="model",
            )

        # Auto-generate README
        model_name = os.path.splitext(os.path.basename(model_file))[0]
        index_info = f"\n- Index file: `{os.path.basename(index_file)}`" if index_file and os.path.exists(index_file) else ""
        readme_content = f"""---
license: mit
tags:
- rvc
- voice-cloning
- advanced-rvc
---

# {model_name}

This model was uploaded to the HuggingFace Hub using [Advanced-RVC-Inference](https://github.com/ArkanDash/Advanced-RVC-Inference).

## Model Files

- Model weights: `{os.path.basename(pth_path)}`{index_info}

## Usage

This model can be used with any RVC-based inference application that supports the RVC v2 format.

## Disclaimer

This model is intended for research and educational purposes only. Please respect the rights of the original voice owners.
"""

        # Upload README
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as tmp:
            tmp.write(readme_content)
            tmp_path = tmp.name

        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="README.md",
            repo_id=hf_repo,
            token=hf_token,
            repo_type="model",
        )
        os.unlink(tmp_path)

        repo_url = f"https://huggingface.co/{hf_repo}"
        gr_info(f"Successfully pushed model to: {repo_url}")
        return f"Model pushed to HuggingFace Hub: {repo_url}"

    except Exception as e:
        logger.error(f"Failed to push model to HuggingFace Hub: {e}")
        return gr_error(f"Failed to push to HuggingFace Hub: {e}")


def update_sample_rate_dropdown(model):
    data = fetch_pretrained_data()
    if not data or model not in data:
        return {"choices": [], "value": None, "__type__": "update"}
    model_data = data[model]
    if not isinstance(model_data, dict):
        return {"choices": [], "value": None, "__type__": "update"}
    if model != translations["success"]: return {"choices": list(model_data.keys()), "value": list(model_data.keys())[0] if model_data else None, "__type__": "update"}
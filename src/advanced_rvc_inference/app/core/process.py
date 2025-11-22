import os
import re
import sys
import shutil
import codecs
import zipfile
import requests

sys.path.append(os.getcwd())

from main.app.variables import logger, translations, configs
from main.app.core.ui import gr_info, gr_warning, gr_error, process_output, replace_punctuation

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

def move_files_from_directory(src_dir, dest_weights, dest_logs, model_name):
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

                shutil.move(file_path, pth_path)
            elif file.endswith(".onnx") and not file.startswith("D_") and not file.startswith("G_"):
                pth_path = process_output(os.path.join(dest_weights, model_name + ".onnx"))

                shutil.move(file_path, pth_path)

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
    gr_info(translations["start"].format(start=translations["zip"]))

    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        zipf.write(pth_path, os.path.basename(pth_path))
        if index: zipf.write(index, os.path.basename(index))

    gr_info(translations["success"])
    return {"visible": True, "value": zip_file_path, "__type__": "update"}

def fetch_pretrained_data():
    try:
        response = requests.get(codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/wfba/phfgbz_cergenvarq.wfba", "rot13"))
        response.raise_for_status()

        return response.json()
    except:
        return {}

def update_sample_rate_dropdown(model):
    data = fetch_pretrained_data()
    if model != translations["success"]: return {"choices": list(data[model].keys()), "value": list(data[model].keys())[0], "__type__": "update"}
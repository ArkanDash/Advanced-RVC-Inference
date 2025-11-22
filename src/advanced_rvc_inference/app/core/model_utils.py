import os
import sys
import json
import torch
import datetime

sys.path.append(os.getcwd())

from main.app.core.ui import gr_info, gr_warning, gr_error
from main.app.variables import config, logger, translations, configs

def fushion_model_pth(name, pth_1, pth_2, ratio):
    if not name.endswith(".pth"): name = name + ".pth"

    if not pth_1 or not os.path.exists(pth_1) or not pth_1.endswith(".pth"):
        gr_warning(translations["provide_file"].format(filename=translations["model"] + " 1"))
        return [translations["provide_file"].format(filename=translations["model"] + " 1"), None]
    
    if not pth_2 or not os.path.exists(pth_2) or not pth_2.endswith(".pth"):
        gr_warning(translations["provide_file"].format(filename=translations["model"] + " 2"))
        return [translations["provide_file"].format(filename=translations["model"] + " 2"), None]

    from collections import OrderedDict

    def extract(ckpt):
        a = ckpt["model"]
        opt = OrderedDict()
        opt["weight"] = {}

        for key in a.keys():
            if "enc_q" in key: continue

            opt["weight"][key] = a[key]

        return opt
    
    try:
        ckpt1 = torch.load(pth_1, map_location="cpu", weights_only=True)
        ckpt2 = torch.load(pth_2, map_location="cpu", weights_only=True)

        if ckpt1["sr"] != ckpt2["sr"]: 
            gr_warning(translations["sr_not_same"])
            return [translations["sr_not_same"], None]

        cfg = ckpt1["config"]
        cfg_f0 = ckpt1["f0"]
        cfg_version = ckpt1["version"]
        cfg_sr = ckpt1["sr"]

        vocoder = ckpt1.get("vocoder", "Default")
        rms_extract = ckpt1.get("energy", False)

        ckpt1 = extract(ckpt1) if "model" in ckpt1 else ckpt1["weight"]
        ckpt2 = extract(ckpt2) if "model" in ckpt2 else ckpt2["weight"]

        if sorted(list(ckpt1.keys())) != sorted(list(ckpt2.keys())): 
            gr_warning(translations["architectures_not_same"])
            return [translations["architectures_not_same"], None]
         
        gr_info(translations["start"].format(start=translations["fushion_model"]))

        opt = OrderedDict()
        opt["weight"] = {}

        for key in ckpt1.keys():
            if key == "emb_g.weight" and ckpt1[key].shape != ckpt2[key].shape:
                min_shape0 = min(ckpt1[key].shape[0], ckpt2[key].shape[0])
                opt["weight"][key] = (ratio * (ckpt1[key][:min_shape0].float()) + (1 - ratio) * (ckpt2[key][:min_shape0].float())).half()
            else: opt["weight"][key] = (ratio * (ckpt1[key].float()) + (1 - ratio) * (ckpt2[key].float())).half()

        opt["config"] = cfg
        opt["sr"] = cfg_sr
        opt["f0"] = cfg_f0
        opt["version"] = cfg_version
        opt["infos"] = translations["model_fushion_info"].format(name=name, pth_1=pth_1, pth_2=pth_2, ratio=ratio)
        opt["vocoder"] = vocoder
        opt["energy"] = rms_extract

        output_model = configs["weights_path"]
        if not os.path.exists(output_model): os.makedirs(output_model, exist_ok=True)

        torch.save(opt, os.path.join(output_model, name))

        gr_info(translations["success"])
        return [translations["success"], os.path.join(output_model, name)]
    except Exception as e:
        gr_error(message=translations["error_occurred"].format(e=e))
        return [e, None]

def fushion_model(name, path_1, path_2, ratio):
    if not name:
        gr_warning(translations["provide_name_is_save"]) 
        return [translations["provide_name_is_save"], None]

    if path_1.endswith(".pth") and path_2.endswith(".pth"): return fushion_model_pth(name, path_1, path_2, ratio)
    else:
        gr_warning(translations["format_not_valid"])
        return [None, None]
    
def onnx_export(model_path):    
    if not model_path.endswith(".pth"): model_path += ".pth"
    if not model_path or not os.path.exists(model_path) or not model_path.endswith(".pth"): return gr_warning(translations["provide_file"].format(filename=translations["model"]))
    
    try:
        gr_info(translations["start_onnx_export"])

        from main.library.onnx.onnx_export import onnx_exporter
        output = onnx_exporter(model_path, model_path.replace(".pth", ".onnx"), is_half=config.is_half, device=config.device)

        gr_info(translations["success"])
        return output
    except Exception as e:
        return gr_error(e)
    
def model_info(path):
    if not path or not os.path.exists(path) or os.path.isdir(path) or not path.endswith((".pth", ".onnx")): return gr_warning(translations["provide_file"].format(filename=translations["model"]))
    
    def prettify_date(date_str):
        if date_str == translations["not_found_create_time"]: return None

        try:
            return datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f").strftime("%Y-%m-%d %H:%M:%S")
        except ValueError as e:
            logger.debug(e)
            return translations["format_not_valid"]
    
    if path.endswith(".pth"): model_data = torch.load(path, map_location="cpu")
    else:        
        import onnx

        model = onnx.load(path)
        model_data = None

        for prop in model.metadata_props:
            if prop.key == "model_info":
                model_data = json.loads(prop.value)
                break

    gr_info(translations["read_info"])

    epochs = model_data.get("epoch", None)
    if epochs is None: 
        epochs = model_data.get("info", None)
        try:
            epoch = epochs.replace("epoch", "").replace("e", "").isdigit()
            if epoch and epochs is None: epochs = translations["not_found"].format(name=translations["epoch"])
        except: 
            pass

    steps = model_data.get("step", translations["not_found"].format(name=translations["step"]))
    sr = model_data.get("sr", translations["not_found"].format(name=translations["sr"]))
    f0 = model_data.get("f0", translations["not_found"].format(name=translations["f0"]))
    version = model_data.get("version", translations["not_found"].format(name=translations["version"]))
    creation_date = model_data.get("creation_date", translations["not_found_create_time"])
    model_hash = model_data.get("model_hash", translations["not_found"].format(name="model_hash"))
    pitch_guidance = translations["trained_f0"] if f0 else translations["not_f0"]
    creation_date_str = prettify_date(creation_date) if creation_date else translations["not_found_create_time"]
    model_name = model_data.get("model_name", translations["unregistered"])
    model_author = model_data.get("author", translations["not_author"])
    vocoder = model_data.get("vocoder", "Default")
    rms_extract = model_data.get("energy", False)

    gr_info(translations["success"])
    return translations["model_info"].format(model_name=model_name, model_author=model_author, epochs=epochs, steps=steps, version=version, sr=sr, pitch_guidance=pitch_guidance, model_hash=model_hash, creation_date_str=creation_date_str, vocoder=vocoder, rms_extract=rms_extract)
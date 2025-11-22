import os
import re
import sys
import json
import torch
import shutil

import gradio as gr
import sounddevice as sd

sys.path.append(os.getcwd())

from main.library.backends import directml, opencl
from main.inference.realtime.audio import list_audio_device
from main.app.variables import config, configs, configs_json, logger, translations, edgetts, google_tts_voice, method_f0, method_f0_full, vr_models, mdx_models, demucs_models, embedders_model, spin_model, whisper_model

def gr_info(message):
    gr.Info(message, duration=2)
    logger.info(message)

def gr_warning(message):
    gr.Warning(message, duration=2)
    logger.warning(message)

def gr_error(message):
    gr.Error(message=message, duration=6)
    logger.error(message)

def get_gpu_info():
    ngpu = torch.cuda.device_count()
    gpu_infos = [
        f"{i}: {torch.cuda.get_device_name(i)} ({int(torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024 + 0.4)} GB)" 
        for i in range(ngpu) 
        if torch.cuda.is_available() or ngpu != 0
    ]

    if len(gpu_infos) == 0:
        if directml.torch_available:
            ngpu = directml.device_count()
            gpu_infos = [f"{i}: {directml.device_name(i)}" for i in range(ngpu) if directml.is_available() or ngpu != 0]
        elif opencl.torch_available:
            ngpu = opencl.device_count()
            gpu_infos = [f"{i}: {opencl.device_name(i)}" for i in range(ngpu) if opencl.is_available() or ngpu != 0]
        else:
            ngpu = 0
            gpu_infos = []

    return "\n".join(gpu_infos) if len(gpu_infos) > 0 and not config.cpu_mode else translations["no_support_gpu"]

def gpu_number_str():
    if config.cpu_mode: return "-"

    ngpu = torch.cuda.device_count()
    if ngpu == 0: ngpu = directml.device_count() if directml.torch_available else opencl.device_count()

    return str("-".join(map(str, range(ngpu))) if torch.cuda.is_available() or directml.is_available() or opencl.is_available() else "-")

def change_f0_choices(): 
    f0_file = sorted([os.path.abspath(os.path.join(root, f)) for root, _, files in os.walk(configs["f0_path"]) for f in files if f.endswith(".txt")])
    return {"value": f0_file[0] if len(f0_file) >= 1 else "", "choices": f0_file, "__type__": "update"}

def change_audios_choices(input_audio): 
    audios = sorted([os.path.abspath(os.path.join(root, f)) for root, _, files in os.walk(configs["audios_path"]) for f in files if os.path.splitext(f)[1].lower() in (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3")])
    return {"value": input_audio if input_audio != "" else (audios[0] if len(audios) >= 1 else ""), "choices": audios, "__type__": "update"}

def change_reference_choices(): 
    reference = sorted([re.sub(r'_v\d+_(?:[A-Za-z0-9_]+?)_(True|False)_(True|False)$', '', name) for name in os.listdir(configs["reference_path"]) if os.path.exists(os.path.join(configs["reference_path"], name)) and os.path.isdir(os.path.join(configs["reference_path"], name))])
    return {"value": reference[0] if len(reference) >= 1 else "", "choices": reference, "__type__": "update"}

def change_models_choices():
    model, index = sorted(list(model for model in os.listdir(configs["weights_path"]) if model.endswith((".pth", ".onnx")) and not model.startswith("G_") and not model.startswith("D_"))), sorted([os.path.join(root, name) for root, _, files in os.walk(configs["logs_path"], topdown=False) for name in files if name.endswith(".index") and "trained" not in name])
    return [{"value": model[0] if len(model) >= 1 else "", "choices": model, "__type__": "update"}, {"value": index[0] if len(index) >= 1 else "", "choices": index, "__type__": "update"}]

def change_pretrained_choices():
    pretrainD = sorted([model for model in os.listdir(configs["pretrained_custom_path"]) if model.endswith(".pth") and "D" in model])
    pretrainG = sorted([model for model in os.listdir(configs["pretrained_custom_path"]) if model.endswith(".pth") and "G" in model])

    return [{"choices": pretrainD, "value": pretrainD[0] if len(pretrainD) >= 1 else "", "__type__": "update"}, {"choices": pretrainG, "value": pretrainG[0] if len(pretrainG) >= 1 else "", "__type__": "update"}]

def change_choices_del():
    return [{"choices": sorted(list(model for model in os.listdir(configs["weights_path"]) if model.endswith(".pth") and not model.startswith("G_") and not model.startswith("D_"))), "__type__": "update"}, {"choices": sorted([os.path.join(configs["logs_path"], f) for f in os.listdir(configs["logs_path"]) if f not in ["mute", "reference"] and os.path.isdir(os.path.join(configs["logs_path"], f))]), "__type__": "update"}]

def change_preset_choices():
    return {"value": "", "choices": sorted(list(f for f in os.listdir(configs["presets_path"]) if f.endswith(".conversion.json"))), "__type__": "update"}

def change_effect_preset_choices():
    return {"value": "", "choices": sorted(list(f for f in os.listdir(configs["presets_path"]) if f.endswith(".effect.json"))), "__type__": "update"}

def change_tts_voice_choices(google):
    return {"choices": google_tts_voice if google else edgetts, "value": google_tts_voice[0] if google else edgetts[0], "__type__": "update"}

def change_backing_choices(backing, merge):
    if backing or merge: return {"value": False, "interactive": False, "__type__": "update"}
    elif not backing or not merge: return  {"interactive": True, "__type__": "update"}
    else: gr_warning(translations["option_not_valid"])

def change_download_choices(select):
    selects = [False]*10

    if select == translations["download_url"]: selects[0] = selects[1] = selects[2] = True
    elif select == translations["download_from_csv"]:  selects[3] = selects[4] = True
    elif select == translations["search_models"]: selects[5] = selects[6] = True
    elif select == translations["upload"]: selects[9] = True
    else: gr_warning(translations["option_not_valid"])

    return [{"visible": selects[i], "__type__": "update"} for i in range(len(selects))]

def change_download_pretrained_choices(select):
    selects = [False]*7

    if select == translations["download_url"]: selects[0] = selects[1] = selects[2] = True
    elif select == translations["list_model"]: selects[3] = selects[4] = selects[5] = True
    elif select == translations["upload"]: selects[6] = True
    else: gr_warning(translations["option_not_valid"])

    return [{"visible": selects[i], "__type__": "update"} for i in range(len(selects))]

def get_index(model):
    model = os.path.basename(model).split("_")[0]
    return {"value": next((f for f in [os.path.join(root, name) for root, _, files in os.walk(configs["logs_path"], topdown=False) for name in files if name.endswith(".index") and "trained" not in name] if model.split(".")[0] in f), ""), "__type__": "update"} if model else None

def index_strength_show(index):
    return {"visible": index and os.path.exists(index) and os.path.isfile(index), "value": 0.5, "__type__": "update"}

def hoplength_show(method, hybrid_method=None):
    visible = False

    for m in ["mangio-crepe", "fcpe", "yin", "piptrack", "mangio-penn"]:
        if m in method: visible = True
        if hybrid_method is not None and m in hybrid_method: visible = True

        if visible: break
        else: visible = False
    
    return {"visible": visible, "__type__": "update"}

def visible(value):
    return {"visible": value, "__type__": "update"}

def valueFalse_interactive(value): 
    return {"value": False, "interactive": value, "__type__": "update"}

def valueEmpty_visible1(value): 
    return {"value": "", "visible": value, "__type__": "update"}

def pitch_guidance_lock(vocoders):
    return {"value": True, "interactive": vocoders == "Default", "__type__": "update"}

def vocoders_lock(pitch, vocoders):
    return {"value": vocoders if pitch else "Default", "interactive": pitch, "__type__": "update"}

def unlock_f0(value):
    return {"choices": method_f0_full if value else method_f0, "value": "rmvpe", "__type__": "update"} 

def unlock_vocoder(value, vocoder):
    return {"value": vocoder if value == "v2" else "Default", "interactive": value == "v2", "__type__": "update"} 

def unlock_ver(value, vocoder):
    return {"value": "v2" if vocoder == "Default" else value, "interactive": vocoder == "Default", "__type__": "update"}

def change_embedders_mode(value):
    if value == "spin":
        return {"value": spin_model[0], "choices": spin_model, "__type__": "update"}
    elif value == "whisper":
        return {"value": whisper_model[0], "choices": whisper_model, "__type__": "update"}
    else:
        return {"value": embedders_model[0], "choices": embedders_model, "__type__": "update"}

def change_fp(fp):
    fp16 = fp == "fp16"

    if fp16 and config.device in ["cpu", "mps", "ocl:0"]: 
        gr_warning(translations["fp16_not_support"])
        return "fp32"
    else:
        gr_info(translations["start_update_precision"])

        configs = json.load(open(configs_json, "r"))
        configs["fp16"] = config.is_half = fp16

        with open(configs_json, "w") as f:
            json.dump(configs, f, indent=4)

        gr_info(translations["success"])
        return "fp16" if fp16 else "fp32"
    
def process_output(file_path):
    if config.configs.get("delete_exists_file", True):
        if os.path.exists(file_path) and os.path.isfile(file_path): os.remove(file_path)
        return file_path
    else:
        if not os.path.exists(file_path): return file_path
        file = os.path.splitext(os.path.basename(file_path))

        index = 1
        while 1:
            file_path = os.path.join(os.path.dirname(file_path), f"{file[0]}_{index}{file[1]}")
            if not os.path.exists(file_path): return file_path
            index += 1

def shutil_move(input_path, output_path):
    output_path = os.path.join(output_path, os.path.basename(input_path)) if os.path.isdir(output_path) else output_path

    return shutil.move(input_path, process_output(output_path)) if os.path.exists(output_path) else shutil.move(input_path, output_path)

def separate_change(model_name, karaoke_model, reverb_model, enable_post_process, separate_backing, separate_reverb, enable_denoise):
    model_type = "vr" if model_name in list(vr_models.keys()) else "mdx" if model_name in list(mdx_models.keys()) else "demucs" if model_name in list(demucs_models.keys()) else ""
    karaoke_type = ("vr" if karaoke_model.startswith("VR") else "mdx") if separate_backing else None
    reverb_type = ("vr" if not reverb_model.startswith("MDX") else "mdx") if separate_reverb else None

    all_types = {model_type, karaoke_type, reverb_type}

    is_vr = "vr" in all_types
    is_mdx = "mdx" in all_types
    is_demucs = "demucs" in all_types

    return [
        visible(separate_backing),
        visible(separate_reverb),
        visible(is_mdx or is_demucs),
        visible(is_mdx or is_demucs),
        visible(is_mdx),
        visible(is_mdx or is_vr),
        visible(is_demucs),
        visible(is_vr),
        visible(is_vr),
        visible(is_vr and enable_post_process),
        visible(is_vr and enable_denoise),
        valueFalse_interactive(is_vr),
        valueFalse_interactive(is_vr),
        valueFalse_interactive(is_vr)
    ]

def create_dataset_change(model_name, reverb_model, enable_post_process, separate_reverb, enable_denoise):
    model_type = "vr" if model_name in list(vr_models.keys()) else "mdx" if model_name in list(mdx_models.keys()) else "demucs" if model_name in list(demucs_models.keys()) else ""
    reverb_type = ("vr" if not reverb_model.startswith("MDX") else "mdx") if separate_reverb else None
    all_types = {model_type, reverb_type}

    is_vr = "vr" in all_types
    is_mdx = "mdx" in all_types
    is_demucs = "demucs" in all_types

    return [
        visible(separate_reverb),
        visible(is_mdx or is_demucs),
        visible(is_mdx or is_demucs),
        visible(is_mdx),
        visible(is_mdx or is_vr),
        visible(is_demucs),
        visible(is_vr),
        visible(is_vr),
        visible(is_vr and enable_post_process),
        visible(is_vr and enable_denoise),
        valueFalse_interactive(is_vr),
        valueFalse_interactive(is_vr),
        valueFalse_interactive(is_vr)
    ]

def audio_device():
    try:
        input_devices, output_devices = list_audio_device()

        def priority(name):
            n = name.lower()
            if "virtual" in n:
                return 0
            if "vb" in n:
                return 1
            return 2

        output_sorted = sorted(output_devices, key=lambda d: priority(d.name))
        input_sorted = sorted(
            input_devices, key=lambda d: priority(d.name), reverse=True
        )

        input_device_list = {
            f"{input_sorted.index(d)+1}: {d.name} ({d.host_api})": [d.index, d.max_input_channels] for d in input_sorted
        }
        output_device_list = {
            f"{output_sorted.index(d)+1}: {d.name} ({d.host_api})": [d.index, d.max_output_channels] for d in output_sorted
        }

        return input_device_list, output_device_list
    except Exception:
        return [], []

def update_audio_device(input_device, output_device, monitor_device, monitor):
    input_channels_map, output_channels_map = audio_device()

    input_is_asio = "ASIO" in input_device if input_device else False
    output_is_asio = "ASIO" in output_device if output_device else False
    monitor_is_asio = "ASIO" in monitor_device if monitor_device else False

    try:
        input_max_ch = input_channels_map.get(input_device, [])[1]
        output_max_ch = output_channels_map.get(output_device, [])[1]
        monitor_max_ch = output_channels_map.get(monitor_device, [])[1] if monitor else 128
    except:
        input_max_ch = output_max_ch = monitor_max_ch = -1

    return [
        visible(monitor),
        visible(monitor),
        visible(monitor_is_asio),
        visible(input_is_asio or output_is_asio or monitor_is_asio),
        gr.update(visible=input_is_asio, maximum=input_max_ch),
        gr.update(visible=output_is_asio, maximum=output_max_ch),
        gr.update(visible=monitor_is_asio, maximum=monitor_max_ch)
    ]

def change_audio_device_choices():
    sd._terminate()
    sd._initialize()

    input_channels_map, output_channels_map = audio_device()
    input_channels_map, output_channels_map = list(input_channels_map.keys()), list(output_channels_map.keys())

    return [
        {"value": input_channels_map[0] if len(input_channels_map) >= 1 else "", "choices": input_channels_map, "__type__": "update"}, 
        {"value": output_channels_map[0] if len(output_channels_map) >= 1 else "", "choices": output_channels_map, "__type__": "update"},
        {"value": output_channels_map[0] if len(output_channels_map) >= 1 else "", "choices": output_channels_map, "__type__": "update"}
    ]

def replace_punctuation(filename):
    return filename.replace(" ", "_").replace("-", "").replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace(",", "").replace('"', "").replace("'", "").replace("|", "_").replace("{", "").replace("}", "").replace("-_-", "_").replace("_-_", "_").replace("-", "_").replace("---", "_").replace("___", "_").strip()

def replace_url(url):
    return url.replace("/blob/", "/resolve/").replace("?download=true", "").strip()

def replace_modelname(modelname):
    return replace_punctuation(modelname.replace(".onnx", "").replace(".pth", "").replace(".index", "").replace(".zip", ""))

def replace_export_format(audio_path, export_format = "wav"):
    export_format = f".{export_format}"

    return audio_path if audio_path.endswith(export_format) else audio_path.replace(f".{os.path.basename(audio_path).split('.')[-1]}", export_format)
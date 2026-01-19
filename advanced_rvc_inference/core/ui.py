import os
import re
import sys
import json
import torch
import shutil
import logging
from typing import Dict, List, Any, Optional, Tuple, Union

import gradio as gr
import sounddevice as sd

sys.path.append(os.getcwd())

from advanced_rvc_inference.library.backends import directml, opencl
from advanced_rvc_inference.rvc.realtime.audio import list_audio_device
from advanced_rvc_inference.utils.variables import (
    config, configs, configs_json, logger, translations, 
    edgetts, google_tts_voice, method_f0, method_f0_full, 
    vr_models, mdx_models, embedders_model, 
    spin_model, whisper_model
)

# Constants for supported audio formats
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", 
                    ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3")

# Constants for model types
MODEL_TYPES = {
    "vr": list(vr_models.keys()),
    "mdx": list(mdx_models.keys()),
}

# Constants for F0 methods that require hop length
HOPLENGTH_METHODS = ["mangio-crepe", "fcpe", "yin", "piptrack", "mangio-penn"]

# Helper functions for UI updates
def gr_info(message: str) -> None:
    """Display info message in UI and log it"""
    gr.Info(message, duration=2)
    logger.info(message)

def gr_warning(message: str) -> None:
    """Display warning message in UI and log it"""
    gr.Warning(message, duration=2)
    logger.warning(message)

def gr_error(message: str) -> None:
    """Display error message in UI and log it"""
    gr.Error(message=message, duration=6)
    logger.error(message)

def get_gpu_info() -> str:
    """Get information about available GPUs"""
    if config.cpu_mode:
        return translations["no_support_gpu"]
    
    gpu_infos = []
    
    # Check CUDA GPUs
    if torch.cuda.is_available():
        ngpu = torch.cuda.device_count()
        gpu_infos = [
            f"{i}: {torch.cuda.get_device_name(i)} ({int(torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024 + 0.4)} GB)" 
            for i in range(ngpu)
        ]
    
    # Check DirectML GPUs if CUDA not available
    if not gpu_infos and directml.torch_available and directml.is_available():
        ngpu = directml.device_count()
        gpu_infos = [f"{i}: {directml.device_name(i)}" for i in range(ngpu)]
    
    # Check OpenCL GPUs if others not available
    if not gpu_infos and opencl.torch_available and opencl.is_available():
        ngpu = opencl.device_count()
        gpu_infos = [f"{i}: {opencl.device_name(i)}" for i in range(ngpu)]
    
    return "\n".join(gpu_infos) if gpu_infos else translations["no_support_gpu"]

def gpu_number_str() -> str:
    """Get a string representation of available GPU count"""
    if config.cpu_mode:
        return "-"
    
    # Check CUDA GPUs
    if torch.cuda.is_available():
        ngpu = torch.cuda.device_count()
        return str("-".join(map(str, range(ngpu))))
    
    # Check DirectML GPUs if CUDA not available
    if directml.torch_available and directml.is_available():
        ngpu = directml.device_count()
        return str("-".join(map(str, range(ngpu))))
    
    # Check OpenCL GPUs if others not available
    if opencl.torch_available and opencl.is_available():
        ngpu = opencl.device_count()
        return str("-".join(map(str, range(ngpu))))
    
    return "-"

def _get_files_from_directory(directory: str, extensions: Optional[Tuple[str, ...]] = None) -> List[str]:
    """Helper function to get sorted list of files from a directory"""
    if not os.path.exists(directory):
        return []
    
    try:
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if extensions is None or filename.lower().endswith(extensions):
                    files.append(os.path.abspath(os.path.join(root, filename)))
        return sorted(files)
    except Exception as e:
        logger.error(f"Error getting files from {directory}: {str(e)}")
        return []

def change_f0_choices() -> Dict[str, Any]:
    """Update F0 file choices"""
    f0_files = _get_files_from_directory(configs["f0_path"], (".txt",))
    return {"value": f0_files[0] if f0_files else "", "choices": f0_files, "__type__": "update"}

def change_audios_choices(input_audio: str) -> Dict[str, Any]:
    """Update audio file choices"""
    audios = _get_files_from_directory(configs["audios_path"], AUDIO_EXTENSIONS)
    value = input_audio if input_audio and input_audio in audios else (audios[0] if audios else "")
    return {"value": value, "choices": audios, "__type__": "update"}

def change_reference_choices() -> Dict[str, Any]:
    """Update reference model choices"""
    if not os.path.exists(configs["reference_path"]):
        return {"choices": [], "value": "", "__type__": "update"}
    
    try:
        references = []
        for name in os.listdir(configs["reference_path"]):
            dir_path = os.path.join(configs["reference_path"], name)
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                # Clean the name by removing version and parameter info
                clean_name = re.sub(r'_v\d+_(?:[A-Za-z0-9_]+?)_(True|False)_(True|False)$', '', name)
                references.append(clean_name)
        
        references = sorted(set(references))  # Remove duplicates and sort
        value = references[0] if references else ""
        
        return {"choices": references, "value": value, "__type__": "update"}
    except Exception as e:
        logger.error(f"Error updating reference choices: {str(e)}")
        return {"choices": [], "value": "", "__type__": "update"}

def change_models_choices() -> List[Dict[str, Any]]:
    """Update model and index file choices"""
    try:
        # Get model files
        models = []
        if os.path.exists(configs["weights_path"]):
            models = [
                model for model in os.listdir(configs["weights_path"]) 
                if model.endswith((".pth", ".onnx")) and not model.startswith("G_") and not model.startswith("D_")
            ]
            models = sorted(models)
        
        # Get index files
        indexes = []
        if os.path.exists(configs["logs_path"]):
            for root, _, files in os.walk(configs["logs_path"], topdown=False):
                for name in files:
                    if name.endswith(".index") and "trained" not in name:
                        indexes.append(os.path.join(root, name))
            indexes = sorted(indexes)
        
        model_update = {
            "choices": models, 
            "value": models[0] if models else "", 
            "__type__": "update"
        }
        
        index_update = {
            "choices": indexes, 
            "value": indexes[0] if indexes else "", 
            "__type__": "update"
        }
        
        return [model_update, index_update]
    except Exception as e:
        logger.error(f"Error updating model choices: {str(e)}")
        return [{"choices": [], "value": "", "__type__": "update"}] * 2

def change_pretrained_choices() -> List[Dict[str, Any]]:
    """Update pretrained model choices"""
    try:
        pretrainD = []
        pretrainG = []
        
        if os.path.exists(configs["pretrained_custom_path"]):
            pretrainD = sorted([
                model for model in os.listdir(configs["pretrained_custom_path"]) 
                if model.endswith(".pth") and "D" in model
            ])
            
            pretrainG = sorted([
                model for model in os.listdir(configs["pretrained_custom_path"]) 
                if model.endswith(".pth") and "G" in model
            ])
        
        pretrainD_update = {
            "choices": pretrainD, 
            "value": pretrainD[0] if pretrainD else "", 
            "__type__": "update"
        }
        
        pretrainG_update = {
            "choices": pretrainG, 
            "value": pretrainG[0] if pretrainG else "", 
            "__type__": "update"
        }
        
        return [pretrainD_update, pretrainG_update]
    except Exception as e:
        logger.error(f"Error updating pretrained choices: {str(e)}")
        return [{"choices": [], "value": "", "__type__": "update"}] * 2

def change_choices_del() -> List[Dict[str, Any]]:
    """Update choices for model deletion"""
    try:
        models = []
        if os.path.exists(configs["weights_path"]):
            models = sorted([
                model for model in os.listdir(configs["weights_path"]) 
                if model.endswith(".pth") and not model.startswith("G_") and not model.startswith("D_")
            ])
        
        directories = []
        if os.path.exists(configs["logs_path"]):
            directories = sorted([
                f for f in os.listdir(configs["logs_path"]) 
                if f not in ["mute", "reference"] and os.path.isdir(os.path.join(configs["logs_path"], f))
            ])
        
        model_update = {"choices": models, "__type__": "update"}
        directory_update = {"choices": directories, "__type__": "update"}
        
        return [model_update, directory_update]
    except Exception as e:
        logger.error(f"Error updating delete choices: {str(e)}")
        return [{"choices": [], "__type__": "update"}] * 2

def change_preset_choices() -> Dict[str, Any]:
    """Update preset file choices"""
    try:
        presets = []
        if os.path.exists(configs["presets_path"]):
            presets = sorted([
                f for f in os.listdir(configs["presets_path"]) 
                if f.endswith(".conversion.json")
            ])
        
        return {"value": "", "choices": presets, "__type__": "update"}
    except Exception as e:
        logger.error(f"Error updating preset choices: {str(e)}")
        return {"value": "", "choices": [], "__type__": "update"}

def change_effect_preset_choices() -> Dict[str, Any]:
    """Update effect preset file choices"""
    try:
        presets = []
        if os.path.exists(configs["presets_path"]):
            presets = sorted([
                f for f in os.listdir(configs["presets_path"]) 
                if f.endswith(".effect.json")
            ])
        
        return {"value": "", "choices": presets, "__type__": "update"}
    except Exception as e:
        logger.error(f"Error updating effect preset choices: {str(e)}")
        return {"value": "", "choices": [], "__type__": "update"}

def change_tts_voice_choices(google: bool) -> Dict[str, Any]:
    """Update TTS voice choices based on provider"""
    voices = google_tts_voice if google else edgetts
    return {
        "choices": voices, 
        "value": voices[0] if voices else "", 
        "__type__": "update"
    }

def change_backing_choices(backing: bool, merge: bool) -> Dict[str, Any]:
    """Update backing track choices based on options"""
    if backing or merge:
        return {"value": False, "interactive": False, "__type__": "update"}
    elif not backing or not merge:
        return {"interactive": True, "__type__": "update"}
    else:
        gr_warning(translations["option_not_valid"])
        return {"__type__": "update"}

def change_download_choices(select: str) -> List[Dict[str, Any]]:
    """Update download UI based on selected option"""
    selects = [False] * 10
    
    if select == translations["download_url"]:
        selects[0] = selects[1] = selects[2] = True
    elif select == translations["download_from_csv"]:
        selects[3] = selects[4] = True
    elif select == translations["search_models"]:
        selects[5] = selects[6] = True
    elif select == translations["upload"]:
        selects[9] = True
    else:
        gr_warning(translations["option_not_valid"])
    
    return [{"visible": selects[i], "__type__": "update"} for i in range(len(selects))]

def change_download_pretrained_choices(select: str) -> List[Dict[str, Any]]:
    """Update pretrained download UI based on selected option"""
    selects = [False] * 7
    
    if select == translations["download_url"]:
        selects[0] = selects[1] = selects[2] = True
    elif select == translations["list_model"]:
        selects[3] = selects[4] = selects[5] = True
    elif select == translations["upload"]:
        selects[6] = True
    else:
        gr_warning(translations["option_not_valid"])
    
    return [{"visible": selects[i], "__type__": "update"} for i in range(len(selects))]

def get_index(model: str) -> Optional[Dict[str, Any]]:
    """Update index file choice based on selected model"""
    if not model:
        return None
    
    try:
        model_name = os.path.basename(model).split("_")[0]
        
        # Find matching index file
        indexes = []
        if os.path.exists(configs["logs_path"]):
            for root, _, files in os.walk(configs["logs_path"], topdown=False):
                for name in files:
                    if name.endswith(".index") and "trained" not in name:
                        if model_name.split(".")[0] in name:
                            indexes.append(os.path.join(root, name))
        
        indexes = sorted(indexes)
        value = indexes[0] if indexes else ""
        
        return {"value": value, "__type__": "update"}
    except Exception as e:
        logger.error(f"Error getting index for model {model}: {str(e)}")
        return {"value": "", "__type__": "update"}

def index_strength_show(index: str) -> Dict[str, Any]:
    """Update visibility of index strength slider"""
    is_visible = (
        index and 
        os.path.exists(index) and 
        os.path.isfile(index)
    )
    return {
        "visible": is_visible, 
        "value": 0.5 if is_visible else 0, 
        "__type__": "update"
    }

def hoplength_show(method: str, hybrid_method: Optional[str] = None) -> Dict[str, Any]:
    """Update visibility of hop length parameter"""
    is_visible = (
        any(m in method for m in HOPLENGTH_METHODS) or
        (hybrid_method and any(m in hybrid_method for m in HOPLENGTH_METHODS))
    )
    
    return {"visible": is_visible, "__type__": "update"}

def visible(value: bool) -> Dict[str, Any]:
    """Return a Gradio update for visibility"""
    return {"visible": value, "__type__": "update"}

def valueFalse_interactive(value: bool) -> Dict[str, Any]:
    """Return a Gradio update for value and interactivity"""
    return {"value": False, "interactive": value, "__type__": "update"}

def valueEmpty_visible1(value: bool) -> Dict[str, Any]:
    """Return a Gradio update for empty value and visibility"""
    return {"value": "", "visible": value, "__type__": "update"}

def pitch_guidance_lock(vocoders: str) -> Dict[str, Any]:
    """Update pitch guidance based on selected vocoder"""
    is_default = vocoders == "Default"
    return {"value": True, "interactive": is_default, "__type__": "update"}

def vocoders_lock(pitch: bool, vocoders: str) -> Dict[str, Any]:
    """Update vocoder choice based on pitch guidance"""
    value = vocoders if pitch else "Default"
    return {"value": value, "interactive": pitch, "__type__": "update"}

def unlock_f0(value: bool) -> Dict[str, Any]:
    """Unlock F0 options based on value"""
    choices = method_f0_full if value else method_f0
    return {
        "choices": choices, 
        "value": "rmvpe", 
        "__type__": "update"
    }

def unlock_vocoder(value: str, vocoder: str) -> Dict[str, Any]:
    """Unlock vocoder options based on value"""
    is_v2 = value == "v2"
    selected_vocoder = vocoder if is_v2 else "Default"
    return {"value": selected_vocoder, "interactive": is_v2, "__type__": "update"}

def unlock_ver(value: str, vocoder: str) -> Dict[str, Any]:
    """Unlock version options based on vocoder"""
    is_default = vocoder == "Default"
    selected_version = "v2" if is_default else value
    return {"value": selected_version, "interactive": is_default, "__type__": "update"}

def change_embedders_mode(value: str) -> Dict[str, Any]:
    """Update embedder model choices based on selected type"""
    try:
        if value == "spin":
            choices = spin_model
            default = spin_model[0] if spin_model else ""
        elif value == "whisper":
            choices = whisper_model
            default = whisper_model[0] if whisper_model else ""
        else:
            choices = embedders_model
            default = embedders_model[0] if embedders_model else ""
        
        return {
            "choices": choices, 
            "value": default, 
            "__type__": "update"
        }
    except Exception as e:
        logger.error(f"Error updating embedder mode: {str(e)}")
        return {"choices": [], "value": "", "__type__": "update"}

def change_fp(fp: str) -> str:
    """Update model precision and save to config"""
    fp16 = fp == "fp16"
    
    # Check if FP16 is supported on the current device
    if fp16 and config.device in ["cpu", "mps", "ocl:0"]:
        gr_warning(translations["fp16_not_support"])
        return "fp32"
    
    try:
        gr_info(translations["start_update_precision"])
        
        # Load config
        with open(configs_json, "r") as f:
            configs_data = json.load(f)
        
        # Update config
        configs_data["fp16"] = fp16
        config.is_half = fp16
        
        # Save config
        with open(configs_json, "w") as f:
            json.dump(configs_data, f, indent=4)
        
        gr_info(translations["success"])
        return "fp16" if fp16 else "fp32"
    except Exception as e:
        logger.error(f"Error updating precision: {str(e)}")
        gr_error(f"Error updating precision: {str(e)}")
        return "fp32" if not fp16 else "fp16"

def process_output(file_path: str) -> str:
    """Process output file path to avoid overwriting existing files"""
    try:
        if config.configs.get("delete_exists_file", True):
            if os.path.exists(file_path) and os.path.isfile(file_path):
                os.remove(file_path)
            return file_path
        else:
            if not os.path.exists(file_path):
                return file_path
            
            # Generate a new filename to avoid overwriting
            base, ext = os.path.splitext(os.path.basename(file_path))
            directory = os.path.dirname(file_path)
            
            counter = 1
            while True:
                new_file_path = os.path.join(directory, f"{base}_{counter}{ext}")
                if not os.path.exists(new_file_path):
                    return new_file_path
                counter += 1
    except Exception as e:
        logger.error(f"Error processing output path {file_path}: {str(e)}")
        return file_path

def shutil_move(input_path: str, output_path: str) -> str:
    """Safely move a file to a new location"""
    try:
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, os.path.basename(input_path))
        
        processed_path = process_output(output_path)
        return shutil.move(input_path, processed_path)
    except Exception as e:
        logger.error(f"Error moving file from {input_path} to {output_path}: {str(e)}")
        raise

def _get_model_type(model_name: str) -> str:
    """Determine the type of a model based on its name"""
    for model_type, models in MODEL_TYPES.items():
        if model_name in models:
            return model_type
    return ""

def separate_change(model_name: str, karaoke_model: str, reverb_model: str,
                   enable_post_process: bool, separate_backing: bool, 
                   separate_reverb: bool, enable_denoise: bool) -> List[Dict[str, Any]]:
    """Update UI components for audio separation based on selected models"""
    try:
        model_type = _get_model_type(model_name)
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
    except Exception as e:
        logger.error(f"Error in separate_change: {str(e)}")
        return [visible(False) for _ in range(14)]

def create_dataset_change(model_name: str, reverb_model: str, 
                         enable_post_process: bool, separate_reverb: bool, 
                         enable_denoise: bool) -> List[Dict[str, Any]]:
    """Update UI components for dataset creation based on selected models"""
    try:
        model_type = _get_model_type(model_name)
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
    except Exception as e:
        logger.error(f"Error in create_dataset_change: {str(e)}")
        return [visible(False) for _ in range(13)]

def audio_device() -> Tuple[Dict[str, List], Dict[str, List]]:
    """Get available input and output audio devices"""
    try:
        input_devices, output_devices = list_audio_device()
        
        # Priority function to sort devices
        def priority(name):
            n = name.lower()
            if "virtual" in n:
                return 0
            if "vb" in n:
                return 1
            return 2
        
        # Sort devices
        output_sorted = sorted(output_devices, key=lambda d: priority(d.name))
        input_sorted = sorted(input_devices, key=lambda d: priority(d.name), reverse=True)
        
        # Create device dictionaries
        input_device_dict = {
            f"{input_sorted.index(d)+1}: {d.name} ({d.host_api})": [d.index, d.max_input_channels] 
            for d in input_sorted
        }
        output_device_dict = {
            f"{output_sorted.index(d)+1}: {d.name} ({d.host_api})": [d.index, d.max_output_channels] 
            for d in output_sorted
        }
        
        return input_device_dict, output_device_dict
    except Exception as e:
        logger.error(f"Error getting audio devices: {str(e)}")
        return {}, {}

def update_audio_device(input_device: str, output_device: str, 
                       monitor_device: str, monitor: bool) -> List[Dict[str, Any]]:
    """Update audio device UI based on selected devices"""
    try:
        input_devices, output_devices = audio_device()
        
        # Check if devices are ASIO
        input_is_asio = "ASIO" in input_device if input_device else False
        output_is_asio = "ASIO" in output_device if output_device else False
        monitor_is_asio = "ASIO" in monitor_device if monitor_device else False
        
        # Get max channels
        input_max_ch = input_devices.get(input_device, [None, 0])[1]
        output_max_ch = output_devices.get(output_device, [None, 0])[1]
        monitor_max_ch = output_devices.get(monitor_device, [None, 0])[1] if monitor else 128
        
        return [
            visible(monitor),
            visible(monitor),
            visible(monitor_is_asio),
            visible(input_is_asio or output_is_asio or monitor_is_asio),
            gr.update(visible=input_is_asio, maximum=input_max_ch),
            gr.update(visible=output_is_asio, maximum=output_max_ch),
            gr.update(visible=monitor_is_asio, maximum=monitor_max_ch)
        ]
    except Exception as e:
        logger.error(f"Error updating audio device UI: {str(e)}")
        return [visible(False) for _ in range(7)]

def change_audio_device_choices() -> List[Dict[str, Any]]:
    """Refresh audio device list and return updates for dropdowns"""
    try:
        sd._terminate()
        sd._initialize()
        
        input_devices, output_devices = audio_device()
        input_choices = list(input_devices.keys())
        output_choices = list(output_devices.keys())
        
        input_update = {
            "choices": input_choices, 
            "value": input_choices[0] if input_choices else "", 
            "__type__": "update"
        }
        
        output_update = {
            "choices": output_choices, 
            "value": output_choices[0] if output_choices else "", 
            "__type__": "update"
        }
        
        return [input_update, output_update, output_update.copy()]
    except Exception as e:
        logger.error(f"Error changing audio device choices: {str(e)}")
        return [{"choices": [], "value": "", "__type__": "update"}] * 3

def replace_punctuation(filename: str) -> str:
    """Sanitize filename by removing/replacing problematic characters"""
    try:
        return (filename.replace(" ", "_")
                       .replace("-", "")
                       .replace("(", "")
                       .replace(")", "")
                       .replace("[", "")
                       .replace("]", "")
                       .replace(",", "")
                       .replace('"', "")
                       .replace("'", "")
                       .replace("|", "_")
                       .replace("{", "")
                       .replace("}", "")
                       .replace("-_-", "_")
                       .replace("_-_", "_")
                       .replace("-", "_")
                       .replace("---", "_")
                       .replace("___", "_")
                       .strip())
    except Exception as e:
        logger.error(f"Error replacing punctuation in {filename}: {str(e)}")
        return filename

def replace_url(url: str) -> str:
    """Sanitize URL for downloading"""
    try:
        return url.replace("/blob/", "/resolve/").replace("?download=true", "").strip()
    except Exception as e:
        logger.error(f"Error replacing URL in {url}: {str(e)}")
        return url

def replace_modelname(modelname: str) -> str:
    """Sanitize model name by removing extensions and problematic characters"""
    try:
        clean_name = modelname.replace(".onnx", "").replace(".pth", "").replace(".index", "").replace(".zip", "")
        return replace_punctuation(clean_name)
    except Exception as e:
        logger.error(f"Error replacing model name in {modelname}: {str(e)}")
        return modelname

def replace_export_format(audio_path: str, export_format: str = "wav") -> str:
    """Change the export format of an audio file path"""
    try:
        export_format = f".{export_format}"
        if audio_path.endswith(export_format):
            return audio_path
        
        # Replace the extension
        base_path = os.path.splitext(audio_path)[0]
        return f"{base_path}{export_format}"
    except Exception as e:
        logger.error(f"Error replacing export format in {audio_path}: {str(e)}")
        return audio_path

def update_dropdowns_from_json(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Update dropdown choices from JSON data"""
    try:
        if not data:
            return [
                gr.update(choices=[], value=None), 
                gr.update(choices=[], value=None), 
                gr.update(choices=[], value=None)
            ]
        
        inputs = list(data.get("inputs", {}).keys())
        outputs = list(data.get("outputs", {}).keys())
        
        return [
            gr.update(choices=inputs, value=inputs[0] if inputs else None),
            gr.update(choices=outputs, value=outputs[0] if outputs else None),
            gr.update(choices=outputs, value=outputs[0] if outputs else None),
        ]
    except Exception as e:
        logger.error(f"Error updating dropdowns from JSON: {str(e)}")
        return [gr.update(choices=[], value=None)] * 3

def update_button_from_json(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Update button states from JSON data"""
    try:
        if not data:
            return [gr.update(interactive=True), gr.update(interactive=False)]
        
        return [
            gr.update(interactive=data.get("start_button", True)),
            gr.update(interactive=data.get("stop_button", False))
        ]
    except Exception as e:
        logger.error(f"Error updating buttons from JSON: {str(e)}")
        return [gr.update(interactive=False)] * 2

import os
import re
import sys
import json
import torch
import shutil
import gradio as gr
import sounddevice as sd
from typing import Dict, List, Tuple, Any, Optional

# Import application-specific modules
sys.path.append(os.getcwd())
from advanced_rvc_inference.library.backends import directml, opencl
from advanced_rvc_inference.infer.realtime.audio import list_audio_device
from advanced_rvc_inference.variables import (
    config, configs, configs_json, logger, translations, 
    edgetts, google_tts_voice, method_f0, method_f0_full, 
    vr_models, mdx_models, demucs_models, embedders_model, 
    spin_model, whisper_model
)

class UIHelper:
    """Helper class for UI operations and updates"""
    
    @staticmethod
    def show_info(message: str) -> None:
        """Display info message in UI and log it"""
        gr.Info(message, duration=2)
        logger.info(message)
    
    @staticmethod
    def show_warning(message: str) -> None:
        """Display warning message in UI and log it"""
        gr.Warning(message, duration=2)
        logger.warning(message)
    
    @staticmethod
    def show_error(message: str) -> None:
        """Display error message in UI and log it"""
        gr.Error(message=message, duration=6)
        logger.error(message)
    
    @staticmethod
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
    
    @staticmethod
    def get_gpu_count_str() -> str:
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
    
    @staticmethod
    def get_file_list(directory: str, extensions: List[str] = None, 
                     is_directory: bool = False, name_filter: str = None) -> List[str]:
        """Get a sorted list of files or directories from a path"""
        if not os.path.exists(directory):
            return []
        
        try:
            if is_directory:
                items = [os.path.join(directory, name) for name in os.listdir(directory) 
                        if os.path.isdir(os.path.join(directory, name))]
            else:
                items = []
                for root, _, files in os.walk(directory):
                    for file in files:
                        if extensions and not any(file.lower().endswith(ext.lower()) for ext in extensions):
                            continue
                        if name_filter and name_filter not in file:
                            continue
                        items.append(os.path.join(root, file))
            
            return sorted(items)
        except Exception as e:
            logger.error(f"Error getting file list from {directory}: {e}")
            return []
    
    @staticmethod
    def update_dropdown_choices(directory: str, extensions: List[str] = None, 
                               is_directory: bool = False, name_filter: str = None,
                               current_value: str = "") -> Dict[str, Any]:
        """Update dropdown choices with files from a directory"""
        choices = UIHelper.get_file_list(directory, extensions, is_directory, name_filter)
        value = current_value if current_value and current_value in choices else (choices[0] if choices else "")
        
        return {"choices": choices, "value": value, "__type__": "update"}
    
    @staticmethod
    def update_visibility(value: bool) -> Dict[str, Any]:
        """Return a Gradio update for visibility"""
        return {"visible": value, "__type__": "update"}
    
    @staticmethod
    def update_value_and_interactivity(value: Any, interactive: bool) -> Dict[str, Any]:
        """Return a Gradio update for value and interactivity"""
        return {"value": value, "interactive": interactive, "__type__": "update"}
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename by removing/replacing problematic characters"""
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
    
    @staticmethod
    def sanitize_url(url: str) -> str:
        """Sanitize URL for downloading"""
        return url.replace("/blob/", "/resolve/").replace("?download=true", "").strip()
    
    @staticmethod
    def sanitize_model_name(modelname: str) -> str:
        """Sanitize model name by removing extensions and problematic characters"""
        clean_name = modelname.replace(".onnx", "").replace(".pth", "").replace(".index", "").replace(".zip", "")
        return UIHelper.sanitize_filename(clean_name)
    
    @staticmethod
    def change_export_format(audio_path: str, export_format: str = "wav") -> str:
        """Change the export format of an audio file path"""
        export_format = f".{export_format}"
        if audio_path.endswith(export_format):
            return audio_path
        
        # Replace the extension
        base_path = os.path.splitext(audio_path)[0]
        return f"{base_path}{export_format}"
    
    @staticmethod
    def get_unique_filepath(filepath: str) -> str:
        """Get a unique filepath by appending a number if the file exists"""
        if not os.path.exists(filepath):
            return filepath
        
        if not config.configs.get("delete_exists_file", True):
            # If we shouldn't delete existing files, create a new filename
            base, ext = os.path.splitext(os.path.basename(filepath))
            directory = os.path.dirname(filepath)
            
            counter = 1
            while True:
                new_filepath = os.path.join(directory, f"{base}_{counter}{ext}")
                if not os.path.exists(new_filepath):
                    return new_filepath
                counter += 1
        
        # If we should delete existing files, just return the original path
        return filepath
    
    @staticmethod
    def safe_move_file(input_path: str, output_path: str) -> str:
        """Safely move a file to a new location"""
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, os.path.basename(input_path))
        
        output_path = UIHelper.get_unique_filepath(output_path)
        return shutil.move(input_path, output_path)

class AudioDeviceManager:
    """Manager for audio device operations"""
    
    @staticmethod
    def get_audio_devices() -> Tuple[Dict[str, List], Dict[str, List]]:
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
            logger.error(f"Error getting audio devices: {e}")
            return {}, {}
    
    @staticmethod
    def refresh_audio_devices() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Refresh audio device list and return updates for dropdowns"""
        sd._terminate()
        sd._initialize()
        
        input_devices, output_devices = AudioDeviceManager.get_audio_devices()
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
        
        return input_update, output_update, output_update.copy()
    
    @staticmethod
    def update_audio_device_ui(input_device: str, output_device: str, 
                              monitor_device: str, monitor: bool) -> List[Dict[str, Any]]:
        """Update audio device UI based on selected devices"""
        input_devices, output_devices = AudioDeviceManager.get_audio_devices()
        
        # Check if devices are ASIO
        input_is_asio = "ASIO" in input_device if input_device else False
        output_is_asio = "ASIO" in output_device if output_device else False
        monitor_is_asio = "ASIO" in monitor_device if monitor_device else False
        
        # Get max channels
        input_max_ch = input_devices.get(input_device, [None, 0])[1]
        output_max_ch = output_devices.get(output_device, [None, 0])[1]
        monitor_max_ch = output_devices.get(monitor_device, [None, 0])[1] if monitor else 128
        
        return [
            UIHelper.update_visibility(monitor),
            UIHelper.update_visibility(monitor),
            UIHelper.update_visibility(monitor_is_asio),
            UIHelper.update_visibility(input_is_asio or output_is_asio or monitor_is_asio),
            gr.update(visible=input_is_asio, maximum=input_max_ch),
            gr.update(visible=output_is_asio, maximum=output_max_ch),
            gr.update(visible=monitor_is_asio, maximum=monitor_max_ch)
        ]

class ModelManager:
    """Manager for model-related operations"""
    
    @staticmethod
    def get_model_type(model_name: str) -> str:
        """Determine the type of a model based on its name"""
        if model_name in list(vr_models.keys()):
            return "vr"
        elif model_name in list(mdx_models.keys()):
            return "mdx"
        elif model_name in list(demucs_models.keys()):
            return "demucs"
        return ""
    
    @staticmethod
    def update_separation_ui(model_name: str, karaoke_model: str, reverb_model: str,
                           enable_post_process: bool, separate_backing: bool, 
                           separate_reverb: bool, enable_denoise: bool) -> List[Dict[str, Any]]:
        """Update UI components for audio separation based on selected models"""
        model_type = ModelManager.get_model_type(model_name)
        karaoke_type = ("vr" if karaoke_model.startswith("VR") else "mdx") if separate_backing else None
        reverb_type = ("vr" if not reverb_model.startswith("MDX") else "mdx") if separate_reverb else None
        
        all_types = {model_type, karaoke_type, reverb_type}
        
        is_vr = "vr" in all_types
        is_mdx = "mdx" in all_types
        is_demucs = "demucs" in all_types
        
        return [
            UIHelper.update_visibility(separate_backing),
            UIHelper.update_visibility(separate_reverb),
            UIHelper.update_visibility(is_mdx or is_demucs),
            UIHelper.update_visibility(is_mdx or is_demucs),
            UIHelper.update_visibility(is_mdx),
            UIHelper.update_visibility(is_mdx or is_vr),
            UIHelper.update_visibility(is_demucs),
            UIHelper.update_visibility(is_vr),
            UIHelper.update_visibility(is_vr),
            UIHelper.update_visibility(is_vr and enable_post_process),
            UIHelper.update_visibility(is_vr and enable_denoise),
            UIHelper.update_value_and_interactivity(False, is_vr),
            UIHelper.update_value_and_interactivity(False, is_vr),
            UIHelper.update_value_and_interactivity(False, is_vr)
        ]
    
    @staticmethod
    def update_dataset_ui(model_name: str, reverb_model: str, 
                         enable_post_process: bool, separate_reverb: bool, 
                         enable_denoise: bool) -> List[Dict[str, Any]]:
        """Update UI components for dataset creation based on selected models"""
        model_type = ModelManager.get_model_type(model_name)
        reverb_type = ("vr" if not reverb_model.startswith("MDX") else "mdx") if separate_reverb else None
        
        all_types = {model_type, reverb_type}
        
        is_vr = "vr" in all_types
        is_mdx = "mdx" in all_types
        is_demucs = "demucs" in all_types
        
        return [
            UIHelper.update_visibility(separate_reverb),
            UIHelper.update_visibility(is_mdx or is_demucs),
            UIHelper.update_visibility(is_mdx or is_demucs),
            UIHelper.update_visibility(is_mdx),
            UIHelper.update_visibility(is_mdx or is_vr),
            UIHelper.update_visibility(is_demucs),
            UIHelper.update_visibility(is_vr),
            UIHelper.update_visibility(is_vr),
            UIHelper.update_visibility(is_vr and enable_post_process),
            UIHelper.update_visibility(is_vr and enable_denoise),
            UIHelper.update_value_and_interactivity(False, is_vr),
            UIHelper.update_value_and_interactivity(False, is_vr),
            UIHelper.update_value_and_interactivity(False, is_vr)
        ]

# UI Update Functions
def update_f0_choices() -> Dict[str, Any]:
    """Update F0 file choices"""
    return UIHelper.update_dropdown_choices(configs["f0_path"], [".txt"])

def update_audio_choices(input_audio: str) -> Dict[str, Any]:
    """Update audio file choices"""
    audio_extensions = [".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", 
                       ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3"]
    return UIHelper.update_dropdown_choices(configs["audios_path"], audio_extensions, 
                                           current_value=input_audio)

def update_reference_choices() -> Dict[str, Any]:
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
        logger.error(f"Error updating reference choices: {e}")
        return {"choices": [], "value": "", "__type__": "update"}

def update_model_choices() -> List[Dict[str, Any]]:
    """Update model and index file choices"""
    # Get model files
    models = UIHelper.get_file_list(
        configs["weights_path"], 
        [".pth", ".onnx"], 
        name_filter=lambda f: not f.startswith("G_") and not f.startswith("D_")
    )
    
    # Get index files
    indexes = []
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

def update_pretrained_choices() -> List[Dict[str, Any]]:
    """Update pretrained model choices"""
    pretrainD = UIHelper.get_file_list(
        configs["pretrained_custom_path"], 
        [".pth"], 
        name_filter=lambda f: "D" in f
    )
    
    pretrainG = UIHelper.get_file_list(
        configs["pretrained_custom_path"], 
        [".pth"], 
        name_filter=lambda f: "G" in f
    )
    
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

def update_delete_choices() -> List[Dict[str, Any]]:
    """Update choices for model deletion"""
    # Get model files
    models = UIHelper.get_file_list(
        configs["weights_path"], 
        [".pth"], 
        name_filter=lambda f: not f.startswith("G_") and not f.startswith("D_")
    )
    
    # Get model directories
    directories = []
    for f in os.listdir(configs["logs_path"]):
        if f not in ["mute", "reference"] and os.path.isdir(os.path.join(configs["logs_path"], f)):
            directories.append(f)
    directories = sorted(directories)
    
    model_update = {"choices": models, "__type__": "update"}
    directory_update = {"choices": directories, "__type__": "update"}
    
    return [model_update, directory_update]

def update_preset_choices() -> Dict[str, Any]:
    """Update preset file choices"""
    return UIHelper.update_dropdown_choices(
        configs["presets_path"], 
        [".conversion.json"]
    )

def update_effect_preset_choices() -> Dict[str, Any]:
    """Update effect preset file choices"""
    return UIHelper.update_dropdown_choices(
        configs["presets_path"], 
        [".effect.json"]
    )

def update_tts_voice_choices(google: bool) -> Dict[str, Any]:
    """Update TTS voice choices based on provider"""
    voices = google_tts_voice if google else edgetts
    return {
        "choices": voices, 
        "value": voices[0] if voices else "", 
        "__type__": "update"
    }

def update_backing_choices(backing: bool, merge: bool) -> Dict[str, Any]:
    """Update backing track choices based on options"""
    if backing or merge:
        return UIHelper.update_value_and_interactivity(False, False)
    elif not backing or not merge:
        return {"interactive": True, "__type__": "update"}
    else:
        UIHelper.show_warning(translations["option_not_valid"])
        return {"__type__": "update"}

def update_download_choices(select: str) -> List[Dict[str, Any]]:
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
        UIHelper.show_warning(translations["option_not_valid"])
    
    return [{"visible": selects[i], "__type__": "update"} for i in range(len(selects))]

def update_download_pretrained_choices(select: str) -> List[Dict[str, Any]]:
    """Update pretrained download UI based on selected option"""
    selects = [False] * 7
    
    if select == translations["download_url"]:
        selects[0] = selects[1] = selects[2] = True
    elif select == translations["list_model"]:
        selects[3] = selects[4] = selects[5] = True
    elif select == translations["upload"]:
        selects[6] = True
    else:
        UIHelper.show_warning(translations["option_not_valid"])
    
    return [{"visible": selects[i], "__type__": "update"} for i in range(len(selects))]

def update_index_for_model(model: str) -> Dict[str, Any]:
    """Update index file choice based on selected model"""
    if not model:
        return {"__type__": "update"}
    
    model_name = os.path.basename(model).split("_")[0]
    
    # Find matching index file
    indexes = []
    for root, _, files in os.walk(configs["logs_path"], topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                if model_name.split(".")[0] in name:
                    indexes.append(os.path.join(root, name))
    
    indexes = sorted(indexes)
    value = indexes[0] if indexes else ""
    
    return {"value": value, "__type__": "update"}

def update_index_strength_visibility(index: str) -> Dict[str, Any]:
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

def update_hoplength_visibility(method: str, hybrid_method: Optional[str] = None) -> Dict[str, Any]:
    """Update visibility of hop length parameter"""
    hoplength_methods = ["mangio-crepe", "fcpe", "yin", "piptrack", "mangio-penn"]
    
    is_visible = (
        any(m in method for m in hoplength_methods) or
        (hybrid_method and any(m in hybrid_method for m in hoplength_methods))
    )
    
    return UIHelper.update_visibility(is_visible)

def update_pitch_guidance(vocoders: str) -> Dict[str, Any]:
    """Update pitch guidance based on selected vocoder"""
    is_default = vocoders == "Default"
    return UIHelper.update_value_and_interactivity(True, is_default)

def update_vocoders(pitch: bool, vocoders: str) -> Dict[str, Any]:
    """Update vocoder choice based on pitch guidance"""
    value = vocoders if pitch else "Default"
    return UIHelper.update_value_and_interactivity(value, pitch)

def unlock_f0_options(value: bool) -> Dict[str, Any]:
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
    return UIHelper.update_value_and_interactivity(selected_vocoder, is_v2)

def unlock_version(value: str, vocoder: str) -> Dict[str, Any]:
    """Unlock version options based on vocoder"""
    is_default = vocoder == "Default"
    selected_version = "v2" if is_default else value
    return UIHelper.update_value_and_interactivity(selected_version, is_default)

def update_embedder_model(value: str) -> Dict[str, Any]:
    """Update embedder model choices based on selected type"""
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

def update_precision(fp: str) -> str:
    """Update model precision and save to config"""
    fp16 = fp == "fp16"
    
    # Check if FP16 is supported on the current device
    if fp16 and config.device in ["cpu", "mps", "ocl:0"]:
        UIHelper.show_warning(translations["fp16_not_support"])
        return "fp32"
    
    try:
        UIHelper.show_info(translations["start_update_precision"])
        
        # Load config
        with open(configs_json, "r") as f:
            configs_data = json.load(f)
        
        # Update config
        configs_data["fp16"] = fp16
        config.is_half = fp16
        
        # Save config
        with open(configs_json, "w") as f:
            json.dump(configs_data, f, indent=4)
        
        UIHelper.show_info(translations["success"])
        return "fp16" if fp16 else "fp32"
    except Exception as e:
        logger.error(f"Error updating precision: {e}")
        UIHelper.show_error(f"Error updating precision: {str(e)}")
        return "fp32" if not fp16 else "fp16"

def update_dropdowns_from_json(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Update dropdown choices from JSON data"""
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

def update_buttons_from_json(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Update button states from JSON data"""
    if not data:
        return [gr.update(interactive=True), gr.update(interactive=False)]
    
    return [
        gr.update(interactive=data.get("start_button", True)),
        gr.update(interactive=data.get("stop_button", False))
    ]

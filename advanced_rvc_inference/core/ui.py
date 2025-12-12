import os
import re
import sys
import json
import torch
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from functools import lru_cache

import gradio as gr
import sounddevice as sd

sys.path.append(os.getcwd())

from advanced_rvc_inference.library.backends import directml, opencl
from advanced_rvc_inference.infer.realtime.audio import list_audio_device
from advanced_rvc_inference.variables import (
    config, configs, configs_json, logger, translations, 
    edgetts, google_tts_voice, method_f0, method_f0_full,
    vr_models, mdx_models, demucs_models, embedders_model, 
    spin_model, whisper_model
)

# Constants
SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".ogg", ".opus", 
    ".m4a", ".mp4", ".aac", ".alac", ".wma", 
    ".aiff", ".webm", ".ac3"
}
CONVERSION_PRESET_EXT = ".conversion.json"
EFFECT_PRESET_EXT = ".effect.json"

@dataclass
class ModelInfo:
    name: str
    path: str
    is_onnx: bool = False
    is_pth: bool = False

# UI Notification Functions
def gr_info(message: str) -> None:
    """Display info message in Gradio UI."""
    gr.Info(message, duration=2)
    logger.info(message)

def gr_warning(message: str) -> None:
    """Display warning message in Gradio UI."""
    gr.Warning(message, duration=2)
    logger.warning(message)

def gr_error(message: str) -> None:
    """Display error message in Gradio UI."""
    gr.Error(message=message, duration=6)
    logger.error(message)

# GPU/Device Management
def get_gpu_info() -> str:
    """Get GPU information with fallback to alternative backends."""
    gpu_info_list = []
    
    if torch.cuda.is_available():
        ngpu = torch.cuda.device_count()
        gpu_info_list = [
            f"{i}: {torch.cuda.get_device_name(i)} "
            f"({torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB)"
            for i in range(ngpu)
        ]
    elif directml.torch_available:
        ngpu = directml.device_count()
        gpu_info_list = [f"{i}: {directml.device_name(i)}" 
                        for i in range(ngpu) if directml.is_available()]
    elif opencl.torch_available:
        ngpu = opencl.device_count()
        gpu_info_list = [f"{i}: {opencl.device_name(i)}" 
                        for i in range(ngpu) if opencl.is_available()]
    
    if gpu_info_list and not config.cpu_mode:
        return "\n".join(gpu_info_list)
    return translations["no_support_gpu"]

def gpu_number_str() -> str:
    """Get GPU numbers as string for display."""
    if config.cpu_mode:
        return "-"
    
    ngpu = torch.cuda.device_count()
    if ngpu == 0 and directml.torch_available:
        ngpu = directml.device_count()
    elif ngpu == 0 and opencl.torch_available:
        ngpu = opencl.device_count()
    
    if ngpu > 0:
        return "-".join(map(str, range(ngpu)))
    return "-"

# File System Utilities
@lru_cache(maxsize=32)
def _scan_directory(path: str, extensions: Optional[set] = None, 
                    is_dir: bool = False) -> List[str]:
    """Scan directory for files with caching."""
    if not os.path.exists(path):
        return []
    
    if is_dir:
        return sorted([
            name for name in os.listdir(path) 
            if os.path.isdir(os.path.join(path, name))
        ])
    
    files = []
    for root, _, filenames in os.walk(path):
        for f in filenames:
            if not extensions or Path(f).suffix.lower() in extensions:
                files.append(os.path.abspath(os.path.join(root, f)))
    return sorted(files)

def change_f0_choices() -> Dict[str, Any]:
    """Update F0 file choices."""
    f0_files = _scan_directory(configs["f0_path"], {".txt"})
    return {
        "value": f0_files[0] if f0_files else "",
        "choices": f0_files,
        "__type__": "update"
    }

def change_audios_choices(input_audio: str = "") -> Dict[str, Any]:
    """Update audio file choices."""
    audios = _scan_directory(configs["audios_path"], SUPPORTED_AUDIO_EXTENSIONS)
    value = input_audio if input_audio and input_audio in audios else ""
    if not value and audios:
        value = audios[0]
    return {
        "value": value,
        "choices": audios,
        "__type__": "update"
    }

def change_reference_choices() -> Dict[str, Any]:
    """Update reference model choices."""
    references = []
    ref_path = Path(configs["reference_path"])
    
    if ref_path.exists():
        for item in ref_path.iterdir():
            if item.is_dir():
                # Clean up the name
                clean_name = re.sub(
                    r'_v\d+_(?:[A-Za-z0-9_]+?)_(True|False)_(True|False)$',
                    '', 
                    item.name
                )
                references.append(clean_name)
    
    return {
        "value": references[0] if references else "",
        "choices": sorted(references),
        "__type__": "update"
    }

def change_models_choices() -> List[Dict[str, Any]]:
    """Update model and index choices."""
    weights_path = Path(configs["weights_path"])
    logs_path = Path(configs["logs_path"])
    
    # Find models
    models = []
    if weights_path.exists():
        models = sorted([
            f.name for f in weights_path.iterdir()
            if f.suffix in (".pth", ".onnx") 
            and not f.name.startswith(("G_", "D_"))
        ])
    
    # Find index files
    index_files = []
    if logs_path.exists():
        for root, _, files in os.walk(logs_path):
            for file in files:
                if file.endswith(".index") and "trained" not in file:
                    index_files.append(os.path.join(root, file))
    
    return [
        {
            "value": models[0] if models else "",
            "choices": models,
            "__type__": "update"
        },
        {
            "value": index_files[0] if index_files else "",
            "choices": sorted(index_files),
            "__type__": "update"
        }
    ]

def change_pretrained_choices() -> List[Dict[str, Any]]:
    """Update pretrained model choices."""
    pretrain_path = Path(configs["pretrained_custom_path"])
    
    pretrainD = []
    pretrainG = []
    
    if pretrain_path.exists():
        for item in pretrain_path.iterdir():
            if item.suffix == ".pth":
                if "D" in item.name:
                    pretrainD.append(item.name)
                elif "G" in item.name:
                    pretrainG.append(item.name)
    
    return [
        {
            "choices": sorted(pretrainD),
            "value": pretrainD[0] if pretrainD else "",
            "__type__": "update"
        },
        {
            "choices": sorted(pretrainG),
            "value": pretrainG[0] if pretrainG else "",
            "__type__": "update"
        }
    ]

def change_preset_choices() -> Dict[str, Any]:
    """Update preset choices."""
    presets = _scan_directory(
        configs["presets_path"], 
        {CONVERSION_PRESET_EXT}
    )
    return {
        "value": "",
        "choices": sorted([Path(p).name for p in presets]),
        "__type__": "update"
    }

def change_effect_preset_choices() -> Dict[str, Any]:
    """Update effect preset choices."""
    effect_presets = _scan_directory(
        configs["presets_path"], 
        {EFFECT_PRESET_EXT}
    )
    return {
        "value": "",
        "choices": sorted([Path(p).name for p in effect_presets]),
        "__type__": "update"
    }

# TTS Voice Management
def change_tts_voice_choices(use_google: bool) -> Dict[str, Any]:
    """Update TTS voice choices based on provider."""
    voices = google_tts_voice if use_google else edgetts
    return {
        "choices": voices,
        "value": voices[0] if voices else "",
        "__type__": "update"
    }

# Audio Device Management
def audio_device() -> Tuple[Dict[str, List], Dict[str, List]]:
    """Get audio device information with priority sorting."""
    try:
        input_devices, output_devices = list_audio_device()
        
        def device_priority(name: str) -> int:
            """Priority for device sorting (lower is better)."""
            name_lower = name.lower()
            if "virtual" in name_lower:
                return 0
            if "vb" in name_lower:
                return 1
            return 2
        
        # Sort devices
        input_sorted = sorted(input_devices, 
                            key=lambda d: device_priority(d.name))
        output_sorted = sorted(output_devices, 
                             key=lambda d: device_priority(d.name))
        
        # Create mapping dictionaries
        input_map = {
            f"{i+1}: {d.name} ({d.host_api})": [d.index, d.max_input_channels]
            for i, d in enumerate(input_sorted)
        }
        
        output_map = {
            f"{i+1}: {d.name} ({d.host_api})": [d.index, d.max_output_channels]
            for i, d in enumerate(output_sorted)
        }
        
        return input_map, output_map
    except Exception as e:
        logger.error(f"Error getting audio devices: {e}")
        return {}, {}

def update_audio_device(input_device: str, output_device: str, 
                       monitor_device: str, monitor: bool) -> List[Dict[str, Any]]:
    """Update audio device visibility and channel settings."""
    input_channels_map, output_channels_map = audio_device()
    
    # Check for ASIO devices
    input_is_asio = "ASIO" in input_device if input_device else False
    output_is_asio = "ASIO" in output_device if output_device else False
    monitor_is_asio = "ASIO" in monitor_device if monitor_device else False
    
    # Get max channels with safe defaults
    try:
        input_max_ch = input_channels_map.get(input_device, [None, 2])[1]
        output_max_ch = output_channels_map.get(output_device, [None, 2])[1]
        monitor_max_ch = output_channels_map.get(monitor_device, [None, 2])[1] if monitor else 128
    except (IndexError, TypeError):
        input_max_ch = output_max_ch = monitor_max_ch = 2
    
    return [
        {"visible": monitor, "__type__": "update"},
        {"visible": monitor, "__type__": "update"},
        {"visible": monitor_is_asio, "__type__": "update"},
        {"visible": input_is_asio or output_is_asio or monitor_is_asio, 
         "__type__": "update"},
        gr.update(visible=input_is_asio, maximum=input_max_ch),
        gr.update(visible=output_is_asio, maximum=output_max_ch),
        gr.update(visible=monitor_is_asio, maximum=monitor_max_ch)
    ]

def change_audio_device_choices() -> List[Dict[str, Any]]:
    """Refresh audio device choices."""
    try:
        sd._terminate()
        sd._initialize()
        
        input_channels_map, output_channels_map = audio_device()
        input_keys = list(input_channels_map.keys())
        output_keys = list(output_channels_map.keys())
        
        return [
            {
                "value": input_keys[0] if input_keys else "",
                "choices": input_keys,
                "__type__": "update"
            },
            {
                "value": output_keys[0] if output_keys else "",
                "choices": output_keys,
                "__type__": "update"
            },
            {
                "value": output_keys[0] if output_keys else "",
                "choices": output_keys,
                "__type__": "update"
            }
        ]
    except Exception as e:
        logger.error(f"Error refreshing audio devices: {e}")
        return [
            {"value": "", "choices": [], "__type__": "update"} 
            for _ in range(3)
        ]

# Model and Processing Utilities
def get_index(model_name: str) -> Dict[str, Any]:
    """Get corresponding index file for a model."""
    if not model_name:
        return {"value": "", "__type__": "update"}
    
    model_base = Path(model_name).stem.split(".")[0]
    index_files = _scan_directory(configs["logs_path"], {".index"})
    
    for index_file in index_files:
        if model_base in index_file and "trained" not in index_file:
            return {"value": index_file, "__type__": "update"}
    
    return {"value": "", "__type__": "update"}

def index_strength_show(index_path: str) -> Dict[str, Any]:
    """Show/hide index strength slider based on index file existence."""
    is_visible = bool(
        index_path and 
        os.path.exists(index_path) and 
        os.path.isfile(index_path)
    )
    return {"visible": is_visible, "value": 0.5, "__type__": "update"}

# String Processing Utilities
def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing problematic characters."""
    # Define characters to replace and remove
    replace_chars = {
        " ": "_",
        "|": "_",
        "-": "_"
    }
    
    remove_chars = '()[]{},;"\''
    
    # Replace characters
    for old, new in replace_chars.items():
        filename = filename.replace(old, new)
    
    # Remove characters
    for char in remove_chars:
        filename = filename.replace(char, "")
    
    # Clean up multiple underscores
    while "__" in filename:
        filename = filename.replace("__", "_")
    
    # Remove leading/trailing underscores and whitespace
    return filename.strip("_").strip()

def clean_url(url: str) -> str:
    """Clean and normalize URL."""
    replacements = [
        ("/blob/", "/resolve/"),
        ("?download=true", "")
    ]
    
    for old, new in replacements:
        url = url.replace(old, new)
    
    return url.strip()

def clean_model_name(modelname: str) -> str:
    """Clean model name by removing extensions and sanitizing."""
    extensions = [".onnx", ".pth", ".index", ".zip"]
    
    for ext in extensions:
        modelname = modelname.replace(ext, "")
    
    return sanitize_filename(modelname)

# Configuration Management
def change_fp(precision: str) -> str:
    """Change floating point precision with validation."""
    fp16 = precision == "fp16"
    
    # Validate hardware support for fp16
    if fp16 and config.device in ["cpu", "mps", "ocl:0"]:
        gr_warning(translations["fp16_not_support"])
        return "fp32"
    
    gr_info(translations["start_update_precision"])
    
    try:
        # Update configs
        with open(configs_json, "r") as f:
            configs_data = json.load(f)
        
        configs_data["fp16"] = config.is_half = fp16
        
        with open(configs_json, "w") as f:
            json.dump(configs_data, f, indent=4)
        
        gr_info(translations["success"])
        return "fp16" if fp16 else "fp32"
    except Exception as e:
        gr_error(f"Error updating precision: {e}")
        return "fp32"

# File Management
def process_output(file_path: str) -> str:
    """
    Process output file path based on deletion setting.
    If file exists and delete is enabled, remove it.
    Otherwise, generate unique filename.
    """
    delete_exists = config.configs.get("delete_exists_file", True)
    
    if not os.path.exists(file_path):
        return file_path
    
    if delete_exists:
        try:
            os.remove(file_path)
            return file_path
        except Exception as e:
            logger.warning(f"Could not delete file {file_path}: {e}")
            # Fall through to rename logic
    
    # Generate unique filename
    path = Path(file_path)
    name_template = f"{path.stem}_{{}}{path.suffix}"
    counter = 1
    
    while True:
        new_path = path.with_stem(f"{path.stem}_{counter}")
        if not new_path.exists():
            return str(new_path)
        counter += 1

def safe_move_file(source: str, destination: str) -> str:
    """Safely move file with conflict resolution."""
    src_path = Path(source)
    
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {source}")
    
    dest_path = Path(destination)
    
    # If destination is a directory, append source filename
    if dest_path.is_dir():
        dest_path = dest_path / src_path.name
    
    # Process output path
    final_path = process_output(str(dest_path))
    
    try:
        shutil.move(str(src_path), final_path)
        return final_path
    except Exception as e:
        raise IOError(f"Failed to move file: {e}")

# UI Component Helpers
def visible(is_visible: bool) -> Dict[str, Any]:
    """Helper for visibility updates."""
    return {"visible": is_visible, "__type__": "update"}

def update_component(value: Any = None, **kwargs) -> Dict[str, Any]:
    """Generic component update helper."""
    update_dict = {"__type__": "update"}
    if value is not None:
        update_dict["value"] = value
    update_dict.update(kwargs)
    return update_dict

# Separation Model Logic
def separate_change(
    model_name: str,
    karaoke_model: str,
    reverb_model: str,
    enable_post_process: bool,
    separate_backing: bool,
    separate_reverb: bool,
    enable_denoise: bool
) -> List[Dict[str, Any]]:
    """Update separation UI components based on model selections."""
    # Determine model types
    model_type = (
        "vr" if model_name in vr_models else
        "mdx" if model_name in mdx_models else
        "demucs" if model_name in demucs_models else ""
    )
    
    karaoke_type = (
        "vr" if karaoke_model.startswith("VR") else "mdx"
    ) if separate_backing else None
    
    reverb_type = (
        "vr" if not reverb_model.startswith("MDX") else "mdx"
    ) if separate_reverb else None
    
    # Collect all active types
    active_types = {model_type, karaoke_type, reverb_type}
    
    is_vr = "vr" in active_types
    is_mdx = "mdx" in active_types
    is_demucs = "demucs" in active_types
    
    # Build update list
    updates = [
        visible(separate_backing),          # 0
        visible(separate_reverb),           # 1
        visible(is_mdx or is_demucs),      # 2
        visible(is_mdx or is_demucs),      # 3
        visible(is_mdx),                   # 4
        visible(is_mdx or is_vr),          # 5
        visible(is_demucs),                # 6
        visible(is_vr),                    # 7
        visible(is_vr),                    # 8
        visible(is_vr and enable_post_process),  # 9
        visible(is_vr and enable_denoise),       # 10
        update_component(False, interactive=is_vr),  # 11
        update_component(False, interactive=is_vr),  # 12
        update_component(False, interactive=is_vr)   # 13
    ]
    
    return updates

# JSON Configuration Handling
def update_dropdowns_from_json(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Update dropdowns from JSON configuration."""
    if not data:
        return [
            update_component(choices=[], value=None),
            update_component(choices=[], value=None),
            update_component(choices=[], value=None)
        ]
    
    inputs = list(data.get("inputs", {}).keys())
    outputs = list(data.get("outputs", {}).keys())
    
    return [
        update_component(choices=inputs, value=inputs[0] if inputs else None),
        update_component(choices=outputs, value=outputs[0] if outputs else None),
        update_component(choices=outputs, value=outputs[0] if outputs else None)
    ]

def update_button_from_json(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Update button states from JSON configuration."""
    if not data:
        return [
            update_component(interactive=True),
            update_component(interactive=False)
        ]
    
    return [
        update_component(interactive=data.get("start_button", True)),
        update_component(interactive=data.get("stop_button", False))
    ]

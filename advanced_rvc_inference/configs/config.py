import os
import sys
import json
import torch
import onnxruntime

from advanced_rvc_inference.library.backends import directml, opencl, zluda

def get_package_dir():
    """Get the package directory (where the installed package files are located)."""
    # Get the directory where this config.py file is located
    return os.path.dirname(os.path.abspath(__file__))

def get_package_config_path(filename):
    """Get the path to a config file within the installed package."""
    package_dir = get_package_dir()
    return os.path.join(package_dir, filename)

def resolve_path(relative_path, fallback_path=None):
    """
    Resolve a relative path to an absolute path.
    First tries the package directory, then falls back to cwd if provided.
    """
    package_dir = get_package_dir()
    package_path = os.path.join(package_dir, "..", "..", relative_path)
    package_path = os.path.normpath(package_path)
    
    if os.path.exists(package_path):
        return package_path
    
    if fallback_path and os.path.exists(fallback_path):
        return fallback_path
    
    return package_path

version_config_paths = [os.path.join(version, size) for version in ["v1", "v2"] for size in ["32000.json", "40000.json", "48000.json"]]

def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances: instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

@singleton
class Config:
    def __init__(self):
        self.configs_path = get_package_config_path("config.json")
        self.configs = json.load(open(self.configs_path, "r"))

        self.cpu_mode = self.configs.get("cpu_mode", False)
        self.brain = self.configs.get("brain", False)
        self.debug_mode = self.configs.get("debug_mode", False)

        # Resolve all paths from config.json to use package directory (before other methods that need paths)
        self.resolve_config_paths()

        self.json_config = self.load_config_json()
        self.translations = self.multi_language()

        self.gpu_mem = None
        self.per_preprocess = 3.7
        self.device = self.get_default_device()
        self.providers = self.get_providers()
        self.is_half = self.is_fp16()
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()
    
    def multi_language(self):
        try:
            lang = self.configs.get("language", "vi-VN")
            if len([l for l in os.listdir(self.configs["language_path"]) if l.endswith(".json")]) < 1: raise FileNotFoundError("Không tìm thấy bất cứ gói ngôn ngữ nào(No package languages found)")

            if not lang: lang = "vi-VN"
            if lang not in self.configs["support_language"]: raise ValueError("Ngôn ngữ không được hỗ trợ(Language not supported)")

            lang_path = os.path.join(self.configs["language_path"], f"{lang}.json")
            if not os.path.exists(lang_path): lang_path = os.path.join(self.configs["language_path"], "vi-VN.json")

            with open(lang_path, encoding="utf-8") as f:
                translations = json.load(f)
        except json.JSONDecodeError:
            print(self.translations["empty_json"].format(file=lang))
            pass

        return translations
    
    def resolve_config_paths(self):
        """Resolve all relative paths in config.json to absolute paths."""
        package_dir = get_package_dir()
        path_keys = [
            "convert_path", "separate_path", "create_dataset_path", "preprocess_path",
            "extract_path", "create_index_path", "train_path", "create_reference_path",
            "csv_path", "weights_path", "logs_path", "binary_path", "f0_path",
            "language_path", "presets_path", "embedders_path", "predictors_path",
            "pretrained_custom_path", "pretrained_v1_path", "pretrained_v2_path",
            "speaker_diarization_path", "uvr5_path", "audios_path", "uvr_path",
            "reference_path"
        ]
        
        for key in path_keys:
            if key in self.configs:
                relative_path = self.configs[key]
                # Resolve relative to package directory
                resolved = os.path.normpath(os.path.join(package_dir, "..", "..", relative_path))
                self.configs[key] = resolved
    
    def is_fp16(self):
        fp16 = self.configs.get("fp16", False)

        if self.device in ["cpu", "mps"] and fp16:
            self.configs["fp16"] = False
            fp16 = False

            with open(self.configs_path, "w") as f:
                json.dump(self.configs, f, indent=4)
        
        if not fp16: self.per_preprocess = 3.0
        return fp16

    def load_config_json(self):
        configs = {}
        package_dir = os.path.dirname(os.path.abspath(__file__))

        for config_file in version_config_paths:
            try:
                config_path = os.path.join(package_dir, config_file)
                # Fallback to cwd path if running from source
                if not os.path.exists(config_path):
                    config_path = os.path.join(os.getcwd(), "advanced_rvc_inference", "configs", config_file)
                
                with open(config_path, "r") as f:
                    configs[config_file] = json.load(f)
            except json.JSONDecodeError:
                print(self.translations["empty_json"].format(file=config_file))
                pass

        return configs

    def device_config(self):
        if self.gpu_mem is not None and self.gpu_mem <= 4: 
            self.per_preprocess = 3.0
            return 1, 5, 30, 32
        
        return (3, 10, 60, 65) if self.is_half else (1, 6, 38, 41)
    
    def get_default_device(self):
        if not self.cpu_mode:
            if torch.cuda.is_available():
                device = "cuda:0"
                self.gpu_mem = torch.cuda.get_device_properties(int(device.split(":")[-1])).total_memory // (1024**3)
            elif directml.is_available(): 
                device = "privateuseone:0"
            elif opencl.is_available(): 
                device = "ocl:0"
            elif torch.backends.mps.is_available(): 
                device = "mps"
            else: 
                device = "cpu"
        else:
            torch.cuda.is_available = lambda : False
            directml.is_available = lambda : False
            opencl.is_available = lambda : False
            torch.backends.mps.is_available = lambda : False

            device = "cpu"

        return device 

    def get_providers(self):
        ort_providers = onnxruntime.get_available_providers()

        if "CUDAExecutionProvider" in ort_providers and self.device.startswith("cuda"): 
            providers = ["CUDAExecutionProvider"]
        elif "ROCMExecutionProvider" in ort_providers and self.device.startswith("cuda"):
            providers = ["ROCMExecutionProvider"]
        elif "DmlExecutionProvider" in ort_providers and self.device.startswith(("ocl", "privateuseone")): 
            providers = ["DmlExecutionProvider"]
        elif "CoreMLExecutionProvider" in ort_providers and self.device.startswith("mps"): 
            providers = ["CoreMLExecutionProvider"]
        else: 
            providers = ["CPUExecutionProvider"]

        return providers

import os
import sys
import time
import yaml
import torch
import codecs
import hashlib
import requests
import warnings
import onnxruntime

from importlib import import_module

now_dir = os.getcwd()
sys.path.append(now_dir)

from main.library.utils import clear_gpu_cache
from main.library.backends import directml, opencl
from main.tools.huggingface import HF_download_file
from main.app.variables import config, translations

warnings.filterwarnings("ignore")

class Separator: 
    def __init__(
            self, 
            logger, 
            model_file_dir=config.configs["uvr5_path"], 
            output_dir=None, 
            output_format="wav", 
            output_bitrate=None, 
            normalization_threshold=0.9, 
            sample_rate=44100, 
            mdx_params={
                "hop_length": 1024, 
                "segment_size": 256, 
                "overlap": 0.25, 
                "batch_size": 1, 
                "enable_denoise": False
            }, 
            demucs_params={
                "segment_size": "Default", 
                "shifts": 2, 
                "overlap": 0.25, 
                "segments_enabled": True
            }, 
            vr_params={
                "batch_size": 1, 
                "window_size": 512, 
                "aggression": 5, 
                "enable_tta": False, 
                "enable_post_process": False, 
                "post_process_threshold": 0.2, 
                "high_end_process": False
            }
    ):
        self.logger = logger
        self.logger.info(translations["separator_info"].format(output_dir=output_dir, output_format=output_format))
        self.model_file_dir = model_file_dir
        self.output_dir = output_dir if output_dir is not None else now_dir
        os.makedirs(self.model_file_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_format = output_format if output_format is not None else "wav"
        self.output_bitrate = output_bitrate
        self.normalization_threshold = normalization_threshold
        if normalization_threshold <= 0 or normalization_threshold > 1: raise ValueError
        self.sample_rate = int(sample_rate)
        self.arch_specific_params = {"MDX": mdx_params, "Demucs": demucs_params, "VR": vr_params}
        self.torch_device = None
        self.torch_device_cpu = None
        self.torch_device_mps = None
        self.onnx_execution_provider = None
        self.model_instance = None
        self.setup_torch_device()

    def setup_torch_device(self):
        hardware_acceleration_enabled = False
        ort_providers = onnxruntime.get_available_providers()
        self.torch_device_cpu = torch.device("cpu")

        if not config.cpu_mode:
            if torch.cuda.is_available():
                self.configure_cuda(ort_providers)
                hardware_acceleration_enabled = True
            elif opencl.is_available() or directml.is_available():
                hardware_acceleration_enabled = True
                self.configure_amd(ort_providers)
            elif torch.backends.mps.is_available():
                self.configure_mps(ort_providers)
                hardware_acceleration_enabled = True

        if not hardware_acceleration_enabled:
            self.logger.info(translations["running_in_cpu"])
            self.torch_device = self.torch_device_cpu
            self.onnx_execution_provider = ["CPUExecutionProvider"]

    def configure_cuda(self, ort_providers):
        self.logger.info(translations["running_in_cuda"])
        self.torch_device = torch.device("cuda")

        if "CUDAExecutionProvider" in ort_providers:
            self.logger.info(translations["onnx_have"].format(have='CUDAExecutionProvider'))
            self.onnx_execution_provider = ["CUDAExecutionProvider"]
        else: self.logger.warning(translations["onnx_not_have"].format(have='CUDAExecutionProvider'))

    def configure_amd(self, ort_providers):
        self.logger.info(translations["running_in_amd"])
        self.torch_device = torch.device(config.device)

        if "DmlExecutionProvider" in ort_providers:
            self.logger.info(translations["onnx_have"].format(have='DmlExecutionProvider'))
            self.onnx_execution_provider = ["DmlExecutionProvider"]
        else: self.logger.warning(translations["onnx_not_have"].format(have='DmlExecutionProvider'))

    def configure_mps(self, ort_providers):
        self.logger.info(translations["set_torch_mps"])
        self.torch_device_mps = torch.device("mps")
        self.torch_device = self.torch_device_mps

        if "CoreMLExecutionProvider" in ort_providers:
            self.logger.info(translations["onnx_have"].format(have='CoreMLExecutionProvider'))
            self.onnx_execution_provider = ["CoreMLExecutionProvider"]
        else: self.logger.warning(translations["onnx_not_have"].format(have='CoreMLExecutionProvider'))

    def get_model_hash(self, model_path):
        try:
            with open(model_path, "rb") as f:
                f.seek(-10000 * 1024, 2)
                return hashlib.md5(f.read()).hexdigest()
        except IOError as e:
            return hashlib.md5(open(model_path, "rb").read()).hexdigest()

    def download_file_if_not_exists(self, url, output_path):
        if os.path.isfile(output_path): return
        HF_download_file(url, output_path)

    def list_supported_model_files(self):
        response = requests.get(codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/enj/znva/wfba/hie_zbqryf.wfba", "rot13"))
        response.raise_for_status()
        model_downloads_list = response.json()

        return {
            "MDX": {
                **model_downloads_list["mdx_download_list"], 
                **model_downloads_list["mdx_download_vip_list"]
            }, 
            "Demucs": {key: value for key, value in model_downloads_list["demucs_download_list"].items() if key.startswith("Demucs v4")},
            "VR": {
                **model_downloads_list["vr_download_list"]
            }
        }
    
    def download_model_files(self, model_filename):
        model_path = os.path.join(self.model_file_dir, model_filename)
        supported_models = self.list_supported_model_files()
        model_repo = codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/hie5_zbqryf", "rot13")

        for model_type, model_list in supported_models.items():
            for _, files in model_list.items():
                if isinstance(files, str) and files == model_filename:
                    try:
                        self.download_file_if_not_exists(f"{model_repo}/MDX/{model_filename}", model_path)
                    except:
                        try:
                            self.download_file_if_not_exists(f"{model_repo}/VR/{model_filename}", model_path)
                        except:
                            self.download_file_if_not_exists(f"{model_repo}/Demucs/{model_filename}", model_path)

                    return model_filename, model_type, model_path

                elif isinstance(files, dict) and any(model_filename in (k, v) for k, v in files.items()):
                    for file_key, file_val in files.items():
                        out_path = os.path.join(self.model_file_dir, file_key)

                        if file_val.startswith("http"):
                            self.download_file_if_not_exists(file_val, out_path)
                        else:
                            self.download_file_if_not_exists(f"{model_repo}/Demucs/{file_val}", os.path.join(self.model_file_dir, file_val))

                    return model_filename, model_type, model_path

        raise ValueError

    def load_model_data_from_yaml(self, yaml_config_filename):
        model_data_yaml_filepath = os.path.join(self.model_file_dir, yaml_config_filename) if not os.path.exists(yaml_config_filename) else yaml_config_filename
        model_data = yaml.load(open(model_data_yaml_filepath, encoding="utf-8"), Loader=yaml.FullLoader)

        return model_data

    def load_model_data_using_hash(self, model_path):
        model_hash = self.get_model_hash(model_path)
        response = requests.get(codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/enj/znva/wfba/zbqry_qngn.wfba", "rot13"))
        response.raise_for_status()
        model_data_object = response.json()

        if model_hash in model_data_object: model_data = model_data_object[model_hash]
        else: raise ValueError

        return model_data

    def load_model(self, model_filename):
        self.logger.info(translations["loading_model"].format(model_filename=model_filename))
        model_filename, model_type, model_path = self.download_model_files(model_filename)

        yaml_config_filename = model_path if model_path.lower().endswith(".yaml") else None

        common_params = {
            "logger": self.logger, 
            "torch_device": self.torch_device, 
            "torch_device_cpu": self.torch_device_cpu, 
            "torch_device_mps": self.torch_device_mps, 
            "onnx_execution_provider": self.onnx_execution_provider, 
            "model_name": model_filename.split(".")[0], 
            "model_path": model_path, 
            "model_data": self.load_model_data_from_yaml(yaml_config_filename) if yaml_config_filename is not None else self.load_model_data_using_hash(model_path), 
            "output_format": self.output_format, 
            "output_bitrate": self.output_bitrate, 
            "output_dir": self.output_dir, 
            "normalization_threshold": self.normalization_threshold, 
            "output_single_stem": None, 
            "invert_using_spec": False, 
            "sample_rate": self.sample_rate
        }
        separator_classes = {"MDX": "mdx_separator.MDXSeparator", "Demucs": "demucs_separator.DemucsSeparator", "VR": "vr_separator.VRSeparator"}

        if model_type not in self.arch_specific_params or model_type not in separator_classes: raise ValueError(translations["model_type_not_support"].format(model_type=model_type))

        module_name, class_name = separator_classes[model_type].split(".")
        separator_class = getattr(import_module(f"main.library.architectures.{module_name}"), class_name)
        self.model_instance = separator_class(common_config=common_params, arch_config=self.arch_specific_params[model_type])

    def separate(self, audio_file_path):
        self.logger.info(f"{translations['starting_separator']}: {audio_file_path}")
        separate_start_time = time.perf_counter()

        with torch.amp.autocast(self.torch_device.type if self.torch_device.type != "ocl" else "cpu", enabled=config.is_half, dtype=torch.float16 if config.is_half else torch.float32):
            output_files = self.model_instance.separate(audio_file_path)

        clear_gpu_cache()
        self.model_instance.clear_file_specific_paths()

        self.logger.debug(translations["separator_success_3"])
        self.logger.info(f"{translations['separator_duration']}: {time.strftime('%H:%M:%S', time.gmtime(int(time.perf_counter() - separate_start_time)))}")
        return output_files
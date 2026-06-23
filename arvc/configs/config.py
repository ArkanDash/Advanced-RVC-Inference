import os
import sys
import json
import logging

logger = logging.getLogger(__name__)


def get_package_dir():
    """Get the package directory (where the installed package files are located)."""
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
    package_path = os.path.normpath(os.path.join(package_dir, "..", "..", relative_path))

    if os.path.exists(package_path):
        return package_path

    if fallback_path and os.path.exists(fallback_path):
        return fallback_path

    return package_path


version_config_paths = [os.path.join(version, size) for version in ["v1", "v2"] for size in ["24000.json", "32000.json", "40000.json", "44100.json", "48000.json"]]


class Config:
    """Configuration manager for Advanced RVC Inference.

    Uses lazy imports for heavy dependencies (onnxruntime, GPU backends)
    so headless/CLI mode works without them installed.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Singleton pattern that preserves constructor arguments."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Avoid re-initializing the singleton
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.configs_path = get_package_config_path("config.json")
        try:
            with open(self.configs_path, "r", encoding="utf-8") as f:
                self.configs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            import warnings
            warnings.warn(f"Could not load config from {self.configs_path}: {e}. Using defaults.")
            self.configs = {}

        self.cpu_mode = self.configs.get("cpu_mode", False)
        self.brain = self.configs.get("brain", False)
        self.debug_mode = self.configs.get("debug_mode", False)

        # Resolve all paths from config.json to use package directory
        self.resolve_config_paths()

        self.json_config = self.load_config_json()
        self.translations = self.multi_language()

        self.gpu_mem = None
        # ACCURACY PATCH (Applio parity): was 3.7 — produces ~26% FEWER
        # training chunks than Applio. Match Applio's PERCENTAGE=3.0 here
        # so a 10-min dataset yields ~222 chunks instead of ~176. This is
        # the largest plausible cause of "less accurate than Applio on
        # small datasets".
        self.per_preprocess = 3.0

        # Lazy backend detection
        self._is_zluda = None
        self.device = self.get_default_device()
        self.gpu_name = self._get_gpu_name()
        self.providers = self.get_providers()
        self.is_half = self.is_fp16()
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

        self._initialized = True

    @property
    def is_zluda(self):
        """Lazy ZLUDA detection."""
        if self._is_zluda is None:
            try:
                from arvc.engine.models.backends import zluda
                self._is_zluda = zluda.is_available()
            except Exception:
                self._is_zluda = False
        return self._is_zluda

    def multi_language(self):
        try:
            lang = self.configs.get("language", "vi-VN")
            lang_dir = self.configs.get("language_path", "")
            if lang_dir and os.path.isdir(lang_dir):
                lang_files = [l for l in os.listdir(lang_dir) if l.endswith(".json")]
                if len(lang_files) < 1:
                    raise FileNotFoundError("No language packages found")

            if not lang:
                lang = "vi-VN"
            if lang not in self.configs.get("support_language", [lang]):
                import warnings
                warnings.warn(f"Language '{lang}' not in supported list, using it anyway.")

            lang_path = os.path.join(self.configs.get("language_path", ""), f"{lang}.json")
            if not os.path.exists(lang_path):
                lang_path = os.path.join(self.configs.get("language_path", ""), "vi-VN.json")

            with open(lang_path, encoding="utf-8") as f:
                translations = json.load(f)
        except json.JSONDecodeError:
            import warnings
            warnings.warn(f"Could not parse language file: {lang}")
            translations = {}
        except Exception as e:
            import warnings
            warnings.warn(f"Could not load language: {e}")
            translations = {}

        return translations

    def resolve_config_paths(self):
        """Resolve all relative paths in config.json to absolute paths."""
        package_dir = get_package_dir()
        path_keys = [
            "convert_path", "separate_path", "create_dataset_path", "preprocess_path",
            "extract_path", "create_index_path", "train_path", "create_reference_path",
            "csv_path", "weights_path", "logs_path", "datasets_path", "binary_path", "f0_path",
            "language_path", "presets_path", "embedders_path", "predictors_path",
            "pretrained_custom_path", "pretrained_v1_path", "pretrained_v2_path",
            "speaker_diarization_path", "uvr5_path", "audios_path", "uvr_path",
            "reference_path"
        ]

        for key in path_keys:
            if key in self.configs:
                relative_path = self.configs[key]
                resolved = os.path.normpath(os.path.join(package_dir, "..", "..", relative_path))
                self.configs[key] = resolved

    def is_fp16(self):
        fp16 = self.configs.get("fp16", False)

        if self.device in ["cpu", "mps"] and fp16:
            self.configs["fp16"] = False
            fp16 = False
            try:
                with open(self.configs_path, "w", encoding="utf-8") as f:
                    json.dump(self.configs, f, indent=4)
            except OSError:
                pass

        if not fp16:
            self.per_preprocess = 3.0
        return fp16

    def _get_gpu_name(self) -> str:
        """Get the GPU device name for diagnostics."""
        try:
            import torch
            if torch.cuda.is_available() and self.device.startswith("cuda"):
                return torch.cuda.get_device_name(0)
        except Exception:
            pass
        return ""

    def load_config_json(self):
        configs = {}
        package_dir = os.path.dirname(os.path.abspath(__file__))

        for config_file in version_config_paths:
            try:
                config_path = os.path.join(package_dir, config_file)
                if not os.path.exists(config_path):
                    config_path = os.path.join(os.getcwd(), "arvc", "configs", config_file)

                with open(config_path, "r") as f:
                    configs[config_file] = json.load(f)
            except json.JSONDecodeError:
                import warnings
                warnings.warn(f"Could not parse config file: {config_file}")
            except Exception:
                pass

        return configs

    def device_config(self):
        if self.gpu_mem is not None and self.gpu_mem <= 4:
            self.per_preprocess = 3.0
            return 1, 5, 30, 32

        return (3, 10, 60, 65) if self.is_half else (1, 6, 38, 41)

    def get_default_device(self):
        """Determine the best available device with lazy backend imports."""
        try:
            import torch
        except ImportError:
            return "cpu"

        if not self.cpu_mode:
            if torch.cuda.is_available():
                device = "cuda:0"
                try:
                    self.gpu_mem = torch.cuda.get_device_properties(
                        int(device.split(":")[-1])
                    ).total_memory // (1024**3)
                except Exception:
                    self.gpu_mem = None
                return device

            # Check alternative backends lazily
            try:
                from arvc.engine.models.backends import directml
                if directml.is_available():
                    return "privateuseone:0"
            except Exception:
                pass

            try:
                from arvc.engine.models.backends import opencl
                if opencl.is_available():
                    return "ocl:0"
            except Exception:
                pass

            if torch.backends.mps.is_available():
                return "mps"

        return "cpu"

    def get_providers(self):
        """Get ONNX runtime providers with lazy import."""
        try:
            import onnxruntime
            ort_providers = onnxruntime.get_available_providers()
        except ImportError:
            return ["CPUExecutionProvider"]
        except Exception:
            return ["CPUExecutionProvider"]

        # ZLUDA: CUDA EP may not work with AMD hardware, prefer CPU/ROCm
        if self.is_zluda:
            if "ROCMExecutionProvider" in ort_providers:
                providers = ["ROCMExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
        elif "CUDAExecutionProvider" in ort_providers and self.device.startswith("cuda"):
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

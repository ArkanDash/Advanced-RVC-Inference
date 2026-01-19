"""
Global variables and configuration for Advanced RVC Inference.

This module provides singleton configuration and shared state
for the entire application.
"""

import os
import sys
import csv
import json
import codecs
import logging
import urllib.request
import logging.handlers
from pathlib import Path
from typing import Dict, Any, List, Optional

# Package root directory
PACKAGE_ROOT = Path(__file__).parent.parent.resolve()
ASSETS_PATH = PACKAGE_ROOT / "assets"
CONFIGS_PATH = PACKAGE_ROOT / "configs"
LOGS_PATH = ASSETS_PATH / "logs"
WEIGHTS_PATH = ASSETS_PATH / "weights"

# Initialize logger
logger = logging.getLogger(__name__)
logger.propagate = False

# Create singleton config instance
_config_instance = None


def get_config():
    """Get the singleton Config instance."""
    global _config_instance
    if _config_instance is None:
        from advanced_rvc_inference.configs.config import Config

        _config_instance = Config()
    return _config_instance


class Config:
    """Configuration manager for Advanced RVC Inference."""

    def __init__(self):
        self.configs_path = CONFIGS_PATH / "config.json"
        self.configs = self._load_json(self.configs_path)

        self.cpu_mode = self.configs.get("cpu_mode", False)
        self.brain = self.configs.get("brain", False)
        self.debug_mode = self.configs.get("debug_mode", False)

        self.json_config = self._load_version_configs()
        self.translations = self._load_translations()

        self.gpu_mem = None
        self.per_preprocess = 3.7
        self.device = self._get_default_device()
        self.providers = self._get_providers()
        self.is_half = self._is_fp16()
        self.x_pad, self.x_query, self.x_center, self.x_max = self._device_config()

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        """Load JSON file safely."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Failed to load {path}: {e}")
            return {}

    def _load_version_configs(self) -> Dict[str, Any]:
        """Load version-specific configuration files."""
        configs = {}
        version_config_paths = [
            CONFIGS_PATH / version / size
            for version in ["v1", "v2"]
            for size in ["32000.json", "40000.json", "48000.json"]
        ]

        for config_file in version_config_paths:
            try:
                if config_file.exists():
                    configs[str(config_file)] = self._load_json(config_file)
            except Exception as e:
                logger.debug(f"Could not load {config_file}: {e}")

        return configs

    def _load_translations(self) -> Dict[str, Any]:
        """Load language translations."""
        try:
            lang = self.configs.get("language", "vi-VN")
            lang_path = CONFIGS_PATH / "languages" / f"{lang}.json"

            if not lang_path.exists():
                lang_path = CONFIGS_PATH / "languages" / "vi-VN.json"

            if lang_path.exists():
                return self._load_json(lang_path)
        except Exception as e:
            logger.warning(f"Failed to load translations: {e}")

        return {}

    def _is_fp16(self) -> bool:
        """Check if half precision is supported."""
        fp16 = self.configs.get("fp16", False)

        if self.device in ["cpu", "mps"] and fp16:
            self.configs["fp16"] = False
            fp16 = False
            self._save_configs()

        if not fp16:
            self.per_preprocess = 3.0
        return fp16

    def _device_config(self):
        """Get device-specific configuration."""
        if self.gpu_mem is not None and self.gpu_mem <= 4:
            self.per_preprocess = 3.0
            return 1, 5, 30, 32

        return (3, 10, 60, 65) if self.is_half else (1, 6, 38, 41)

    def _get_default_device(self) -> str:
        """Determine the best available device."""
        if not self.cpu_mode:
            import torch

            if torch.cuda.is_available():
                device = "cuda:0"
                self.gpu_mem = (
                    torch.cuda.get_device_properties(int(device.split(":")[-1])).total_memory
                    // (1024**3)
                )
                return device

            # Check for other accelerators
            try:
                from advanced_rvc_inference.library.backends import directml, opencl

                if directml.is_available():
                    return "privateuseone:0"
                if opencl.is_available():
                    return "ocl:0"
            except ImportError:
                pass

            if torch.backends.mps.is_available():
                return "mps"

        # Fallback to CPU
        import torch

        torch.cuda.is_available = lambda: False
        try:
            from advanced_rvc_inference.library.backends import directml, opencl

            directml.is_available = lambda: False
            opencl.is_available = lambda: False
        except ImportError:
            pass
        try:
            import torch

            torch.backends.mps.is_available = lambda: False
        except AttributeError:
            pass

        return "cpu"

    def _get_providers(self):
        """Get ONNX runtime providers."""
        try:
            import onnxruntime

            ort_providers = onnxruntime.get_available_providers()

            if "CUDAExecutionProvider" in ort_providers and self.device.startswith(
                "cuda"
            ):
                return ["CUDAExecutionProvider"]
            elif "ROCMExecutionProvider" in ort_providers and self.device.startswith(
                "cuda"
            ):
                return ["ROCMExecutionProvider"]
            elif "DmlExecutionProvider" in ort_providers and self.device.startswith(
                ("ocl", "privateuseone")
            ):
                return ["DmlExecutionProvider"]
            elif "CoreMLExecutionProvider" in ort_providers and self.device.startswith(
                "mps"
            ):
                return ["CoreMLExecutionProvider"]
            else:
                return ["CPUExecutionProvider"]
        except ImportError:
            return ["CPUExecutionProvider"]

    def _save_configs(self):
        """Save current configuration to file."""
        try:
            with open(self.configs_path, "w", encoding="utf-8") as f:
                json.dump(self.configs, f, indent=4)
        except Exception as e:
            logger.warning(f"Failed to save config: {e}")


# Initialize logging
def setup_logging(config: Config):
    """Configure logging handlers."""
    if not logger.hasHandlers():
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            fmt="%(asctime)s.%(msecs)03d | %(levelname)s | %(module)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.DEBUG if config.debug_mode else logging.INFO)

        log_path = LOGS_PATH
        if not log_path.exists():
            log_path.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_path / "app.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_formatter = logging.Formatter(
            fmt="%(asctime)s.%(msecs)03d | %(module)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)


# Load config and setup logging
config = get_config()
setup_logging(config)

# Python executable
python = sys.executable

# Translations
translations = config.translations

# Load configs JSON
configs_json = CONFIGS_PATH / "config.json"
configs = config.configs

# Adjust for CPU/MPS with FP16
if config.device in ["cpu", "mps", "ocl:0"] and configs.get("fp16", False):
    logger.warning(translations.get("fp16_not_support", "FP16 not supported on this device"))
    configs["fp16"] = config.is_half = False
    config._save_configs()

# Global state
models = {}
model_name = {}
model_options = {}

# F0 methods
method_f0 = [
    "mangio-crepe-full",
    "crepe-full",
    "fcpe",
    "rmvpe",
    "harvest",
    "pyin",
    "hybrid",
]
method_f0_full = [
    "pm-ac",
    "pm-cc",
    "pm-shs",
    "dio",
    "mangio-crepe-tiny",
    "mangio-crepe-small",
    "mangio-crepe-medium",
    "mangio-crepe-large",
    "mangio-crepe-full",
    "crepe-tiny",
    "crepe-small",
    "crepe-medium",
    "crepe-large",
    "crepe-full",
    "fcpe",
    "fcpe-legacy",
    "fcpe-previous",
    "rmvpe",
    "rmvpe-clipping",
    "rmvpe-medfilt",
    "rmvpe-clipping-medfilt", 
    "hpa-rmvpe", 
    "hpa-rmvpe-medfilt", 
    "harvest",
    "yin",
    "pyin",
    "swipe",
    "piptrack",
    "penn",
    "mangio-penn",
    "djcm",
    "djcm-clipping",
    "djcm-medfilt",
    "djcm-clipping-medfilt",
    "swift",
    "pesto",
    "hybrid",
]
hybrid_f0_method = [
    "hybrid[pm+dio]",
    "hybrid[pm+crepe-tiny]",
    "hybrid[pm+crepe]",
    "hybrid[pm+fcpe]",
    "hybrid[pm+rmvpe]",
    "hybrid[pm+harvest]",
    "hybrid[pm+yin]",
    "hybrid[dio+crepe-tiny]",
    "hybrid[dio+crepe]",
    "hybrid[dio+fcpe]",
    "hybrid[dio+rmvpe]",
    "hybrid[dio+harvest]",
    "hybrid[dio+yin]",
    "hybrid[crepe-tiny+crepe]",
    "hybrid[crepe-tiny+fcpe]",
    "hybrid[crepe-tiny+rmvpe]",
    "hybrid[crepe-tiny+harvest]",
    "hybrid[crepe+fcpe]",
    "hybrid[crepe+rmvpe]",
    "hybrid[crepe+harvest]",
    "hybrid[crepe+yin]",
    "hybrid[fcpe+rmvpe]",
    "hybrid[fcpe+harvest]",
    "hybrid[fcpe+yin]",
    "hybrid[rmvpe+harvest]",
    "hybrid[rmvpe+yin]",
    "hybrid[harvest+yin]",
]

# Embedders
embedders_mode = ["fairseq", "onnx", "transformers", "spin", "whisper"]
embedders_model = [
    "contentvec_base",
    "hubert_base",
    "vietnamese_hubert_base",
    "japanese_hubert_base",
    "korean_hubert_base",
    "chinese_hubert_base",
    "portuguese_hubert_base",
    "custom",
]
spin_model = ["spin-v1", "spin-v2"]
whisper_model = [
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
]

# File extensions
audio_extensions = (
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".opus",
    ".m4a",
    ".mp4",
    ".aac",
    ".alac",
    ".wma",
    ".aiff",
    ".webm",
    ".ac3",
)

# Paths for files
def _scan_paths_for_files(base_path: Path, extensions: tuple) -> List[str]:
    """Scan directory for files with given extensions."""
    if not base_path.exists():
        return []

    paths = []
    for root, _, files in os.walk(base_path):
        for f in files:
            if Path(f).suffix.lower() in extensions:
                paths.append(os.path.abspath(os.path.join(root, f)))

    return sorted(paths)


def _list_model_names(base_path: Path) -> List[str]:
    """List model files in a directory."""
    if not base_path.exists():
        return []

    return sorted(
        [
            name
            for name in os.listdir(base_path)
            if name.endswith((".pth", ".onnx"))
            and not name.startswith("G_")
            and not name.startswith("D_")
        ]
    )


def _list_index_files(logs_path: Path) -> List[str]:
    """List index files in logs directory."""
    if not logs_path.exists():
        return []

    return sorted(
        [
            os.path.abspath(os.path.join(root, name))
            for root, _, files in os.walk(logs_path, topdown=False)
            for name in files
            if name.endswith(".index") and "trained" not in name
        ]
    )


# Initialize paths
paths_for_files = _scan_paths_for_files(Path(configs.get("audios_path", ASSETS_PATH / "audios")), audio_extensions)
model_name = _list_model_names(Path(configs.get("weights_path", WEIGHTS_PATH)))
index_path = _list_index_files(Path(configs.get("logs_path", LOGS_PATH)))

# Reference list
reference_path = Path(configs.get("reference_path", ASSETS_PATH / "reference"))
if reference_path.exists():
    reference_list = sorted(
        [
            name
            for name in os.listdir(reference_path)
            if (reference_path / name).exists() and (reference_path / name).is_dir()
        ]
    )
else:
    reference_list = []

# Pretrained models
pretrained_custom_path = Path(configs.get("pretrained_custom_path", ASSETS_PATH / "pretrained_custom"))
if pretrained_custom_path.exists():
    pretrainedD = [m for m in os.listdir(pretrained_custom_path) if m.endswith(".pth") and "D" in m]
    pretrainedG = [m for m in os.listdir(pretrained_custom_path) if m.endswith(".pth") and "G" in m]
else:
    pretrainedD = []
    pretrainedG = []

# Presets
presets_path = Path(configs.get("presets_path", ASSETS_PATH / "presets"))
if presets_path.exists():
    presets_file = sorted([f for f in os.listdir(presets_path) if f.endswith(".conversion.json")])
    audio_effect_presets_file = sorted([f for f in os.listdir(presets_path) if f.endswith(".effect.json")])
else:
    presets_file = []
    audio_effect_presets_file = []

# F0 files
f0_path = Path(configs.get("f0_path", ASSETS_PATH / "f0"))
if f0_path.exists():
    f0_file = sorted(
        [
            os.path.abspath(os.path.join(root, f))
            for root, _, files in os.walk(f0_path)
            for f in files
            if f.endswith(".txt")
        ]
    )
else:
    f0_file = []

# Export formats
file_types = list(audio_extensions)
export_format_choices = [ext[1:] for ext in audio_extensions]

# Language and theme
language = configs.get("language", "vi-VN")
theme = configs.get("theme", "NeoPy/Soft")

# TTS voices
edgetts = configs.get("edge_tts", ["vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"])
google_tts_voice = configs.get("google_tts_voice", ["vi", "en"])

# VR models
vr_models = configs.get("vr_models", "")
mdx_models = configs.get("mdx_models", "")
karaoke_models = configs.get("karaoke_models", "")
reverb_models = configs.get("reverb_models", "")
denoise_models = configs.get("denoise_models", "")
uvr_model = list(vr_models.keys()) + list(mdx_models.keys())

# Sample rates
sample_rate_choice = [8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 96000]

# CSV path
csv_path = configs.get("csv_path", CONFIGS_PATH / "models.csv")

# Allow all disk access on Windows
allow_disk = []
if "--allow_all_disk" in sys.argv and sys.platform == "win32":
    try:
        import win32api
    except ImportError:
        os.system(f"{python} -m pip install pywin32")
        import win32api

    allow_disk = win32api.GetLogicalDriveStrings().split("\x00")[:-1]

# Load model URLs from CSV
def _load_model_urls():
    """Load model URLs from CSV file."""
    try:
        if os.path.exists(csv_path):
            reader = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
        else:
            # Try to load from URL (rot13 encoded)
            try:
                reader = list(
                    csv.DictReader(
                        [
                            line.decode("utf-8")
                            for line in urllib.request.urlopen(
                                codecs.decode(
                                    "uggcf://qbpf.tbbtyr.pbz/fcernqfurrgf/q/1gNHnDeRULtEfz1Yieaw14USUQjWJy0Oq9k0DrCrjApb/rkcbeg?sbezng=pfi&tvq=1977693859",
                                    "rot13",
                                )
                            ).readlines()
                        ]
                    )
                )
                writer = csv.DictWriter(
                    open(csv_path, mode="w", newline="", encoding="utf-8"),
                    fieldnames=reader[0].keys(),
                )
                writer.writeheader()
                writer.writerows(reader)
            except Exception:
                reader = []

        for row in reader:
            filename = row.get("Filename", "")
            url = None

            for value in row.values():
                if isinstance(value, str) and "huggingface" in value:
                    url = value
                    break

            if url and filename:
                models[filename] = url

    except Exception as e:
        logger.debug(f"Could not load model URLs: {e}")


_load_model_urls()

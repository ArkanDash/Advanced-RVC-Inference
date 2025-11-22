import os
import sys
import csv
import json
import codecs
import logging
import urllib.request
import logging.handlers
from pathlib import Path

# Add parent directories to Python path
sys.path.append(os.getcwd())
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False

try:
    from configs.config import Config
except ImportError:
    # Fallback if config import fails
    class Config:
        def __init__(self):
            if torch_available:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = "cpu"
            self.debug_mode = False
            self.is_half = True
            
            # Default translations
            self.translations = {
                "fp16_not_support": "FP16 not supported on this device"
            }

logger = logging.getLogger(__name__)
logger.propagate = False

config = Config()
python = sys.executable
translations = config.translations 
configs_json = "configs/config.json"
if not os.path.exists(configs_json):
    configs_json = os.path.join("configs", "config.json")

try:
    configs = json.load(open(configs_json, "r"))
except FileNotFoundError:
    # Default configuration
    configs = {
        "language": "vi_VN",
        "app_port": 7860,
        "server_name": "0.0.0.0",
        "theme": "NoCrypt/miku",
        "app_show_error": True,
        "discord_presence": True,
        "num_of_restart": 5,
        "logs_path": "logs",
        "audios_path": "audios",
        "weights_path": "weights",
        "reference_path": "assets/logs/reference",
        "pretrained_custom_path": "weights/pretrained",
        "presets_path": "assets/presets",
        "f0_path": "assets/f0",
        "csv_path": "assets/spreadsheet.csv",
        "font": "https://fonts.googleapis.com/css2?family=Courgette&display=swap",
        "edge_tts": ["vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"],
        "google_tts_voice": ["vi", "en"],
        "fp16": True
    }

if not logger.hasHandlers():
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(fmt="\n%(asctime)s.%(msecs)03d | %(levelname)s | %(module)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.DEBUG if config.debug_mode else logging.INFO)
    file_handler = logging.handlers.RotatingFileHandler(os.path.join(configs["logs_path"], "app.log"), maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
    file_formatter = logging.Formatter(fmt="\n%(asctime)s.%(msecs)03d | %(levelname)s | %(module)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)

if config.device in ["cpu", "mps", "ocl:0"] and configs.get("fp16", False):
    logger.warning(translations["fp16_not_support"])
    configs["fp16"] = config.is_half = False

    with open(configs_json, "w") as f:
        json.dump(configs, f, indent=4)

models = {}
model_options = {}

method_f0 = ["mangio-crepe-full", "crepe-full", "fcpe", "rmvpe", "harvest", "pyin", "hybrid"]
method_f0_full = ["pm-ac", "pm-cc", "pm-shs", "dio", "mangio-crepe-tiny", "mangio-crepe-small", "mangio-crepe-medium", "mangio-crepe-large", "mangio-crepe-full", "crepe-tiny", "crepe-small", "crepe-medium", "crepe-large", "crepe-full", "fcpe", "fcpe-legacy", "fcpe-previous", "rmvpe", "rmvpe-clipping", "rmvpe-medfilt", "rmvpe-clipping-medfilt", "harvest", "yin", "pyin", "swipe", "piptrack", "penn", "mangio-penn", "djcm", "djcm-clipping", "djcm-medfilt", "djcm-clipping-medfilt", "swift", "pesto", "hybrid"]
hybrid_f0_method = ["hybrid[pm+dio]", "hybrid[pm+crepe-tiny]", "hybrid[pm+crepe]", "hybrid[pm+fcpe]", "hybrid[pm+rmvpe]", "hybrid[pm+harvest]", "hybrid[pm+yin]", "hybrid[dio+crepe-tiny]", "hybrid[dio+crepe]", "hybrid[dio+fcpe]", "hybrid[dio+rmvpe]", "hybrid[dio+harvest]", "hybrid[dio+yin]", "hybrid[crepe-tiny+crepe]", "hybrid[crepe-tiny+fcpe]", "hybrid[crepe-tiny+rmvpe]", "hybrid[crepe-tiny+harvest]", "hybrid[crepe+fcpe]", "hybrid[crepe+rmvpe]", "hybrid[crepe+harvest]", "hybrid[crepe+yin]", "hybrid[fcpe+rmvpe]", "hybrid[fcpe+harvest]", "hybrid[fcpe+yin]", "hybrid[rmvpe+harvest]", "hybrid[rmvpe+yin]", "hybrid[harvest+yin]"]

embedders_mode = ["fairseq", "onnx", "transformers", "spin", "whisper"]
embedders_model = ["contentvec_base", "hubert_base", "vietnamese_hubert_base", "japanese_hubert_base", "korean_hubert_base", "chinese_hubert_base", "portuguese_hubert_base", "custom"]
spin_model = ["spin-v1", "spin-v2"]
whisper_model = ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large-v1", "large-v2", "large-v3", "large-v3-turbo"]

paths_for_files = sorted([os.path.abspath(os.path.join(root, f)) for root, _, files in os.walk(configs["audios_path"]) for f in files if os.path.splitext(f)[1].lower() in (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3")])
reference_list = sorted([name for name in os.listdir(configs["reference_path"]) if os.path.exists(os.path.join(configs["reference_path"], name)) and os.path.isdir(os.path.join(configs["reference_path"], name))])
model_name = sorted(list(model for model in os.listdir(configs["weights_path"]) if model.endswith((".pth", ".onnx")) and not model.startswith("G_") and not model.startswith("D_"))) 
index_path = sorted([os.path.join(root, name) for root, _, files in os.walk(configs["logs_path"], topdown=False) for name in files if name.endswith(".index") and "trained" not in name])

pretrainedD = [model for model in os.listdir(configs["pretrained_custom_path"]) if model.endswith(".pth") and "D" in model]
pretrainedG = [model for model in os.listdir(configs["pretrained_custom_path"]) if model.endswith(".pth") and "G" in model]

presets_file = sorted(list(f for f in os.listdir(configs["presets_path"]) if f.endswith(".conversion.json")))
audio_effect_presets_file = sorted(list(f for f in os.listdir(configs["presets_path"]) if f.endswith(".effect.json")))
f0_file = sorted([os.path.abspath(os.path.join(root, f)) for root, _, files in os.walk(configs["f0_path"]) for f in files if f.endswith(".txt")])

file_types = [".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3"]
export_format_choices = ["wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"]

language = configs.get("language", "vi-VN")
theme = configs.get("theme", "NoCrypt/miku")

edgetts = configs.get("edge_tts", ["vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"])
google_tts_voice = configs.get("google_tts_voice", ["vi", "en"])

vr_models = configs.get("vr_models", "")
demucs_models = configs.get("demucs_models", "")
mdx_models = configs.get("mdx_models", "")
karaoke_models = configs.get("karaoke_models", "")
reverb_models = configs.get("reverb_models", "")
denoise_models = configs.get("denoise_models", "")
uvr_model = list(demucs_models.keys()) + list(vr_models.keys()) + list(mdx_models.keys())

font = configs.get("font", "https://fonts.googleapis.com/css2?family=Courgette&display=swap")
sample_rate_choice = [8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 96000]
csv_path = configs["csv_path"]

if "--allow_all_disk" in sys.argv and sys.platform == "win32":
    try:
        import win32api
    except:
        os.system(f"{python} -m pip install pywin32")
        import win32api

    allow_disk = win32api.GetLogicalDriveStrings().split('\x00')[:-1]
else: allow_disk = []

try:
    if os.path.exists(csv_path): reader = list(csv.DictReader(open(csv_path, newline='', encoding='utf-8')))
    else:
        reader = list(csv.DictReader([line.decode('utf-8') for line in urllib.request.urlopen(codecs.decode("uggcf://qbpf.tbbtyr.pbz/fcernqfurrgf/q/1gNHnDeRULtEfz1Yieaw14USUQjWJy0Oq9k0DrCrjApb/rkcbeg?sbezng=pfi&tvq=1977693859", "rot13")).readlines()]))
        writer = csv.DictWriter(open(csv_path, mode='w', newline='', encoding='utf-8'), fieldnames=reader[0].keys())
        writer.writeheader()
        writer.writerows(reader)

    for row in reader:
        filename = row['Filename']
        url = None

        for value in row.values():
            if isinstance(value, str) and "huggingface" in value:
                url = value
                break

        if url: models[filename] = url
except:
    pass
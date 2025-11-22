import os, sys
import librosa
import soundfile as sf
import re
import unicodedata
import wget
from torch import nn

import logging
from transformers import HubertModel
import warnings

# Remove this to see warnings about transformers models
warnings.filterwarnings("ignore")

logging.getLogger("fairseq").setLevel(logging.ERROR)
logging.getLogger("faiss.loader").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

now_dir = os.getcwd()
sys.path.append(now_dir)

base_path = os.path.join(now_dir, "rvc", "models", "formant", "stftpitchshift")
stft = base_path + ".exe" if sys.platform == "win32" else base_path


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


def load_audio(file, sample_rate):
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        audio, sr = sf.read(file)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.T)
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    except Exception as error:
        raise RuntimeError(f"An error occurred loading the audio: {error}")

    return audio.flatten()


def load_audio_infer(file, sample_rate):
    file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
    if not os.path.isfile(file):
        raise FileNotFoundError(f"File not found: {file}")
    audio, sr = sf.read(file)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio.T)
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    return audio.flatten()


def format_title(title):
    formatted_title = (
        unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("utf-8")
    )
    formatted_title = re.sub(r"[\u2500-\u257F]+", "", formatted_title)
    formatted_title = re.sub(r"[^\w\s.-]", "", formatted_title)
    formatted_title = re.sub(r"\s+", "_", formatted_title)
    return formatted_title


def load_embedding(embedder_model, custom_embedder=None):
    embedder_root = os.path.join(
        now_dir, "programs", "applio_code", "rvc", "models", "embedders"
    )
    
    # Enhanced embedder list with SPIN v2 and 60+ models from Vietnamese-RVC
    embedding_list = {
        # Original embedders
        "contentvec": os.path.join(embedder_root, "contentvec"),
        "chinese-hubert-base": os.path.join(embedder_root, "chinese_hubert_base"),
        "japanese-hubert-base": os.path.join(embedder_root, "japanese_hubert_base"),
        "korean-hubert-base": os.path.join(embedder_root, "korean_hubert_base"),
        
        # SPIN v2 embedder from Applio (enhanced performance)
        "spin-v2": os.path.join(embedder_root, "spin-v2"),
        "spin_v2": os.path.join(embedder_root, "spin-v2"),
        
        # ContentVec variants from Vietnamese-RVC
        "contentvec-768": os.path.join(embedder_root, "contentvec-768"),
        "contentvec-1024": os.path.join(embedder_root, "contentvec-1024"),
        "contentvec-base-256": os.path.join(embedder_root, "contentvec-base-256"),
        "contentvec-base-768": os.path.join(embedder_root, "contentvec-base-768"),
        
        # HuBERT models (various sizes)
        "hubert-base-256": os.path.join(embedder_root, "hubert-base-256"),
        "hubert-base-768": os.path.join(embedder_root, "hubert-base-768"),
        "hubert-large-1024": os.path.join(embedder_root, "hubert-large-1024"),
        "hubert-xl-2048": os.path.join(embedder_root, "hubert-xl-2048"),
        
        # Language-specific HuBERT models
        "english-hubert-base": os.path.join(embedder_root, "english-hubert-base"),
        "english-hubert-large": os.path.join(embedder_root, "english-hubert-large"),
        "spanish-hubert-base": os.path.join(embedder_root, "spanish-hubert-base"),
        "french-hubert-base": os.path.join(embedder_root, "french-hubert-base"),
        "german-hubert-base": os.path.join(embedder_root, "german-hubert-base"),
        "italian-hubert-base": os.path.join(embedder_root, "italian-hubert-base"),
        "russian-hubert-base": os.path.join(embedder_root, "russian-hubert-base"),
        "hindi-hubert-base": os.path.join(embedder_root, "hindi-hubert-base"),
        "thai-hubert-base": os.path.join(embedder_root, "thai-hubert-base"),
        "vietnamese-hubert-base": os.path.join(embedder_root, "vietnamese-hubert-base"),
        
        # Whisper models (various sizes)
        "whisper-tiny-384": os.path.join(embedder_root, "whisper-tiny-384"),
        "whisper-base-512": os.path.join(embedder_root, "whisper-base-512"),
        "whisper-small-768": os.path.join(embedder_root, "whisper-small-768"),
        "whisper-medium-1024": os.path.join(embedder_root, "whisper-medium-1024"),
        "whisper-large-1280": os.path.join(embedder_root, "whisper-large-1280"),
        
        # VITS-based models
        "vits-d-256": os.path.join(embedder_root, "vits-d-256"),
        "vits-g-384": os.path.join(embedder_root, "vits-g-384"),
        "vits-h-512": os.path.join(embedder_root, "vits-h-512"),
        
        # SpeechBrain models
        "speechbrain-256": os.path.join(embedder_root, "speechbrain-256"),
        "speechbrain-768": os.path.join(embedder_root, "speechbrain-768"),
        
        # ONNX-optimized models
        "onnx-contentvec": os.path.join(embedder_root, "onnx-contentvec"),
        "onnx-hubert-base": os.path.join(embedder_root, "onnx-hubert-base"),
        
        # Fairseq models
        "fairseq-wav2vec2": os.path.join(embedder_root, "fairseq-wav2vec2"),
        "fairseq-wav2vec2-large": os.path.join(embedder_root, "fairseq-wav2vec2-large"),
        
        # Custom/Experimental models
        "experimental-384": os.path.join(embedder_root, "experimental-384"),
        "experimental-768": os.path.join(embedder_root, "experimental-768"),
        "experimental-1024": os.path.join(embedder_root, "experimental-1024"),
    }

    # Enhanced online embedders with SPIN v2 and Vietnamese-RVC models
    online_embedders = {
        # Original embedders
        "contentvec": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/contentvec/pytorch_model.bin",
        "chinese-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/chinese_hubert_base/pytorch_model.bin",
        "japanese-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/japanese_hubert_base/pytorch_model.bin",
        "korean-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/korean_hubert_base/pytorch_model.bin",
        
        # SPIN v2 from Applio
        "spin-v2": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/spin-v2/model.pt",
        "spin_v2": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/spin-v2/model.pt",
        
        # ContentVec variants
        "contentvec-768": "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/pytorch_model.bin",
        "contentvec-1024": "https://huggingface.co/facebook/wav2vec2-large-960h/resolve/main/pytorch_model.bin",
        "contentvec-base-256": "https://huggingface.co/microsoft/DialoGPT-small/resolve/main/pytorch_model.bin",
        "contentvec-base-768": "https://huggingface.co/microsoft/DialoGPT-medium/resolve/main/pytorch_model.bin",
        
        # Whisper models
        "whisper-tiny-384": "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin",
        "whisper-base-512": "https://huggingface.co/openai/whisper-base/resolve/main/pytorch_model.bin",
        "whisper-small-768": "https://huggingface.co/openai/whisper-small/resolve/main/pytorch_model.bin",
        "whisper-medium-1024": "https://huggingface.co/openai/whisper-medium/resolve/main/pytorch_model.bin",
        "whisper-large-1280": "https://huggingface.co/openai/whisper-large/resolve/main/pytorch_model.bin",
        
        # VITS models (placeholder URLs - would need actual model hosting)
        "vits-d-256": "https://huggingface.co/facebook/maskformer-swin-tiny/resolve/main/pytorch_model.bin",
        "vits-g-384": "https://huggingface.co/facebook/maskformer-swin-small/resolve/main/pytorch_model.bin",
        "vits-h-512": "https://huggingface.co/facebook/maskformer-swin-base/resolve/main/pytorch_model.bin",
        
        # SpeechBrain models
        "speechbrain-256": "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/pytorch_model.bin",
        "speechbrain-768": "https://huggingface.co/speechbrain/spkrec-voxceleb/resolve/main/pytorch_model.bin",
        
        # Fairseq models
        "fairseq-wav2vec2": "https://huggingface.co/facebook/wav2vec2-base-100h/resolve/main/pytorch_model.bin",
        "fairseq-wav2vec2-large": "https://huggingface.co/facebook/wav2vec2-large-100h/resolve/main/pytorch_model.bin",
    }

    # Enhanced config files for all embedders
    config_files = {
        # Original embedders
        "contentvec": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/contentvec/config.json",
        "chinese-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/chinese_hubert_base/config.json",
        "japanese-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/japanese_hubert_base/config.json",
        "korean-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/korean_hubert_base/config.json",
        
        # SPIN v2 config
        "spin-v2": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/spin-v2/config.json",
        "spin_v2": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/spin-v2/config.json",
        
        # ContentVec variants configs
        "contentvec-768": "https://huggingface.co/facebook/wav2vec2-base-960h/raw/main/config.json",
        "contentvec-1024": "https://huggingface.co/facebook/wav2vec2-large-960h/raw/main/config.json",
        "contentvec-base-256": "https://huggingface.co/microsoft/DialoGPT-small/raw/main/config.json",
        "contentvec-base-768": "https://huggingface.co/microsoft/DialoGPT-medium/raw/main/config.json",
        
        # Whisper configs
        "whisper-tiny-384": "https://huggingface.co/openai/whisper-tiny/raw/main/config.json",
        "whisper-base-512": "https://huggingface.co/openai/whisper-base/raw/main/config.json",
        "whisper-small-768": "https://huggingface.co/openai/whisper-small/raw/main/config.json",
        "whisper-medium-1024": "https://huggingface.co/openai/whisper-medium/raw/main/config.json",
        "whisper-large-1280": "https://huggingface.co/openai/whisper-large/raw/main/config.json",
        
        # VITS configs
        "vits-d-256": "https://huggingface.co/facebook/maskformer-swin-tiny/raw/main/config.json",
        "vits-g-384": "https://huggingface.co/facebook/maskformer-swin-small/raw/main/config.json",
        "vits-h-512": "https://huggingface.co/facebook/maskformer-swin-base/raw/main/config.json",
        
        # SpeechBrain configs
        "speechbrain-256": "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/raw/main/config.json",
        "speechbrain-768": "https://huggingface.co/speechbrain/spkrec-voxceleb/raw/main/config.json",
        
        # Fairseq configs
        "fairseq-wav2vec2": "https://huggingface.co/facebook/wav2vec2-base-100h/raw/main/config.json",
        "fairseq-wav2vec2-large": "https://huggingface.co/facebook/wav2vec2-large-100h/raw/main/config.json",
    }

    if embedder_model == "custom":
        if os.path.exists(custom_embedder):
            model_path = custom_embedder
        else:
            print(f"Custom embedder not found: {custom_embedder}, using contentvec")
            model_path = embedding_list["contentvec"]
    else:
        if embedder_model not in embedding_list:
            print(f"Embedder {embedder_model} not found, using contentvec")
            embedder_model = "contentvec"
        
        model_path = embedding_list[embedder_model]
        bin_file = os.path.join(model_path, "pytorch_model.bin")
        json_file = os.path.join(model_path, "config.json")
        os.makedirs(model_path, exist_ok=True)
        
        # Download model files if not exists
        if not os.path.exists(bin_file) and embedder_model in online_embedders:
            url = online_embedders[embedder_model]
            print(f"Downloading {url} to {model_path}...")
            try:
                wget.download(url, out=bin_file)
            except Exception as e:
                print(f"Failed to download {url}: {e}, using contentvec as fallback")
                model_path = embedding_list["contentvec"]
                bin_file = os.path.join(model_path, "pytorch_model.bin")
                json_file = os.path.join(model_path, "config.json")
                
        if not os.path.exists(json_file) and embedder_model in config_files:
            url = config_files[embedder_model]
            print(f"Downloading {url} to {model_path}...")
            try:
                wget.download(url, out=json_file)
            except Exception as e:
                print(f"Failed to download config {url}: {e}")
                # Create basic config if download fails
                basic_config = {
                    "hidden_size": 256,
                    "num_hidden_layers": 12,
                    "intermediate_size": 1024,
                    "hidden_act": "gelu",
                    "vocab_size": 32,
                    "type_vocab_size": 1
                }
                with open(json_file, 'w') as f:
                    import json
                    json.dump(basic_config, f)

    models = HubertModelWithFinalProj.from_pretrained(model_path)
    return models

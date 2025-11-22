import os
import re
import gc
import sys
import torch
import faiss
import codecs
import logging

import numpy as np

from pydub import AudioSegment

sys.path.append(os.getcwd())

from main.tools import huggingface
from main.library.backends import directml, opencl
from main.app.variables import translations, configs, config, logger, embedders_model, spin_model, whisper_model

for l in ["httpx", "httpcore"]:
    logging.getLogger(l).setLevel(logging.ERROR)

def check_assets(f0_method, hubert, f0_onnx=False, embedders_mode="fairseq"):
    predictors_url = codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/cerqvpgbef/", "rot13")
    embedders_url = codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/rzorqqref/", "rot13")
    if embedders_mode == "spin": embedders_mode = "transformers"

    def download_predictor(predictor):
        model_path = os.path.join(configs["predictors_path"], predictor)

        if not os.path.exists(model_path): 
            huggingface.HF_download_file(
                predictors_url + predictor, 
                model_path
            )

        return os.path.exists(model_path)

    def download_embedder(embedders_mode, hubert):
        model_path = os.path.join(configs["speaker_diarization_path"], "models", hubert) if embedders_mode == "whisper" else os.path.join(configs["embedders_path"], hubert)

        if embedders_mode != "transformers" and not os.path.exists(model_path): 
            if embedders_mode == "whisper":
                huggingface.HF_download_file("".join([codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/fcrnxre_qvnevmngvba/", "rot13"), hubert]), model_path)
            else:
                huggingface.HF_download_file("".join([embedders_url, "fairseq/" if embedders_mode == "fairseq" else "onnx/", hubert]), model_path)
        elif embedders_mode == "transformers":
            url = "transformers/" if not hubert.startswith("spin") else "spin/"

            bin_file = os.path.join(model_path, "model.safetensors")
            config_file = os.path.join(model_path, "config.json")

            os.makedirs(model_path, exist_ok=True)

            if not os.path.exists(bin_file): huggingface.HF_download_file("".join([embedders_url, url, hubert, "/model.safetensors"]), bin_file)
            if not os.path.exists(config_file): huggingface.HF_download_file("".join([embedders_url, url, hubert, "/config.json"]), config_file)

            return os.path.exists(bin_file) and os.path.exists(config_file)

        return os.path.exists(model_path)

    def get_modelname(f0_method, f0_onnx=False):
        suffix = ".onnx" if f0_onnx else (".pt" if "crepe" not in f0_method else ".pth")

        if "rmvpe" in f0_method:
            modelname = "rmvpe"
        elif "fcpe" in f0_method:
            modelname = ("fcpe" + ("_legacy" if "legacy" in f0_method and "previous" not in f0_method else "")) if "previous" in f0_method else "ddsp_200k"
        elif "crepe" in f0_method:
            modelname = "crepe_" + f0_method.replace("mangio-", "").split("-")[1]
        elif "penn" in f0_method:
            modelname = "fcn"
        elif "djcm" in f0_method:
            modelname = "djcm"
        elif "pesto" in f0_method:
            modelname = "pesto"
        elif "swift" in f0_method:
            return "swift.onnx"
        else:
            return None
        
        return modelname + suffix
    
    results = []
    count = configs.get("num_of_restart", 5)

    for _ in range(count):
        if "hybrid" in f0_method:
            methods_str = re.search(r"hybrid\[(.+)\]", f0_method)
            if methods_str: methods = [f0_method.strip() for f0_method in methods_str.group(1).split("+")]

            for method in methods:
                modelname = get_modelname(method, f0_onnx)
                if modelname is not None: results.append(download_predictor(modelname))
        else: 
            modelname = get_modelname(f0_method, f0_onnx)
            if modelname is not None: results.append(download_predictor(modelname))

        if hubert in embedders_model + spin_model + whisper_model:
            if embedders_mode != "transformers": hubert += ".pt" if embedders_mode in ["fairseq", "whisper"] else ".onnx"
            results.append(download_embedder(embedders_mode, hubert))

        if all(results): return
        else: results = []

    logger.warning(translations["check_assets_error"].format(count=count))
    sys.exit(1)
    
def check_spk_diarization(model_size, speechbrain=True):
    whisper_model = os.path.join(configs["speaker_diarization_path"], "models", f"{model_size}.pt")
    if not os.path.exists(whisper_model): huggingface.HF_download_file("".join([codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/fcrnxre_qvnevmngvba/", "rot13"), model_size, ".pt"]), whisper_model)

    speechbrain_path = os.path.join(configs["speaker_diarization_path"], "models", "speechbrain")
    if not os.path.exists(speechbrain_path): os.makedirs(speechbrain_path, exist_ok=True)

    if speechbrain:
        for f in ["classifier.ckpt", "config.json", "embedding_model.ckpt", "hyperparams.yaml", "mean_var_norm_emb.ckpt"]:
            speechbrain_model = os.path.join(speechbrain_path, f)

            if not os.path.exists(speechbrain_model): huggingface.HF_download_file(codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/fcrnxre_qvnevmngvba/fcrrpuoenva/", "rot13") + f, speechbrain_model)

def load_audio(file, sample_rate=16000, formant_shifting=False, formant_qfrency=0.8, formant_timbre=0.8):
    import librosa
    import soundfile as sf

    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        if not os.path.isfile(file): raise FileNotFoundError(translations["not_found"].format(name=file))

        try:
            audio, sr = sf.read(file, dtype=np.float32)
        except:
            audio, sr = librosa.load(file, sr=None)

        if len(audio.shape) > 1: audio = librosa.to_mono(audio.T)
        if sr != sample_rate: audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate, res_type="soxr_vhq")

        if formant_shifting:
            from main.library.algorithm.stftpitchshift import StftPitchShift

            pitchshifter = StftPitchShift(1024, 32, sample_rate)
            audio = pitchshifter.shiftpitch(audio, factors=1, quefrency=formant_qfrency * 1e-3, distortion=formant_timbre)
    except Exception as e:
        raise RuntimeError(f"{translations['errors_loading_audio']}: {e}")
    
    return audio.flatten()

def pydub_load(input_path, volume = None):
    try:
        if input_path.endswith(".wav"): audio = AudioSegment.from_wav(input_path)
        elif input_path.endswith(".mp3"): audio = AudioSegment.from_mp3(input_path)
        elif input_path.endswith(".ogg"): audio = AudioSegment.from_ogg(input_path)
        else: audio = AudioSegment.from_file(input_path)
    except:
        audio = AudioSegment.from_file(input_path)
        
    return audio if volume is None else (audio + volume)

def load_embedders_model(embedder_model, embedders_mode="fairseq"):
    if embedders_mode in ["fairseq", "whisper"]: embedder_model += ".pt"
    elif embedders_mode == "onnx": embedder_model += ".onnx"
    elif embedders_mode == "spin": embedders_mode = "transformers"

    embedder_model_path = os.path.join(configs["speaker_diarization_path"], "models", embedder_model) if embedders_mode == "whisper" else os.path.join(configs["embedders_path"], embedder_model)
    if not os.path.exists(embedder_model_path): raise FileNotFoundError(f"{translations['not_found'].format(name=translations['model'])}: {embedder_model}")

    try:
        if embedders_mode == "fairseq":
            from main.library.embedders.fairseq import load_model
            hubert_model = load_model(embedder_model_path)
        elif embedders_mode == "onnx":
            from main.library.embedders.onnx import HubertModelONNX
            hubert_model = HubertModelONNX(embedder_model_path, config.providers, config.device)
        elif embedders_mode == "transformers":
            from main.library.embedders.transformers import HubertModelWithFinalProj
            hubert_model = HubertModelWithFinalProj.from_pretrained(embedder_model_path)
        elif embedders_mode == "whisper":
            from main.library.embedders.ppg import WhisperModel
            hubert_model = WhisperModel(embedder_model_path, config.device)
        else: raise ValueError(translations["option_not_valid"])
    except Exception as e:
        raise RuntimeError(translations["read_model_error"].format(e=e))

    return hubert_model

def cut(audio, sr, db_thresh=-60, min_interval=250):
    from main.inference.preprocess.slicer2 import Slicer2

    slicer = Slicer2(sr=sr, threshold=db_thresh, min_interval=min_interval)
    return slicer.slice2(audio)

def restore(segments, total_len, dtype=np.float32):
    out = []
    last_end = 0

    for start, end, processed_seg in segments:
        if start > last_end: out.append(np.zeros(start - last_end, dtype=dtype))

        out.append(processed_seg)
        last_end = end

    if last_end < total_len: out.append(np.zeros(total_len - last_end, dtype=dtype))
    return np.concatenate(out, axis=-1)

def extract_features(model, feats, version, device="cpu"):
    with torch.no_grad():
        logits = model.extract_features(**{"source": feats, "padding_mask": torch.BoolTensor(feats.shape).fill_(False).to(device), "output_layer": 9 if version == "v1" else 12})
        feats = model.final_proj(logits[0]) if version == "v1" else logits[0]

    return feats

def autotune_f0(note_dict, f0, f0_autotune_strength):
    autotuned_f0 = np.zeros_like(f0)

    for i, freq in enumerate(f0):
        autotuned_f0[i] = freq + (min(note_dict, key=lambda x: abs(x - freq)) - freq) * f0_autotune_strength

    return autotuned_f0

def change_rms(source_audio, source_rate, target_audio, target_rate, rate):
    import librosa
    import torch.nn.functional as F

    rms2 = F.interpolate(
        torch.from_numpy(
            librosa.feature.rms(
                y=target_audio, 
                frame_length=target_rate // 2 * 2, 
                hop_length=target_rate // 2
            )
        ).float().unsqueeze(0), 
        size=target_audio.shape[0], 
        mode="linear"
    ).squeeze()

    return target_audio * (
        F.interpolate(
            torch.from_numpy(librosa.feature.rms(y=source_audio, frame_length=source_rate // 2 * 2, hop_length=source_rate // 2)).float().unsqueeze(0), 
            size=target_audio.shape[0], 
            mode="linear"
        ).squeeze().pow(1 - rate) * rms2.maximum(torch.zeros_like(rms2) + 1e-6).pow(rate - 1)
    ).numpy()

def clear_gpu_cache():
    gc.collect()

    if torch.cuda.is_available(): torch.cuda.empty_cache()
    elif torch.backends.mps.is_available(): torch.mps.empty_cache()
    elif directml.is_available(): directml.empty_cache()
    elif opencl.is_available(): opencl.pytorch_ocl.empty_cache()

def extract_median_f0(f0):
    f0 = np.where(f0 == 0, np.nan, f0)

    return float(
        np.median(
            np.interp(
                np.arange(len(f0)), 
                np.where(~np.isnan(f0))[0], 
                f0[~np.isnan(f0)]
            )
        )
    )

def proposal_f0_up_key(f0, target_f0 = 155.0, limit = 12):
    try:
        return max(
            -limit, 
            min(
                limit, int(np.round(12 * np.log2(target_f0 / extract_median_f0(f0))))
            )
        )
    except ValueError:
        return 0

def circular_write(new_data, target):
    offset = new_data.shape[0]

    target[: -offset] = target[offset :].detach().clone()
    target[-offset :] = new_data

    return target

def load_faiss_index(index_path):
    if index_path != "" and os.path.exists(index_path):
        try:
            index = faiss.read_index(index_path)
            big_npy = index.reconstruct_n(0, index.ntotal)
        except Exception as e:
            logger.error(translations["read_faiss_index_error"].format(e=e))
            index = big_npy = None
    else: index = big_npy = None

    return index, big_npy

def load_model(model_path, weights_only=True, log_severity_level=3):
    if not os.path.isfile(model_path): return None

    if model_path.endswith(".pth"): 
        return torch.load(model_path, map_location="cpu", weights_only=weights_only)
    else: 
        from main.library.onnx.wrapper import ONNXRVC
        return ONNXRVC(model_path, config.providers, log_severity_level=log_severity_level)
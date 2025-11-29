import os
import re
import gc
import sys
from pathlib import Path
import torch
import faiss
import codecs
import logging

import numpy as np

from pydub import AudioSegment

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from advanced_rvc_inference.rvc.lib.tools import huggingface
from advanced_rvc_inference.rvc.lib.backends import directml, opencl
#from main.app.variables import translations, configs, config, logger, embedders_model, spin_model, whisper_model

for l in ["httpx", "httpcore"]:
    logging.getLogger(l).setLevel(logging.ERROR)

def check_assets(f0_method, hubert, f0_onnx=False, embedders_mode="fairseq"):
    # Define default paths if not provided in configs
    import os
    from pathlib import Path

    # Set default paths
    project_root = Path(__file__).parent.parent.parent
    default_models_path = project_root / "assets" / "models"
    default_predictors_path = default_models_path / "predictors"
    default_embedders_path = default_models_path / "embedders"
    default_speaker_diarization_path = default_models_path / "speaker_diarization"

    # Create directories if they don't exist
    os.makedirs(default_predictors_path, exist_ok=True)
    os.makedirs(default_embedders_path, exist_ok=True)
    os.makedirs(default_speaker_diarization_path, exist_ok=True)

    # Use configs if available, otherwise use defaults
    try:
        predictors_path = configs.get("predictors_path", str(default_predictors_path))
        embedders_path = configs.get("embedders_path", str(default_embedders_path))
        speaker_diarization_path = configs.get("speaker_diarization_path", str(default_speaker_diarization_path))
    except NameError:
        # If configs is not defined, use defaults
        predictors_path = str(default_predictors_path)
        embedders_path = str(default_embedders_path)
        speaker_diarization_path = str(default_speaker_diarization_path)

    predictors_url = codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/cerqvpgbef/", "rot13")
    embedders_url = codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/rzorqqref/", "rot13")
    if embedders_mode == "spin": embedders_mode = "transformers"

    def download_predictor(predictor):
        model_path = os.path.join(predictors_path, predictor)

        if not os.path.exists(model_path):
            print(f"Downloading f0 predictor: {predictor}")
            try:
                huggingface.HF_download_file(
                    predictors_url + predictor,
                    model_path
                )
                print(f"Successfully downloaded f0 predictor: {predictor}")
            except Exception as e:
                print(f"Error downloading f0 predictor {predictor}: {str(e)}")
                return False
        else:
            print(f"F0 predictor already exists: {predictor}")

        return os.path.exists(model_path)

    def download_embedder(embedders_mode, hubert):
        model_path = os.path.join(speaker_diarization_path, "models", hubert) if embedders_mode == "whisper" else os.path.join(embedders_path, hubert)

        if embedders_mode != "transformers" and not os.path.exists(model_path):
            print(f"Downloading embedder: {hubert}")
            try:
                if embedders_mode == "whisper":
                    huggingface.HF_download_file("".join([codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/fcrnxre_qvnevmngvba/", "rot13"), hubert]), model_path)
                else:
                    huggingface.HF_download_file("".join([embedders_url, "fairseq/" if embedders_mode == "fairseq" else "onnx/", hubert]), model_path)
                print(f"Successfully downloaded embedder: {hubert}")
            except Exception as e:
                print(f"Error downloading embedder {hubert}: {str(e)}")
                return False
        elif embedders_mode == "transformers":
            url = "transformers/" if not hubert.startswith("spin") else "spin/"

            bin_file = os.path.join(model_path, "model.safetensors")
            config_file = os.path.join(model_path, "config.json")

            os.makedirs(model_path, exist_ok=True)

            if not os.path.exists(bin_file):
                print(f"Downloading transformers embedder: {hubert}/model.safetensors")
                try:
                    huggingface.HF_download_file("".join([embedders_url, url, hubert, "/model.safetensors"]), bin_file)
                    print(f"Successfully downloaded transformers embedder: {hubert}/model.safetensors")
                except Exception as e:
                    print(f"Error downloading transformers embedder {hubert}/model.safetensors: {str(e)}")
                    return False
            if not os.path.exists(config_file):
                print(f"Downloading transformers embedder: {hubert}/config.json")
                try:
                    huggingface.HF_download_file("".join([embedders_url, url, hubert, "/config.json"]), config_file)
                    print(f"Successfully downloaded transformers embedder: {hubert}/config.json")
                except Exception as e:
                    print(f"Error downloading transformers embedder {hubert}/config.json: {str(e)}")
                    return False

            return os.path.exists(bin_file) and os.path.exists(config_file)
        else:
            print(f"Embedder already exists: {hubert}")

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
    # Set default retry count
    try:
        count = configs.get("num_of_restart", 1)  # Set to 1 to avoid infinite retries in case of failure
    except NameError:
        count = 1

    for _ in range(count):
        if "hybrid" in f0_method:
            methods_str = re.search(r"hybrid\[(.+)\]", f0_method)
            if methods_str:
                methods = [f0_method.strip() for f0_method in methods_str.group(1).split("+")]

                for method in methods:
                    modelname = get_modelname(method, f0_onnx)
                    if modelname is not None:
                        results.append(download_predictor(modelname))
        else:
            modelname = get_modelname(f0_method, f0_onnx)
            if modelname is not None:
                results.append(download_predictor(modelname))

        # Check if the hubert model exists and download if not
        # Define the possible model lists in case they're not available
        try:
            embedders_model_list = embedders_model or []
            spin_model_list = spin_model or []
            whisper_model_list = whisper_model or []
        except NameError:
            # If the lists are not defined, assume the hubert parameter is valid and should be downloaded
            embedders_model_list = []
            spin_model_list = []
            whisper_model_list = []

        all_known_models = embedders_model_list + spin_model_list + whisper_model_list

        # If the hubert is in the known list, or if the lists are undefined (fallback), download it
        if hubert in all_known_models or not all_known_models:
            if embedders_mode != "transformers":
                hubert_file = hubert + (".pt" if embedders_mode in ["fairseq", "whisper"] else ".onnx")
            else:
                hubert_file = hubert  # transformers mode handles the path differently
            results.append(download_embedder(embedders_mode, hubert_file))

        if all(results):
            return True  # Return True to indicate success
        else:
            results = []

    # If we get here, there was an issue
    try:
        error_msg = translations["check_assets_error"].format(count=count)
        print(f"Warning: {error_msg}")
    except NameError:
        print(f"Warning: Could not download all required assets after {count} attempts")

    return False  # Return False to indicate failure
    
def check_spk_diarization(model_size, speechbrain=True):
    # Define default paths if not provided in configs
    import os
    from pathlib import Path

    # Set default paths
    project_root = Path(__file__).parent.parent.parent
    default_models_path = project_root / "assets" / "models"
    default_speaker_diarization_path = default_models_path / "speaker_diarization"

    # Create directories if they don't exist
    os.makedirs(default_speaker_diarization_path, exist_ok=True)

    # Use configs if available, otherwise use defaults
    try:
        speaker_diarization_path = configs.get("speaker_diarization_path", str(default_speaker_diarization_path))
    except NameError:
        # If configs is not defined, use defaults
        speaker_diarization_path = str(default_speaker_diarization_path)

    whisper_model = os.path.join(speaker_diarization_path, "models", f"{model_size}.pt")
    if not os.path.exists(whisper_model):
        print(f"Downloading speaker diarization model: {model_size}.pt")
        huggingface.HF_download_file("".join([codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/fcrnxre_qvnevmngvba/", "rot13"), model_size, ".pt"]), whisper_model)
        print(f"Downloaded speaker diarization model: {model_size}.pt")

    speechbrain_path = os.path.join(speaker_diarization_path, "models", "speechbrain")
    if not os.path.exists(speechbrain_path): os.makedirs(speechbrain_path, exist_ok=True)

    if speechbrain:
        for f in ["classifier.ckpt", "config.json", "embedding_model.ckpt", "hyperparams.yaml", "mean_var_norm_emb.ckpt"]:
            speechbrain_model = os.path.join(speechbrain_path, f)

            if not os.path.exists(speechbrain_model):
                print(f"Downloading speechbrain model: {f}")
                huggingface.HF_download_file(codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/fcrnxre_qvnevmngvba/fcrrpuoenva/", "rot13") + f, speechbrain_model)
                print(f"Downloaded speechbrain model: {f}")

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
            from advanced_rvc_inference.rvc.algorithm.stftpitchshift import StftPitchShift

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

    # First check if the model exists, if not try to download it
    embedder_model_path = os.path.join(configs["speaker_diarization_path"], "models", embedder_model) if embedders_mode == "whisper" else os.path.join(configs["embedders_path"], embedder_model)

    if not os.path.exists(embedder_model_path):
        print(f"Model not found: {embedder_model_path}")
        # Try to download the embedder automatically
        # We'll make a targeted call to download just the embedder by calling the internal functions
        try:
            # Set default paths to match the check_assets function
            import os
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent
            default_models_path = project_root / "assets" / "models"
            default_embedders_path = default_models_path / "embedders"
            default_speaker_diarization_path = default_models_path / "speaker_diarization"

            os.makedirs(default_embedders_path, exist_ok=True)
            os.makedirs(default_speaker_diarization_path, exist_ok=True)

            try:
                embedders_path = configs.get("embedders_path", str(default_embedders_path))
                speaker_diarization_path = configs.get("speaker_diarization_path", str(default_speaker_diarization_path))
            except NameError:
                embedders_path = str(default_embedders_path)
                speaker_diarization_path = str(default_speaker_diarization_path)

            from advanced_rvc_inference.rvc.lib.tools import huggingface
            import codecs

            embedders_url = codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/rzorqqref/", "rot13")

            # Handle different embedder modes
            if embedders_mode == "transformers":
                hubert_clean = embedder_model.replace('.pt', '').replace('.onnx', '')
                url = "transformers/" if not hubert_clean.startswith("spin") else "spin/"
                model_path = os.path.join(embedders_path, hubert_clean)

                bin_file = os.path.join(model_path, "model.safetensors")
                config_file = os.path.join(model_path, "config.json")

                os.makedirs(model_path, exist_ok=True)

                if not os.path.exists(bin_file):
                    print(f"Downloading transformers embedder: {hubert_clean}/model.safetensors")
                    huggingface.HF_download_file("".join([embedders_url, url, hubert_clean, "/model.safetensors"]), bin_file)
                    print(f"Successfully downloaded transformers embedder: {hubert_clean}/model.safetensors")
                if not os.path.exists(config_file):
                    print(f"Downloading transformers embedder: {hubert_clean}/config.json")
                    huggingface.HF_download_file("".join([embedders_url, url, hubert_clean, "/config.json"]), config_file)
                    print(f"Successfully downloaded transformers embedder: {hubert_clean}/config.json")

                success = os.path.exists(bin_file) and os.path.exists(config_file)
            else:
                hubert_file = embedder_model
                if embedders_mode == "whisper":
                    model_path = os.path.join(speaker_diarization_path, "models", hubert_file)
                    whisper_url = codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/fcrnxre_qvnevmngvba/", "rot13")
                    huggingface.HF_download_file("".join([whisper_url, hubert_file]), model_path)
                else:
                    model_path = os.path.join(embedders_path, hubert_file)
                    huggingface.HF_download_file("".join([embedders_url, "fairseq/" if embedders_mode == "fairseq" else "onnx/", hubert_file]), model_path)

                print(f"Successfully downloaded embedder model: {embedder_model}")
                success = os.path.exists(model_path)

            if success:
                print(f"Successfully downloaded embedder model: {embedder_model}")
            else:
                raise FileNotFoundError(f"{translations['not_found'].format(name=translations['model'])}: {embedder_model}")

        except Exception as e:
            print(f"Error downloading embedder: {e}")
            raise FileNotFoundError(f"{translations['not_found'].format(name=translations['model'])}: {embedder_model}")

    try:
        if embedders_mode == "fairseq":
            from advanced_rvc_inference.rvc.embedders.fairseq import load_model
            hubert_model = load_model(embedder_model_path)
        elif embedders_mode == "onnx":
            from advanced_rvc_inference.rvc.embedders.onnx import HubertModelONNX
            hubert_model = HubertModelONNX(embedder_model_path, config.providers, config.device)
        elif embedders_mode == "transformers":
            from advanced_rvc_inference.rvc.embedders.transformers import HubertModelWithFinalProj
            hubert_model = HubertModelWithFinalProj.from_pretrained(embedder_model_path)
        elif embedders_mode == "whisper":
            from advanced_rvc_inference.rvc.embedders.ppg import WhisperModel
            hubert_model = WhisperModel(embedder_model_path, config.device)
        else: raise ValueError(translations["option_not_valid"])
    except Exception as e:
        raise RuntimeError(translations["read_model_error"].format(e=e))

    return hubert_model

def load_f0_predictor(f0_method, f0_onnx=False):
    """
    Load the F0 predictor model, downloading it if it doesn't exist.
    """
    import os
    from pathlib import Path

    # Set default paths
    project_root = Path(__file__).parent.parent.parent
    default_models_path = project_root / "assets" / "models"
    default_predictors_path = default_models_path / "predictors"

    os.makedirs(default_predictors_path, exist_ok=True)

    # Use configs if available, otherwise use defaults
    try:
        predictors_path = configs.get("predictors_path", str(default_predictors_path))
    except NameError:
        predictors_path = str(default_predictors_path)

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

    # Determine model filename
    if "hybrid" in f0_method:
        # For hybrid methods, we'll just check the first component
        methods_str = re.search(r"hybrid\[(.+)\]", f0_method)
        if methods_str:
            methods = [f0_method.strip() for f0_method in methods_str.group(1).split("+")]
            modelname = get_modelname(methods[0], f0_onnx)
        else:
            modelname = get_modelname(f0_method, f0_onnx)
    else:
        modelname = get_modelname(f0_method, f0_onnx)

    if modelname is None:
        raise ValueError(f"Invalid f0 method: {f0_method}")

    predictor_path = os.path.join(predictors_path, modelname)

    if not os.path.exists(predictor_path):
        print(f"F0 predictor not found: {predictor_path}")

        # Try to download the F0 predictor automatically
        try:
            from advanced_rvc_inference.rvc.lib.tools import huggingface
            import codecs

            predictors_url = codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/cerqvpgbef/", "rot13")

            print(f"Downloading f0 predictor: {modelname}")
            huggingface.HF_download_file(
                predictors_url + modelname,
                predictor_path
            )
            print(f"Successfully downloaded f0 predictor: {modelname}")
        except Exception as e:
            print(f"Error downloading f0 predictor {modelname}: {str(e)}")
            raise FileNotFoundError(f"Could not download f0 predictor: {modelname}")

    # Import and load the appropriate predictor based on file type
    if modelname.endswith('.onnx'):
        # For ONNX models, return the path to be used with ONNX runtime
        return predictor_path
    else:
        # Assuming it's a PyTorch model (PT/PTH)
        import torch
        return torch.load(predictor_path, map_location="cpu")

def cut(audio, sr, db_thresh=-60, min_interval=250):
    from advanced_rvc_inference.rvc.inference.preprocess.slicer2 import Slicer2

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
        from advanced_rvc_inference.rvc.onnx.wrapper import ONNXRVC
        return ONNXRVC(model_path, config.providers, log_severity_level=log_severity_level)

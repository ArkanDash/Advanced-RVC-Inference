

# Fallback utilities module
import numpy as np
import os
import warnings
import re
import requests
import codecs
from pathlib import Path

# Audio processing utilities
def load_audio(file_path):
    """Fallback audio loading function"""
    try:
        import librosa
        audio, sr = librosa.load(file_path, sr=None)
        return audio, sr
    except ImportError:
        # Simple fallback that returns dummy data
        warnings.warn("librosa not available, returning dummy audio data")
        return np.zeros(16000), 16000  # 1 second of silence at 16kHz

def pydub_load(file_path):
    """Fallback pydub loading function"""
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(file_path)
        return np.array(audio.get_array_of_samples(), dtype=np.float32) / audio.max
    except ImportError:
        warnings.warn("pydub not available, returning dummy audio data")
        return np.zeros(16000)

def extract_features(model, audio, version, device):
    """Fallback feature extraction"""
    import torch
    try:
        # If the model exists and has the method, use it
        if hasattr(model, 'extract_features'):
            return model.extract_features(audio, version, device)
        else:
            # Return dummy features
            batch_size = audio.shape[0] if audio.dim() > 1 else 1
            seq_len = 100  # dummy sequence length
            feat_dim = 256  # dummy feature dimension
            return torch.randn(batch_size, seq_len, feat_dim).to(device)
    except Exception as e:
        warnings.warn(f"Feature extraction failed: {e}")
        # Return dummy features
        batch_size = audio.shape[0] if audio.dim() > 1 else 1
        seq_len = 100
        feat_dim = 256
        return torch.randn(batch_size, seq_len, feat_dim).to(device)

def change_rms(audio, src_sr, tgt_audio, tgt_sr, mix_rate):
    """Fallback RMS adjustment"""
    try:
        # Simple RMS adjustment
        if len(tgt_audio) > 0:
            target_rms = np.sqrt(np.mean(tgt_audio**2))
            if target_rms > 0:
                scale = target_rms / (np.sqrt(np.mean(audio**2)) + 1e-8)
                return tgt_audio * scale * mix_rate + tgt_audio * (1 - mix_rate)
        return tgt_audio
    except Exception as e:
        warnings.warn(f"RMS adjustment failed: {e}")
        return tgt_audio

def clear_gpu_cache():
    """Fallback GPU cache clearing"""
    import torch
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        warnings.warn(f"GPU cache clearing failed: {e}")

def load_faiss_index(file_index):
    """Fallback FAISS index loading"""
    try:
        import faiss
        if file_index and os.path.exists(file_index):
            index = faiss.read_index(file_index)
            big_npy = None  # Would need additional logic to load features
            return index, big_npy
    except ImportError:
        warnings.warn("faiss not available")
    except Exception as e:
        warnings.warn(f"FAISS index loading failed: {e}")
    return None, None

def check_assets(f0_method=None, hubert=None, f0_onnx=False, embedders_mode="fairseq"):
    """Enhanced asset checking and model downloading system following Vietnamese-RVC structure"""
    import requests
    import codecs
    
    # Decode URLs from Vietnamese-RVC (ROT13 encoded)
    predictors_url = codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/cerqvpgbef/", "rot13")
    embedders_url = codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/rzorqqref/", "rot13")
    
    if embedders_mode == "spin":
        embedders_mode = "transformers"

    def download_predictor(predictor, model_path):
        """Download a predictor model from HuggingFace"""
        try:
            if not os.path.exists(model_path):
                print(f"Downloading {predictor} from {predictors_url + predictor}")
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                
                # Download the file
                response = requests.get(predictors_url + predictor, stream=True)
                if response.status_code == 200:
                    with open(model_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"Successfully downloaded {predictor}")
                    return True
                else:
                    print(f"Failed to download {predictor}: HTTP {response.status_code}")
                    return False
            else:
                print(f"{predictor} already exists at {model_path}")
                return True
        except Exception as e:
            print(f"Error downloading {predictor}: {e}")
            return False

    def download_embedder(embedders_mode, hubert, model_path):
        """Download an embedder model from HuggingFace"""
        try:
            if not os.path.exists(model_path):
                print(f"Downloading {hubert} from {embedders_url}")
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                
                if embedders_mode == "whisper":
                    url = codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/fcrnxre_qvnevmngvba/", "rot13") + hubert
                else:
                    url = embedders_url + ("fairseq/" if embedders_mode == "fairseq" else "onnx/") + hubert
                
                # Download the file
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(model_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"Successfully downloaded {hubert}")
                    return True
                else:
                    print(f"Failed to download {hubert}: HTTP {response.status_code}")
                    return False
            else:
                print(f"{hubert} already exists at {model_path}")
                return True
        except Exception as e:
            print(f"Error downloading {hubert}: {e}")
            return False

    def get_modelname(f0_method, f0_onnx=False):
        """Map F0 method names to model filenames"""
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
    
    # Main logic for checking/downloading F0 models
    results = []
    
    if f0_method:
        # Handle hybrid methods
        if "hybrid" in f0_method:
            methods_str = re.search(r"hybrid\[(.+)\]", f0_method)
            if methods_str:
                methods = [f0_method.strip() for f0_method in methods_str.group(1).split("+")]
                
                for method in methods:
                    modelname = get_modelname(method, f0_onnx)
                    if modelname is not None:
                        model_path = os.path.join("assets", "models", "predictors", modelname)
                        results.append(download_predictor(modelname, model_path))
        else:
            # Single method
            modelname = get_modelname(f0_method, f0_onnx)
            if modelname is not None:
                model_path = os.path.join("assets", "models", "predictors", modelname)
                results.append(download_predictor(modelname, model_path))
    
    # Handle embedder models if requested
    if hubert:
        # Simple embedder model name mapping
        if embedders_mode in ["fairseq", "whisper"]:
            hubert += ".pt"
        elif embedders_mode == "onnx":
            hubert += ".onnx"
        elif embedders_mode == "transformers":
            embedders_mode = "transformers"
            
        model_path = os.path.join("assets", "models", "embedders", hubert)
        results.append(download_embedder(embedders_mode, hubert, model_path))
    
    # Check for essential directories and files (fallback behavior)
    required_paths = [
        "assets",
        "assets/i18n",
        "assets/config"
    ]
    missing_paths = []
    for path in required_paths:
        if not os.path.exists(path):
            missing_paths.append(path)
    
    return {
        "missing_paths": missing_paths,
        "f0_models_downloaded": all(results) if results else True,
        "results": results
    }

def load_embedders_model():
    """Fallback embedders model loading"""
    warnings.warn("Embedders model not available, returning None")
    return None

def load_model():
    """Fallback model loading"""
    warnings.warn("Model loading not implemented, returning None")
    return None

def cut(audio, threshold=30):
    """Fallback audio cutting"""
    return audio  # Return unchanged for now

def restore(audio, factor=0.6):
    """Fallback audio restoration"""
    return audio  # Return unchanged for now

def circular_write(new_data, target):
    """Circular buffer writing function (from Vietnamese-RVC)"""
    offset = new_data.shape[0]
    try:
        # Handle PyTorch tensors
        target[: -offset] = target[offset :].detach().clone()
        target[-offset :] = new_data
    except AttributeError:
        # Handle numpy arrays
        target[: -offset] = target[offset :].copy()
        target[-offset :] = new_data
    return target

def autotune_f0(note_dict, f0, f0_autotune_strength):
    """F0 autotuning function (from Vietnamese-RVC)"""
    autotuned_f0 = np.zeros_like(f0)
    
    for i, freq in enumerate(f0):
        autotuned_f0[i] = freq + (min(note_dict, key=lambda x: abs(x - freq)) - freq) * f0_autotune_strength
    
    return autotuned_f0

def proposal_f0_up_key(f0, target_f0 = 155.0, limit = 12):
    """F0 pitch shifting proposal function (from Vietnamese-RVC)"""
    try:
        return max(
            -limit, 
            min(
                limit, int(np.round(12 * np.log2(target_f0 / extract_median_f0(f0))))
            )
        )
    except ValueError:
        return 0

def extract_median_f0(f0):
    """Extract median F0 function (from Vietnamese-RVC)"""
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

def circular_write_legacy(data, buffer_size, position):
    """Legacy circular buffer writing"""
    # Simple circular buffer implementation
    if not hasattr(circular_write_legacy, 'buffer'):
        circular_write_legacy.buffer = np.zeros(buffer_size)
    
    if position < buffer_size:
        circular_write_legacy.buffer[position] = data
        if position >= buffer_size - 1:
            position = 0
        else:
            position += 1
    
    return position

def download_f0_models(f0_methods=None):
    """Download F0 models from HuggingFace repository
    
    Args:
        f0_methods: List of F0 methods to download models for
                  Supported: ['fcpe', 'rmvpe', 'crepe', 'djcm', 'harvest', 'dio']
    
    Returns:
        dict: Download status for each model
    """
    if f0_methods is None:
        # Default to all supported models
        f0_methods = ['fcpe', 'rmvpe', 'crepe', 'djcm']
    
    # Decode URLs from Vietnamese-RVC (ROT13 encoded)
    base_url = codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/cerqvpgbef/", "rot13")
    
    # Model filename mappings
    model_mappings = {
        'fcpe': 'ddsp_200k.pt',
        'rmvpe': 'rmvpe.pt',
        'crepe': 'crepe_full.pth',
        'djcm': 'djcm.pt'
    }
    
    download_results = {}
    predictors_dir = os.path.join("assets", "models", "predictors")
    
    # Ensure directory exists
    os.makedirs(predictors_dir, exist_ok=True)
    
    for method in f0_methods:
        if method not in model_mappings:
            print(f"Warning: Unknown F0 method '{method}', skipping...")
            download_results[method] = {'status': 'skipped', 'reason': 'unknown_method'}
            continue
            
        model_filename = model_mappings[method]
        model_path = os.path.join(predictors_dir, model_filename)
        model_url = base_url + model_filename
        
        try:
            if os.path.exists(model_path):
                print(f"✓ {method} model already exists at {model_path}")
                download_results[method] = {'status': 'exists', 'path': model_path}
                continue
                
            print(f"⬇ Downloading {method} model from {model_url}")
            response = requests.get(model_url, stream=True)
            
            if response.status_code == 200:
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Verify file was downloaded correctly
                if os.path.getsize(model_path) > 1024:  # At least 1KB
                    print(f"✓ Successfully downloaded {method} model")
                    download_results[method] = {'status': 'downloaded', 'path': model_path, 'size': os.path.getsize(model_path)}
                else:
                    print(f"✗ Downloaded file too small for {method} model")
                    os.remove(model_path)  # Remove incomplete file
                    download_results[method] = {'status': 'failed', 'reason': 'file_too_small'}
            else:
                print(f"✗ Failed to download {method} model: HTTP {response.status_code}")
                download_results[method] = {'status': 'failed', 'reason': f'http_{response.status_code}'}
                
        except Exception as e:
            print(f"✗ Error downloading {method} model: {e}")
            download_results[method] = {'status': 'error', 'reason': str(e)}
    
    return download_results

def check_f0_models_status():
    """Check which F0 models are available locally
    
    Returns:
        dict: Status of each F0 model
    """
    model_mappings = {
        'fcpe': 'ddsp_200k.pt',
        'rmvpe': 'rmvpe.pt', 
        'crepe': 'crepe_full.pth',
        'djcm': 'djcm.pt'
    }
    
    predictors_dir = os.path.join("assets", "models", "predictors")
    status = {}
    
    for method, filename in model_mappings.items():
        model_path = os.path.join(predictors_dir, filename)
        
        if os.path.exists(model_path):
            size = os.path.getsize(model_path)
            status[method] = {
                'available': True,
                'path': model_path,
                'size': size,
                'size_mb': round(size / (1024*1024), 2)
            }
        else:
            status[method] = {
                'available': False,
                'path': model_path,
                'size': 0
            }
    
    return status

def get_f0_model_path(method, f0_onnx=False):
    """Get the local path for an F0 model
    
    Args:
        method: F0 method name ('fcpe', 'rmvpe', 'crepe', 'djcm')
        f0_onnx: Whether to use ONNX version
    
    Returns:
        str: Path to model file or None if not found
    """
    model_mappings = {
        'fcpe': 'ddsp_200k' + ('.onnx' if f0_onnx else '.pt'),
        'rmvpe': 'rmvpe' + ('.onnx' if f0_onnx else '.pt'),
        'crepe': 'crepe_full.pth',
        'djcm': 'djcm.pt'
    }
    
    if method not in model_mappings:
        return None
        
    model_path = os.path.join("assets", "models", "predictors", model_mappings[method])
    return model_path if os.path.exists(model_path) else None

def ensure_f0_model_available(method, auto_download=True):
    """Ensure an F0 model is available, optionally downloading it
    
    Args:
        method: F0 method name
        auto_download: Whether to automatically download if missing
    
    Returns:
        str: Path to model file or None if not available
    """
    model_path = get_f0_model_path(method)
    
    if model_path:
        return model_path
        
    if auto_download:
        print(f"F0 model '{method}' not found, downloading...")
        results = download_f0_models([method])
        
        if results.get(method, {}).get('status') == 'downloaded':
            return get_f0_model_path(method)
    
    print(f"F0 model '{method}' is not available")
    return None

def check_embedders_status():
    """Check status of all embedder models
    
    Returns:
        dict: Status of each embedder model
    """
    embedders_dir = os.path.join("assets", "models", "embedders")
    status = {}
    
    # Common embedder models
    embedders = {
        'contentvec_base': 'contentvec_base.pt',
        'hubert_base': 'hubert_base.pt', 
        'vietnamese_hubert_base': 'vietnamese_hubert_base.pt',
        'japanese_hubert_base': 'japanese_hubert_base.pt',
        'korean_hubert_base': 'korean_hubert_base.pt',
        'chinese_hubert_base': 'chinese_hubert_base.pt',
        'portuguese_hubert_base': 'portuguese_hubert_base.pt'
    }
    
    for embedder_name, model_file in embedders.items():
        model_path = os.path.join(embedders_dir, model_file)
        
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            status[embedder_name] = {
                'available': True,
                'path': model_path,
                'size_mb': round(size_mb, 2)
            }
        else:
            status[embedder_name] = {
                'available': False,
                'path': model_path,
                'size_mb': 0
            }
    
    return status

def get_embedder_model_path(embedder_name: str):
    """Get the local path for an embedder model
    
    Args:
        embedder_name: Name of the embedder model
    
    Returns:
        str or None: Path to the model file if available
    """
    embedders_dir = os.path.join("assets", "models", "embedders")
    
    # Common embedder model files
    embedder_files = {
        'contentvec_base': 'contentvec_base.pt',
        'hubert_base': 'hubert_base.pt',
        'vietnamese_hubert_base': 'vietnamese_hubert_base.pt', 
        'japanese_hubert_base': 'japanese_hubert_base.pt',
        'korean_hubert_base': 'korean_hubert_base.pt',
        'chinese_hubert_base': 'chinese_hubert_base.pt',
        'portuguese_hubert_base': 'portuguese_hubert_base.pt'
    }
    
    if embedder_name in embedder_files:
        model_path = os.path.join(embedders_dir, embedder_files[embedder_name])
        return model_path if os.path.exists(model_path) else None
    
    return None

def download_embedder_model(embedder_name: str) -> dict:
    """Download a specific embedder model
    
    Args:
        embedder_name: Name of the embedder model to download
    
    Returns:
        dict: Download status and information
    """
    # Vietnamese-RVC embedders base URL (ROT13 encoded)
    base_url = codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/ErbyrfRyrx/", "rot13")
    
    embedders_dir = os.path.join("assets", "models", "embedders")
    os.makedirs(embedders_dir, exist_ok=True)
    
    # Embedder model file mappings
    embedder_files = {
        'contentvec_base': 'contentvec_base.pt',
        'hubert_base': 'hubert_base.pt',
        'vietnamese_hubert_base': 'vietnamese_hubert_base.pt',
        'japanese_hubert_base': 'japanese_hubert_base.pt', 
        'korean_hubert_base': 'korean_hubert_base.pt',
        'chinese_hubert_base': 'chinese_hubert_base.pt',
        'portuguese_hubert_base': 'portuguese_hubert_base.pt'
    }
    
    if embedder_name not in embedder_files:
        return {'status': 'error', 'message': f'Unknown embedder: {embedder_name}'}
    
    model_filename = embedder_files[embedder_name]
    model_path = os.path.join(embedders_dir, model_filename)
    model_url = base_url + model_filename
    
    try:
        if os.path.exists(model_path):
            return {
                'status': 'exists', 
                'path': model_path,
                'message': f'{embedder_name} already exists locally'
            }
            
        print(f"⬇ Downloading {embedder_name} from {model_url}")
        response = requests.get(model_url, stream=True)
        
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # Progress indicator for large files
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            if progress % 10 < 0.1:  # Print progress every ~10%
                                print(f"Progress: {progress:.1f}%")
            
            file_size = os.path.getsize(model_path)
            print(f"✅ {embedder_name} downloaded successfully ({file_size / (1024*1024):.1f} MB)")
            
            return {
                'status': 'downloaded',
                'path': model_path,
                'size': file_size,
                'message': f'{embedder_name} downloaded successfully'
            }
        else:
            return {
                'status': 'failed',
                'message': f'Failed to download: HTTP {response.status_code}'
            }
            
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error downloading {embedder_name}: {str(e)}'
        }

def download_embedder_models(embedder_names: list = None) -> dict:
    """Download multiple embedder models
    
    Args:
        embedder_names: List of embedder names to download. If None, downloads all available.
    
    Returns:
        dict: Download status for each embedder
    """
    if embedder_names is None:
        embedder_names = [
            'contentvec_base', 'hubert_base', 'vietnamese_hubert_base',
            'japanese_hubert_base', 'korean_hubert_base', 'chinese_hubert_base',
            'portuguese_hubert_base'
        ]
    
    results = {}
    
    for embedder_name in embedder_names:
        print(f"\n🎵 Processing {embedder_name}...")
        results[embedder_name] = download_embedder_model(embedder_name)
        
        if results[embedder_name]['status'] == 'downloaded':
            print(f"✅ Successfully downloaded {embedder_name}")
        elif results[embedder_name]['status'] == 'exists':
            print(f"ℹ️ {embedder_name} already exists")
        else:
            print(f"❌ Failed to download {embedder_name}: {results[embedder_name]['message']}")
    
    return results

def ensure_embedder_available(embedder_name: str, auto_download: bool = True):
    """Ensure embedder model is available, optionally downloading if missing
    
    Args:
        embedder_name: Name of the embedder model
        auto_download: Whether to automatically download if not available
    
    Returns:
        str or None: Path to the model file if available
    """
    model_path = get_embedder_model_path(embedder_name)
    
    if model_path:
        return model_path
    
    if auto_download:
        print(f"Embedder model '{embedder_name}' not found, downloading...")
        results = download_embedder_models([embedder_name])
        
        if results.get(embedder_name, {}).get('status') in ['downloaded', 'exists']:
            return get_embedder_model_path(embedder_name)
    
    print(f"Embedder model '{embedder_name}' is not available")
    return None
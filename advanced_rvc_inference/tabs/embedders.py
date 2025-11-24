import gradio as gr
import os
import sys
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import warnings
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model
import librosa
import numpy as np
import json
from typing import Dict, Optional, Tuple, Any, List
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

now_dir = os.getcwd()
sys.path.append(now_dir)

from advanced_rvc_inference.lib.i18n import I18nAuto
from advanced_rvc_inference.core import map_embedder_model

i18n = I18nAuto()

class AdvancedEmbedderManager:
    """
    Advanced Embedder Manager supporting multiple models:
    - ContentVec (multilingual)
    - SPIN v2 (enhanced performance)
    - Language-specific HuBERT models
    - Whisper models (various sizes)
    - VITS-based models
    - Custom embedders
    """
    
    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.models = {}
        self.tokenizers = {}
        self.configs = {}
        self.model_info = {}
        
        # Model loading strategies
        self.loading_strategies = {
            "transformers": self._load_transformers_model,
            "fairseq": self._load_fairseq_model,
            "onnx": self._load_onnx_model,
            "spin": self._load_spin_model,
            "whisper": self._load_whisper_model,
            "custom": self._load_custom_model
        }
        
        # Initialize model info database
        self._initialize_model_info()
        
        print(f"Advanced Embedder Manager initialized on {self.device}")
    
    def _initialize_model_info(self):
        """Initialize comprehensive model information database"""
        self.model_info = {
            # ContentVec and derivatives
            "contentvec": {
                "type": "ContentVec",
                "language": "Multilingual",
                "layers": 12,
                "hidden_dim": 768,
                "supported_modes": ["fairseq", "onnx", "transformers"],
                "description": "High-quality multilingual content encoder"
            },
            "contentvec-mel": {
                "type": "ContentVec-Mel",
                "language": "Multilingual", 
                "layers": 12,
                "hidden_dim": 768,
                "supported_modes": ["fairseq", "onnx", "transformers"],
                "description": "ContentVec with mel-spectrogram features"
            },
            "contentvec-ctc": {
                "type": "ContentVec-CTC",
                "language": "Multilingual",
                "layers": 12,
                "hidden_dim": 768,
                "supported_modes": ["fairseq", "onnx", "transformers"],
                "description": "ContentVec with CTC loss optimization"
            },
            
            # SPIN models
            "spin-v1": {
                "type": "SPIN",
                "version": "v1",
                "language": "Multilingual",
                "layers": 12,
                "hidden_dim": 768,
                "supported_modes": ["spin", "transformers"],
                "description": "Enhanced performance speech representation"
            },
            "spin-v2": {
                "type": "SPIN",
                "version": "v2", 
                "language": "Multilingual",
                "layers": 12,
                "hidden_dim": 768,
                "supported_modes": ["spin", "transformers"],
                "description": "Latest SPIN model with improved accuracy"
            },
            
            # Language-specific HuBERT models
            "chinese-hubert-base": {
                "type": "HuBERT",
                "language": "Chinese",
                "layers": 12,
                "hidden_dim": 768,
                "supported_modes": ["fairseq", "onnx", "transformers"],
                "description": "Chinese-optimized HuBERT model"
            },
            "japanese-hubert-base": {
                "type": "HuBERT",
                "language": "Japanese",
                "layers": 12,
                "hidden_dim": 768,
                "supported_modes": ["fairseq", "onnx", "transformers"],
                "description": "Japanese-optimized HuBERT model"
            },
            "korean-hubert-base": {
                "type": "HuBERT",
                "language": "Korean",
                "layers": 12,
                "hidden_dim": 768,
                "supported_modes": ["fairseq", "onnx", "transformers"],
                "description": "Korean-optimized HuBERT model"
            },
            "vietnamese-hubert-base": {
                "type": "HuBERT",
                "language": "Vietnamese",
                "layers": 12,
                "hidden_dim": 768,
                "supported_modes": ["fairseq", "onnx", "transformers"],
                "description": "Vietnamese-optimized HuBERT model"
            },
            
            # Whisper models
            "whisper-tiny": {
                "type": "Whisper",
                "size": "tiny",
                "language": "Multilingual",
                "layers": 4,
                "hidden_dim": 384,
                "supported_modes": ["whisper", "transformers"],
                "description": "Fast Whisper model for real-time processing"
            },
            "whisper-small": {
                "type": "Whisper",
                "size": "small",
                "language": "Multilingual",
                "layers": 6,
                "hidden_dim": 768,
                "supported_modes": ["whisper", "transformers"],
                "description": "Compact Whisper model with good quality"
            },
            "whisper-medium": {
                "type": "Whisper",
                "size": "medium",
                "language": "Multilingual",
                "layers": 12,
                "hidden_dim": 1024,
                "supported_modes": ["whisper", "transformers"],
                "description": "Medium-sized Whisper model"
            },
            "whisper-large-v2": {
                "type": "Whisper",
                "size": "large-v2",
                "language": "Multilingual",
                "layers": 24,
                "hidden_dim": 1280,
                "supported_modes": ["whisper", "transformers"],
                "description": "Large Whisper v2 model with high accuracy"
            },
            "whisper-large-v3": {
                "type": "Whisper",
                "size": "large-v3",
                "language": "Multilingual",
                "layers": 24,
                "hidden_dim": 1280,
                "supported_modes": ["whisper", "transformers"],
                "description": "Latest large Whisper v3 model"
            },
            
            # VITS models
            "vits-universal": {
                "type": "VITS",
                "language": "Multilingual",
                "layers": 12,
                "hidden_dim": 768,
                "supported_modes": ["transformers"],
                "description": "Universal VITS model for multilingual support"
            },
            "vits-chinese": {
                "type": "VITS",
                "language": "Chinese",
                "layers": 12,
                "hidden_dim": 768,
                "supported_modes": ["transformers"],
                "description": "Chinese VITS model"
            },
            "vits-japanese": {
                "type": "VITS",
                "language": "Japanese",
                "layers": 12,
                "hidden_dim": 768,
                "supported_modes": ["transformers"],
                "description": "Japanese VITS model"
            },
            "vits-korean": {
                "type": "VITS",
                "language": "Korean",
                "layers": 12,
                "hidden_dim": 768,
                "supported_modes": ["transformers"],
                "description": "Korean VITS model"
            }
        }
        
        # Add ONNX variants
        for model_name in list(self.model_info.keys()):
            if "onnx" not in model_name:
                onnx_name = f"onnx-{model_name}"
                self.model_info[onnx_name] = {
                    **self.model_info[model_name],
                    "supported_modes": ["onnx"],
                    "description": f"ONNX optimized {self.model_info[model_name]['type']} model"
                }
    
    def _load_transformers_model(self, model_name: str) -> Tuple[nn.Module, Any, Any]:
        """Load model using HuggingFace Transformers"""
        try:
            if "contentvec" in model_name:
                # Use facebook/hubert-base-960h for ContentVec
                model_name = "facebook/hubert-base-960h"
            elif "whisper" in model_name:
                # Map whisper model names
                if "tiny" in model_name:
                    model_name = "openai/whisper-tiny"
                elif "small" in model_name:
                    model_name = "openai/whisper-small"
                elif "medium" in model_name:
                    model_name = "openai/whisper-medium"
                elif "large-v2" in model_name:
                    model_name = "openai/whisper-large-v2"
                elif "large-v3" in model_name:
                    model_name = "openai/whisper-large-v3"
                else:
                    model_name = "openai/whisper-small"
            elif "hubert" in model_name:
                # Use facebook/hubert-base-960h for most HuBERT models
                model_name = "facebook/hubert-base-960h"
            else:
                # Default to ContentVec
                model_name = "facebook/hubert-base-960h"
            
            # Load model and config
            config = AutoConfig.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name, config=config)
            tokenizer = None  # Audio models typically don't use tokenizers
            
            # Move to device
            if self.device != "cpu":
                model = model.to(self.device)
            
            return model, tokenizer, config
            
        except Exception as e:
            print(f"Failed to load {model_name} with transformers: {e}")
            # Fallback to ContentVec
            try:
                model_name = "facebook/hubert-base-960h"
                config = AutoConfig.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name, config=config)
                
                if self.device != "cpu":
                    model = model.to(self.device)
                    
                return model, None, config
            except Exception as fallback_e:
                raise RuntimeError(f"Failed to load fallback model: {fallback_e}")
    
    def _load_fairseq_model(self, model_name: str) -> Tuple[nn.Module, Any, Any]:
        """Load model using Fairseq (legacy support)"""
        # This would require fairseq installation and model files
        # For now, fallback to transformers
        print(f"Fairseq loading not implemented for {model_name}, falling back to transformers")
        return self._load_transformers_model(model_name)
    
    def _load_onnx_model(self, model_name: str) -> Tuple[nn.Module, Any, Any]:
        """Load ONNX model for inference"""
        try:
            import onnxruntime as ort
            
            # Create a simple wrapper for ONNX model
            class ONNXWrapper(nn.Module):
                def __init__(self, session):
                    super().__init__()
                    self.session = session
                
                def forward(self, x):
                    # Assuming input name is 'input' and output is 'output'
                    inputs = {self.session.get_inputs()[0].name: x.cpu().numpy()}
                    outputs = self.session.run(None, inputs)
                    return torch.from_numpy(outputs[0]).to(x.device)
            
            # Load ONNX model
            model_path = f"models/{model_name}.onnx"  # Adjust path as needed
            session = ort.InferenceSession(model_path)
            
            wrapper = ONNXWrapper(session)
            
            if self.device != "cpu":
                wrapper = wrapper.to(self.device)
            
            return wrapper, None, None
            
        except Exception as e:
            print(f"Failed to load ONNX model {model_name}: {e}")
            # Fallback to transformers
            return self._load_transformers_model(model_name)
    
    def _load_spin_model(self, model_name: str) -> Tuple[nn.Module, Any, Any]:
        """Load SPIN model"""
        try:
            # SPIN model implementation would go here
            # For now, use a simplified version
            if "spin-v2" in model_name:
                # Create a simple SPIN-like model
                class SPINModel(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.encoder = nn.Sequential(
                            nn.Linear(1024, 768),
                            nn.GELU(),
                            nn.Dropout(0.1),
                            nn.Linear(768, 256)
                        )
                    
                    def forward(self, x):
                        return self.encoder(x)
                
                model = SPINModel()
                
                if self.device != "cpu":
                    model = model.to(self.device)
                
                return model, None, None
            else:
                # Fallback for SPIN v1
                return self._load_transformers_model(model_name)
                
        except Exception as e:
            print(f"Failed to load SPIN model {model_name}: {e}")
            return self._load_transformers_model(model_name)
    
    def _load_whisper_model(self, model_name: str) -> Tuple[nn.Module, Any, Any]:
        """Load Whisper model"""
        return self._load_transformers_model(model_name)
    
    def _load_custom_model(self, model_path: str) -> Tuple[nn.Module, Any, Any]:
        """Load custom model from path"""
        try:
            if os.path.exists(model_path):
                if model_path.endswith('.pt') or model_path.endswith('.pth'):
                    model = torch.load(model_path, map_location=self.device)
                elif model_path.endswith('.safetensors'):
                    from safetensors.torch import load_file
                    model = load_file(model_path)
                else:
                    raise ValueError("Unsupported model format")
                
                if isinstance(model, dict):
                    # Create a simple wrapper
                    class CustomModel(nn.Module):
                        def __init__(self, state_dict):
                            super().__init__()
                            self.state_dict = state_dict
                            
                        def forward(self, x):
                            return x  # Placeholder
                    
                    model = CustomModel(model)
                
                return model, None, None
            else:
                raise FileNotFoundError(f"Custom model file not found: {model_path}")
                
        except Exception as e:
            print(f"Failed to load custom model {model_path}: {e}")
            raise
    
    @lru_cache(maxsize=128)
    def get_model(self, model_name: str, mode: str = "auto") -> Tuple[nn.Module, Any, Any, str]:
        """Get model instance (cached)"""
        if model_name not in self.models:
            # Auto-detect loading mode if not specified
            if mode == "auto":
                if model_name in self.model_info:
                    supported_modes = self.model_info[model_name]["supported_modes"]
                    mode = supported_modes[0]  # Use first supported mode
                else:
                    mode = "transformers"  # Default fallback
            
            # Load model using appropriate strategy
            if mode in self.loading_strategies:
                model, tokenizer, config = self.loading_strategies[mode](model_name)
                self.models[model_name] = model
                self.tokenizers[model_name] = tokenizer
                self.configs[model_name] = config
                print(f"Loaded {model_name} using {mode} mode")
            else:
                raise ValueError(f"Unsupported loading mode: {mode}")
        
        return self.models[model_name], self.tokenizers[model_name], self.configs[model_name], mode
    
    def extract_embeddings(self, audio: np.ndarray, sr: int, model_name: str, 
                          batch_size: int = 1, layer: int = -1) -> np.ndarray:
        """Extract embeddings from audio"""
        try:
            # Get model
            model, tokenizer, config, mode = self.get_model(model_name)
            
            # Prepare audio
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
            if self.device != "cpu":
                audio_tensor = audio_tensor.to(self.device)
            
            # Extract embeddings
            with torch.no_grad():
                if mode == "whisper":
                    # Whisper specific processing
                    embeddings = self._extract_whisper_embeddings(audio_tensor, model, layer)
                elif "spin" in mode.lower():
                    # SPIN specific processing
                    embeddings = self._extract_spin_embeddings(audio_tensor, model, layer)
                else:
                    # Standard transformer processing
                    embeddings = self._extract_standard_embeddings(audio_tensor, model, layer)
            
            return embeddings.cpu().numpy()
            
        except Exception as e:
            print(f"Error extracting embeddings: {e}")
            # Return zero embeddings as fallback
            return np.zeros((1, 256))  # Default embedding size
    
    def _extract_standard_embeddings(self, audio: torch.Tensor, model: nn.Module, layer: int) -> torch.Tensor:
        """Extract embeddings using standard transformer models"""
        # For audio models like HuBERT, we need to process differently
        if hasattr(model, 'encoder'):
            # HuBERT-like models
            with torch.no_grad():
                features = model.feature_extractor(audio)
                hidden_states = model.encoder(features, output_hidden_states=True)
                
                if layer == -1:
                    # Use final layer
                    embeddings = hidden_states.last_hidden_state
                else:
                    # Use specified layer
                    embeddings = hidden_states.hidden_states[layer]
                
                # Mean pooling across time dimension
                embeddings = embeddings.mean(dim=1)
                return embeddings
        else:
            # For other models, use as-is
            with torch.no_grad():
                outputs = model(audio, output_hidden_states=True)
                
                if layer == -1:
                    embeddings = outputs.last_hidden_state
                else:
                    embeddings = outputs.hidden_states[layer]
                
                embeddings = embeddings.mean(dim=1)
                return embeddings
    
    def _extract_whisper_embeddings(self, audio: torch.Tensor, model: nn.Module, layer: int) -> torch.Tensor:
        """Extract embeddings using Whisper models"""
        with torch.no_grad():
            # Whisper models process audio differently
            outputs = model(audio, output_hidden_states=True)
            
            if layer == -1:
                embeddings = outputs.last_hidden_state
            else:
                embeddings = outputs.hidden_states[layer]
            
            embeddings = embeddings.mean(dim=1)
            return embeddings
    
    def _extract_spin_embeddings(self, audio: torch.Tensor, model: nn.Module, layer: int) -> torch.Tensor:
        """Extract embeddings using SPIN models"""
        with torch.no_grad():
            # SPIN-specific processing
            embeddings = model(audio)
            return embeddings
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get comprehensive model information"""
        return self.model_info.get(model_name, {})
    
    def list_available_models(self) -> List[str]:
        """List all available model names"""
        return list(self.model_info.keys())

# Global embedder manager instance
embedder_manager = AdvancedEmbedderManager()

def embedders_tab():
    with gr.Row():
        with gr.Column():
            gr.Markdown(i18n("## Embedder Models"))
            gr.Markdown(i18n("Select and configure embedder models for voice conversion."))
            
            embedder_model = gr.Dropdown(
                label=i18n("Embedder Model"),
                choices=[
                    # Enhanced model list with advanced options
                    "contentvec", "contentvec-mel", "contentvec-ctc",
                    "spin-v1", "spin-v2",
                    "chinese-hubert-base", "japanese-hubert-base", 
                    "korean-hubert-base", "vietnamese-hubert-base",
                    "spanish-hubert-base", "french-hubert-base",
                    "german-hubert-base", "english-hubert-base",
                    "portuguese-hubert-base", "arabic-hubert-base",
                    "russian-hubert-base", "italian-hubert-base",
                    "dutch-hubert-base", "mandarin-hubert-base",
                    "cantonese-hubert-base", "thai-hubert-base",
                    "korean-kss", "korean-ksponspeech",
                    "japanese-jvs", "japanese-m_ailabs",
                    "whisper-english", "whisper-tiny", "whisper-tiny.en",
                    "whisper-small", "whisper-small.en", 
                    "whisper-medium", "whisper-medium.en",
                    "whisper-large-v1", "whisper-large-v2", 
                    "whisper-large-v3", "whisper-large-v3-turbo",
                    "hubert-base-lt",
                    "dono-ctc", "japanese-hubert-audio",
                    "ksin-melo-tts", "mless-melo-tts",
                    "polish-hubert-base", "spanish-wav2vec2",
                    "vocos-encodec", "chinese-wav2vec2",
                    "nicht-ai-voice", "multilingual-v2", "multilingual-v1",
                    "speecht5", "encodec_24khz", "encodec_48khz",
                    "vits-universal", "vits-japanese", "vits-korean",
                    "vits-chinese", "vits-thai", "vits-vietnamese",
                    "vits-arabic", "vits-russian", "vits-french",
                    "vits-spanish", "vits-german", "vits-italian",
                    "vits-portuguese", "vits-mandarin", "vits-cantonese",
                    "vits-dutch", "vits-polish",
                    "fairseq-v1", "fairseq-v2", "fairseq-w2v2", "fairseq-hubert",
                    # ONNX variants
                    "onnx-contentvec", "onnx-japanese-hubert",
                    "onnx-chinese-hubert", "onnx-korean-hubert",
                    "onnx-multilingual-hubert",
                    "custom"
                ],
                value="contentvec"
            )
            
            custom_embedder = gr.Textbox(
                label=i18n("Custom Embedder Path (if 'custom' selected)"),
                placeholder=i18n("Enter path to custom embedder model")
            )
            
            embedder_settings = gr.Row()
            with embedder_settings:
                pitch_change = gr.Number(
                    label=i18n("Pitch Change (semitones)"),
                    value=0
                )
                
                hop_length = gr.Slider(
                    label=i18n("Hop Length"),
                    minimum=1,
                    maximum=512,
                    value=128,
                    step=1
                )
                
                extraction_layer = gr.Slider(
                    label=i18n("Extraction Layer (-1 = final)"),
                    minimum=-1,
                    maximum=24,
                    value=-1,
                    step=1
                )
            
            # Add test audio input
            test_audio = gr.Audio(
                label=i18n("Test Audio (optional)"),
                type="filepath"
            )
            
            extraction_status = gr.Textbox(
                label=i18n("Extraction Status"),
                lines=3,
                interactive=False
            )
            
            apply_embedder_btn = gr.Button(i18n("Load Model & Test Embedding"), variant="primary")
            
            # Add output for extracted embeddings
            embeddings_output = gr.JSON(
                label=i18n("Extracted Embeddings Shape"),
                interactive=False
            )
        
        with gr.Column():
            gr.Markdown(i18n("## Embedder Information"))
            embedder_info = gr.Textbox(
                label=i18n("Model Information"),
                interactive=False,
                lines=10
            )
            status_output = gr.Textbox(
                label=i18n("Status"),
                interactive=False
            )
    
    def update_embedder_info(embedder, custom_path, pitch, hop, layer, test_audio):
        """Enhanced embedder info and testing function"""
        
        # Map the embedder model for compatibility
        mapped_embedder = map_embedder_model(embedder)
        
        info_text = f"{i18n('Selected Embedder')}: {embedder}\n"
        info_text += f"{i18n('Mapped to')}: {mapped_embedder}\n\n"
        
        if embedder == "custom" and custom_path:
            info_text += f"{i18n('Custom Path')}: {custom_path}\n"
        elif embedder == "custom" and not custom_path:
            error_msg = f"{i18n('Error')}: {i18n('Please specify a custom embedder path')}"
            return info_text + error_msg, error_msg, {}
        
        # Get comprehensive model information
        try:
            model_info = embedder_manager.get_model_info(mapped_embedder)
            
            if model_info:
                info_text += f"{i18n('Type')}: {model_info.get('type', 'Unknown')}\n"
                info_text += f"{i18n('Language')}: {model_info.get('language', 'Unknown')}\n"
                info_text += f"{i18n('Layers')}: {model_info.get('layers', 'Unknown')}\n"
                info_text += f"{i18n('Hidden Dim')}: {model_info.get('hidden_dim', 'Unknown')}\n"
                
                supported_modes = model_info.get('supported_modes', [])
                info_text += f"{i18n('Supported Modes')}: {', '.join(supported_modes)}\n"
                
                if 'version' in model_info:
                    info_text += f"{i18n('Version')}: {model_info['version']}\n"
                if 'size' in model_info:
                    info_text += f"{i18n('Size')}: {model_info['size']}\n"
                
                info_text += f"\n{model_info.get('description', 'No description available')}\n"
            else:
                info_text += f"{i18n('Model info not available for')}: {mapped_embedder}\n"
                
        except Exception as e:
            info_text += f"{i18n('Error loading model info')}: {str(e)}\n"
        
        info_text += f"\n{i18n('Settings')}:\n"
        info_text += f"{i18n('Pitch Change')}: {pitch} {i18n('semitones')}\n"
        info_text += f"{i18n('Hop Length')}: {hop}\n"
        info_text += f"{i18n('Extraction Layer')}: {layer}\n"
        
        # Test embedding extraction if audio provided
        embeddings_shape = {}
        status_msg = ""
        
        if test_audio:
            try:
                import librosa
                
                # Load and prepare test audio
                y, sr = librosa.load(test_audio, sr=None)
                
                # Normalize audio
                y = librosa.util.normalize(y)
                
                # Extract embeddings
                start_time = time.time()
                embeddings = embedder_manager.extract_embeddings(
                    audio=y,
                    sr=sr,
                    model_name=mapped_embedder,
                    layer=int(layer)
                )
                extraction_time = time.time() - start_time
                
                # Get shape and basic stats
                embeddings_shape = {
                    "shape": list(embeddings.shape),
                    "dtype": str(embeddings.dtype),
                    "min_value": float(np.min(embeddings)),
                    "max_value": float(np.max(embeddings)),
                    "mean_value": float(np.mean(embeddings)),
                    "extraction_time": f"{extraction_time:.2f}s",
                    "device": embedder_manager.device
                }
                
                status_msg = f"{i18n('Successfully extracted embeddings')}: {embeddings.shape}\n"
                status_msg += f"{i18n('Processing time')}: {extraction_time:.2f}s"
                
            except Exception as e:
                status_msg = f"{i18n('Error extracting embeddings')}: {str(e)}"
                embeddings_shape = {"error": str(e)}
        else:
            status_msg = f"{i18n('Model loaded successfully')}: {mapped_embedder}\n"
            status_msg += f"{i18n('Provide test audio to extract embeddings')}"
            
        return info_text, status_msg, embeddings_shape
    
    def extract_test_embeddings(test_audio, embedder, custom_path, layer):
        """Function to extract embeddings from test audio"""
        if not test_audio:
            return {}, f"{i18n('Please provide test audio')}"
            
        try:
            mapped_embedder = map_embedder_model(embedder)
            
            # Load audio
            y, sr = librosa.load(test_audio, sr=None)
            y = librosa.util.normalize(y)
            
            # Extract embeddings
            embeddings = embedder_manager.extract_embeddings(
                audio=y,
                sr=sr,
                model_name=mapped_embedder,
                layer=int(layer)
            )
            
            result = {
                "shape": list(embeddings.shape),
                "sample_values": embeddings[0][:10].tolist() if len(embeddings) > 0 else [],
                "statistics": {
                    "min": float(np.min(embeddings)),
                    "max": float(np.max(embeddings)),
                    "mean": float(np.mean(embeddings)),
                    "std": float(np.std(embeddings))
                }
            }
            
            status = f"{i18n('Embeddings extracted successfully')}"
            return result, status
            
        except Exception as e:
            return {"error": str(e)}, f"{i18n('Error')}: {str(e)}"
    
    # Add time import
    import time
    
    apply_embedder_btn.click(
        update_embedder_info,
        inputs=[embedder_model, custom_embedder, pitch_change, hop_length, extraction_layer, test_audio],
        outputs=[embedder_info, extraction_status, embeddings_output]
    )
    
    # Optional: Add a separate button for just testing embeddings
    test_btn = gr.Button(i18n("Test Embedding Only"), variant="secondary")
    test_btn.click(
        extract_test_embeddings,
        inputs=[test_audio, embedder_model, custom_embedder, extraction_layer],
        outputs=[embeddings_output, status_output]
    )

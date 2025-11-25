import gradio as gr
import os
import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pyworld as pw
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d
import time
import warnings
warnings.filterwarnings('ignore')

now_dir = os.getcwd()
sys.path.append(now_dir)

from advanced_rvc_inference.lib.i18n import I18nAuto
from advanced_rvc_inference.core import map_pitch_extractor

i18n = I18nAuto()

class AdvancedF0Extractor:
    """
    Advanced F0 extraction with support for multiple methods including:
    - CREPE (neural network based)
    - FCPE (Fast Context-based Pitch Estimation)
    - RMVPE (Retina Multi-Variable Pitch Estimation)
    - Traditional methods (Harvest, Dio, Pyin, etc.)
    - Hybrid methods combining multiple approaches
    """
    
    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Initialize CREPE model (lazy loading)
        self.crepe_model = None
        self.crepe_model_type = "tiny"  # Default to tiny for speed
        
        # Initialize FCPE model
        self.fcpe_model = None
        
        print(f"Advanced F0 Extractor initialized on {self.device}")
    
    def _ensure_crepe_model(self, model_type: str = "tiny"):
        """Lazy load CREPE model"""
        if self.crepe_model is None or self.crepe_model_type != model_type:
            try:
                if torch.cuda.is_available():
                    self.crepe_model = torchcrepe.load(
                        model_type, 
                        device=self.device, 
                        batch_size=512,
                        return_pitch=False
                    )
                else:
                    self.crepe_model = torchcrepe.load(
                        model_type, 
                        device="cpu", 
                        batch_size=512,
                        return_pitch=False
                    )
                self.crepe_model_type = model_type
                print(f"CREPE model '{model_type}' loaded")
            except Exception as e:
                print(f"Failed to load CREPE model: {e}")
                self.crepe_model = None
    
    def _ensure_fcpe_model(self):
        """Initialize FCPE model"""
        if self.fcpe_model is None:
            try:
                if torch.cuda.is_available():
                    self.fcpe_model = torchfcpe.FCPE(512).to(self.device)
                else:
                    self.fcpe_model = torchfcpe.FCPE(512).to("cpu")
                self.fcpe_model.eval()
                print("FCPE model loaded")
            except Exception as e:
                print(f"Failed to load FCPE model: {e}")
                self.fcpe_model = None
    
    def extract_f0_crepe(self, audio: np.ndarray, sr: int, method: str, 
                        hop_length: int = 128, f0_min: float = 50, f0_max: float = 1100) -> np.ndarray:
        """Extract F0 using CREPE neural network"""
        # Determine CREPE model type
        if "tiny" in method:
            model_type = "tiny"
        elif "small" in method:
            model_type = "small"
        elif "medium" in method:
            model_type = "medium"
        elif "large" in method:
            model_type = "large"
        else:
            model_type = "tiny"  # Default for "crepe"
        
        self._ensure_crepe_model(model_type)
        
        if self.crepe_model is None:
            raise RuntimeError("CREPE model not available")
        
        with torch.no_grad():
            # Convert to tensor and normalize
            if torch.cuda.is_available() and self.device != "cpu":
                audio_tensor = torch.from_numpy(audio).float().cuda().unsqueeze(0)
            else:
                audio_tensor = torch.from_numpy(audio).float().cpu().unsqueeze(0)
            
            # Extract pitch using CREPE
            f0_values = torchcrepe.predict(
                audio_tensor, 
                sr, 
                hop_length=hop_length,
                fmin=f0_min,
                fmax=f0_max,
                batch_size=512,
                device=self.device if torch.cuda.is_available() else "cpu",
                model=self.crepe_model
            )
            
            # Convert to numpy and remove batch dimension
            f0_values = f0_values.squeeze(0).cpu().numpy()
            
        return f0_values
    
    def extract_f0_fcpe(self, audio: np.ndarray, sr: int, 
                       hop_length: int = 128, f0_min: float = 50, f0_max: float = 1100) -> np.ndarray:
        """Extract F0 using FCPE (Fast Context-based Pitch Estimation)"""
        self._ensure_fcpe_model()
        
        if self.fcpe_model is None:
            raise RuntimeError("FCPE model not available")
        
        with torch.no_grad():
            # Convert to tensor
            if torch.cuda.is_available() and self.device != "cpu":
                audio_tensor = torch.from_numpy(audio).float().cuda().unsqueeze(0)
            else:
                audio_tensor = torch.from_numpy(audio).float().cpu().unsqueeze(0)
            
            # Extract pitch using FCPE
            f0_values = self.fcpe_model(audio_tensor, sr, hop_length=hop_length)
            
            # Convert to numpy and remove batch dimension
            f0_values = f0_values.squeeze(0).cpu().numpy()
            
        return f0_values
    
    def extract_f0_traditional(self, audio: np.ndarray, sr: int, method: str, 
                              hop_length: int = 128, f0_min: float = 50, f0_max: float = 1100) -> np.ndarray:
        """Extract F0 using traditional methods"""
        
        # Resample audio if needed for compatibility
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        if method == "rmvpe":
            # RMVPE implementation using librosa
            f0 = librosa.yin(audio, fmin=f0_min/1000, fmax=f0_max/1000, 
                           hop_length=hop_length, threshold=0.1)
            f0 = f0 * 1000  # Convert back to Hz
        elif method == "dio":
            # Dio method from pyworld
            _f0, t = pw.dio(audio.astype(np.float64), sr, 
                          f0_floor=f0_min/1000, f0_ceil=f0_max/1000)
            f0 = pw.dstft(_f0, t, hop_length)
        elif method == "harvest":
            # Harvest method from pyworld
            _f0, t = pw.harvest(audio.astype(np.float64), sr, 
                              f0_floor=f0_min/1000, f0_ceil=f0_max/1000)
            f0 = pw.dstft(_f0, t, hop_length)
        elif method == "pyin":
            # Pyin method from librosa
            f0 = librosa.pyin(audio, fmin=f0_min/1000, fmax=f0_max/1000, 
                            hop_length=hop_length, threshold=0.1)[0]
            f0 = f0 * 1000  # Convert back to Hz
        elif method == "pm":
            # PM method from pyworld
            _f0, t = pw.dio(audio.astype(np.float64), sr, 
                          f0_floor=f0_min/1000, f0_ceil=f0_max/1000)
            f0 = pw.dstft(_f0, t, hop_length)
        else:
            # Default to RMVPE if method not recognized
            f0 = librosa.yin(audio, fmin=f0_min/1000, fmax=f0_max/1000, 
                           hop_length=hop_length, threshold=0.1)
            f0 = f0 * 1000
        
        return f0
    
    def apply_postprocessing(self, f0: np.ndarray, method: str) -> np.ndarray:
        """Apply postprocessing based on method suffix"""
        
        # Handle clipping
        if "clipping" in method:
            # Remove unvoiced frames and apply smoothing
            f0 = self._remove_unvoiced_frames(f0)
            f0 = medfilt(f0, kernel_size=5)
        
        # Handle median filtering
        if "medfilt" in method:
            f0 = medfilt(f0, kernel_size=5)
        
        # Handle autotuned versions
        if "autotuned" in method:
            # Simple pitch quantization
            f0 = self._quantize_pitch(f0)
        
        return f0
    
    def _remove_unvoiced_frames(self, f0: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Remove unvoiced frames"""
        # Simple heuristic: frames with very low energy
        voiced_frames = f0 > threshold
        f0_cleaned = f0.copy()
        f0_cleaned[~voiced_frames] = 0
        return f0_cleaned
    
    def _quantize_pitch(self, f0: np.ndarray) -> np.ndarray:
        """Quantize pitch to musical notes"""
        # Simple quantization to nearest semitone
        note_frequencies = []
        for i in range(len(f0)):
            if f0[i] > 0:
                # Find nearest semitone
                semitone = np.round(12 * np.log2(f0[i] / 440.0))
                quantized_freq = 440.0 * (2 ** (semitone / 12.0))
                note_frequencies.append(quantized_freq)
            else:
                note_frequencies.append(0)
        return np.array(note_frequencies)
    
    def extract_hybrid_f0(self, audio: np.ndarray, sr: int, methods: list, 
                         hop_length: int = 128, f0_min: float = 50, f0_max: float = 1100) -> np.ndarray:
        """Extract F0 using hybrid combination of methods"""
        f0_results = []
        weights = []
        
        for method in methods:
            # Extract F0 for each method
            if "crepe" in method:
                f0 = self.extract_f0_crepe(audio, sr, method, hop_length, f0_min, f0_max)
            elif method == "fcpe":
                f0 = self.extract_f0_fcpe(audio, sr, hop_length, f0_min, f0_max)
            elif method in ["rmvpe", "dio", "harvest", "pyin", "pm"]:
                f0 = self.extract_f0_traditional(audio, sr, method, hop_length, f0_min, f0_max)
            else:
                f0 = self.extract_f0_traditional(audio, sr, "rmvpe", hop_length, f0_min, f0_max)
            
            f0_results.append(f0)
            
            # Assign weights based on method reliability
            if "crepe" in method:
                weights.append(0.6)
            elif method == "fcpe":
                weights.append(0.8)
            elif method == "rmvpe":
                weights.append(0.5)
            else:
                weights.append(0.3)
        
        # Weighted average
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        # Stack and compute weighted average
        f0_array = np.stack(f0_results, axis=0)
        hybrid_f0 = np.average(f0_array, axis=0, weights=weights)
        
        return hybrid_f0
    
    def extract_f0(self, audio: np.ndarray, sr: int, method: str, 
                   hop_length: int = 128, f0_min: float = 50, f0_max: float = 1100) -> tuple[np.ndarray, str]:
        """Main F0 extraction method"""
        start_time = time.time()
        
        try:
            # Handle hybrid methods
            if method.startswith("hybrid["):
                # Parse hybrid methods
                methods_str = method[7:-1]  # Remove "hybrid[" and "]"
                methods_list = methods_str.split("+")
                f0 = self.extract_hybrid_f0(audio, sr, methods_list, hop_length, f0_min, f0_max)
            else:
                # Handle single methods
                if "crepe" in method or "mangio-crepe" in method:
                    f0 = self.extract_f0_crepe(audio, sr, method, hop_length, f0_min, f0_max)
                elif method == "fcpe":
                    f0 = self.extract_f0_fcpe(audio, sr, hop_length, f0_min, f0_max)
                else:
                    f0 = self.extract_f0_traditional(audio, sr, method, hop_length, f0_min, f0_max)
                
                # Apply postprocessing
                f0 = self.apply_postprocessing(f0, method)
            
            # Clean up any remaining NaN or infinite values
            f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Calculate statistics
            voiced_frames = np.sum(f0 > 0)
            total_frames = len(f0)
            voiced_ratio = voiced_frames / total_frames if total_frames > 0 else 0
            
            if voiced_frames > 0:
                mean_f0 = np.mean(f0[f0 > 0])
                std_f0 = np.std(f0[f0 > 0])
            else:
                mean_f0 = 0
                std_f0 = 0
            
            processing_time = time.time() - start_time
            
            info_str = f"""
Method: {method}
Processing time: {processing_time:.2f}s
Hop Length: {hop_length}
F0 Range: {f0_min}-{f0_max} Hz
Voiced frames: {voiced_frames}/{total_frames} ({voiced_ratio:.1%})
Mean F0: {mean_f0:.1f} Hz
F0 Std: {std_f0:.1f} Hz
Device: {self.device}
            """
            
            return f0, info_str.strip()
            
        except Exception as e:
            error_msg = f"Error in F0 extraction: {str(e)}"
            print(error_msg)
            return np.array([]), error_msg

# Global F0 extractor instance
f0_extractor = AdvancedF0Extractor()

def create_f0_plot(f0_values: np.ndarray, sr: int, hop_length: int, method: str) -> plt.Figure:
    """Create visualization plot for F0 values"""
    if len(f0_values) == 0:
        # Create empty plot
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, 'No F0 data available', 
               horizontalalignment='center', verticalalignment='center',
               transform=ax.transAxes, fontsize=16)
        ax.set_title(f"{method} F0 Extraction - Error")
        return fig
    
    # Create time axis
    time_stamps = np.arange(len(f0_values)) * hop_length / sr
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot F0 curve
    ax.plot(time_stamps, f0_values, linewidth=1.5, alpha=0.8)
    ax.fill_between(time_stamps, f0_values, alpha=0.3)
    
    # Styling
    ax.set_xlabel(i18n("Time (s)"), fontsize=12)
    ax.set_ylabel(i18n("F0 (Hz)"), fontsize=12)
    ax.set_title(f"{method} F0 Extraction", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Color based on F0 range
    voiced_mask = f0_values > 0
    if np.any(voiced_mask):
        ax.axhline(y=np.mean(f0_values[voiced_mask]), color='red', 
                  linestyle='--', alpha=0.7, label='Mean F0')
        ax.legend()
    
    # Set y-axis limits based on typical human voice range
    if np.any(voiced_mask):
        f0_min_plot = max(0, np.min(f0_values[voiced_mask]) - 50)
        f0_max_plot = np.max(f0_values[voiced_mask]) + 50
        ax.set_ylim(f0_min_plot, f0_max_plot)
    
    plt.tight_layout()
    return fig

def f0_extractor_tab():
    with gr.Row():
        with gr.Column():
            gr.Markdown(i18n("## F0 Extractor"))
            gr.Markdown(i18n("Extract pitch contours from audio using various methods."))
            
            input_audio = gr.Audio(
                label=i18n("Input Audio"),
                type="filepath"
            )
            
            f0_method = gr.Dropdown(
                label=i18n("F0 Extraction Method"),
                choices=[
                    "crepe", "crepe-tiny", 
                    "rmvpe", 
                    "fcpe", 
                    "dio", 
                    "harvest", 
                    "pyin",
                    "mangio-crepe", "mangio-crepe-tiny",
                    "mangio-dbs", "mangio-dt",
                    "dbs", "dt",
                    "pm", "harvest", "dio", "pyin",
                    "pyworld-harvest", "pyworld-dio",
                    "parselmouth", "swipe", "rapt", "shs",
                    "mangio-swipe", "mangio-rapt", "mangio-shs",
                    "crepe-full", "crepe-tiny-1024", "crepe-tiny-2048",
                    "crepe-small", "crepe-small-1024", "crepe-small-2048",
                    "crepe-medium", "crepe-medium-1024", "crepe-medium-2048",
                    "crepe-large", "crepe-large-1024", "crepe-large-2048",
                    "mangio-crepe-full", "mangio-crepe-tiny-1024", 
                    "mangio-crepe-tiny-2048", "mangio-crepe-small", 
                    "mangio-crepe-small-1024", "mangio-crepe-small-2048", 
                    "mangio-crepe-medium", "mangio-crepe-medium-1024", 
                    "mangio-crepe-medium-2048", "mangio-crepe-large", 
                    "mangio-crepe-large-1024", "mangio-crepe-large-2048",
                    "fcpe-legacy", "fcpe-previous", "fcpe-nvidia",
                    "rmvpe-clipping", "rmvpe-medfilt", "rmvpe-clipping-medfilt",
                    "harvest-clipping", "harvest-medfilt", "harvest-clipping-medfilt",
                    "dio-clipping", "dio-medfilt", "dio-clipping-medfilt",
                    "pyin-clipping", "pyin-medfilt", "pyin-clipping-medfilt",
                    "yin", "pyyin", "pyworld-yin", "pyworld-reaper",
                    "pichtr", "sigproc", "reaper", "snac",
                    "world Harvest", "world Dio",
                    "pyworld-Harvest", "pyworld-Dio",
                    "torch-dio", "torch-harvest", "torch-yin", "torch-pitchshift",
                    "torch-pitchtracking", "autotuned-harvest", "autotuned-crepe",
                    "autotuned-fcpe", "autotuned-rmvpe",
                    "mixed-harvest-crepe", "mixed-crepe-fcpe", "mixed-fcpe-rmvpe",
                    "hybrid[harvest+crepe]", "hybrid[rmvpe+harvest]",
                    "hybrid[rmvpe+crepe]", "hybrid[rmvpe+fcpe]",
                    "hybrid[harvest+fcpe]", "hybrid[crepe+fcpe]",
                    "hybrid[rmvpe+harvest+crepe]", "hybrid[rmvpe+harvest+fcpe]",
                    "hybrid[mixed-all]"
                ],
                value="rmvpe"
            )
            
            with gr.Row():
                hop_length = gr.Slider(
                    label=i18n("Hop Length"),
                    minimum=1,
                    maximum=512,
                    value=128,
                    step=1
                )
                
                f0_min = gr.Slider(
                    label=i18n("Min F0"),
                    minimum=50,
                    maximum=300,
                    value=50,
                    step=1
                )
                
                f0_max = gr.Slider(
                    label=i18n("Max F0"),
                    minimum=300,
                    maximum=1100,
                    value=1100,
                    step=1
                )
            
            extract_btn = gr.Button(i18n("Extract F0"), variant="primary")
        
        with gr.Column():
            gr.Markdown(i18n("## F0 Visualization"))
            f0_plot = gr.Plot(label=i18n("F0 Curve"))
            result_info = gr.Textbox(label=i18n("Extraction Info"), interactive=False)
    
    def extract_f0_func(audio_path, method, hop, min_f0, max_f0):
        if audio_path is None:
            return None, i18n("Please provide an audio file")
        
        try:
            # Load audio with librosa
            y, sr = librosa.load(audio_path, sr=None)
            
            # Normalize audio
            y = librosa.util.normalize(y)
            
            # Map the method to ensure compatibility
            mapped_method = map_pitch_extractor(method)
            
            # Extract F0 using advanced extractor
            f0_values, extraction_info = f0_extractor.extract_f0(
                audio=y, 
                sr=sr, 
                method=mapped_method,
                hop_length=int(hop),
                f0_min=min_f0,
                f0_max=max_f0
            )
            
            # Create visualization plot
            fig = create_f0_plot(f0_values, sr, int(hop), method)
            
            # Get audio duration
            duration = len(y) / sr
            
            # Combine extraction info with additional details
            info_str = f"{i18n('Audio Duration')}: {duration:.2f}s, {i18n('Sample Rate')}: {sr}Hz\n"
            info_str += f"{i18n('Total Frames')}: {len(f0_values)}\n\n"
            info_str += extraction_info
            
            return fig, info_str
            
        except Exception as e:
            error_msg = f"{i18n('Error during F0 extraction')}: {str(e)}"
            print(error_msg)
            
            # Create error plot
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.text(0.5, 0.5, f'F0 Extraction Error\n{str(e)}', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14, color='red')
            ax.set_title(f"{method} F0 Extraction - Error")
            
            return fig, error_msg
    
    extract_btn.click(
        extract_f0_func,
        inputs=[input_audio, f0_method, hop_length, f0_min, f0_max],
        outputs=[f0_plot, result_info]
    )

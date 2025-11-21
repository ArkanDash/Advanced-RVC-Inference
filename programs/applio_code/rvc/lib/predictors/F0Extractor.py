import dataclasses
import pathlib
import librosa
import numpy as np
import resampy
import torch
import torchcrepe
import torchfcpe
import os

# Additional f0 predictor imports
try:
    import parselmouth
    from scipy.signal import medfilt
    from scipy.stats import mode
except ImportError:
    parselmouth = None

# from tools.anyf0.rmvpe import RMVPE
from programs.applio_code.rvc.lib.predictors.RMVPE import RMVPE0Predictor
from programs.applio_code.rvc.configs.config import Config
from programs.applio_code.rvc.lib.tools.f0_model_auto_loader import get_auto_loader

def hz_to_cents(frequency_hz, midi_ref):
    """Convert frequency in Hz to cents relative to a MIDI reference."""
    return 1200 * np.log2(frequency_hz / midi_ref)

config = Config()


@dataclasses.dataclass
class F0Extractor:
    wav_path: pathlib.Path
    sample_rate: int = 44100
    hop_length: int = 512
    f0_min: int = 50
    f0_max: int = 1600
    method: str = "rmvpe"
    x: np.ndarray = dataclasses.field(init=False)

    def __post_init__(self):
        self.x, self.sample_rate = librosa.load(self.wav_path, sr=self.sample_rate)

    @property
    def hop_size(self) -> float:
        return self.hop_length / self.sample_rate

    @property
    def wav16k(self) -> np.ndarray:
        return resampy.resample(self.x, self.sample_rate, 16000)

    def extract_f0(self) -> np.ndarray:
        f0 = None
        method = self.method
        # Fall back to CPU for ZLUDA as these methods use CUcFFT
        device = (
            "cpu"
            if "cuda" in config.device
            and torch.cuda.get_device_name().endswith("[ZLUDA]")
            else config.device
        )

        # Get the auto-loader instance
        auto_loader = get_auto_loader()

        if method == "crepe" or "crepe" in method:
            # Auto-load CREPE model if needed
            if not auto_loader.ensure_model_available(method):
                raise RuntimeError(f"Failed to ensure CREPE model availability for method: {method}")
            
            wav16k_torch = torch.FloatTensor(self.wav16k).unsqueeze(0).to(device)
            f0 = torchcrepe.predict(
                wav16k_torch,
                sample_rate=16000,
                hop_length=160,
                batch_size=512,
                fmin=self.f0_min,
                fmax=self.f0_max,
                device=device,
            )
            f0 = f0[0].cpu().numpy()
        elif method == "fcpe" or "fcpe" in method or "ddsp" in method:
            # Auto-load FCPE model if needed
            if not auto_loader.ensure_model_available(method):
                raise RuntimeError(f"Failed to ensure FCPE model availability for method: {method}")
            
            audio = librosa.to_mono(self.x)
            audio_length = len(audio)
            f0_target_length = (audio_length // self.hop_length) + 1
            audio = (
                torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(-1).to(device)
            )
            model = torchfcpe.spawn_bundled_infer_model(device=device)

            f0 = model.infer(
                audio,
                sr=self.sample_rate,
                decoder_mode="local_argmax",
                threshold=0.006,
                f0_min=self.f0_min,
                f0_max=self.f0_max,
                interp_uv=False,
                output_interp_target_length=f0_target_length,
            )
            f0 = f0.squeeze().cpu().numpy()
        elif method == "rmvpe" or "rmvpe" in method:
            # Auto-load RMVPE model if needed
            if not auto_loader.ensure_model_available(method):
                raise RuntimeError(f"Failed to ensure RMVPE model availability for method: {method}")
            
            is_half = False if device == "cpu" else config.is_half
            
            # Load RMVPE model using auto-loader
            model_rmvpe = auto_loader.load_f0_model(
                method, 
                device=device, 
                is_half=is_half
            )
            
            if model_rmvpe is None:
                raise RuntimeError(f"Failed to load RMVPE model for method: {method}")
            
            f0 = model_rmvpe.infer_from_audio(self.wav16k, thred=0.03)
        elif method in ["penn", "djcm", "swift", "pesto"]:
            # Auto-load other model-based methods
            if not auto_loader.ensure_model_available(method):
                raise RuntimeError(f"Failed to ensure model availability for method: {method}")
            
            # Load the specific model
            model = auto_loader.load_f0_model(method, device=device)
            
            if model is None:
                raise RuntimeError(f"Failed to load model for method: {method}")
            
            # Extract F0 based on the model type
            if method == "penn":
                f0 = model.infer(self.wav16k)
            elif method == "djcm":
                f0 = model.infer(self.wav16k)
            elif method == "swift":
                # SWIFT is ONNX-based, handle accordingly
                audio_input = torch.FloatTensor(self.wav16k).unsqueeze(0).numpy()
                f0 = model.run(None, {'input': audio_input})[0]
            elif method == "pesto":
                f0 = model.infer(self.wav16k)
        elif method == "world":
            # Use WORLD-based F0 extraction
            if parselmouth is None:
                raise ImportError("parselmouth is required for world method")
            
            sound = parselmouth.Sound(self.wav16k, sampling_frequency=16000)
            pitch = sound.to_pitch_ac(
                time_step=self.hop_size,
                pitch_floor=self.f0_min,
                pitch_ceiling=self.f0_max,
                very_accurate=False,
                octave_cost=0.01,
                voiced_cost=0.14,
                voiced_unvoiced_cost=0.14,
                silence_cost=0.115
            )
            f0 = pitch.selected_array["frequency"]
            # Fill unvoiced frames with interpolated values
            f0[f0 == 0] = np.nan
            f0 = np.interp(np.arange(len(f0)), np.arange(len(f0))[~np.isnan(f0)], f0[~np.isnan(f0)])
        elif method == "pyin":
            # Use librosa's pyin method
            f0, voiced_flag, voiced_probs = librosa.pyin(
                self.wav16k,
                fmin=self.f0_min,
                fmax=self.f0_max,
                hop_length=self.hop_length,
                threshold=0.1
            )
            # Fill unvoiced frames with interpolated values
            f0[~voiced_flag] = np.interp(
                np.arange(len(f0)),
                np.arange(len(f0))[voiced_flag],
                f0[voiced_flag]
            )
        elif method == "yin":
            # Use librosa's yin method
            f0 = librosa.yin(
                self.wav16k,
                fmin=self.f0_min,
                fmax=self.f0_max,
                hop_length=self.hop_length
            )
        elif method == "harvest":
            # Use WORLD's harvest method
            try:
                import pyworld as pw
            except ImportError:
                raise ImportError("pyworld is required for harvest method")
            
            f0, voiced_flag = pw.dio(
                self.wav16k.astype(np.float64),
                16000,
                fmin=self.f0_min,
                fmax=self.f0_max
            )
            f0 = pw.stonemask(self.wav16k.astype(np.float64), f0, voiced_flag, 16000)
            # Fill unvoiced frames with interpolated values
            f0[~voiced_flag] = np.interp(
                np.arange(len(f0)),
                np.arange(len(f0))[voiced_flag],
                f0[voiced_flag]
            )
        elif method == "parselmouth":
            # Use Parselmouth for F0 extraction
            if parselmouth is None:
                raise ImportError("parselmouth is required for parselmouth method")
            
            sound = parselmouth.Sound(self.wav16k, sampling_frequency=16000)
            pitch = sound.to_pitch()
            f0 = pitch.selected_array["frequency"]
            # Fill unvoiced frames with interpolated values
            f0[f0 == 0] = np.nan
            f0 = np.interp(np.arange(len(f0)), np.arange(len(f0))[~np.isnan(f0)], f0[~np.isnan(f0)])
        elif method == "swipe":
            # Use SWIPE (if available)
            try:
                from swipepy import swipe
            except ImportError:
                raise ImportError("swipepy is required for swipe method")
            
            f0 = swipe(
                self.wav16k,
                16000,
                fmin=self.f0_min,
                fmax=self.f0_max,
                step=self.hop_length
            )
        elif method.startswith("hybrid[") and method.endswith("]"):
            # Handle hybrid methods by combining multiple F0 estimates
            import re
            
            # Extract individual methods from hybrid specification
            methods_str = re.search(r'\[(.+)\]', method)
            if not methods_str:
                raise ValueError(f"Invalid hybrid method format: {method}")
            
            individual_methods = [m.strip() for m in methods_str.group(1).split('+')]
            
            if len(individual_methods) < 2:
                raise ValueError(f"Hybrid method must contain at least 2 methods: {method}")
            
            # Extract F0 for each method and combine
            f0_estimates = []
            weights = []
            
            for individual_method in individual_methods:
                # Create temporary F0Extractor for this method
                temp_extractor = F0Extractor(
                    wav_path=self.wav_path,
                    sample_rate=self.sample_rate,
                    hop_length=self.hop_length,
                    f0_min=self.f0_min,
                    f0_max=self.f0_max,
                    method=individual_method
                )
                
                # Extract F0 for this method
                temp_f0 = temp_extractor.extract_f0()
                f0_estimates.append(temp_f0)
                
                # Assign weights based on method quality/reliability
                if individual_method in ["rmvpe", "fcpe", "crepe-large", "crepe-full"]:
                    weights.append(1.0)  # High quality methods get higher weight
                elif individual_method in ["harvest", "yin", "pyin"]:
                    weights.append(0.8)  # Good quality methods
                else:
                    weights.append(0.6)  # Other methods
            
            # Weighted average of F0 estimates
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize weights
            
            # Calculate weighted average
            f0_combined = np.average(f0_estimates, axis=0, weights=weights)
            
            # Apply median filtering to reduce outliers
            try:
                from scipy.signal import medfilt
                f0 = medfilt(f0_combined, kernel_size=3)
            except ImportError:
                f0 = f0_combined  # Fallback if scipy is not available
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return hz_to_cents(f0, librosa.midi_to_hz(0))

    def plot_f0(self, f0):
        from matplotlib import pyplot as plt

        plt.figure(figsize=(10, 4))
        plt.plot(f0)
        plt.title(self.method)
        plt.xlabel("Time (frames)")
        plt.ylabel("F0 (cents)")
        plt.show()

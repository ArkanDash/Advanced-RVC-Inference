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

        if method == "crepe":
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
        elif method == "fcpe":
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
        elif method == "rmvpe":
            is_half = False if device == "cpu" else config.is_half
            # Get the directory where this F0Extractor.py file is located
            predictor_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up to the models directory level (2 levels up from predictors)
            models_dir = os.path.join(predictor_dir, "..", "..", "models", "predictors")
            rmvpe_model_path = os.path.join(models_dir, "rmvpe.pt")
            model_rmvpe = RMVPE0Predictor(
                rmvpe_model_path,
                is_half=is_half,
                device=device,
                # hop_length=80
            )
            f0 = model_rmvpe.infer_from_audio(self.wav16k, thred=0.03)
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

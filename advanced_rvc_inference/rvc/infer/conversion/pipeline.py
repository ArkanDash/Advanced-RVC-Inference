import os
import sys
import torch
import time
import warnings
import logging
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch.nn.functional as F
from scipy import signal

# Add current directory to path
sys.path.append(os.getcwd())

# Import Rich logging
try:
    from ....lib.rich_logging import logger as rich_logger, RICH_AVAILABLE
except ImportError:
    rich_logger = logging.getLogger(__name__)
    RICH_AVAILABLE = False

# Import Vietnamese-RVC compatible utilities
from ....lib.utils import (
    extract_features, 
    change_rms, 
    clear_gpu_cache, 
    load_faiss_index,
    extract_median_f0,
    proposal_f0_up_key,
    autotune_f0
)

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)

class Pipeline:
    """
    Enhanced Vietnamese-RVC Compatible Pipeline with Rich logging
    """
    def __init__(self, tgt_sr, config):
        # Vietnamese-RVC configuration parameters
        self.x_pad = config.x_pad
        self.x_query = config.x_query
        self.x_center = config.x_center
        self.x_max = config.x_max
        self.sample_rate = 16000
        self.window = 160
        self.t_pad = self.sample_rate * self.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sample_rate * self.x_query
        self.t_center = self.sample_rate * self.x_center
        self.t_max = self.sample_rate * self.x_max
        self.f0_min = 50
        self.f0_max = 1100
        self.device = config.device
        self.is_half = config.is_half
        self.tgt_sr = tgt_sr
        
        # Enhanced features
        self.f0_generator = None
        self.conversion_stats = {
            "total_conversions": 0,
            "total_time": 0.0,
            "average_time": 0.0
        }
        
        rich_logger.debug(f"Pipeline initialized - Target SR: {tgt_sr}Hz, Device: {self.device}")
    
    def _get_f0_generator(self):
        """Get or create F0 generator with auto-download support"""
        if self.f0_generator is None:
            rich_logger.debug("Initializing F0 generator with auto-download...")
            try:
                from ....lib.predictors.Generator import Generator
                self.f0_generator = Generator(
                    sample_rate=self.sample_rate,
                    hop_length=self.window,
                    f0_min=self.f0_min,
                    f0_max=self.f0_max,
                    device=self.device,
                    f0_onnx_mode=False,
                    auto_download_models=True
                )
                rich_logger.success("F0 generator initialized successfully")
            except Exception as e:
                rich_logger.error(f"Failed to initialize F0 generator: {e}")
                raise
        return self.f0_generator

    def voice_conversion(self, model, net_g, sid, audio0, pitch, pitchf, index, big_npy, index_rate, version, protect, energy):
        """
        Enhanced voice conversion with Vietnamese-RVC compatibility and Rich logging
        """
        pitch_guidance = pitch is not None and pitchf is not None
        energy_use = energy is not None

        # Log conversion parameters
        rich_logger.debug(f"Voice conversion - Pitch guidance: {pitch_guidance}, Energy: {energy_use}")

        # Prepare audio features
        feats = torch.from_numpy(audio0).to(self.device).to(torch.float16 if self.is_half else torch.float32)
        feats = feats.mean(-1) if feats.dim() == 2 else feats
        assert feats.dim() == 1, f"Expected 1D audio, got {feats.dim()}D"

        with torch.no_grad():
            # Extract hubert features
            rich_logger.debug("Extracting hubert features...")
            feats = extract_features(model, feats.view(1, -1), version, self.device)
            
            # Prepare features for protection if needed
            feats0 = feats.clone() if protect < 0.5 and pitch_guidance else None

            # Apply index-based feature enhancement
            if index is not None and big_npy is not None and index_rate != 0:
                rich_logger.debug(f"Applying index enhancement (rate: {index_rate})...")
                npy = feats[0].cpu().numpy()
                if self.is_half: 
                    npy = npy.astype(np.float32)

                # Search and retrieve similar features
                score, ix = index.search(npy, k=8)
                weight = np.square(1 / score)
                
                # Weighted combination of retrieved features
                npy = np.sum(big_npy[ix] * np.expand_dims(weight / weight.sum(axis=1, keepdims=True), axis=2), axis=1)
                if self.is_half: 
                    npy = npy.astype(np.float16)
                
                # Blend with original features
                feats = (torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate + (1 - index_rate) * feats)

            # Resize features for upsampling
            feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
            p_len = min(audio0.shape[0] // self.window, feats.shape[1])

            # Trim pitch arrays to match feature length
            if pitch_guidance: 
                pitch, pitchf = pitch[:, :p_len], pitchf[:, :p_len]
            if energy_use: 
                energy = energy[:p_len].unsqueeze(0)
            
            # Apply pitch protection
            if feats0 is not None:
                rich_logger.debug("Applying pitch protection...")
                pitchff = pitchf.clone()
                pitchff[pitchf > 0] = 1
                pitchff[pitchf < 1] = protect
                pitchff = pitchff.unsqueeze(-1)
                
                feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
                feats = (feats * pitchff + feats0 * (1 - pitchff)).to(feats0.dtype)

            # Prepare for inference
            p_len = torch.tensor([p_len], device=self.device).long()
            feats = feats.to(torch.float16 if self.is_half else torch.float32)
            
            # Perform voice conversion inference
            rich_logger.debug("Running voice conversion inference...")
            audio1 = (
                (
                    net_g.infer(
                        feats, 
                        p_len, 
                        pitch if pitch_guidance else None, 
                        pitchf.to(torch.float16 if self.is_half else torch.float32) if pitch_guidance else None,
                        sid,
                        energy.to(torch.float16 if self.is_half else torch.float32) if energy_use else None
                    )[0][0, 0]
                ).data.cpu().float().numpy()
            )

        # Cleanup
        del feats, feats0, p_len
        clear_gpu_cache()

        return audio1
    
    def pipeline(self, logger, model, net_g, sid, audio, f0_up_key, f0_method, file_index, index_rate, pitch_guidance, filter_radius, rms_mix_rate, version, protect, hop_length, f0_autotune, f0_autotune_strength, f0_file=None, f0_onnx=False, pbar=None, proposal_pitch=False, proposal_pitch_threshold=255.0, energy_use=False, del_onnx=True, alpha=0.5):
        """
        Main conversion pipeline with enhanced logging and error handling
        """
        start_time = time.time()
        self.conversion_stats["total_conversions"] += 1
        
        try:
            rich_logger.debug(f"Pipeline conversion started - Method: {f0_method}, Pitch guidance: {pitch_guidance}")
            
            # Load FAISS index if needed
            index, big_npy = load_faiss_index(file_index) if index_rate != 0 else (None, None)
            if pbar: 
                pbar.update(1)

            # Audio preprocessing
            rich_logger.debug("Applying high-pass filter...")
            opt_ts, audio_opt = [], []
            audio = signal.filtfilt(bh, ah, audio)
            audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")

            # Handle long audio files
            if audio_pad.shape[0] > self.t_max:
                rich_logger.debug("Processing long audio file with optimization...")
                audio_sum = np.zeros_like(audio)

                for i in range(self.window):
                    audio_sum += audio_pad[i : i - self.window]

                for t in range(self.t_center, audio.shape[0], self.t_center):
                    opt_ts.append(t - self.t_query + np.where(np.abs(audio_sum[t - self.t_query : t + self.t_query]) == np.abs(audio_sum[t - self.t_query : t + self.t_query]).min())[0][0])

            s = 0
            t, inp_f0 = None, None
            audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
            sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
            p_len = audio_pad.shape[0] // self.window

            # Load F0 file if provided
            if hasattr(f0_file, "name"):
                try:
                    rich_logger.debug(f"Loading F0 from file: {f0_file.name}")
                    with open(f0_file.name, "r") as f:
                        raw_lines = f.read()

                        if len(raw_lines) > 0:
                            inp_f0 = []

                            for line in raw_lines.strip("\n").split("\n"):
                                inp_f0.append([float(i) for i in line.split(",")])

                            inp_f0 = np.array(inp_f0, dtype=np.float32)
                except Exception as e:
                    rich_logger.error(f"Failed to read F0 file: {e}")
                    inp_f0 = None

            if pbar: 
                pbar.update(1)

            # Extract F0 if pitch guidance is enabled
            if pitch_guidance:
                rich_logger.debug(f"Extracting F0 using method: {f0_method}")
                f0_generator = self._get_f0_generator()
                
                pitch, pitchf = f0_generator.calculator(
                    self.x_pad, f0_method, audio_pad, f0_up_key, p_len, 
                    filter_radius, f0_autotune, f0_autotune_strength, 
                    manual_f0=inp_f0, proposal_pitch=proposal_pitch, 
                    proposal_pitch_threshold=proposal_pitch_threshold
                )
                
                if self.device == "mps": 
                    pitchf = pitchf.astype(np.float32)
                pitch, pitchf = torch.tensor(pitch[:p_len], device=self.device).unsqueeze(0).long(), torch.tensor(pitchf[:p_len], device=self.device).unsqueeze(0).float()

            if pbar: 
                pbar.update(1)

            # Extract energy if needed
            if energy_use:
                rich_logger.debug("Extracting energy features...")
                try:
                    if not hasattr(self, "rms_extract"): 
                        from ...lib.infer.extracting.rms import RMSEnergyExtractor
                        self.rms_extract = RMSEnergyExtractor(frame_length=2048, hop_length=self.window, center=True, pad_mode = "reflect").to(self.device).eval()

                    energy = self.rms_extract(torch.from_numpy(audio_pad).to(self.device).unsqueeze(0))[:p_len].to(self.device).float()
                except Exception as e:
                    rich_logger.error(f"Failed to extract energy: {e}")
                    energy = None

            if pbar: 
                pbar.update(1)

            # Process audio in chunks
            for t in opt_ts:
                t = t // self.window * self.window
                rich_logger.debug(f"Processing audio chunk: {s} to {t + self.t_pad2 + self.window}")
                
                audio_opt.append(
                    self.voice_conversion(
                        model, 
                        net_g, 
                        sid, 
                        audio_pad[s : t + self.t_pad2 + self.window], 
                        pitch[:, s // self.window : (t + self.t_pad2) // self.window] if pitch_guidance else None, 
                        pitchf[:, s // self.window : (t + self.t_pad2) // self.window] if pitch_guidance else None, 
                        index, 
                        big_npy, 
                        index_rate, 
                        version, 
                        protect, 
                        energy[:, s // self.window : (t + self.t_pad2) // self.window] if energy_use else None
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )    
                s = t
                
            # Process final chunk
            rich_logger.debug(f"Processing final audio chunk from {s} to end")
            audio_opt.append(
                self.voice_conversion(
                    model, 
                    net_g, 
                    sid, 
                    audio_pad[t:], 
                    (pitch[:, t // self.window :] if t is not None else pitch) if pitch_guidance else None, 
                    (pitchf[:, t // self.window :] if t is not None else pitchf) if pitch_guidance else None, 
                    index, 
                    big_npy, 
                    index_rate, 
                    version, 
                    protect, 
                    (energy[:, t // self.window :] if t is not None else energy) if energy_use else None
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )

            if pbar: 
                pbar.update(1)

            # Combine audio chunks
            audio_opt = np.concatenate(audio_opt)
            
            # Apply RMS mixing
            if rms_mix_rate != 1:
                rich_logger.debug(f"Applying RMS mixing (rate: {rms_mix_rate})")
                audio_opt = change_rms(audio, self.sample_rate, audio_opt, self.tgt_sr, rms_mix_rate)

            # Normalize output
            audio_max = np.abs(audio_opt).max() / 0.99
            if audio_max > 1: 
                audio_opt /= audio_max

            # Cleanup
            if pitch_guidance: 
                del pitch, pitchf
            del sid
            clear_gpu_cache()
            
            # Update performance stats
            elapsed_time = time.time() - start_time
            self.conversion_stats["total_time"] += elapsed_time
            self.conversion_stats["average_time"] = (
                self.conversion_stats["total_time"] / self.conversion_stats["total_conversions"]
            )
            
            rich_logger.debug(f"Pipeline conversion completed in {elapsed_time:.2f}s")

            return audio_opt
            
        except Exception as e:
            import traceback
            rich_logger.error(f"Pipeline conversion failed: {e}")
            rich_logger.debug(traceback.format_exc())
            raise
    
    def get_conversion_stats(self) -> dict:
        """Get conversion performance statistics"""
        return {
            "total_conversions": self.conversion_stats["total_conversions"],
            "total_time": self.conversion_stats["total_time"],
            "average_time": self.conversion_stats["average_time"],
            "conversions_per_second": 1.0 / self.conversion_stats["average_time"] if self.conversion_stats["average_time"] > 0 else 0
        }
    
    def reset_stats(self):
        """Reset conversion statistics"""
        self.conversion_stats = {
            "total_conversions": 0,
            "total_time": 0.0,
            "average_time": 0.0
        }
        rich_logger.info("Conversion statistics reset")

class EnhancedConfig:
    """
    Enhanced configuration class matching Vietnamese-RVC structure
    """
    
    def __init__(self, device="cpu", is_half=False):
        # Vietnamese-RVC default configuration
        self.x_pad = 1.25
        self.x_query = 10
        self.x_center = 60
        self.x_max = 65
        
        # Enhanced settings
        self.device = torch.device(device)
        self.is_half = is_half
        
        # Performance settings
        self.enable_optimizations = True
        self.use_krvc = False  # Will be detected automatically
        
        rich_logger.debug(f"Enhanced config initialized - Device: {self.device}, Half precision: {self.is_half}")
    
    def detect_hardware(self):
        """Detect and configure hardware-specific settings"""
        
        # Detect CUDA
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.is_half = True
            rich_logger.success(f"CUDA detected: {torch.cuda.get_device_name()}")
        
        # Detect Apple Silicon
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.is_half = False
            rich_logger.success("Apple Silicon MPS detected")
        
        # Fallback to CPU
        else:
            self.device = torch.device("cpu")
            self.is_half = False
            rich_logger.info("Using CPU")
        
        return self.device
import os
import sys
import time
import torch

import numpy as np
import torch.nn.functional as F
import torchaudio.transforms as tat

sys.path.append(os.getcwd())

from main.library.utils import circular_write
from main.app.variables import config, translations
from main.inference.realtime.pipeline import create_pipeline

class RVC_Realtime:
    def __init__(self, model_path, index_path = None, f0_method = "rmvpe", f0_onnx = False, embedder_model = "hubert_base", embedders_mode = "fairseq", sample_rate = 16000, hop_length = 160, silent_threshold = 0, input_sample_rate = 48000, output_sample_rate = 48000, vad_enabled = False, vad_sensitivity = 3, vad_frame_ms = 30, clean_audio=False, clean_strength=0.7):
        self.model_path = model_path
        self.index_path = index_path
        self.f0_method = f0_method
        self.f0_onnx = f0_onnx
        self.embedder_model = embedder_model
        self.embedders_mode = embedders_mode
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.pipeline = None
        self.convert_buffer = None
        self.pitch_buffer = None
        self.pitchf_buffer = None
        self.return_length = 0
        self.skip_head = 0
        self.silence_front = 0
        self.resample_in = None
        self.resample_out = None
        self.vad = None
        self.tg = None
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.vad_enabled = vad_enabled
        self.vad_sensitivity = vad_sensitivity
        self.vad_frame_ms = vad_frame_ms
        self.clean_audio = clean_audio
        self.clean_strength = clean_strength
        self.input_sensitivity = 10 ** (silent_threshold / 20)
        self.window_size = sample_rate // 100
        self.dtype = torch.float16 if config.is_half else torch.float32

    def initialize(self):
        if self.vad_enabled:
            from main.inference.realtime.vad_utils import VADProcessor
            self.vad = VADProcessor(sensitivity_mode=self.vad_sensitivity, sample_rate=self.sample_rate, frame_duration_ms=self.vad_frame_ms)
        else: self.vad = None

        if self.clean_audio:
            from main.tools.noisereduce import TorchGate
            self.tg = TorchGate(self.sample_rate, prop_decrease=self.clean_strength).to(config.device)
        else: self.tg = None

        self.pipeline = create_pipeline(
            model_path=self.model_path, 
            index_path=self.index_path, 
            f0_method=self.f0_method, 
            f0_onnx=self.f0_onnx, 
            embedder_model=self.embedder_model, 
            embedders_mode=self.embedders_mode, 
            sample_rate=self.sample_rate, 
            hop_length=self.hop_length, 
        )

        self.resample_in = tat.Resample(
            orig_freq=self.input_sample_rate,
            new_freq=self.sample_rate,
            dtype=torch.float32
        ).to(config.device)
        self.resample_out = tat.Resample(
            orig_freq=self.pipeline.tgt_sr,
            new_freq=self.output_sample_rate,
            dtype=torch.float32
        ).to(config.device)

    def realloc(self, block_frame, extra_frame, crossfade_frame, sola_search_frame):
        block_frame_16k = int(block_frame / self.input_sample_rate * self.sample_rate)
        crossfade_frame_16k = int(crossfade_frame / self.input_sample_rate * self.sample_rate)
        sola_search_frame_16k = int(sola_search_frame / self.input_sample_rate * self.sample_rate)
        extra_frame_16k = int(extra_frame / self.input_sample_rate * self.sample_rate)

        convert_size_16k = block_frame_16k + sola_search_frame_16k + extra_frame_16k + crossfade_frame_16k
        if (modulo := convert_size_16k % self.window_size) != 0: convert_size_16k = convert_size_16k + (self.window_size - modulo)

        self.convert_feature_size_16k = convert_size_16k // self.window_size
        self.skip_head = extra_frame_16k // self.window_size
        self.return_length = self.convert_feature_size_16k - self.skip_head
        self.silence_front = extra_frame_16k - (self.window_size * 5) if self.silence_front else 0

        audio_buffer_size = block_frame_16k + crossfade_frame_16k

        self.audio_buffer = torch.zeros(audio_buffer_size, dtype=self.dtype, device=config.device)
        self.convert_buffer = torch.zeros(convert_size_16k, dtype=self.dtype, device=config.device)
        self.pitch_buffer = torch.zeros(self.convert_feature_size_16k + 1, dtype=torch.int64, device=config.device)
        self.pitchf_buffer = torch.zeros(self.convert_feature_size_16k + 1, dtype=self.dtype, device=config.device)

    def inference(self, audio_in, f0_up_key = 0, index_rate = 0.5, protect = 0.5, filter_radius = 3, rms_mix_rate = 1, f0_autotune = False, f0_autotune_strength = 1, proposal_pitch = False, proposal_pitch_threshold = 255.0):
        if self.pipeline is None:
            raise RuntimeError(translations["create_pipeline_error"])

        audio_in_16k = self.resample_in(torch.as_tensor(audio_in, dtype=torch.float32, device=config.device)).to(self.dtype)
        circular_write(audio_in_16k, self.audio_buffer)

        vol_t = self.audio_buffer.square().mean().sqrt()
        vol = max(vol_t.item(), 0)

        if self.vad is not None:
            is_speech = self.vad.is_speech(audio_in_16k.cpu().numpy().copy())
            if not is_speech: 
                self.pipeline.execute(
                    self.convert_buffer,
                    self.pitch_buffer,
                    self.pitchf_buffer,
                    f0_up_key,
                    index_rate,
                    self.convert_feature_size_16k,
                    self.silence_front,
                    self.skip_head,
                    self.return_length,
                    protect,
                    filter_radius,
                    rms_mix_rate,
                    f0_autotune, 
                    f0_autotune_strength, 
                    proposal_pitch, 
                    proposal_pitch_threshold
                )
                return None, vol

        if vol < self.input_sensitivity:
            self.pipeline.execute(
                self.convert_buffer,
                self.pitch_buffer,
                self.pitchf_buffer,
                f0_up_key,
                index_rate,
                self.convert_feature_size_16k,
                self.silence_front,
                self.skip_head,
                self.return_length,
                protect,
                filter_radius,
                rms_mix_rate,
                f0_autotune, 
                f0_autotune_strength, 
                proposal_pitch, 
                proposal_pitch_threshold
            )

            return None, vol

        circular_write(audio_in_16k, self.convert_buffer)

        audio_model = self.pipeline.execute(
            self.convert_buffer,
            self.pitch_buffer,
            self.pitchf_buffer,
            f0_up_key,
            index_rate,
            self.convert_feature_size_16k,
            self.silence_front,
            self.skip_head,
            self.return_length,
            protect,
            filter_radius,
            rms_mix_rate,
            f0_autotune, 
            f0_autotune_strength, 
            proposal_pitch, 
            proposal_pitch_threshold
        )

        if self.tg is not None: audio_model = self.tg(audio_model.unsqueeze(0)).squeeze(0)
        audio_out = self.resample_out(audio_model * vol_t.sqrt())

        return audio_out, vol
    
class VoiceChanger:
    def __init__(self, read_chunk_size, cross_fade_overlap_size, input_sample_rate, extra_convert_size):
        self.block_frame = read_chunk_size * 128
        self.crossfade_frame = int(cross_fade_overlap_size * input_sample_rate)
        self.extra_frame = int(extra_convert_size * input_sample_rate)
        self.sola_search_frame = input_sample_rate // 100
        self.vc_model = None
        self.sola_buffer = None
        self.generate_strength()

    def initialize(self, vc_model):
        self.vc_model = vc_model
        self.vc_model.realloc(self.block_frame, self.extra_frame, self.crossfade_frame, self.sola_search_frame)
        self.vc_model.initialize()

    def generate_strength(self):
        self.fade_in_window = (0.5 * np.pi * torch.linspace(0.0, 1.0, steps=self.crossfade_frame, device=config.device, dtype=torch.float32)).sin() ** 2
        self.fade_out_window = 1 - self.fade_in_window
        self.sola_buffer = torch.zeros(self.crossfade_frame, device=config.device, dtype=torch.float32)

    def process_audio(self, audio_in, f0_up_key = 0, index_rate = 0.5, protect = 0.5, filter_radius = 3, rms_mix_rate = 1, f0_autotune = False, f0_autotune_strength = 1, proposal_pitch = False, proposal_pitch_threshold = 255.0):
        block_size = audio_in.shape[0]
        audio, vol = self.vc_model.inference(audio_in, f0_up_key, index_rate, protect, filter_radius, rms_mix_rate, f0_autotune, f0_autotune_strength, proposal_pitch, proposal_pitch_threshold)

        if audio is None: return np.zeros(block_size, dtype=np.float32), vol

        conv_input = audio[None, None, : self.crossfade_frame + self.sola_search_frame]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = (F.conv1d(conv_input ** 2, torch.ones(1, 1, self.crossfade_frame, device=config.device)) + 1e-8).sqrt()
        sola_offset = (cor_nom[0, 0] / cor_den[0, 0]).argmax()

        audio = audio[sola_offset:]
        audio[: self.crossfade_frame] *= self.fade_in_window
        audio[: self.crossfade_frame] += (self.sola_buffer * self.fade_out_window)

        self.sola_buffer[:] = audio[block_size : block_size + self.crossfade_frame]
        return audio[: block_size].detach().cpu().numpy(), vol
    
    @torch.no_grad()
    def on_request(self, audio_in, f0_up_key = 0, index_rate = 0.5, protect = 0.5, filter_radius = 3, rms_mix_rate = 1, f0_autotune = False, f0_autotune_strength = 1, proposal_pitch = False, proposal_pitch_threshold = 255.0):
        if self.vc_model is None:
            raise RuntimeError(translations["voice_changer_selected_error"])

        start = time.perf_counter()
        result, vol = self.process_audio(audio_in, f0_up_key, index_rate, protect, filter_radius, rms_mix_rate, f0_autotune, f0_autotune_strength, proposal_pitch, proposal_pitch_threshold)
        end = time.perf_counter()

        return result, vol, [0, (end - start) * 1000, 0]
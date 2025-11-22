import os
import sys
import threading

import numpy as np

sys.path.append(os.getcwd())

from main.library.utils import check_assets
from main.inference.realtime.audio import Audio
from main.app.variables import logger, translations
from main.inference.realtime.realtime import VoiceChanger, RVC_Realtime

class AudioCallbacks:
    def emit_to(self, performance):
        self.latency = performance[1]
    
    def __init__(self, pass_through = False, read_chunk_size = 192, cross_fade_overlap_size = 0.1, input_sample_rate = 48000, output_sample_rate = 48000, extra_convert_size = 0.5, model_path = None, index_path = None, f0_method = "rmvpe", f0_onnx = False, embedder_model = "hubert_base", embedders_mode = "fairseq", sample_rate = 16000, hop_length = 160, silent_threshold = -90, f0_up_key = 0, index_rate = 0.5, protect = 0.5, filter_radius = 3, rms_mix_rate = 1, f0_autotune = False, f0_autotune_strength = 1, proposal_pitch = False, proposal_pitch_threshold = 255.0, input_audio_gain = 1.0, output_audio_gain = 1.0, monitor_audio_gain = 1.0, monitor = False, vad_enabled = False, vad_sensitivity = 3, vad_frame_ms = 30, clean_audio = False, clean_strength = 0.7):
        self.pass_through = pass_through
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.lock = threading.Lock()
        self.vc = VoiceChanger(
            read_chunk_size, 
            cross_fade_overlap_size, 
            input_sample_rate, 
            extra_convert_size
        )
        self.audio = Audio(
            self, 
            f0_up_key, 
            index_rate, 
            protect, 
            filter_radius, 
            rms_mix_rate,
            f0_autotune, 
            f0_autotune_strength, 
            proposal_pitch, 
            proposal_pitch_threshold,
            input_audio_gain, 
            output_audio_gain, 
            monitor_audio_gain,
            monitor
        )
        self.initialize(
            model_path, 
            index_path, 
            f0_method, 
            f0_onnx, 
            embedder_model, 
            embedders_mode, 
            sample_rate, 
            hop_length, 
            silent_threshold,
            vad_enabled,
            vad_sensitivity,
            vad_frame_ms,
            clean_audio, 
            clean_strength
        )

    def initialize(self, model_path, index_path = None, f0_method = "rmvpe", f0_onnx = False, embedder_model = "hubert_base", embedders_mode = "fairseq", sample_rate = 16000, hop_length = 160, silent_threshold = -90, vad_enabled = False, vad_sensitivity = 3, vad_frame_ms = 30, clean_audio = False, clean_strength = 0.7):
        check_assets(f0_method, embedder_model, f0_onnx, embedders_mode)

        self.vc.initialize(
            RVC_Realtime(
                model_path, 
                index_path, 
                f0_method, 
                f0_onnx, 
                embedder_model, 
                embedders_mode, 
                sample_rate, 
                hop_length, 
                silent_threshold, 
                self.input_sample_rate, 
                self.output_sample_rate,
                vad_enabled, 
                vad_sensitivity,
                vad_frame_ms,
                clean_audio, 
                clean_strength
            )
        )

    def change_voice(self, received_data, f0_up_key = 0, index_rate = 0.5, protect = 0.5, filter_radius = 3, rms_mix_rate = 1, f0_autotune = False, f0_autotune_strength = 1, proposal_pitch = False, proposal_pitch_threshold = 255.0):
        if self.pass_through:
            vol = float(np.sqrt(np.square(received_data).mean(dtype=np.float32)))
            return received_data, vol, [0, 0, 0], None

        try:
            with self.lock:
                audio, vol, perf = self.vc.on_request(received_data, f0_up_key, index_rate, protect, filter_radius, rms_mix_rate, f0_autotune, f0_autotune_strength, proposal_pitch, proposal_pitch_threshold)

            return audio, vol, perf, None
        except RuntimeError as e:
            import traceback

            logger.debug(traceback.format_exc())
            logger.error(translations["error_occurred"].format(e=e))

            return np.zeros(1, dtype=np.float32), 0, [0, 0, 0], None
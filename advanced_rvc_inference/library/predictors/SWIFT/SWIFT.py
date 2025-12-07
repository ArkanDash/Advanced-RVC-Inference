import librosa
import onnxruntime

import numpy as np

SAMPLE_RATE, HOP_LENGTH, FRAME_LENGTH = 16000, 256, 1024

class SWIFT:
    def __init__(self, model_path, fmin = 50, fmax = 1100, confidence_threshold = 0.9, providers = ["CPUExecutionProvider"]):
        self.fmin = fmin
        self.fmax = fmax
        self.confidence_threshold = confidence_threshold
        session_options = onnxruntime.SessionOptions()
        session_options.inter_op_num_threads = 1
        session_options.intra_op_num_threads = 1
        self.pitch_session = onnxruntime.InferenceSession(model_path, session_options, providers=providers)
        self.pitch_input_name = self.pitch_session.get_inputs()[0].name

    def _extract_pitch_and_confidence(self, audio_16k):
        if audio_16k.ndim != 1 or len(audio_16k) == 0: raise ValueError
        if len(audio_16k) < 256: audio_16k = np.pad(audio_16k, (0, max(0, 256 - len(audio_16k))), mode="constant")

        outputs = self.pitch_session.run(None, {self.pitch_input_name: audio_16k[None, :].astype(np.float32)})
        if len(outputs) < 2: raise RuntimeError

        return outputs[0][0], outputs[1][0]

    def _compute_voicing(self, pitch_hz, confidence):
        return (confidence > self.confidence_threshold) & (pitch_hz >= self.fmin) & (pitch_hz <= self.fmax)

    def _calculate_timestamps(self, n_frames):
        frame_centers = np.arange(n_frames) * HOP_LENGTH + ((FRAME_LENGTH - 1) / 2 - ((FRAME_LENGTH - HOP_LENGTH) // 2))
        return frame_centers / SAMPLE_RATE

    def detect_from_array(self, audio_array, sample_rate=SAMPLE_RATE):
        if audio_array.ndim > 1: audio_array = np.mean(audio_array, axis=-1)

        audio_16k = librosa.resample(audio_array.astype(np.float32), orig_sr=sample_rate, target_sr=SAMPLE_RATE) if sample_rate != SAMPLE_RATE else audio_array
        pitch_hz, confidence = self._extract_pitch_and_confidence(audio_16k)

        return pitch_hz, self._compute_voicing(pitch_hz, confidence), self._calculate_timestamps(len(pitch_hz))
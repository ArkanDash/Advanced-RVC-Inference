import webrtcvad

import numpy as np

class VADProcessor:
    def __init__(self, sensitivity_mode=3, sample_rate=16000, frame_duration_ms=30):
        if sample_rate not in [8000, 16000]: raise ValueError
        if frame_duration_ms not in [10, 20, 30]: raise ValueError

        self.vad = webrtcvad.Vad(sensitivity_mode)
        self.sample_rate = sample_rate
        self.frame_length = int(sample_rate * (frame_duration_ms / 1000.0))

    def is_speech(self, audio_chunk):
        if audio_chunk.ndim > 1 and audio_chunk.shape[1] == 1: audio_chunk = audio_chunk.flatten()
        elif audio_chunk.ndim > 1: audio_chunk = np.mean(audio_chunk, axis=1)

        if np.max(np.abs(audio_chunk)) > 1.0: audio_chunk = np.clip(audio_chunk, -1.0, 1.0)

        audio_chunk = (audio_chunk * 32767).astype(np.int16)
        num_frames = len(audio_chunk) // self.frame_length

        if num_frames == 0 and len(audio_chunk) > 0:
            audio_chunk = np.concatenate((audio_chunk, np.zeros(self.frame_length - len(audio_chunk), dtype=np.int16)))
            num_frames = 1
        elif num_frames == 0 and len(audio_chunk) == 0: return False

        try:
            for i in range(num_frames):
                start = i * self.frame_length
                if self.vad.is_speech(audio_chunk[start:start + self.frame_length].tobytes(), self.sample_rate): return True
            
            return False
        except Exception:
            return False
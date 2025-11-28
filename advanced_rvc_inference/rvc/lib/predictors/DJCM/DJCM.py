import os
import sys
import torch

import numpy as np

from scipy.signal import medfilt

sys.path.append(os.getcwd())

from main.library.predictors.DJCM.model import DJCMM
from main.library.predictors.DJCM.spec import Spectrogram
from main.library.predictors.DJCM.utils import WINDOW_LENGTH, SAMPLE_RATE, N_CLASS

class DJCM:
    def __init__(self, model_path, device = "cpu", is_half = False, onnx = False, providers = ["CPUExecutionProvider"], batch_size = 1, segment_len = 5.12, kernel_size = 3):
        super(DJCM, self).__init__()
        self.onnx = onnx

        if self.onnx:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            self.model = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        else:
            model = DJCMM(1, 1, 1)
            model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            model = model.to(device).eval()
            self.model = model.half() if is_half else model.float()

        self.batch_size = batch_size
        self.seg_len = int(segment_len * SAMPLE_RATE)
        self.seg_frames = int(self.seg_len // int(SAMPLE_RATE // 100))

        self.device = device
        self.is_half = is_half
        self.kernel_size = kernel_size

        self.spec_extractor = Spectrogram(int(SAMPLE_RATE // 100), WINDOW_LENGTH).to(device)
        cents_mapping = 20 * np.arange(N_CLASS) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4))

    def spec2hidden(self, spec):
        if self.onnx:
            hidden = torch.as_tensor(
                self.model.run([self.model.get_outputs()[0].name], {self.model.get_inputs()[0].name: spec.cpu().numpy().astype(np.float32)})[0], device=self.device
            )
        else:
            hidden = self.model(
                spec.half() if self.is_half else spec.float()
            )

        return hidden

    def infer_from_audio(self, audio, thred=0.03):
        if torch.is_tensor(audio): audio = audio.cpu().numpy()
        if audio.ndim > 1: audio = audio.squeeze()

        with torch.no_grad():
            padded_audio = self.pad_audio(audio)
            hidden = self.inference(padded_audio)[:(audio.shape[-1] // int(SAMPLE_RATE // 100) + 1)]

            f0 = self.decode(hidden.squeeze(0).cpu().numpy(), thred)
            if self.kernel_size is not None: f0 = medfilt(f0, kernel_size=self.kernel_size)

            return f0
        
    def infer_from_audio_with_pitch(self, audio, thred=0.03, f0_min=50, f0_max=1100):
        f0 = self.infer_from_audio(audio, thred)
        f0[(f0 < f0_min) | (f0 > f0_max)] = 0

        return f0

    def to_local_average_cents(self, salience, thred=0.05):
        center = np.argmax(salience, axis=1)
        salience = np.pad(salience, ((0, 0), (4, 4)))
        center += 4
        todo_salience, todo_cents_mapping = [], []
        starts = center - 4
        ends = center + 5

        for idx in range(salience.shape[0]):
            todo_salience.append(salience[:, starts[idx] : ends[idx]][idx])
            todo_cents_mapping.append(self.cents_mapping[starts[idx] : ends[idx]])

        todo_salience = np.array(todo_salience)
        devided = np.sum(todo_salience * np.array(todo_cents_mapping), 1) / np.sum(todo_salience, 1)
        devided[np.max(salience, axis=1) <= thred] = 0

        return devided
        
    def decode(self, hidden, thred=0.03):
        f0 = 10 * (2 ** (self.to_local_average_cents(hidden, thred=thred) / 1200))
        f0[f0 == 10] = 0
        return f0

    def pad_audio(self, audio):
        audio_len = audio.shape[-1]

        seg_nums = int(np.ceil(audio_len / self.seg_len)) + 1
        pad_len = int(seg_nums * self.seg_len - audio_len + self.seg_len // 2)

        left_pad = np.zeros(int(self.seg_len // 4), dtype=np.float32)
        right_pad = np.zeros(int(pad_len - self.seg_len // 4), dtype=np.float32)
        padded_audio = np.concatenate([left_pad, audio, right_pad], axis=-1)

        segments = [padded_audio[start: start + int(self.seg_len)] for start in range(0, len(padded_audio) - int(self.seg_len) + 1, int(self.seg_len // 2))]
        segments = np.stack(segments, axis=0)
        segments = torch.from_numpy(segments).unsqueeze(1).to(self.device)

        return segments

    def inference(self, segments):
        hidden_segments = torch.cat([
            self.spec2hidden(self.spec_extractor(segments[i:i + self.batch_size].float()))
            for i in range(0, len(segments), self.batch_size)
        ], dim=0)

        hidden = torch.cat([
            seg[self.seg_frames // 4: int(self.seg_frames * 0.75)]
            for seg in hidden_segments
        ], dim=0)

        return hidden
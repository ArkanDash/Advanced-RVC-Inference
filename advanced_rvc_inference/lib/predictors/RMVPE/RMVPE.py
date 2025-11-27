import os
import sys
import torch

import numpy as np
import torch.nn.functional as F

sys.path.append(os.getcwd())

from main.library.predictors.RMVPE.mel import MelSpectrogram

N_MELS, N_CLASS = 128, 360

class RMVPE:
    def __init__(self, model_path, is_half, device=None, providers=None, onnx=False):
        self.onnx = onnx

        if self.onnx:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            self.model = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        else:
            from main.library.predictors.RMVPE.e2e import E2E

            model = E2E(4, 1, (2, 2))
            ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt)
            model.eval()
            if is_half: model = model.half()
            self.model = model.to(device)

        self.is_half = is_half
        self.device = device
        self.mel_extractor = MelSpectrogram(N_MELS, 16000, 1024, 160, None, 30, 8000).to(device)
        cents_mapping = 20 * np.arange(N_CLASS) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4))

    def mel2hidden(self, mel):
        with torch.no_grad():
            n_frames = mel.shape[-1]
            n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
            if n_pad > 0: mel = F.pad(mel, (0, n_pad), mode="constant")
            
            if self.onnx:
                hidden = self.model.run(
                    [self.model.get_outputs()[0].name], 
                    {
                        self.model.get_inputs()[0].name: mel.cpu().numpy().astype(np.float32)
                    }
                )[0] 
            else: 
                hidden = self.model(
                    mel.half() if self.is_half else mel.float()
                )

            return hidden[:, :n_frames]

    def decode(self, hidden, thred=0.03):
        f0 = 10 * (2 ** (self.to_local_average_cents(hidden, thred=thred) / 1200))
        f0[f0 == 10] = 0

        return f0

    def infer_from_audio(self, audio, thred=0.03):
        hidden = self.mel2hidden(self.mel_extractor(torch.from_numpy(audio).float().to(self.device).unsqueeze(0), center=True))

        return self.decode(hidden.squeeze(0).cpu().numpy().astype(np.float32) if not self.onnx else hidden[0], thred=thred)
    
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
import os
import sys
import torch

sys.path.append(os.getcwd())

from main.library.speaker_diarization.whisper import Whisper, ModelDimensions, log_mel_spectrogram, pad_or_trim

class WhisperModel(torch.nn.Module):
    def __init__(self, model_path, device):
        super().__init__()
        checkpoint = torch.load(model_path, map_location="cpu")
        dims = ModelDimensions(**checkpoint["dims"])
        self.final_proj = torch.nn.Linear(dims.n_text_state, 768)
        self.model = Whisper(dims)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(device)
        del self.model.decoder

    def forward(self, audio):
        ppgln = audio.shape[1] // 320
        mel = log_mel_spectrogram(pad_or_trim(audio[0])).to(audio.device)

        with torch.no_grad():
            ppg_raw = self.model.encoder(mel.unsqueeze(0))
            ppg_projected = self.final_proj(ppg_raw)
            ppg = ppg_projected.data.float()
            ppg = ppg[:, :ppgln, :]

        return [ppg]
    
    def extract_features(self, source, padding_mask = None, output_layer = None):
        return self.forward(source)
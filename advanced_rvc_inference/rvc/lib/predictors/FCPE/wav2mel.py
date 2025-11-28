import os
import sys
import torch

from torchaudio.transforms import Resample

sys.path.append(os.getcwd())

from main.library.predictors.FCPE.stft import STFT

class Wav2Mel:
    def __init__(self, device=None, dtype=torch.float32):
        self.sample_rate = 16000
        self.hop_size = 160
        if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = dtype
        self.stft = STFT(16000, 128, 1024, 1024, 160, 0, 8000)
        self.resample_kernel = {}

    def extract_nvstft(self, audio, keyshift=0, train=False):
        return self.stft.get_mel(audio, keyshift=keyshift, train=train).transpose(1, 2)

    def extract_mel(self, audio, sample_rate, keyshift=0, train=False):
        audio = audio.to(self.dtype).to(self.device)
        if sample_rate == self.sample_rate: audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel: self.resample_kernel[key_str] = Resample(sample_rate, self.sample_rate, lowpass_filter_width=128)
            self.resample_kernel[key_str] = (self.resample_kernel[key_str].to(self.dtype).to(self.device))
            audio_res = self.resample_kernel[key_str](audio)

        mel = self.extract_nvstft(audio_res, keyshift=keyshift, train=train) 
        n_frames = int(audio.shape[1] // self.hop_size) + 1
        mel = (torch.cat((mel, mel[:, -1:, :]), 1) if n_frames > int(mel.shape[1]) else mel)
        return mel[:, :n_frames, :] if n_frames < int(mel.shape[1]) else mel

    def __call__(self, audio, sample_rate, keyshift=0, train=False):
        return self.extract_mel(audio, sample_rate, keyshift=keyshift, train=train)
import os
import sys
import torch
import librosa
import functools
import torchaudio

import numpy as np
import torch.nn.functional as F

sys.path.append(os.getcwd())

from main.library.predictors.PENN.core import bins_to_cents, cents_to_frequency
from main.library.predictors.PENN.core import PITCH_BINS, CENTS_PER_BIN, OCTAVE, frequency_to_bins, seconds_to_samples, entropy, interpolate

SAMPLE_RATE, WINDOW_SIZE = 8000, 1024

class Viterbi:
    def __init__(self, pitch_bins=1440, hop_length=80, sample_rate=8000, local_pitch_window_size=19, octaves=1200, max_octaves_per_second=32, cents_per_bin=5):
        self.pitch_bins = pitch_bins
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.window_size = local_pitch_window_size
        self.octaves = octaves
        self.max_octave = max_octaves_per_second
        self.cents_per_bin = cents_per_bin

    def __call__(self, logits):
        distributions = F.softmax(logits, dim=1).permute(2, 1, 0)

        bins = np.array([
            librosa.sequence.viterbi(sequence, self.transition).astype(np.int64)
            for sequence in distributions.cpu().numpy()
        ])
        bins = torch.tensor(bins, device=distributions.device)

        pitch = self.local_expected_value_from_bins(bins.T, logits).T
        return pitch.T

    @functools.cached_property
    def transition(self):
        return self.triangular_transition_matrix().cpu().numpy()
    
    def local_expected_value_from_bins(self, bins, logits):
        padded = F.pad(logits.squeeze(2), (self.window_size // 2, self.window_size // 2), value=-float('inf'))

        if str(bins.device).startswith("ocl"):
            indices = (bins.cpu().repeat(1, self.window_size) + torch.arange(self.window_size, device="cpu")[None]).to(bins.device)
        else:
            indices = bins.repeat(1, self.window_size) + torch.arange(self.window_size, device=bins.device)[None]

        return self.expected_value(padded.gather(1, indices), bins_to_cents(torch.clip(indices - self.window_size // 2, 0)))
    
    def triangular_transition_matrix(self):
        xx, yy = torch.meshgrid(torch.arange(self.pitch_bins), torch.arange(self.pitch_bins), indexing='ij')
        transition = torch.clip(((self.max_octave * self.hop_length / self.sample_rate) * (self.octaves / self.cents_per_bin) + 1) - (xx - yy).abs(), 0)
        return transition / transition.sum(dim=1, keepdims=True)

    def expected_value(self, logits, cents):
        return cents_to_frequency((F.softmax(logits, dim=1) * cents).sum(dim=1, keepdims=True))

class PENN:
    def __init__(self, model_path, hop_length = 80, batch_size = None, f0_min = 31, f0_max = 1984, sample_rate = 8000, interp_unvoiced_at = None, device = None, providers = None, onnx = False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hopsize = hop_length / SAMPLE_RATE
        self.batch_size = batch_size
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.sample_rate = sample_rate
        self.interp_unvoiced_at = interp_unvoiced_at
        self.onnx = onnx
        self.resample_audio = None
        self.decoder = Viterbi(PITCH_BINS, hop_length, SAMPLE_RATE, 19, OCTAVE, 32, CENTS_PER_BIN)

        if self.onnx:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            self.model = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        else:
            from main.library.predictors.PENN.fcn import FCN

            model = FCN(256, PITCH_BINS, (2, 2))
            ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt['model'])
            model.eval()
            self.model = model.to(device)

    def expected_frames(self, samples, sample_rate, hopsize, center):
        hopsize_resampled = seconds_to_samples(hopsize, sample_rate)
        if center == 'half-window': samples = samples - ((WINDOW_SIZE / SAMPLE_RATE * sample_rate) - hopsize_resampled)
        elif center == 'half-hop': samples = samples
        elif center == 'zero': samples = samples + hopsize_resampled
        else: raise ValueError

        return max(1, int(samples / hopsize_resampled))
    
    def resample(self, audio, target_sample_rate=SAMPLE_RATE):
        if self.sample_rate == target_sample_rate: return audio
        if self.resample_audio is None: self.resample_audio = torchaudio.transforms.Resample(self.sample_rate, target_sample_rate).to(audio.device)
        
        return self.resample_audio(audio)
    
    def preprocess(self, audio, sample_rate=SAMPLE_RATE, hopsize=0.01, batch_size=None, center='half-window'):
        total_frames = self.expected_frames(audio.shape[-1], self.sample_rate, hopsize, center)
        if self.sample_rate != sample_rate: audio = self.resample(audio, sample_rate)

        hopsize = seconds_to_samples(hopsize)

        if center in ['half-hop', 'zero']:
            padding = int((WINDOW_SIZE - hopsize) / 2) if center == 'half-hop' else int(WINDOW_SIZE / 2)
            audio = torch.nn.functional.pad(audio, (padding, padding), mode='reflect')

        if isinstance(hopsize, int) or hopsize.is_integer():
            hopsize = int(round(hopsize))
            start_idxs = None
        else: start_idxs = torch.tensor([hopsize * i for i in range(total_frames + 1)]).round().int()

        batch_size = total_frames if batch_size is None else batch_size
        for i in range(0, total_frames, batch_size):
            batch = min(total_frames - i, batch_size)

            if start_idxs is None:
                start = i * hopsize
                end = min(start + int((batch - 1) * hopsize) + WINDOW_SIZE, audio.shape[-1])
                batch_audio = audio[:, start:end]

                if end - start < WINDOW_SIZE:
                    padding = WINDOW_SIZE - (end - start)
                    if (end - start) % hopsize: padding += end - start - hopsize
                    batch_audio = torch.nn.functional.pad(batch_audio, (0, padding))

                frames = torch.nn.functional.unfold(batch_audio[:, None, None], kernel_size=(1, WINDOW_SIZE), stride=(1, hopsize)).permute(2, 0, 1)
            else:
                frames = torch.zeros(batch, 1, WINDOW_SIZE)

                for j in range(batch):
                    start = start_idxs[i + j]
                    end = min(start + WINDOW_SIZE, audio.shape[-1])
                    frames[j, :, : end - start] = audio[:, start:end]

            yield frames

    def postprocess(self, logits, fmin, fmax):
        with torch.no_grad():
            logits[:, :frequency_to_bins(torch.tensor(fmin))] = -float('inf')
            logits[:, frequency_to_bins(torch.tensor(fmax), torch.ceil):] = -float('inf')

            pitch = self.decoder(logits)
            periodicity = entropy(logits)

            return pitch.T, periodicity.T
        
    def compute_f0(self, audio, center="half-window"):
        if self.batch_size is not None: logits = []

        for frames in self.preprocess(audio, SAMPLE_RATE, self.hopsize, self.batch_size, center):
            inferred = self.infer(frames.to(self.device)).detach()

            if self.batch_size is None: pitch, periodicity = self.postprocess(inferred, self.f0_min, self.f0_max)
            else: logits.append(inferred.cpu())

        if self.batch_size is not None:
            pitch, periodicity = self.postprocess(torch.cat(logits, 0), self.f0_min, self.f0_max)

        if self.interp_unvoiced_at is not None: 
            pitch = interpolate(pitch, periodicity, self.interp_unvoiced_at)
            return pitch

        return pitch, periodicity
    
    def infer(self, frames):
        if self.onnx:
            inferred = torch.tensor(
                self.model.run(
                    [self.model.get_outputs()[0].name], 
                    {
                        self.model.get_inputs()[0].name: frames.cpu().numpy()
                    }
                )[0]
            )
        else:
            with torch.no_grad():
                inferred = self.model(frames)
        
        return inferred
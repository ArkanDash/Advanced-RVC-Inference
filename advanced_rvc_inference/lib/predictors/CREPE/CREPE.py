import os
import sys
import torch
import librosa
import scipy.stats

import numpy as np

sys.path.append(os.getcwd())

from main.library.predictors.CREPE.model import MODEL

CENTS_PER_BIN, PITCH_BINS, SAMPLE_RATE, WINDOW_SIZE = 20, 360, 16000, 1024

class CREPE:
    def __init__(self, model_path, model_size="full", hop_length=512, batch_size=None, f0_min=50, f0_max=1100, device=None, sample_rate=16000, providers=None, onnx=False, return_periodicity=False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hop_length = hop_length
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.onnx = onnx
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.return_periodicity = return_periodicity

        if self.onnx:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            self.model = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        else:
            model = MODEL(model_size)
            ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt)
            model.eval()
            self.model = model.to(device)

    def bins_to_frequency(self, bins):
        if str(bins.device).startswith(("ocl", "privateuseone")): bins = bins.to(torch.float32)

        cents = CENTS_PER_BIN * bins + 1997.3794084376191
        return 10 * 2 ** ((cents + cents.new_tensor(scipy.stats.triang.rvs(c=0.5, loc=-CENTS_PER_BIN, scale=2 * CENTS_PER_BIN, size=cents.size()))) / 1200)

    def frequency_to_bins(self, frequency, quantize_fn=torch.floor):
        return quantize_fn(((1200 * (frequency / 10).log2()) - 1997.3794084376191) / CENTS_PER_BIN).int()

    def viterbi(self, logits):
        if not hasattr(self, 'transition'):
            xx, yy = np.meshgrid(range(360), range(360))
            transition = np.maximum(12 - abs(xx - yy), 0)
            self.transition = transition / transition.sum(axis=1, keepdims=True)

        with torch.no_grad():
            probs = torch.nn.functional.softmax(logits, dim=1)

        bins = torch.tensor(np.array([librosa.sequence.viterbi(sequence, self.transition).astype(np.int64) for sequence in probs.cpu().numpy()]), device=probs.device)
        return bins, self.bins_to_frequency(bins)
    
    def preprocess(self, audio, pad=True):
        hop_length = (self.sample_rate // 100) if self.hop_length is None else self.hop_length

        if self.sample_rate != SAMPLE_RATE:
            audio = torch.tensor(librosa.resample(audio.detach().cpu().numpy().squeeze(0), orig_sr=self.sample_rate, target_sr=SAMPLE_RATE, res_type="soxr_vhq"), device=audio.device).unsqueeze(0)
            hop_length = int(hop_length * SAMPLE_RATE / self.sample_rate)

        if pad:
            total_frames = 1 + int(audio.size(1) // hop_length)
            audio = torch.nn.functional.pad(audio, (WINDOW_SIZE // 2, WINDOW_SIZE // 2))
        else: total_frames = 1 + int((audio.size(1) - WINDOW_SIZE) // hop_length)

        batch_size = total_frames if self.batch_size is None else self.batch_size

        for i in range(0, total_frames, batch_size):
            frames = torch.nn.functional.unfold(audio[:, None, None, max(0, i * hop_length):min(audio.size(1), (i + batch_size - 1) * hop_length + WINDOW_SIZE)], kernel_size=(1, WINDOW_SIZE), stride=(1, hop_length))
            
            if self.device.startswith(("ocl", "privateuseone")):
                frames = frames.transpose(1, 2).contiguous().reshape(-1, WINDOW_SIZE).to(self.device)
            else:
                frames = frames.transpose(1, 2).reshape(-1, WINDOW_SIZE).to(self.device)

            frames -= frames.mean(dim=1, keepdim=True)
            frames /= torch.tensor(1e-10, device=frames.device).max(frames.std(dim=1, keepdim=True))

            yield frames

    def periodicity(self, probabilities, bins):
        probs_stacked = probabilities.transpose(1, 2).reshape(-1, PITCH_BINS)
        periodicity = probs_stacked.gather(1, bins.reshape(-1, 1).to(torch.int64))
        
        return periodicity.reshape(probabilities.size(0), probabilities.size(2))

    def postprocess(self, probabilities):
        probabilities = probabilities.detach()
        probabilities[:, :self.frequency_to_bins(torch.tensor(self.f0_min))] = -float('inf')
        probabilities[:, self.frequency_to_bins(torch.tensor(self.f0_max), torch.ceil):] = -float('inf')

        bins, pitch = self.viterbi(probabilities)

        if not self.return_periodicity: return pitch
        return pitch, self.periodicity(probabilities, bins)

    def compute_f0(self, audio, pad=True):
        results = []

        for frames in self.preprocess(audio, pad):
            if self.onnx:
                model = torch.tensor(
                    self.model.run(
                        [self.model.get_outputs()[0].name], 
                        {
                            self.model.get_inputs()[0].name: frames.cpu().numpy()
                        }
                    )[0].transpose(1, 0)[None],
                    device=self.device
                )
            else:
                with torch.no_grad():
                    model = self.model(
                        frames, 
                        embed=False
                    ).reshape(audio.size(0), -1, PITCH_BINS).transpose(1, 2)

            result = self.postprocess(model)
            results.append((result[0].to(audio.device), result[1].to(audio.device)) if isinstance(result, tuple) else result.to(audio.device))
        
        if self.return_periodicity:
            pitch, periodicity = zip(*results)
            return torch.cat(pitch, 1), torch.cat(periodicity, 1)
        
        return torch.cat(results, 1)
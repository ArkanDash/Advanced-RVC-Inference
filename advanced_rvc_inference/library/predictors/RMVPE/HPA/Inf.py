import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

now_dir = os.getcwd()
sys.path.append(now_dir)

from advanced_rvc_inference.library.predictors.RMVPE.HPA.constants import *
from advanced_rvc_inference.library.predictors.RMVPE.HPA.spec import MelSpectrogram

class HPARMVPE:
    """
    A predictor for fundamental frequency (F0) based on the HPA-RMVPE model.

    Args:
        model_path (str): Path to the HPA-RMVPE model file.
        device (str, optional): Device to use for computation. Defaults to None, which uses CUDA if available.
        is_half (bool, optional): Use Half to save resources and speed up.
        onnx (bool, optional): Using the ONNX model.
        providers (list, optional): Providers of onnx model. default is CPUExecutionProvider.
    """

    def __init__(
        self, 
        model_path: str, 
        device = "cpu",  # Changed from str | torch.device to just device for compatibility
        is_half: bool = False, 
        # onnx: bool = False, 
        providers = ["CPUExecutionProvider"], 
        hop_length: int = 160,
        n_gru: int = 1, 
        in_channels: int = 1, 
        en_out_channels: int = 16,
    ):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.onnx = model_path.endswith(".onnx") # onnx

        if self.onnx:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            self.model = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        else:
            from src.model import E2E0

            model = E2E0(n_gru, in_channels, en_out_channels)
            # Made weights_only conditional based on PyTorch version
            try:
                model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            except TypeError:
                model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()
            model = model.to(device).eval()
            self.model = model.half() if is_half else model.float()

        self.device = device
        self.is_half = is_half
        self.mel_extractor = MelSpectrogram(N_MELS, SAMPLE_RATE, WINDOW_LENGTH, hop_length, None, MEL_FMIN, MEL_FMAX).to(device)
        cents_mapping = 20 * np.arange(N_CLASS) + CONST
        self.cents_mapping = np.pad(cents_mapping, (4, 4))

    def mel2hidden(self, mel, chunk_size=SAMPLE_RATE*2):
        """
        Converts Mel-spectrogram features to hidden representation.

        Args:
            mel (torch.Tensor): Mel-spectrogram features.
        """
        with torch.no_grad():
            n_frames = mel.shape[-1]
            # print('n_frames', n_frames)
            # print('mel shape before padding', mel.shape)
            mel = F.pad(
                mel, (0, 32 * ((n_frames - 1) // 32 + 1) - n_frames), mode="reflect"
            )
            # print('mel shape after padding', mel.shape)

            output_chunks = []
            pad_frames = mel.shape[-1]
            for start in range(0, pad_frames, chunk_size):
                # print('chunk @', start)
                end = min(start + chunk_size, pad_frames)
                mel_chunk = mel[..., start:end]
                assert (
                    mel_chunk.shape[-1] % 32 == 0
                ), "chunk_size must be divisible by 32"
                # print(' before padding', mel_chunk.shape)
                # mel_chunk = F.pad(mel_chunk, (320, 320), mode="reflect")
                # print(' after padding', mel_chunk.shape)
                if self.onnx:
                    # Fixed numpy to torch tensor conversion for compatibility
                    onnx_output = self.model.run(
                        [self.model.get_outputs()[0].name], 
                        {
                            self.model.get_inputs()[0].name: mel_chunk.cpu().numpy().astype(np.float32)
                        }
                    )[0]
                    out_chunk = torch.from_numpy(onnx_output).to(self.device)
                else: 
                    out_chunk = self.model(
                        mel_chunk.half() if self.is_half else mel_chunk.float()
                    )
                # print(' result chunk', out_chunk.shape)
                # out_chunk = out_chunk[:, 320:-320, :]
                # print(' trimmed chunk', out_chunk.shape)
                output_chunks.append(out_chunk)

            hidden = torch.cat(output_chunks, dim=1)
        # print('output', hidden[:, :n_frames].shape)
        return hidden[:, :n_frames]

    def decode(self, hidden, thred=0.03):
        """
        Decodes hidden representation to F0.

        Args:
            hidden (np.ndarray): Hidden representation.
            thred (float, optional): Threshold for salience. Defaults to 0.03.
        """
        cents_pred = self.to_local_average_cents(hidden, thred=thred)
        f0 = 10 * (2 ** (cents_pred / 1200))
        f0[f0 == 10] = 0
        return f0

    def infer_from_audio(self, audio, thred=0.03):
        """
        Infers F0 from audio.

        Args:
            audio (np.ndarray): Audio signal.
            thred (float, optional): Threshold for salience. Defaults to 0.03.
        """
        audio = torch.from_numpy(audio).float().to(self.device).unsqueeze(0)
        mel = self.mel_extractor(audio, center=True)
        del audio
        with torch.no_grad():
            # Added CUDA availability check
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        hidden = self.mel2hidden(mel)
        hidden = hidden.squeeze(0).cpu().numpy()
        f0 = self.decode(hidden, thred=thred)
        return f0

    def infer_from_audio_with_pitch(self, audio, thred=0.03, f0_min=50, f0_max=1100):
        """
        Infers F0 from audio with pitch.

        Args:
            audio (np.ndarray): Audio signal.
            thred (float, optional): Threshold for salience. Defaults to 0.03.
            f0_min (float, int, optional): Minimum F0 threshold.
            f0_max (float, int, optional): Maximum F0 threshold.
        """

        f0 = self.infer_from_audio(audio, thred)
        f0[(f0 < f0_min) | (f0 > f0_max)] = 0

        return f0

    def to_local_average_cents(self, salience, thred=0.05):
        """
        Converts salience to local average cents.

        Args:
            salience (np.ndarray): Salience values.
            thred (float, optional): Threshold for salience. Defaults to 0.05.
        """
        center = np.argmax(salience, axis=1)
        salience = np.pad(salience, ((0, 0), (4, 4)))
        center += 4
        todo_salience = []
        todo_cents_mapping = []
        starts = center - 4
        ends = center + 5
        for idx in range(salience.shape[0]):
            todo_salience.append(salience[:, starts[idx] : ends[idx]][idx])
            todo_cents_mapping.append(self.cents_mapping[starts[idx] : ends[idx]])
        todo_salience = np.array(todo_salience)
        todo_cents_mapping = np.array(todo_cents_mapping)
        product_sum = np.sum(todo_salience * todo_cents_mapping, 1)
        weight_sum = np.sum(todo_salience, 1)
        devided = product_sum / weight_sum
        maxx = np.max(salience, axis=1)
        # Fixed boolean indexing for numpy 1.25.2 compatibility
        devided = np.where(maxx <= thred, 0, devided)
        return devided
    
if __name__ == "__main__":
    import librosa
    import matplotlib.pyplot as plt

    model = HPARMVPE("hpa-rmvpe.pt", device="cpu")
    y, sr = librosa.load("voice.mp3", sr=SAMPLE_RATE, mono=True)

    f0 = model.infer_from_audio(y)

    print(f0.shape)

    plt.figure(figsize=(10, 4))
    plt.plot(f0)
    plt.title("RMVPE")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.savefig("f0-rmvpe.png")
    plt.show()

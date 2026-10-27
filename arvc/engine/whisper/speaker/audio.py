import os
import math
import torch
import random
import torchaudio

from io import IOBase

def get_torchaudio_info(file, backend = None):
    if not backend:
        backends = (torchaudio.list_audio_backends())
        backend = "soundfile" if "soundfile" in backends else backends[0]

    info = torchaudio.info(file["audio"], backend=backend)
    if isinstance(file["audio"], IOBase): file["audio"].seek(0)

    return info

class Audio:
    @staticmethod
    def power_normalize(waveform):
        return waveform / (waveform.square().mean(dim=-1, keepdim=True).sqrt() + 1e-8)

    @staticmethod
    def validate_file(file):
        if isinstance(file, (str, os.PathLike)): file = {"audio": str(file), "uri": os.path.splitext(os.path.basename(file))[0]}
        elif isinstance(file, IOBase): return {"audio": file, "uri": "stream"}
        else: raise ValueError

        if "waveform" in file:
            waveform = file["waveform"]
            if len(waveform.shape) != 2 or waveform.shape[0] > waveform.shape[1]: raise ValueError

            sample_rate = file.get("sample_rate", None)
            if sample_rate is None: raise ValueError

            file.setdefault("uri", "waveform")

        elif "audio" in file:
            if isinstance(file["audio"], IOBase): return file

            path = os.path.abspath(file["audio"])
            file.setdefault("uri", os.path.splitext(os.path.basename(path))[0])

        else: raise ValueError

        return file

    def __init__(self, sample_rate = None, mono=None, backend = None):
        super().__init__()
        self.sample_rate = sample_rate
        self.mono = mono

        if not backend:
            backends = (torchaudio.list_audio_backends())  
            backend = "soundfile" if "soundfile" in backends else backends[0]

        self.backend = backend

    def downmix_and_resample(self, waveform, sample_rate):
        num_channels = waveform.shape[0]

        if num_channels > 1:
            if self.mono == "random":
                channel = random.randint(0, num_channels - 1)
                waveform = waveform[channel : channel + 1]
            elif self.mono == "downmix": waveform = waveform.mean(dim=0, keepdim=True)

        if (self.sample_rate is not None) and (self.sample_rate != sample_rate):
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
            sample_rate = self.sample_rate

        return waveform, sample_rate

    def get_duration(self, file):
        file = self.validate_file(file)

        if "waveform" in file:
            frames = len(file["waveform"].T)
            sample_rate = file["sample_rate"]
        else:
            info = file["torchaudio.info"] if "torchaudio.info" in file else get_torchaudio_info(file, backend=self.backend)
            frames = info.num_frames
            sample_rate = info.sample_rate

        return frames / sample_rate

    def get_num_samples(self, duration, sample_rate = None):
        sample_rate = sample_rate or self.sample_rate
        if sample_rate is None: raise ValueError

        return math.floor(duration * sample_rate)

    def __call__(self, file):
        file = self.validate_file(file)

        if "waveform" in file:
            waveform = file["waveform"]
            sample_rate = file["sample_rate"]
        elif "audio" in file:
            waveform, sample_rate = torchaudio.load(file["audio"], backend=self.backend)
            if isinstance(file["audio"], IOBase): file["audio"].seek(0)

        channel = file.get("channel", None)
        if channel is not None: waveform = waveform[channel : channel + 1]

        return self.downmix_and_resample(waveform, sample_rate)

    def crop(self, file, segment, duration = None, mode="raise"):
        file = self.validate_file(file)

        if "waveform" in file:
            waveform = file["waveform"]
            frames = waveform.shape[1]
            sample_rate = file["sample_rate"]
        elif "torchaudio.info" in file:
            info = file["torchaudio.info"]
            frames = info.num_frames
            sample_rate = info.sample_rate
        else:
            info = get_torchaudio_info(file, backend=self.backend)
            frames = info.num_frames
            sample_rate = info.sample_rate

        channel = file.get("channel", None)
        start_frame = math.floor(segment.start * sample_rate)

        if duration:
            num_frames = math.floor(duration * sample_rate)
            end_frame = start_frame + num_frames
        else:
            end_frame = math.floor(segment.end * sample_rate)
            num_frames = end_frame - start_frame

        if mode == "raise":
            if num_frames > frames: raise ValueError

            if end_frame > frames + math.ceil(0.001 * sample_rate): raise ValueError
            else:
                end_frame = min(end_frame, frames)
                start_frame = end_frame - num_frames

            if start_frame < 0: raise ValueError
        elif mode == "pad":
            pad_start = -min(0, start_frame)
            pad_end = max(end_frame, frames) - frames

            start_frame = max(0, start_frame)
            end_frame = min(end_frame, frames)

            num_frames = end_frame - start_frame

        if "waveform" in file: data = file["waveform"][:, start_frame:end_frame]
        else:
            try:
                data, _ = torchaudio.load(file["audio"], frame_offset=start_frame, num_frames=num_frames, backend=self.backend)
                if isinstance(file["audio"], IOBase): file["audio"].seek(0)
            except RuntimeError:
                if isinstance(file["audio"], IOBase): raise RuntimeError

                waveform, sample_rate = self.__call__(file)
                data = waveform[:, start_frame:end_frame]

                file["waveform"] = waveform
                file["sample_rate"] = sample_rate

        if channel is not None: data = data[channel : channel + 1, :]
        if mode == "pad": data = torch.nn.functional.pad(data, (pad_start, pad_end))

        return self.downmix_and_resample(data, sample_rate)
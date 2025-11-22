import os
import sys
import tqdm
import torch
import random
import concurrent.futures

sys.path.append(os.getcwd())

from main.library.uvr5_lib.demucs.utils import center_trim

class DummyPoolExecutor:
    class DummyResult:
        def __init__(self, func, *args, **kwargs):
            self.func = func
            self.args = args
            self.kwargs = kwargs

        def result(self):
            return self.func(*self.args, **self.kwargs)

    def __init__(self, workers=0):
        pass

    def submit(self, func, *args, **kwargs):
        return DummyPoolExecutor.DummyResult(func, *args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        return

class BagOfModels(torch.nn.Module):
    def __init__(self, models, weights = None, segment = None):
        super().__init__()
        assert len(models) > 0
        first = models[0]

        for other in models:
            assert other.sources == first.sources
            assert other.samplerate == first.samplerate
            assert other.audio_channels == first.audio_channels

            if segment is not None: other.segment = segment

        self.audio_channels = first.audio_channels
        self.samplerate = first.samplerate
        self.sources = first.sources
        self.models = torch.nn.ModuleList(models)

        if weights is None: weights = [[1.0 for _ in first.sources] for _ in models]
        else:
            assert len(weights) == len(models)

            for weight in weights:
                assert len(weight) == len(first.sources)

        self.weights = weights

    def forward(self, x):
        pass

class TensorChunk:
    def __init__(self, tensor, offset=0, length=None):
        total_length = tensor.shape[-1]
        assert offset >= 0
        assert offset < total_length

        length = total_length - offset if length is None else min(total_length - offset, length)

        if isinstance(tensor, TensorChunk):
            self.tensor = tensor.tensor
            self.offset = offset + tensor.offset
        else:
            self.tensor = tensor
            self.offset = offset

        self.length = length
        self.device = tensor.device

    @property
    def shape(self):
        shape = list(self.tensor.shape)
        shape[-1] = self.length
        return shape

    def padded(self, target_length):
        delta = target_length - self.length
        total_length = self.tensor.shape[-1]
        assert delta >= 0

        start = self.offset - delta // 2
        end = start + target_length

        correct_start = max(0, start)
        correct_end = min(total_length, end)

        pad_left = correct_start - start
        pad_right = end - correct_end

        out = torch.nn.functional.pad(self.tensor[..., correct_start:correct_end], (pad_left, pad_right))

        assert out.shape[-1] == target_length
        return out

def tensor_chunk(tensor_or_chunk):
    if isinstance(tensor_or_chunk, TensorChunk): return tensor_or_chunk
    else:
        assert isinstance(tensor_or_chunk, torch.Tensor)
        return TensorChunk(tensor_or_chunk)

def apply_model(model, mix, shifts=1, split=True, overlap=0.25, transition_power=1.0, static_shifts=1, set_progress_bar=None, device=None, progress=False, num_workers=0, pool=None):
    global fut_length, bag_num, prog_bar

    device = mix.device if device is None else torch.device(device)
    if pool is None: pool = concurrent.futures.ThreadPoolExecutor(num_workers) if num_workers > 0 and device.type == "cpu" else DummyPoolExecutor()

    kwargs = {
        "shifts": shifts,
        "split": split,
        "overlap": overlap,
        "transition_power": transition_power,
        "progress": progress,
        "device": device,
        "pool": pool,
        "set_progress_bar": set_progress_bar,
        "static_shifts": static_shifts,
    }

    if isinstance(model, BagOfModels):
        estimates, fut_length, prog_bar, current_model = 0, 0, 0, 0
        totals = [0] * len(model.sources)
        bag_num = len(model.models)

        for sub_model, weight in zip(model.models, model.weights):
            original_model_device = next(iter(sub_model.parameters())).device
            sub_model.to(device)
            fut_length += fut_length
            current_model += 1
            out = apply_model(sub_model, mix, **kwargs)
            sub_model.to(original_model_device)

            for k, inst_weight in enumerate(weight):
                out[:, k, :, :] *= inst_weight
                totals[k] += inst_weight

            estimates += out
            del out

        for k in range(estimates.shape[1]):
            estimates[:, k, :, :] /= totals[k]

        return estimates

    model.to(device)
    model.eval()
    assert transition_power >= 1
    batch, channels, length = mix.shape

    if shifts:
        kwargs["shifts"] = 0
        max_shift = int(0.5 * model.samplerate)
        mix = tensor_chunk(mix)
        padded_mix = mix.padded(length + 2 * max_shift)
        out = 0

        for _ in range(shifts):
            offset = random.randint(0, max_shift)
            shifted = TensorChunk(padded_mix, offset, length + max_shift - offset)
            shifted_out = apply_model(model, shifted, **kwargs)
            out += shifted_out[..., max_shift - offset :]

        out /= shifts
        return out
    elif split:
        kwargs["split"] = False
        out = torch.zeros(batch, len(model.sources), channels, length, device=mix.device)
        sum_weight = torch.zeros(length, device=mix.device)
        segment = int(model.samplerate * model.segment)
        stride = int((1 - overlap) * segment)
        offsets = range(0, length, stride)
        weight = torch.cat([torch.arange(1, segment // 2 + 1, device=device), torch.arange(segment - segment // 2, 0, -1, device=device)])
        assert len(weight) == segment
        weight = (weight / weight.max()) ** transition_power
        futures = []

        for offset in offsets:
            chunk = TensorChunk(mix, offset, segment)
            future = pool.submit(apply_model, model, chunk, **kwargs)
            futures.append((future, offset))
            offset += segment

        if progress: futures = tqdm.tqdm(futures)

        for future, offset in futures:
            if set_progress_bar:
                fut_length = len(futures) * bag_num * static_shifts
                prog_bar += 1
                set_progress_bar(0.1, (0.8 / fut_length * prog_bar))

            chunk_out = future.result()
            chunk_length = chunk_out.shape[-1]

            out[..., offset : offset + segment] += (weight[:chunk_length].to(device) * chunk_out).to(mix.device)
            sum_weight[offset : offset + segment] += weight[:chunk_length].to(mix.device)

        assert sum_weight.min() > 0

        out /= sum_weight
        return out
    else:
        valid_length = model.valid_length(length) if hasattr(model, "valid_length") else length
        mix = tensor_chunk(mix)
        padded_mix = mix.padded(valid_length).to(device)

        with torch.no_grad():
            out = model(padded_mix)

        return center_trim(out, length)

def demucs_segments(demucs_segment, demucs_model):
    if demucs_segment == "Default":
        segment = None

        if isinstance(demucs_model, BagOfModels):
            if segment is not None:
                for sub in demucs_model.models:
                    sub.segment = segment
        else:
            if segment is not None: sub.segment = segment
    else:
        try:
            segment = int(demucs_segment)
            if isinstance(demucs_model, BagOfModels):
                if segment is not None:
                    for sub in demucs_model.models:
                        sub.segment = segment
            else:
                if segment is not None: sub.segment = segment
        except:
            segment = None

            if isinstance(demucs_model, BagOfModels):
                if segment is not None:
                    for sub in demucs_model.models:
                        sub.segment = segment
            else:
                if segment is not None: sub.segment = segment

    return demucs_model
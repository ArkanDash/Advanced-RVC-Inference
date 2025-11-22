import numpy as np

class Slicer:
    def __init__(self, sr, threshold = -40.0, min_length = 5000, min_interval = 300, hop_size = 20, max_sil_kept = 5000):
        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.0)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform, begin, end):
        start_idx = begin * self.hop_size

        return waveform[:, start_idx:min(waveform.shape[1], end * self.hop_size)] if len(waveform.shape) > 1 else waveform[start_idx:min(waveform.shape[0], end * self.hop_size)]

    def slice(self, waveform):
        samples = waveform.mean(axis=0) if len(waveform.shape) > 1 else waveform
        if samples.shape[0] <= self.min_length: return [waveform]
        rms_list = get_rms(y=samples, frame_length=self.win_size, hop_length=self.hop_size).squeeze(0)
        sil_tags = []
        silence_start, clip_start = None, 0

        for i, rms in enumerate(rms_list):
            if rms < self.threshold:
                if silence_start is None: silence_start = i
                continue

            if silence_start is None: continue
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = (i - silence_start >= self.min_interval and i - clip_start >= self.min_length)
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue

            if i - silence_start <= self.max_sil_kept:
                pos = rms_list[silence_start : i + 1].argmin() + silence_start
                sil_tags.append((0, pos) if silence_start == 0 else (pos, pos))   
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = rms_list[i - self.max_sil_kept : silence_start + self.max_sil_kept + 1].argmin()
                pos += i - self.max_sil_kept
                pos_r = (rms_list[i - self.max_sil_kept : i + 1].argmin() + i - self.max_sil_kept)
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min((rms_list[silence_start : silence_start + self.max_sil_kept + 1].argmin() + silence_start), pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_r = (rms_list[i - self.max_sil_kept : i + 1].argmin() + i - self.max_sil_kept)
                sil_tags.append((0, pos_r) if silence_start == 0 else ((rms_list[silence_start : silence_start + self.max_sil_kept + 1].argmin() + silence_start), pos_r))
                clip_start = pos_r

            silence_start = None

        total_frames = rms_list.shape[0]
        if (silence_start is not None and total_frames - silence_start >= self.min_interval): sil_tags.append((rms_list[silence_start : min(total_frames, silence_start + self.max_sil_kept) + 1].argmin() + silence_start, total_frames + 1))

        if not sil_tags: return [waveform]
        else:
            chunks = []
            if sil_tags[0][0] > 0: chunks.append(self._apply_slice(waveform, 0, sil_tags[0][0]))

            for i in range(len(sil_tags) - 1):
                chunks.append(self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0]))

            if sil_tags[-1][1] < total_frames: chunks.append(self._apply_slice(waveform, sil_tags[-1][1], total_frames))
            return chunks

class Slicer2(Slicer):
    def slice2(self, waveform):
        samples = waveform.mean(axis=0) if len(waveform.shape) > 1 else waveform

        if samples.shape[0] <= self.min_length: return [(waveform, 0, samples.shape[0])]
        rms_list = get_rms(y=samples, frame_length=self.win_size, hop_length=self.hop_size).squeeze(0)

        sil_tags = []
        silence_start, clip_start = None, 0

        for i, rms in enumerate(rms_list):
            if rms < self.threshold:
                if silence_start is None: silence_start = i
                continue

            if silence_start is None: continue

            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = (i - silence_start >= self.min_interval and i - clip_start >= self.min_length)

            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue

            if i - silence_start <= self.max_sil_kept:
                pos = rms_list[silence_start : i + 1].argmin() + silence_start
                sil_tags.append((0, pos) if silence_start == 0 else (pos, pos))   
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = rms_list[i - self.max_sil_kept : silence_start + self.max_sil_kept + 1].argmin()
                pos += i - self.max_sil_kept

                pos_r = (rms_list[i - self.max_sil_kept : i + 1].argmin() + i - self.max_sil_kept)

                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min((rms_list[silence_start : silence_start + self.max_sil_kept + 1].argmin() + silence_start), pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_r = (rms_list[i - self.max_sil_kept : i + 1].argmin() + i - self.max_sil_kept)
                sil_tags.append((0, pos_r) if silence_start == 0 else ((rms_list[silence_start : silence_start + self.max_sil_kept + 1].argmin() + silence_start), pos_r))
                clip_start = pos_r

            silence_start = None

        total_frames = rms_list.shape[0]
        if (silence_start is not None and total_frames - silence_start >= self.min_interval): sil_tags.append((rms_list[silence_start : min(total_frames, silence_start + self.max_sil_kept) + 1].argmin() + silence_start, total_frames + 1))

        if not sil_tags: return [(waveform, 0, samples.shape[-1])]
        else:
            chunks = []
            if sil_tags[0][0] > 0: chunks.append((self._apply_slice(waveform, 0, sil_tags[0][0]), 0, sil_tags[0][0] * self.hop_size))

            for i in range(len(sil_tags) - 1):
                chunks.append((self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0]), sil_tags[i][1] * self.hop_size, sil_tags[i + 1][0] * self.hop_size))

            if sil_tags[-1][1] < total_frames: chunks.append((self._apply_slice(waveform, sil_tags[-1][1], total_frames), sil_tags[-1][1] * self.hop_size, samples.shape[-1]))
            return chunks
            
def get_rms(y, frame_length=2048, hop_length=512, pad_mode="constant"):
    y = np.pad(y, (int(frame_length // 2), int(frame_length // 2)), mode=pad_mode)
    axis = -1

    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    xw = np.moveaxis(np.lib.stride_tricks.as_strided(y, shape=tuple(x_shape_trimmed) + tuple([frame_length]), strides=y.strides + tuple([y.strides[axis]])), -1, axis - 1 if axis < 0 else axis + 1)
    
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)

    return np.sqrt(np.mean(np.abs(xw[tuple(slices)]) ** 2, axis=-2, keepdims=True))
import os
import sys
import torch

import numpy as np
import torch.utils.data as tdata

sys.path.append(os.getcwd())

from main.app.variables import translations
from main.inference.training.mel_processing import spectrogram_torch
from main.inference.training.utils import load_filepaths_and_text, load_wav_to_torch

class TextAudioLoader(tdata.Dataset):
    def __init__(self, hparams, pitch_guidance=True, energy=False):
        self.audiopaths_and_text = load_filepaths_and_text(hparams.training_files)
        self.max_wav_value = hparams.max_wav_value
        self.sample_rate = hparams.sample_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sample_rate = hparams.sample_rate
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 5000)
        self.pitch_guidance = pitch_guidance
        self.energy = energy
        self._filter()

    def _filter(self):
        audiopaths_and_text_new, lengths = [], []

        for item in self.audiopaths_and_text:
            audiopath = item[0]
            text = item[1]

            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_and_text_new.append(item)
                lengths.append(os.path.getsize(audiopath) // (3 * self.hop_length))

        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths

    def get_sid(self, sid):
        try:
            sid = torch.LongTensor([int(sid)])
        except ValueError:
            sid = torch.LongTensor([0])

        return sid

    def get_audio_text_pair(self, audiopath_and_text):
        if self.energy:
            if self.pitch_guidance:
                phone, pitch, pitchf, energy = self.get_labels(audiopath_and_text[1], audiopath_and_text[2], audiopath_and_text[3], audiopath_and_text[4])
                spec, wav = self.get_audio(audiopath_and_text[0])
                dv = self.get_sid(audiopath_and_text[5])
            else:
                phone, _, _, energy = self.get_labels(audiopath_and_text[1], energy=audiopath_and_text[2])
                spec, wav = self.get_audio(audiopath_and_text[0])
                dv = self.get_sid(audiopath_and_text[3])
        else:
            if self.pitch_guidance:       
                phone, _, _, _ = self.get_labels(audiopath_and_text[1])
                spec, wav = self.get_audio(audiopath_and_text[0])
                dv = self.get_sid(audiopath_and_text[2])
            else:
                phone, pitch, pitchf, _ = self.get_labels(audiopath_and_text[1], audiopath_and_text[2], audiopath_and_text[3])
                spec, wav = self.get_audio(audiopath_and_text[0])
                dv = self.get_sid(audiopath_and_text[4])

        extra = audiopath_and_text[2:]
        pitch = pitchf = energy = sid = None

        if self.pitch_guidance and self.energy: pitch, pitchf, energy, sid = extra
        elif self.pitch_guidance: pitch, pitchf, sid = extra
        elif self.energy: energy, sid = extra
        else: pitch, pitchf, sid = extra

        spec, wav = self.get_audio(audiopath_and_text[0])
        dv = self.get_sid(sid)

        phone, pitch, pitchf, energy = self.get_labels(
            audiopath_and_text[1],
            pitch=pitch,
            pitchf=pitchf,
            energy=energy
        )

        len_phone = phone.size()[0]
        len_spec = spec.size()[-1]

        if len_phone != len_spec:
            len_min = min(len_phone, len_spec)
            len_wav = len_min * self.hop_length
            spec, wav, phone = spec[:, :len_min], wav[:, :len_wav], phone[:len_min, :]
            if self.pitch_guidance: pitch, pitchf = pitch[:len_min], pitchf[:len_min]
            if self.energy: energy = energy[:len_min]

        outputs = [spec, wav, phone, dv]
        if self.pitch_guidance: outputs[3:3] = [pitch, pitchf]
        if self.energy: outputs.append(energy)

        return tuple(outputs)

    def get_labels(self, phone, pitch=None, pitchf=None, energy=None):
        phone = np.repeat(np.load(phone), 2, axis=0)
        n_num = min(phone.shape[0], 900)

        return (
            torch.FloatTensor(phone[:n_num, :]), 
            torch.LongTensor(np.load(pitch)[:n_num]) if pitch else None, 
            torch.FloatTensor(np.load(pitchf)[:n_num]) if pitchf else None, 
            torch.FloatTensor(np.load(energy)[:n_num]) if energy else None
        )

    def get_audio(self, filename):
        audio, sample_rate = load_wav_to_torch(filename)
        if sample_rate != self.sample_rate: raise ValueError(translations["sr_does_not_match"].format(sample_rate=sample_rate, sample_rate2=self.sample_rate))

        audio_norm = audio.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")

        if os.path.exists(spec_filename):
            try:
                spec = torch.load(spec_filename, weights_only=True)
            except Exception:
                spec = spectrogram_torch(audio_norm, self.filter_length, self.hop_length, self.win_length, center=False).squeeze(0)
                torch.save(spec, spec_filename, _use_new_zipfile_serialization=False)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length, self.hop_length, self.win_length, center=False).squeeze(0)
            torch.save(spec, spec_filename, _use_new_zipfile_serialization=False)

        return spec, audio_norm

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)

class TextAudioCollate:
    def __init__(self, return_ids=False, pitch_guidance=True, energy=False):
        self.return_ids = return_ids
        self.pitch_guidance = pitch_guidance
        self.energy = energy

    def __call__(self, batch):
        _, ids_sorted_decreasing = torch.sort(torch.LongTensor([x[0].size(1) for x in batch]), dim=0, descending=True)
        spec_lengths, wave_lengths = torch.LongTensor(len(batch)), torch.LongTensor(len(batch))
        spec_padded, wave_padded = torch.FloatTensor(len(batch), batch[0][0].size(0), max([x[0].size(1) for x in batch])), torch.FloatTensor(len(batch), 1, max([x[1].size(1) for x in batch]))
        spec_padded.zero_()
        wave_padded.zero_()

        max_phone_len = max([x[2].size(0) for x in batch])
        phone_lengths, phone_padded = torch.LongTensor(len(batch)), torch.FloatTensor(len(batch), max_phone_len, batch[0][2].shape[1])
        phone_padded.zero_()

        if self.pitch_guidance:
            pitch_padded, pitchf_padded = torch.LongTensor(len(batch), max_phone_len), torch.FloatTensor(len(batch), max_phone_len)
            pitch_padded.zero_()
            pitchf_padded.zero_()

        sid = torch.LongTensor(len(batch))

        if self.energy:
            energy_padded = torch.FloatTensor(len(batch), max_phone_len)
            energy_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            spec = row[0]

            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wave = row[1]
            wave_padded[i, :, : wave.size(1)] = wave
            wave_lengths[i] = wave.size(1)

            phone = row[2]
            phone_padded[i, : phone.size(0), :] = phone
            phone_lengths[i] = phone.size(0)

            if self.pitch_guidance:
                pitch = row[3]
                pitch_padded[i, : pitch.size(0)] = pitch
                pitchf = row[4]
                pitchf_padded[i, : pitchf.size(0)] = pitchf

            sid[i] = row[5 if self.pitch_guidance else 3]

            if self.energy:
                energy = row[6 if self.pitch_guidance else 4]
                energy_padded[i, : energy.size(0)] = energy

        outputs = [phone_padded, phone_lengths, spec_padded, spec_lengths, wave_padded, wave_lengths, sid]
        if self.pitch_guidance: outputs[2:2] = [pitch_padded, pitchf_padded]
        if self.energy: outputs.append(energy_padded)

        return tuple(outputs)

class DistributedBucketSampler(tdata.distributed.DistributedSampler):
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]

        for i in range(len(self.lengths)):
            idx_bucket = self._bisect(self.lengths[i])
            if idx_bucket != -1: buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, -1, -1):  
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []

        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            num_samples_per_bucket.append(len_bucket + ((total_batch_size - (len_bucket % total_batch_size)) % total_batch_size))

        return buckets, num_samples_per_bucket

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices, batches = [], []

        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            rem = self.num_samples_per_bucket[i] - len_bucket
            ids_bucket = (ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[: (rem % len_bucket)])[self.rank :: self.num_replicas]

            for j in range(len(ids_bucket) // self.batch_size):
                batches.append([bucket[idx] for idx in ids_bucket[j * self.batch_size : (j + 1) * self.batch_size]])

        if self.shuffle: batches = [batches[i] for i in torch.randperm(len(batches), generator=g).tolist()]
        self.batches = batches
        assert len(self.batches) * self.batch_size == self.num_samples

        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None: hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2

            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]: return mid
            elif x <= self.boundaries[mid]: return self._bisect(x, lo, mid)
            else: return self._bisect(x, mid + 1, hi)
        else: return -1

    def __len__(self):
        return self.num_samples // self.batch_size
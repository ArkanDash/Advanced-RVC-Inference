import os
import sys
import ast
import torch
import itertools
import collections

sys.path.append(os.getcwd())

from main.library.speaker_diarization.speechbrain import if_main_process, ddp_barrier
from main.library.speaker_diarization.features import register_checkpoint_hooks, mark_as_saver, mark_as_loader

@register_checkpoint_hooks
class CategoricalEncoder:
    VALUE_SEPARATOR = " => "
    EXTRAS_SEPARATOR = "================\n"

    def __init__(self, starting_index=0, **special_labels):
        self.lab2ind = {}
        self.ind2lab = {}
        self.starting_index = starting_index
        self.handle_special_labels(special_labels)

    def handle_special_labels(self, special_labels):
        if "unk_label" in special_labels: self.add_unk(special_labels["unk_label"])

    def __len__(self):
        return len(self.lab2ind)

    @classmethod
    def from_saved(cls, path):
        obj = cls()
        obj.load(path)
        return obj

    def update_from_iterable(self, iterable, sequence_input=False):
        label_iterator = itertools.chain.from_iterable(iterable) if sequence_input else iter(iterable)
        for label in label_iterator:
            self.ensure_label(label)

    def update_from_didataset(self, didataset, output_key, sequence_input=False):
        with didataset.output_keys_as([output_key]):
            self.update_from_iterable((data_point[output_key] for data_point in didataset), sequence_input=sequence_input)

    def limited_labelset_from_iterable(self, iterable, sequence_input=False, n_most_common=None, min_count=1):
        label_iterator = itertools.chain.from_iterable(iterable) if sequence_input else iter(iterable)
        counts = collections.Counter(label_iterator)

        for label, count in counts.most_common(n_most_common):
            if count < min_count: break
            self.add_label(label)

        return counts

    def load_or_create(self, path, from_iterables=[], from_didatasets=[], sequence_input=False, output_key=None, special_labels={}):
        try:
            if if_main_process():
                if not self.load_if_possible(path):
                    for iterable in from_iterables:
                        self.update_from_iterable(iterable, sequence_input)

                    for didataset in from_didatasets:
                        if output_key is None: raise ValueError
                        self.update_from_didataset(didataset, output_key, sequence_input)

                    self.handle_special_labels(special_labels)
                    self.save(path)
        finally:
            ddp_barrier()
            self.load(path)

    def add_label(self, label):
        if label in self.lab2ind: raise KeyError
        index = self._next_index()

        self.lab2ind[label] = index
        self.ind2lab[index] = label

        return index

    def ensure_label(self, label):
        if label in self.lab2ind: return self.lab2ind[label]
        else: return self.add_label(label)

    def insert_label(self, label, index):
        if label in self.lab2ind: raise KeyError
        else: self.enforce_label(label, index)

    def enforce_label(self, label, index):
        index = int(index)

        if label in self.lab2ind:
            if index == self.lab2ind[label]: return
            else: del self.ind2lab[self.lab2ind[label]]

        if index in self.ind2lab:
            saved_label = self.ind2lab[index]
            moving_other = True
        else: moving_other = False

        self.lab2ind[label] = index
        self.ind2lab[index] = label

        if moving_other:
            new_index = self._next_index()
            self.lab2ind[saved_label] = new_index
            self.ind2lab[new_index] = saved_label

    def add_unk(self, unk_label="<unk>"):
        self.unk_label = unk_label
        return self.add_label(unk_label)

    def _next_index(self):
        index = self.starting_index
        while index in self.ind2lab:
            index += 1

        return index

    def is_continuous(self):
        indices = sorted(self.ind2lab.keys())
        return self.starting_index in indices and all(j - i == 1 for i, j in zip(indices[:-1], indices[1:]))

    def encode_label(self, label, allow_unk=True):
        self._assert_len()

        try:
            return self.lab2ind[label]
        except KeyError:
            if hasattr(self, "unk_label") and allow_unk: return self.lab2ind[self.unk_label]
            elif hasattr(self, "unk_label") and not allow_unk: raise KeyError
            elif not hasattr(self, "unk_label") and allow_unk: raise KeyError
            else: raise KeyError

    def encode_label_torch(self, label, allow_unk=True):
        return torch.LongTensor([self.encode_label(label, allow_unk)])

    def encode_sequence(self, sequence, allow_unk=True):
        self._assert_len()
        return [self.encode_label(label, allow_unk) for label in sequence]

    def encode_sequence_torch(self, sequence, allow_unk=True):
        return torch.LongTensor([self.encode_label(label, allow_unk) for label in sequence])

    def decode_torch(self, x):
        self._assert_len()
        decoded = []

        if x.ndim == 1:  
            for element in x:
                decoded.append(self.ind2lab[int(element)])
        else:
            for subtensor in x:
                decoded.append(self.decode_torch(subtensor))

        return decoded

    def decode_ndim(self, x):
        self._assert_len()
        try:
            decoded = []
            for subtensor in x:
                decoded.append(self.decode_ndim(subtensor))

            return decoded
        except TypeError:  
            return self.ind2lab[int(x)]

    @mark_as_saver
    def save(self, path):
        self._save_literal(path, self.lab2ind, self._get_extras())

    def load(self, path):
        lab2ind, ind2lab, extras = self._load_literal(path)
        self.lab2ind = lab2ind
        self.ind2lab = ind2lab
        self._set_extras(extras)

    @mark_as_loader
    def load_if_possible(self, path, end_of_epoch=False):
        del end_of_epoch

        try:
            self.load(path)
        except FileNotFoundError:
            return False
        except (ValueError, SyntaxError):
            return False
        
        return True 

    def expect_len(self, expected_len):
        self.expected_len = expected_len

    def ignore_len(self):
        self.expected_len = None

    def _assert_len(self):
        if hasattr(self, "expected_len"):
            if self.expected_len is None: return
            if len(self) != self.expected_len: raise RuntimeError
        else:
            self.ignore_len()
            return

    def _get_extras(self):
        extras = {"starting_index": self.starting_index}
        if hasattr(self, "unk_label"): extras["unk_label"] = self.unk_label

        return extras

    def _set_extras(self, extras):
        if "unk_label" in extras: self.unk_label = extras["unk_label"]
        self.starting_index = extras["starting_index"]

    @staticmethod
    def _save_literal(path, lab2ind, extras):
        with open(path, "w", encoding="utf-8") as f:
            for label, ind in lab2ind.items():
                f.write(repr(label) + CategoricalEncoder.VALUE_SEPARATOR + str(ind) + "\n")

            f.write(CategoricalEncoder.EXTRAS_SEPARATOR)

            for key, value in extras.items():
                f.write(repr(key) + CategoricalEncoder.VALUE_SEPARATOR + repr(value) + "\n")

            f.flush()

    @staticmethod
    def _load_literal(path):
        lab2ind, ind2lab, extras = {}, {}, {}

        with open(path, encoding="utf-8") as f:
            for line in f:
                if line == CategoricalEncoder.EXTRAS_SEPARATOR: break
                literal, ind = line.strip().split(CategoricalEncoder.VALUE_SEPARATOR, maxsplit=1)
                label = ast.literal_eval(literal)
                lab2ind[label] = int(ind)
                ind2lab[ind] = label

            for line in f:
                literal_key, literal_value = line.strip().split(CategoricalEncoder.VALUE_SEPARATOR, maxsplit=1)
                extras[ast.literal_eval(literal_key)] = ast.literal_eval(literal_value)
                
        return lab2ind, ind2lab, extras
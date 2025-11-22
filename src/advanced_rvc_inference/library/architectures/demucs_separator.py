import os
import sys
import yaml
import torch
import warnings

import numpy as np

from hashlib import sha256

sys.path.append(os.getcwd())

from main.app.variables import config
from main.library.utils import clear_gpu_cache
from main.library.uvr5_lib import spec_utils, common_separator
from main.library.uvr5_lib.demucs import hdemucs, states, apply

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.getcwd(), "main", "library", "uvr5_lib"))

DEMUCS_4_SOURCE_MAPPER = {
    common_separator.CommonSeparator.BASS_STEM: 0, 
    common_separator.CommonSeparator.DRUM_STEM: 1, 
    common_separator.CommonSeparator.OTHER_STEM: 2, 
    common_separator.CommonSeparator.VOCAL_STEM: 3
}

class DemucsSeparator(common_separator.CommonSeparator):
    def __init__(self, common_config, arch_config):
        super().__init__(config=common_config)
        self.segment_size = arch_config.get("segment_size", "Default")
        self.shifts = arch_config.get("shifts", 2)
        self.overlap = arch_config.get("overlap", 0.25)
        self.segments_enabled = arch_config.get("segments_enabled", True)
        self.demucs_source_map = DEMUCS_4_SOURCE_MAPPER
        self.audio_file_path = None
        self.audio_file_base = None
        self.demucs_model_instance = None
        if config.configs.get("demucs_cpu_mode", False): self.torch_device = torch.device("cpu")

    def separate(self, audio_file_path):
        source = None
        inst_source = {}
        self.audio_file_path = audio_file_path
        self.audio_file_base = os.path.splitext(os.path.basename(audio_file_path))[0]
        mix = self.prepare_mix(self.audio_file_path)
        self.demucs_model_instance = hdemucs.HDemucs(sources=["drums", "bass", "other", "vocals"])
        self.demucs_model_instance = get_demucs_model(name=os.path.splitext(os.path.basename(self.model_path))[0], repo=os.path.dirname(self.model_path))
        self.demucs_model_instance = apply.demucs_segments(self.segment_size, self.demucs_model_instance)
        self.demucs_model_instance.to(self.torch_device)
        self.demucs_model_instance.eval()
        source = self.demix_demucs(mix)
        del self.demucs_model_instance
        clear_gpu_cache()
        output_files = []

        if isinstance(inst_source, np.ndarray):
            inst_source[self.demucs_source_map[common_separator.CommonSeparator.VOCAL_STEM]] = spec_utils.reshape_sources(inst_source[self.demucs_source_map[common_separator.CommonSeparator.VOCAL_STEM]], source[self.demucs_source_map[common_separator.CommonSeparator.VOCAL_STEM]])
            source = inst_source

        if isinstance(source, np.ndarray):
            source_length = len(source)

            if source_length == 2: 
                self.demucs_source_map = {
                    common_separator.CommonSeparator.INST_STEM: 0, 
                    common_separator.CommonSeparator.VOCAL_STEM: 1
                }
            elif source_length == 6: 
                self.demucs_source_map = {
                    common_separator.CommonSeparator.BASS_STEM: 0, 
                    common_separator.CommonSeparator.DRUM_STEM: 1, 
                    common_separator.CommonSeparator.OTHER_STEM: 2, 
                    common_separator.CommonSeparator.VOCAL_STEM: 3, 
                    common_separator.CommonSeparator.GUITAR_STEM: 4, 
                    common_separator.CommonSeparator.PIANO_STEM: 5
                }
            else: self.demucs_source_map = DEMUCS_4_SOURCE_MAPPER

        for stem_name, stem_value in self.demucs_source_map.items():
            if self.output_single_stem is not None:
                if stem_name.lower() != self.output_single_stem.lower():
                    continue

            stem_path = os.path.join(f"{self.audio_file_base}_({stem_name})_{self.model_name}.{self.output_format.lower()}")
            self.final_process(stem_path, source[stem_value].T, stem_name)
            output_files.append(stem_path)

        return output_files

    def demix_demucs(self, mix):
        processed = {}
        mix = torch.tensor(mix, dtype=torch.float32)
        ref = mix.mean(0)
        mix = (mix - ref.mean()) / ref.std()
        mix_infer = mix

        with torch.no_grad():
            sources = apply.apply_model(model=self.demucs_model_instance, mix=mix_infer[None], shifts=self.shifts, split=self.segments_enabled, overlap=self.overlap, static_shifts=max(self.shifts, 1), set_progress_bar=None, device=self.torch_device, progress=True)[0]

        sources = (sources * ref.std() + ref.mean()).cpu().numpy()
        sources[[0, 1]] = sources[[1, 0]]

        processed[mix] = sources[:, :, 0:None].copy()
        return np.concatenate([s[:, :, 0:None] for s in list(processed.values())], axis=-1)

class LocalRepo:
    def __init__(self, root):
        self.root = root
        self.scan()

    def scan(self):
        self._models, self._checksums = {}, {}
        for filename in os.listdir(self.root):
            filepath = os.path.join(self.root, filename)
            if not os.path.isfile(filepath): continue

            if os.path.splitext(filename)[1] == ".th":
                stem = os.path.splitext(filename)[0]
                
                if "-" in stem:
                    xp_sig, checksum = stem.split("-", 1)
                    self._checksums[xp_sig] = checksum
                else: xp_sig = stem

                if xp_sig in self._models: raise RuntimeError
                self._models[xp_sig] = filepath

    def has_model(self, sig):
        return sig in self._models

    def get_model(self, sig):
        try:
            file = self._models[sig]
        except KeyError:
            raise RuntimeError
        
        if sig in self._checksums: check_checksum(file, self._checksums[sig])
        return states.load_model(file)

class BagOnlyRepo:
    def __init__(self, root, model_repo):
        self.root = root
        self.model_repo = model_repo
        self.scan()

    def scan(self):
        self._bags = {}
        for filename in os.listdir(self.root):
            filepath = os.path.join(self.root, filename)

            if os.path.isfile(filepath) and os.path.splitext(filename)[1] == ".yaml":
                stem = os.path.splitext(filename)[0]
                self._bags[stem] = filepath

    def get_model(self, name):
        try:
            yaml_file = self._bags[name]
        except KeyError:
            raise RuntimeError
        
        with open(yaml_file, 'r') as f:
            bag = yaml.safe_load(f)

        return apply.BagOfModels([self.model_repo.get_model(sig) for sig in bag["models"]], bag.get("weights"), bag.get("segment"))

def check_checksum(path, checksum):
    sha = sha256()

    with open(path, "rb") as file:
        while 1:
            buf = file.read(2 ** 20)
            if not buf: break
            sha.update(buf)

    actual_checksum = sha.hexdigest()[:len(checksum)]
    if actual_checksum != checksum: raise RuntimeError

def get_demucs_model(name, repo = None):
    model_repo = LocalRepo(repo)
    return (model_repo.get_model(name) if model_repo.has_model(name) else BagOnlyRepo(repo, model_repo).get_model(name)).eval()
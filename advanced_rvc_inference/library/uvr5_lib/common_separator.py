import os
import sys
import librosa

import numpy as np
import soundfile as sf

from pydub import AudioSegment

sys.path.append(os.getcwd())

from main.library.uvr5_lib.spec_utils import normalize

class CommonSeparator:
    VOCAL_STEM = "Vocals"
    INST_STEM = "Instrumental"
    OTHER_STEM = "Other"
    BASS_STEM = "Bass"
    DRUM_STEM = "Drums"
    GUITAR_STEM = "Guitar"
    PIANO_STEM = "Piano"
    SYNTH_STEM = "Synthesizer"
    STRINGS_STEM = "Strings"
    WOODWINDS_STEM = "Woodwinds"
    BRASS_STEM = "Brass"
    WIND_INST_STEM = "Wind Inst"
    PRIMARY_STEM = "Primary Stem"
    SECONDARY_STEM = "Secondary Stem"
    LEAD_VOCAL_STEM = "lead_only"
    BV_VOCAL_STEM = "backing_only"
    NO_STEM = "No "
    STEM_PAIR_MAPPER = {VOCAL_STEM: INST_STEM, INST_STEM: VOCAL_STEM, LEAD_VOCAL_STEM: BV_VOCAL_STEM, BV_VOCAL_STEM: LEAD_VOCAL_STEM, PRIMARY_STEM: SECONDARY_STEM}
    NON_ACCOM_STEMS = (VOCAL_STEM, OTHER_STEM, BASS_STEM, DRUM_STEM, GUITAR_STEM, PIANO_STEM, SYNTH_STEM, STRINGS_STEM, WOODWINDS_STEM, BRASS_STEM, WIND_INST_STEM)

    def __init__(self, config):
        self.logger = config.get("logger")
        self.torch_device = config.get("torch_device")
        self.torch_device_cpu = config.get("torch_device_cpu")
        self.torch_device_mps = config.get("torch_device_mps")
        self.onnx_execution_provider = config.get("onnx_execution_provider")
        self.model_name = config.get("model_name")
        self.model_path = config.get("model_path")
        self.model_data = config.get("model_data")
        self.output_dir = config.get("output_dir")
        self.output_format = config.get("output_format")
        self.output_bitrate = config.get("output_bitrate")
        self.normalization_threshold = config.get("normalization_threshold")
        self.enable_denoise = config.get("enable_denoise")
        self.output_single_stem = config.get("output_single_stem")
        self.invert_using_spec = config.get("invert_using_spec")
        self.sample_rate = config.get("sample_rate")
        self.primary_stem_name = None
        self.secondary_stem_name = None

        if "training" in self.model_data and "instruments" in self.model_data["training"]:
            instruments = self.model_data["training"]["instruments"]
            if instruments:
                self.primary_stem_name = instruments[0]
                self.secondary_stem_name = instruments[1] if len(instruments) > 1 else self.secondary_stem(self.primary_stem_name)

        if self.primary_stem_name is None:
            self.primary_stem_name = self.model_data.get("primary_stem", "Vocals")
            self.secondary_stem_name = self.secondary_stem(self.primary_stem_name)

        self.is_karaoke = self.model_data.get("is_karaoke", False)
        self.is_bv_model = self.model_data.get("is_bv_model", False)
        self.bv_model_rebalance = self.model_data.get("is_bv_model_rebalanced", 0)
        self.audio_file_path = None
        self.audio_file_base = None
        self.primary_source = None
        self.secondary_source = None
        self.primary_stem_output_path = None
        self.secondary_stem_output_path = None
        self.cached_sources_map = {}

    def secondary_stem(self, primary_stem):
        primary_stem = primary_stem if primary_stem else self.NO_STEM
        return self.STEM_PAIR_MAPPER[primary_stem] if primary_stem in self.STEM_PAIR_MAPPER else primary_stem.replace(self.NO_STEM, "") if self.NO_STEM in primary_stem else f"{self.NO_STEM}{primary_stem}"

    def separate(self, audio_file_path):
        pass

    def final_process(self, stem_path, source, stem_name):
        self.write_audio(stem_path, source)
        return {stem_name: source}

    def cached_sources_clear(self):
        self.cached_sources_map = {}

    def cached_source_callback(self, model_architecture, model_name=None):
        model, sources = None, None
        mapper = self.cached_sources_map[model_architecture]
        for key, value in mapper.items():
            if model_name in key:
                model = key
                sources = value

        return model, sources

    def cached_model_source_holder(self, model_architecture, sources, model_name=None):
        self.cached_sources_map[model_architecture] = {**self.cached_sources_map.get(model_architecture, {}), **{model_name: sources}}

    def prepare_mix(self, mix):
        if not isinstance(mix, np.ndarray):
            mix, _ = librosa.load(mix, mono=False, sr=self.sample_rate)
        else:
            mix = mix.T

        if mix.ndim == 1:
            mix = np.asfortranarray([mix, mix])

        return mix

    def write_audio(self, stem_path, stem_source):
        duration_seconds = librosa.get_duration(y=librosa.load(self.audio_file_path, sr=None)[0])
        duration_hours = duration_seconds / 3600

        if duration_hours >= 1:
            self.write_audio_soundfile(stem_path, stem_source)
        else:
            self.write_audio_pydub(stem_path, stem_source)

    def write_audio_pydub(self, stem_path, stem_source):
        stem_source = normalize(wave=stem_source, max_peak=self.normalization_threshold)

        if np.max(np.abs(stem_source)) < 1e-6: return

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            stem_path = os.path.join(self.output_dir, stem_path)

        if stem_source.dtype != np.int16: stem_source = (stem_source * 32767).astype(np.int16)
        stem_source_interleaved = np.empty((2 * stem_source.shape[0],), dtype=np.int16)
        stem_source_interleaved[0::2] = stem_source[:, 0] 
        stem_source_interleaved[1::2] = stem_source[:, 1]

        audio_segment = AudioSegment(stem_source_interleaved.tobytes(), frame_rate=self.sample_rate, sample_width=stem_source.dtype.itemsize, channels=2)
        file_format = stem_path.lower().split(".")[-1]

        if file_format == "m4a": file_format = "mp4"
        elif file_format == "mka": file_format = "matroska"

        audio_segment.export(stem_path, format=file_format, bitrate="320k" if file_format == "mp3" and self.output_bitrate is None else self.output_bitrate)

    def write_audio_soundfile(self, stem_path, stem_source):
        if stem_source.shape[1] == 2:
            if stem_source.flags["F_CONTIGUOUS"]: stem_source = np.ascontiguousarray(stem_source)
            else:
                stereo_interleaved = np.empty((2 * stem_source.shape[0],), dtype=np.int16)
                stereo_interleaved[0::2] = stem_source[:, 0]
                stereo_interleaved[1::2] = stem_source[:, 1]
                stem_source = stereo_interleaved

        sf.write(stem_path, stem_source, self.sample_rate)

    def clear_file_specific_paths(self):
        self.audio_file_path = None
        self.audio_file_base = None
        self.primary_source = None
        self.secondary_source = None
        self.primary_stem_output_path = None
        self.secondary_stem_output_path = None
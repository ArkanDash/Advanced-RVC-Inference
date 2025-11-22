import os
import sys
import math
import torch
import librosa
import audioread

import numpy as np

from tqdm import tqdm

sys.path.append(os.getcwd())

from main.app.variables import configs
from main.library.uvr5_lib import spec_utils
from main.library.uvr5_lib.vr_network import nets
from main.library.uvr5_lib.vr_network import nets_new
from main.library.uvr5_lib.common_separator import CommonSeparator
from main.library.uvr5_lib.vr_network.model_param_init import ModelParameters

class VRSeparator(CommonSeparator):
    def __init__(self, common_config, arch_config):
        super().__init__(config=common_config)
        self.model_capacity = 32, 128
        self.is_vr_51_model = False

        if "nout" in self.model_data.keys() and "nout_lstm" in self.model_data.keys():
            self.model_capacity = self.model_data["nout"], self.model_data["nout_lstm"]
            self.is_vr_51_model = True

        self.model_params = ModelParameters(os.path.join(configs["binary_path"], "vr_params.bin"), f"{self.model_data['vr_model_param']}.json")
        self.enable_tta = arch_config.get("enable_tta", False)
        self.enable_post_process = arch_config.get("enable_post_process", False)
        self.post_process_threshold = arch_config.get("post_process_threshold", 0.2)
        self.batch_size = arch_config.get("batch_size", 1)
        self.window_size = arch_config.get("window_size", 512)
        self.high_end_process = arch_config.get("high_end_process", False)
        self.input_high_end_h = None
        self.input_high_end = None
        self.aggression = float(int(arch_config.get("aggression", 5)) / 100)
        self.aggressiveness = {"value": self.aggression, "split_bin": self.model_params.param["band"]["1"]["crop_stop"], "aggr_correction": self.model_params.param.get("aggr_correction")}
        self.model_samplerate = self.model_params.param["sr"]
        self.wav_subtype = "PCM_16"

    def separate(self, audio_file_path, custom_output_names=None):
        self.primary_source = None
        self.secondary_source = None
        self.audio_file_path = audio_file_path
        self.audio_file_base = os.path.splitext(os.path.basename(audio_file_path))[0]
        nn_arch_sizes = [31191, 33966, 56817, 123821, 123812, 129605, 218409, 537238, 537227]
        vr_5_1_models = [56817, 218409]
        model_size = math.ceil(os.stat(self.model_path).st_size / 1024)
        nn_arch_size = min(nn_arch_sizes, key=lambda x: abs(x - model_size))

        if nn_arch_size in vr_5_1_models or self.is_vr_51_model:
            self.model_run = nets_new.CascadedNet(self.model_params.param["bins"] * 2, nn_arch_size, nout=self.model_capacity[0], nout_lstm=self.model_capacity[1])
            self.is_vr_51_model = True
        else:
            self.model_run = nets.determine_model_capacity(self.model_params.param["bins"] * 2, nn_arch_size)

        self.model_run.load_state_dict(torch.load(self.model_path, map_location="cpu", weights_only=True))
        self.model_run.to(self.torch_device)

        y_spec, v_spec = self.inference_vr(self.loading_mix(), self.torch_device, self.aggressiveness)
        y_spec = np.nan_to_num(y_spec, nan=0.0, posinf=0.0, neginf=0.0)
        v_spec = np.nan_to_num(v_spec, nan=0.0, posinf=0.0, neginf=0.0)

        output_files = []
        if self.output_single_stem and (self.output_single_stem.lower() != self.primary_stem_name.lower() and self.output_single_stem.lower() != self.secondary_stem_name.lower()):
            self.output_single_stem = None

        if not self.output_single_stem or self.output_single_stem.lower() == self.primary_stem_name.lower():
            if not isinstance(self.primary_source, np.ndarray):
                self.primary_source = self.spec_to_wav(y_spec).T
                if not self.model_samplerate == 44100:
                    self.primary_source = librosa.resample(self.primary_source.T, orig_sr=self.model_samplerate, target_sr=44100).T

            self.primary_stem_output_path = os.path.join(f"{self.audio_file_base}_({self.primary_stem_name})_{self.model_name}.{self.output_format.lower()}")
            self.final_process(self.primary_stem_output_path, self.primary_source, self.primary_stem_name)
            output_files.append(self.primary_stem_output_path)

        if not self.output_single_stem or self.output_single_stem.lower() == self.secondary_stem_name.lower():
            if not isinstance(self.secondary_source, np.ndarray):
                self.secondary_source = self.spec_to_wav(v_spec).T
                if not self.model_samplerate == 44100:
                    self.secondary_source = librosa.resample(self.secondary_source.T, orig_sr=self.model_samplerate, target_sr=44100).T

            self.secondary_stem_output_path = os.path.join(f"{self.audio_file_base}_({self.secondary_stem_name})_{self.model_name}.{self.output_format.lower()}")
            self.final_process(self.secondary_stem_output_path, self.secondary_source, self.secondary_stem_name)
            output_files.append(self.secondary_stem_output_path)

        return output_files

    def loading_mix(self):
        X_wave, X_spec_s = {}, {}
        bands_n = len(self.model_params.param["band"])

        audio_file = spec_utils.write_array_to_mem(self.audio_file_path, subtype=self.wav_subtype)
        is_mp3 = audio_file.endswith(".mp3") if isinstance(audio_file, str) else False

        self.logger.debug(f"loading_mix iteraring through {bands_n} bands")
        for d in tqdm(range(bands_n, 0, -1)):
            bp = self.model_params.param["band"][str(d)]

            if d == bands_n:
                X_wave[d], _ = librosa.load(audio_file, sr=bp["sr"], mono=False, dtype=np.float32, res_type="soxr_vhq")
                X_spec_s[d] = spec_utils.wave_to_spectrogram(X_wave[d], bp["hl"], bp["n_fft"], self.model_params, band=d, is_v51_model=self.is_vr_51_model)

                if not np.any(X_wave[d]) and is_mp3:
                    X_wave[d] = rerun_mp3(audio_file, bp["sr"])

                if X_wave[d].ndim == 1:
                    X_wave[d] = np.asarray([X_wave[d], X_wave[d]])
            else:
                X_wave[d] = librosa.resample(X_wave[d + 1], orig_sr=self.model_params.param["band"][str(d + 1)]["sr"], target_sr=bp["sr"], res_type="soxr_vhq")
                X_spec_s[d] = spec_utils.wave_to_spectrogram(X_wave[d], bp["hl"], bp["n_fft"], self.model_params, band=d, is_v51_model=self.is_vr_51_model)

            if d == bands_n and self.high_end_process:
                self.input_high_end_h = (bp["n_fft"] // 2 - bp["crop_stop"]) + (self.model_params.param["pre_filter_stop"] - self.model_params.param["pre_filter_start"])
                self.input_high_end = X_spec_s[d][:, bp["n_fft"] // 2 - self.input_high_end_h : bp["n_fft"] // 2, :]

        X_spec = spec_utils.combine_spectrograms(X_spec_s, self.model_params, is_v51_model=self.is_vr_51_model)
        del X_wave, X_spec_s, audio_file

        return X_spec

    def inference_vr(self, X_spec, device, aggressiveness):
        def _execute(X_mag_pad, roi_size):
            X_dataset = []
            patches = (X_mag_pad.shape[2] - 2 * self.model_run.offset) // roi_size

            for i in tqdm(range(patches)):
                start = i * roi_size
                X_mag_window = X_mag_pad[:, :, start : start + self.window_size]
                X_dataset.append(X_mag_window)

            total_iterations = patches // self.batch_size if not self.enable_tta else (patches // self.batch_size) * 2
            X_dataset = np.asarray(X_dataset)
            self.model_run.eval()

            with torch.no_grad():
                mask = []

                for i in tqdm(range(0, patches, self.batch_size)):
                    X_batch = X_dataset[i : i + self.batch_size]
                    X_batch = torch.from_numpy(X_batch).to(device)
                    pred = self.model_run.predict_mask(X_batch)
    
                    if not pred.size()[3] > 0:
                        raise ValueError

                    pred = pred.detach().cpu().numpy()
                    pred = np.concatenate(pred, axis=2)
                    mask.append(pred)

                if len(mask) == 0:
                    raise ValueError

                mask = np.concatenate(mask, axis=2)

            return mask

        def postprocess(mask, X_mag, X_phase):
            is_non_accom_stem = False
            for stem in CommonSeparator.NON_ACCOM_STEMS:
                if stem == self.primary_stem_name: is_non_accom_stem = True

            mask = spec_utils.adjust_aggr(mask, is_non_accom_stem, aggressiveness)
            if self.enable_post_process: mask = spec_utils.merge_artifacts(mask, thres=self.post_process_threshold)

            y_spec = mask * X_mag * np.exp(1.0j * X_phase)
            v_spec = (1 - mask) * X_mag * np.exp(1.0j * X_phase)

            return y_spec, v_spec

        X_mag, X_phase = spec_utils.preprocess(X_spec)
        n_frame = X_mag.shape[2]
        pad_l, pad_r, roi_size = spec_utils.make_padding(n_frame, self.window_size, self.model_run.offset)
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")
        X_mag_pad /= X_mag_pad.max()
        mask = _execute(X_mag_pad, roi_size)

        if self.enable_tta:
            pad_l += roi_size // 2
            pad_r += roi_size // 2
            X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")
            X_mag_pad /= X_mag_pad.max()
            mask_tta = _execute(X_mag_pad, roi_size)
            mask_tta = mask_tta[:, :, roi_size // 2 :]
            mask = (mask[:, :, :n_frame] + mask_tta[:, :, :n_frame]) * 0.5
        else:
            mask = mask[:, :, :n_frame]

        y_spec, v_spec = postprocess(mask, X_mag, X_phase)

        return y_spec, v_spec

    def spec_to_wav(self, spec):
        if self.high_end_process and isinstance(self.input_high_end, np.ndarray) and self.input_high_end_h:
            input_high_end_ = spec_utils.mirroring("mirroring", spec, self.input_high_end, self.model_params)
            wav = spec_utils.cmb_spectrogram_to_wave(spec, self.model_params, self.input_high_end_h, input_high_end_, is_v51_model=self.is_vr_51_model)
        else:
            wav = spec_utils.cmb_spectrogram_to_wave(spec, self.model_params, is_v51_model=self.is_vr_51_model)

        return wav

def rerun_mp3(audio_file, sample_rate=44100):
    with audioread.audio_open(audio_file) as f:
        track_length = int(f.duration)

    return librosa.load(audio_file, duration=track_length, mono=False, sr=sample_rate)[0]
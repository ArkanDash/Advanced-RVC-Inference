import os
import sys
import onnx
import torch
import platform
import warnings
import onnx2torch

import numpy as np
import onnxruntime as ort

from tqdm import tqdm

sys.path.append(os.getcwd())

from main.library.uvr5_lib import spec_utils
from main.library.uvr5_lib.common_separator import CommonSeparator

warnings.filterwarnings("ignore")

class MDXSeparator(CommonSeparator):
    def __init__(self, common_config, arch_config):
        super().__init__(config=common_config)
        self.segment_size = arch_config.get("segment_size")
        self.overlap = arch_config.get("overlap")
        self.batch_size = arch_config.get("batch_size", 1)
        self.hop_length = arch_config.get("hop_length")
        self.enable_denoise = arch_config.get("enable_denoise")
        self.compensate = self.model_data["compensate"]
        self.dim_f = self.model_data["mdx_dim_f_set"]
        self.dim_t = 2 ** self.model_data["mdx_dim_t_set"]
        self.n_fft = self.model_data["mdx_n_fft_scale_set"]
        self.config_yaml = self.model_data.get("config_yaml", None)
        self.load_model()
        self.n_bins = 0
        self.trim = 0
        self.chunk_size = 0
        self.gen_size = 0
        self.stft = None
        self.primary_source = None
        self.secondary_source = None
        self.audio_file_path = None
        self.audio_file_base = None

    def load_model(self):
        if self.segment_size == self.dim_t:
            ort_session_options = ort.SessionOptions()
            ort_session_options.log_severity_level = 3
            ort_inference_session = ort.InferenceSession(self.model_path, providers=self.onnx_execution_provider, sess_options=ort_session_options)
            self.model_run = lambda spek: ort_inference_session.run(None, {"input": spek.cpu().numpy()})[0]
        else:
            self.model_run = onnx2torch.convert(onnx.load(self.model_path)) if platform.system() == 'Windows' else onnx2torch.convert(self.model_path)
            self.model_run.to(self.torch_device).eval()

    def separate(self, audio_file_path):
        self.audio_file_path = audio_file_path
        self.audio_file_base = os.path.splitext(os.path.basename(audio_file_path))[0]
        mix = self.prepare_mix(self.audio_file_path)
        mix = spec_utils.normalize(wave=mix, max_peak=self.normalization_threshold)
        source = self.demix(mix)
        output_files = []

        if not isinstance(self.primary_source, np.ndarray):
            self.primary_source = spec_utils.normalize(wave=source, max_peak=self.normalization_threshold).T

        if not isinstance(self.secondary_source, np.ndarray):
            raw_mix = self.demix(mix, is_match_mix=True)

            if self.invert_using_spec:
                self.secondary_source = spec_utils.invert_stem(raw_mix, source)
            else:
                self.secondary_source = mix.T - source.T

        if not self.output_single_stem or self.output_single_stem.lower() == self.secondary_stem_name.lower():
            self.secondary_stem_output_path = os.path.join(f"{self.audio_file_base}_({self.secondary_stem_name})_{self.model_name}.{self.output_format.lower()}")
            self.final_process(self.secondary_stem_output_path, self.secondary_source, self.secondary_stem_name)
            output_files.append(self.secondary_stem_output_path)

        if not self.output_single_stem or self.output_single_stem.lower() == self.primary_stem_name.lower():
            self.primary_stem_output_path = os.path.join(f"{self.audio_file_base}_({self.primary_stem_name})_{self.model_name}.{self.output_format.lower()}")
            if not isinstance(self.primary_source, np.ndarray): self.primary_source = source.T

            self.final_process(self.primary_stem_output_path, self.primary_source, self.primary_stem_name)
            output_files.append(self.primary_stem_output_path)

        return output_files

    def initialize_model_settings(self):
        self.n_bins = self.n_fft // 2 + 1
        self.trim = self.n_fft // 2
        self.chunk_size = self.hop_length * (self.segment_size - 1)
        self.gen_size = self.chunk_size - 2 * self.trim
        self.stft = STFT(self.n_fft, self.hop_length, self.dim_f, self.torch_device)

    def initialize_mix(self, mix, is_ckpt=False):
        if is_ckpt:
            pad = self.gen_size + self.trim - (mix.shape[-1] % self.gen_size)
            mixture = np.concatenate((np.zeros((2, self.trim), dtype="float32"), mix, np.zeros((2, pad), dtype="float32")), 1)

            num_chunks = mixture.shape[-1] // self.gen_size
            mix_waves = [mixture[:, i * self.gen_size : i * self.gen_size + self.chunk_size] for i in range(num_chunks)]
        else:
            mix_waves = []
            n_sample = mix.shape[1]

            pad = self.gen_size - n_sample % self.gen_size
            mix_p = np.concatenate((np.zeros((2, self.trim)), mix, np.zeros((2, pad)), np.zeros((2, self.trim))), 1)

            i = 0
            while i < n_sample + pad:
                mix_waves.append(np.array(mix_p[:, i : i + self.chunk_size]))
                i += self.gen_size

        mix_waves_tensor = torch.tensor(mix_waves, dtype=torch.float32).to(self.torch_device)
        return mix_waves_tensor, pad

    def demix(self, mix, is_match_mix=False):
        self.initialize_model_settings()
        tar_waves_ = []

        if is_match_mix:
            chunk_size = self.hop_length * (self.segment_size - 1)
            overlap = 0.02
        else:
            chunk_size = self.chunk_size
            overlap = self.overlap

        gen_size = chunk_size - 2 * self.trim
        mixture = np.concatenate((np.zeros((2, self.trim), dtype="float32"), mix, np.zeros((2, gen_size + self.trim - ((mix.shape[-1]) % gen_size)), dtype="float32")), 1)
        step = int((1 - overlap) * chunk_size)

        result = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
        divider = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
        total = 0

        for i in tqdm(range(0, mixture.shape[-1], step), ncols=100, unit="f"):
            total += 1
            start = i
            end = min(i + chunk_size, mixture.shape[-1])

            chunk_size_actual = end - start
            window = None

            if overlap != 0:
                window = np.hanning(chunk_size_actual)
                window = np.tile(window[None, None, :], (1, 2, 1))

            mix_part_ = mixture[:, start:end]
            
            if end != i + chunk_size:
                pad_size = (i + chunk_size) - end
                mix_part_ = np.concatenate((mix_part_, np.zeros((2, pad_size), dtype="float32")), axis=-1)

            mix_waves = torch.tensor([mix_part_], dtype=torch.float32).to(self.torch_device).split(self.batch_size)

            with torch.no_grad():
                batches_processed = 0
                
                for mix_wave in mix_waves:
                    batches_processed += 1
                    tar_waves = self.run_model(mix_wave, is_match_mix=is_match_mix)

                    if window is not None:
                        tar_waves[..., :chunk_size_actual] *= window
                        divider[..., start:end] += window
                    else: divider[..., start:end] += 1

                    result[..., start:end] += tar_waves[..., : end - start]

        tar_waves = result / divider
        tar_waves_.append(tar_waves)
        tar_waves = np.concatenate(np.vstack(tar_waves_)[:, :, self.trim : -self.trim], axis=-1)[:, : mix.shape[-1]]

        source = tar_waves[:, 0:None]

        if not is_match_mix:
            source *= self.compensate

        return source

    def run_model(self, mix, is_match_mix=False):
        spek = self.stft(mix.to(self.torch_device))
        spek[:, :, :3, :] *= 0

        if is_match_mix:
            spec_pred = spek.cpu().numpy()
        else:
            if self.enable_denoise:
                spec_pred_neg = self.model_run(-spek)  
                spec_pred_pos = self.model_run(spek)
                spec_pred = (spec_pred_neg * -0.5) + (spec_pred_pos * 0.5)
            else:
                spec_pred = self.model_run(spek)

        result = self.stft.inverse(torch.tensor(spec_pred).to(self.torch_device)).cpu().detach().numpy()
        return result

class STFT:
    def __init__(self, n_fft, hop_length, dim_f, device):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.dim_f = dim_f
        self.device = device
        self.hann_window = torch.hann_window(window_length=self.n_fft, periodic=True)

    def __call__(self, input_tensor):
        is_non_standard_device = not input_tensor.device.type in ["cuda", "cpu"]
        if is_non_standard_device: input_tensor = input_tensor.cpu()

        batch_dimensions = input_tensor.shape[:-2]
        channel_dim, time_dim = input_tensor.shape[-2:]

        permuted_stft_output = torch.stft(input_tensor.reshape([-1, time_dim]), n_fft=self.n_fft, hop_length=self.hop_length, window=self.hann_window.to(input_tensor.device), center=True, return_complex=False).permute([0, 3, 1, 2])
        final_output = permuted_stft_output.reshape([*batch_dimensions, channel_dim, 2, -1, permuted_stft_output.shape[-1]]).reshape([*batch_dimensions, channel_dim * 2, -1, permuted_stft_output.shape[-1]])

        if is_non_standard_device: final_output = final_output.to(self.device)
        return final_output[..., : self.dim_f, :]

    def pad_frequency_dimension(self, input_tensor, batch_dimensions, channel_dim, freq_dim, time_dim, num_freq_bins):
        return torch.cat([input_tensor, torch.zeros([*batch_dimensions, channel_dim, num_freq_bins - freq_dim, time_dim]).to(input_tensor.device)], -2)

    def calculate_inverse_dimensions(self, input_tensor):
        channel_dim, freq_dim, time_dim = input_tensor.shape[-3:]

        return input_tensor.shape[:-3], channel_dim, freq_dim, time_dim, self.n_fft // 2 + 1

    def prepare_for_istft(self, padded_tensor, batch_dimensions, channel_dim, num_freq_bins, time_dim):
        permuted_tensor = padded_tensor.reshape([*batch_dimensions, channel_dim // 2, 2, num_freq_bins, time_dim]).reshape([-1, 2, num_freq_bins, time_dim]).permute([0, 2, 3, 1])

        return permuted_tensor[..., 0] + permuted_tensor[..., 1] * 1.0j

    def inverse(self, input_tensor):
        is_non_standard_device = not input_tensor.device.type in ["cuda", "cpu"]
        if is_non_standard_device: input_tensor = input_tensor.cpu()

        batch_dimensions, channel_dim, freq_dim, time_dim, num_freq_bins = self.calculate_inverse_dimensions(input_tensor)
        final_output = torch.istft(self.prepare_for_istft(self.pad_frequency_dimension(input_tensor, batch_dimensions, channel_dim, freq_dim, time_dim, num_freq_bins), batch_dimensions, channel_dim, num_freq_bins, time_dim), n_fft=self.n_fft, hop_length=self.hop_length, window=self.hann_window.to(input_tensor.device), center=True).reshape([*batch_dimensions, 2, -1])

        if is_non_standard_device: final_output = final_output.to(self.device)
        return final_output
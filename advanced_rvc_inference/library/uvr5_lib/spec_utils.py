import os
import six
import sys
import librosa
import tempfile
import platform
import audioread
import subprocess

import numpy as np
import soundfile as sf

from scipy.signal import correlate, hilbert

sys.path.append(os.getcwd())

from main.app.variables import translations, logger

OPERATING_SYSTEM = platform.system()
SYSTEM_ARCH = platform.platform()
SYSTEM_PROC = platform.processor()
ARM = "arm"
AUTO_PHASE = "Automatic"
POSITIVE_PHASE = "Positive Phase"
NEGATIVE_PHASE = "Negative Phase"
NONE_P = ("None",)
BASE_PATH_RUB = sys._MEIPASS if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
DEVNULL = open(os.devnull, 'w') if six.PY2 else subprocess.DEVNULL
MAX_SPEC = "Max Spec"
MIN_SPEC = "Min Spec"
AVERAGE = "Average"

is_macos = False
progress_value, last_update_time = 0, 0

if OPERATING_SYSTEM == "Darwin":
    wav_resolution = "polyphase" if SYSTEM_PROC == ARM or ARM in SYSTEM_ARCH else "sinc_fastest"
    wav_resolution_float_resampling = "kaiser_best" if SYSTEM_PROC == ARM or ARM in SYSTEM_ARCH else wav_resolution
    is_macos = True
else:
    wav_resolution = "sinc_fastest"
    wav_resolution_float_resampling = wav_resolution

def crop_center(h1, h2):
    h1_shape = h1.size()
    h2_shape = h2.size()

    if h1_shape[3] == h2_shape[3]: return h1
    elif h1_shape[3] < h2_shape[3]: raise ValueError("h1_shape[3] > h2_shape[3]")

    s_time = (h1_shape[3] - h2_shape[3]) // 2

    h1 = h1[:, :, :, s_time:s_time + h2_shape[3]]
    return h1

def preprocess(X_spec):
    return np.abs(X_spec), np.angle(X_spec)

def make_padding(width, cropsize, offset):
    roi_size = cropsize - offset * 2

    if roi_size == 0: roi_size = cropsize
    return offset, roi_size - (width % roi_size) + offset, roi_size

def normalize(wave, max_peak=1.0):
    maxv = np.abs(wave).max()

    if maxv > max_peak: wave *= max_peak / maxv
    return wave

def auto_transpose(audio_array):
    if audio_array.shape[1] == 2: return audio_array.T
    return audio_array

def write_array_to_mem(audio_data, subtype):
    if isinstance(audio_data, np.ndarray):
        import io

        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio_data, 44100, subtype=subtype, format="WAV")

        audio_buffer.seek(0)
        return audio_buffer
    else: return audio_data

def spectrogram_to_image(spec, mode="magnitude"):
    if mode == "magnitude": y = np.log10((np.abs(spec) if np.iscomplexobj(spec) else spec)**2 + 1e-8)
    elif mode == "phase": y = np.angle(spec) if np.iscomplexobj(spec) else spec

    y -= y.min()
    y *= 255 / y.max()
    img = np.uint8(y)

    if y.ndim == 3:
        img = img.transpose(1, 2, 0)
        img = np.concatenate([np.max(img, axis=2, keepdims=True), img], axis=2)

    return img

def reduce_vocal_aggressively(X, y, softmask):
    y_mag_tmp = np.abs(y)
    v_mag_tmp = np.abs(X - y)

    return np.clip(y_mag_tmp - v_mag_tmp * (v_mag_tmp > y_mag_tmp) * softmask, 0, np.inf) * np.exp(1.0j * np.angle(y))

def merge_artifacts(y_mask, thres=0.01, min_range=64, fade_size=32):
    mask = y_mask

    try:
        if min_range < fade_size * 2: raise ValueError("min_range >= fade_size * 2")

        idx = np.where(y_mask.min(axis=(0, 1)) > thres)[0]
        start_idx = np.insert(idx[np.where(np.diff(idx) != 1)[0] + 1], 0, idx[0])
        end_idx = np.append(idx[np.where(np.diff(idx) != 1)[0]], idx[-1])
        artifact_idx = np.where(end_idx - start_idx > min_range)[0]
        weight = np.zeros_like(y_mask)

        if len(artifact_idx) > 0:
            start_idx = start_idx[artifact_idx]
            end_idx = end_idx[artifact_idx]
            old_e = None

            for s, e in zip(start_idx, end_idx):
                if old_e is not None and s - old_e < fade_size: s = old_e - fade_size * 2

                if s != 0: weight[:, :, s : s + fade_size] = np.linspace(0, 1, fade_size)
                else: s -= fade_size

                if e != y_mask.shape[2]: weight[:, :, e - fade_size : e] = np.linspace(1, 0, fade_size)
                else: e += fade_size

                weight[:, :, s + fade_size : e - fade_size] = 1
                old_e = e

        v_mask = 1 - y_mask
        y_mask += weight * v_mask
        mask = y_mask
    except Exception as e:
        import traceback
        logger.error(f'{translations["not_success"]} {type(e).__name__}: {e}\n{traceback.format_exc()}')

    return mask

def align_wave_head_and_tail(a, b):
    l = min([a[0].size, b[0].size])
    return a[:l, :l], b[:l, :l]

def convert_channels(spec, mp, band):
    cc = mp.param["band"][str(band)].get("convert_channels")

    if "mid_side_c" == cc:
        spec_left = np.add(spec[0], spec[1] * 0.25)
        spec_right = np.subtract(spec[1], spec[0] * 0.25)
    elif "mid_side" == cc:
        spec_left = np.add(spec[0], spec[1]) / 2
        spec_right = np.subtract(spec[0], spec[1])
    elif "stereo_n" == cc:
        spec_left = np.add(spec[0], spec[1] * 0.25) / 0.9375
        spec_right = np.add(spec[1], spec[0] * 0.25) / 0.9375
    else: return spec

    return np.asfortranarray([spec_left, spec_right])

def combine_spectrograms(specs, mp, is_v51_model=False):
    l = min([specs[i].shape[2] for i in specs])
    spec_c = np.zeros(shape=(2, mp.param["bins"] + 1, l), dtype=np.complex64)
    offset = 0
    bands_n = len(mp.param["band"])

    for d in range(1, bands_n + 1):
        h = mp.param["band"][str(d)]["crop_stop"] - mp.param["band"][str(d)]["crop_start"]
        spec_c[:, offset : offset + h, :l] = specs[d][:, mp.param["band"][str(d)]["crop_start"] : mp.param["band"][str(d)]["crop_stop"], :l]
        offset += h

    if offset > mp.param["bins"]: raise ValueError("offset > mp.param['bins']")

    if mp.param["pre_filter_start"] > 0:
        if is_v51_model: spec_c *= get_lp_filter_mask(spec_c.shape[1], mp.param["pre_filter_start"], mp.param["pre_filter_stop"])
        else:
            if bands_n == 1: spec_c = fft_lp_filter(spec_c, mp.param["pre_filter_start"], mp.param["pre_filter_stop"])
            else:
                import math
                gp = 1

                for b in range(mp.param["pre_filter_start"] + 1, mp.param["pre_filter_stop"]):
                    g = math.pow(10, -(b - mp.param["pre_filter_start"]) * (3.5 - gp) / 20.0)
                    gp = g
                    spec_c[:, b, :] *= g

    return np.asfortranarray(spec_c)

def wave_to_spectrogram(wave, hop_length, n_fft, mp, band, is_v51_model=False):
    if wave.ndim == 1: wave = np.asfortranarray([wave, wave])

    if not is_v51_model:
        if mp.param["reverse"]:
            wave_left = np.flip(np.asfortranarray(wave[0]))
            wave_right = np.flip(np.asfortranarray(wave[1]))
        elif mp.param["mid_side"]:
            wave_left = np.asfortranarray(np.add(wave[0], wave[1]) / 2)
            wave_right = np.asfortranarray(np.subtract(wave[0], wave[1]))
        elif mp.param["mid_side_b2"]:
            wave_left = np.asfortranarray(np.add(wave[1], wave[0] * 0.5))
            wave_right = np.asfortranarray(np.subtract(wave[0], wave[1] * 0.5))
        else:
            wave_left = np.asfortranarray(wave[0])
            wave_right = np.asfortranarray(wave[1])
    else:
        wave_left = np.asfortranarray(wave[0])
        wave_right = np.asfortranarray(wave[1])

    spec_left = librosa.stft(wave_left, n_fft=n_fft, hop_length=hop_length)
    spec_right = librosa.stft(wave_right, n_fft=n_fft, hop_length=hop_length)

    spec = np.asfortranarray([spec_left, spec_right])

    if is_v51_model: spec = convert_channels(spec, mp, band)
    return spec

def spectrogram_to_wave(spec, hop_length=1024, mp={}, band=0, is_v51_model=True):
    spec_left = np.asfortranarray(spec[0])
    spec_right = np.asfortranarray(spec[1])

    wave_left = librosa.istft(spec_left, hop_length=hop_length)
    wave_right = librosa.istft(spec_right, hop_length=hop_length)

    if is_v51_model:
        cc = mp.param["band"][str(band)].get("convert_channels")

        if "mid_side_c" == cc: return np.asfortranarray([np.subtract(wave_left / 1.0625, wave_right / 4.25), np.add(wave_right / 1.0625, wave_left / 4.25)])
        elif "mid_side" == cc: return np.asfortranarray([np.add(wave_left, wave_right / 2), np.subtract(wave_left, wave_right / 2)])
        elif "stereo_n" == cc: return np.asfortranarray([np.subtract(wave_left, wave_right * 0.25), np.subtract(wave_right, wave_left * 0.25)])
    else:
        if mp.param["reverse"]: return np.asfortranarray([np.flip(wave_left), np.flip(wave_right)])
        elif mp.param["mid_side"]: return np.asfortranarray([np.add(wave_left, wave_right / 2), np.subtract(wave_left, wave_right / 2)])
        elif mp.param["mid_side_b2"]: return np.asfortranarray([np.add(wave_right / 1.25, 0.4 * wave_left), np.subtract(wave_left / 1.25, 0.4 * wave_right)])

    return np.asfortranarray([wave_left, wave_right])

def cmb_spectrogram_to_wave(spec_m, mp, extra_bins_h=None, extra_bins=None, is_v51_model=False):
    bands_n = len(mp.param["band"])
    offset = 0

    for d in range(1, bands_n + 1):
        bp = mp.param["band"][str(d)]
        spec_s = np.zeros(shape=(2, bp["n_fft"] // 2 + 1, spec_m.shape[2]), dtype=complex)
        h = bp["crop_stop"] - bp["crop_start"]
        spec_s[:, bp["crop_start"] : bp["crop_stop"], :] = spec_m[:, offset : offset + h, :]
        offset += h

        if d == bands_n:
            if extra_bins_h:  
                max_bin = bp["n_fft"] // 2
                spec_s[:, max_bin - extra_bins_h : max_bin, :] = extra_bins[:, :extra_bins_h, :]

            if bp["hpf_start"] > 0:
                if is_v51_model: spec_s *= get_hp_filter_mask(spec_s.shape[1], bp["hpf_start"], bp["hpf_stop"] - 1)
                else: spec_s = fft_hp_filter(spec_s, bp["hpf_start"], bp["hpf_stop"] - 1)

            wave = spectrogram_to_wave(spec_s, bp["hl"], mp, d, is_v51_model) if bands_n == 1 else np.add(wave, spectrogram_to_wave(spec_s, bp["hl"], mp, d, is_v51_model))
        else:
            sr = mp.param["band"][str(d + 1)]["sr"]
            if d == 1: 
                if is_v51_model: spec_s *= get_lp_filter_mask(spec_s.shape[1], bp["lpf_start"], bp["lpf_stop"])
                else: spec_s = fft_lp_filter(spec_s, bp["lpf_start"], bp["lpf_stop"])

                try:
                    wave = librosa.resample(spectrogram_to_wave(spec_s, bp["hl"], mp, d, is_v51_model), orig_sr=bp["sr"], target_sr=sr, res_type="soxr_vhq")
                except ValueError as e:
                    logger.error(f"{translations['resample_error']}: {e}")
                    logger.error(f"{translations['shapes']} Spec_s: {spec_s.shape}, SR: {sr}, {translations['wav_resolution']}: {wav_resolution}")
            else:  
                if is_v51_model:
                    spec_s *= get_hp_filter_mask(spec_s.shape[1], bp["hpf_start"], bp["hpf_stop"] - 1)
                    spec_s *= get_lp_filter_mask(spec_s.shape[1], bp["lpf_start"], bp["lpf_stop"])
                else:
                    spec_s = fft_hp_filter(spec_s, bp["hpf_start"], bp["hpf_stop"] - 1)
                    spec_s = fft_lp_filter(spec_s, bp["lpf_start"], bp["lpf_stop"])

                try:
                    wave = librosa.resample(np.add(wave, spectrogram_to_wave(spec_s, bp["hl"], mp, d, is_v51_model)), orig_sr=bp["sr"], target_sr=sr, res_type="soxr_vhq")
                except ValueError as e:
                    logger.error(f"{translations['resample_error']}: {e}")
                    logger.error(f"{translations['shapes']} Spec_s: {spec_s.shape}, SR: {sr}, {translations['wav_resolution']}: {wav_resolution}")

    return wave

def get_lp_filter_mask(n_bins, bin_start, bin_stop):
    return np.concatenate([np.ones((bin_start - 1, 1)), np.linspace(1, 0, bin_stop - bin_start + 1)[:, None], np.zeros((n_bins - bin_stop, 1))], axis=0)

def get_hp_filter_mask(n_bins, bin_start, bin_stop):
    return np.concatenate([np.zeros((bin_stop + 1, 1)), np.linspace(0, 1, 1 + bin_start - bin_stop)[:, None], np.ones((n_bins - bin_start - 2, 1))], axis=0)

def fft_lp_filter(spec, bin_start, bin_stop):
    g = 1.0

    for b in range(bin_start, bin_stop):
        g -= 1 / (bin_stop - bin_start)
        spec[:, b, :] = g * spec[:, b, :]

    spec[:, bin_stop:, :] *= 0
    return spec

def fft_hp_filter(spec, bin_start, bin_stop):
    g = 1.0

    for b in range(bin_start, bin_stop, -1):
        g -= 1 / (bin_start - bin_stop)
        spec[:, b, :] = g * spec[:, b, :]

    spec[:, 0 : bin_stop + 1, :] *= 0
    return spec

def spectrogram_to_wave_old(spec, hop_length=1024):
    if spec.ndim == 2: wave = librosa.istft(spec, hop_length=hop_length)
    elif spec.ndim == 3: wave = np.asfortranarray([librosa.istft(np.asfortranarray(spec[0]), hop_length=hop_length), librosa.istft(np.asfortranarray(spec[1]), hop_length=hop_length)])

    return wave

def wave_to_spectrogram_old(wave, hop_length, n_fft):
    return np.asfortranarray([librosa.stft(np.asfortranarray(wave[0]), n_fft=n_fft, hop_length=hop_length), librosa.stft(np.asfortranarray(wave[1]), n_fft=n_fft, hop_length=hop_length)])

def mirroring(a, spec_m, input_high_end, mp):
    if "mirroring" == a:
        mirror = np.flip(np.abs(spec_m[:, mp.param["pre_filter_start"] - 10 - input_high_end.shape[1] : mp.param["pre_filter_start"] - 10, :]), 1) * np.exp(1.0j * np.angle(input_high_end))

        return np.where(np.abs(input_high_end) <= np.abs(mirror), input_high_end, mirror)

    if "mirroring2" == a:
        mi = np.multiply(np.flip(np.abs(spec_m[:, mp.param["pre_filter_start"] - 10 - input_high_end.shape[1] : mp.param["pre_filter_start"] - 10, :]), 1), input_high_end * 1.7)

        return np.where(np.abs(input_high_end) <= np.abs(mi), input_high_end, mi)

def adjust_aggr(mask, is_non_accom_stem, aggressiveness):
    aggr = aggressiveness["value"] * 2

    if aggr != 0:
        if is_non_accom_stem:
            aggr = 1 - aggr

        if np.any(aggr > 10) or np.any(aggr < -10): logger.warning(f"{translations['warnings']}: {aggr}")

        aggr = [aggr, aggr]

        if aggressiveness["aggr_correction"] is not None:
            aggr[0] += aggressiveness["aggr_correction"]["left"]
            aggr[1] += aggressiveness["aggr_correction"]["right"]

        for ch in range(2):
            mask[ch, : aggressiveness["split_bin"]] = np.power(mask[ch, : aggressiveness["split_bin"]], 1 + aggr[ch] / 3)
            mask[ch, aggressiveness["split_bin"] :] = np.power(mask[ch, aggressiveness["split_bin"] :], 1 + aggr[ch])

    return mask

def stft(wave, nfft, hl):
    return np.asfortranarray([librosa.stft(np.asfortranarray(wave[0]), n_fft=nfft, hop_length=hl), librosa.stft(np.asfortranarray(wave[1]), n_fft=nfft, hop_length=hl)])

def istft(spec, hl):
    return np.asfortranarray([librosa.istft(np.asfortranarray(spec[0]), hop_length=hl), librosa.istft(np.asfortranarray(spec[1]), hop_length=hl)])

def spec_effects(wave, algorithm="Default", value=None):
    if np.isnan(wave).any() or np.isinf(wave).any(): logger.warning(f"{translations['warnings_2']}: {wave.shape}")
    spec = [stft(wave[0], 2048, 1024), stft(wave[1], 2048, 1024)]

    if algorithm == "Min_Mag": wave = istft(np.where(np.abs(spec[1]) <= np.abs(spec[0]), spec[1], spec[0]), 1024)
    elif algorithm == "Max_Mag": wave = istft(np.where(np.abs(spec[1]) >= np.abs(spec[0]), spec[1], spec[0]), 1024)
    elif algorithm == "Default": wave = (wave[1] * value) + (wave[0] * (1 - value))
    elif algorithm == "Invert_p":
        X_mag, y_mag = np.abs(spec[0]), np.abs(spec[1])
        wave = istft(spec[1] - np.where(X_mag >= y_mag, X_mag, y_mag) * np.exp(1.0j * np.angle(spec[0])), 1024)

    return wave

def spectrogram_to_wave_no_mp(spec, n_fft=2048, hop_length=1024):
    wave = librosa.istft(spec, n_fft=n_fft, hop_length=hop_length)
    if wave.ndim == 1: wave = np.asfortranarray([wave, wave])

    return wave

def wave_to_spectrogram_no_mp(wave):
    spec = librosa.stft(wave, n_fft=2048, hop_length=1024)

    if spec.ndim == 1: spec = np.asfortranarray([spec, spec])
    return spec

def invert_audio(specs, invert_p=True):
    ln = min([specs[0].shape[2], specs[1].shape[2]])
    specs[0] = specs[0][:, :, :ln]
    specs[1] = specs[1][:, :, :ln]

    if invert_p:
        X_mag, y_mag = np.abs(specs[0]), np.abs(specs[1])
        v_spec = specs[1] - np.where(X_mag >= y_mag, X_mag, y_mag) * np.exp(1.0j * np.angle(specs[0]))
    else:
        specs[1] = reduce_vocal_aggressively(specs[0], specs[1], 0.2)
        v_spec = specs[0] - specs[1]

    return v_spec

def invert_stem(mixture, stem):
    return -spectrogram_to_wave_no_mp(invert_audio([wave_to_spectrogram_no_mp(mixture), wave_to_spectrogram_no_mp(stem)])).T

def ensembling(a, inputs, is_wavs=False):
    for i in range(1, len(inputs)):
        if i == 1: input = inputs[0]

        if is_wavs:
            ln = min([input.shape[1], inputs[i].shape[1]])
            input = input[:, :ln]
            inputs[i] = inputs[i][:, :ln]
        else:
            ln = min([input.shape[2], inputs[i].shape[2]])
            input = input[:, :, :ln]
            inputs[i] = inputs[i][:, :, :ln]

        if MIN_SPEC == a: input = np.where(np.abs(inputs[i]) <= np.abs(input), inputs[i], input)
        if MAX_SPEC == a: input = np.where(np.abs(inputs[i]) >= np.abs(input), inputs[i], input)

    return input

def ensemble_for_align(waves):
    specs = []

    for wav in waves:
        spec = wave_to_spectrogram_no_mp(wav.T)
        specs.append(spec)

    wav_aligned = spectrogram_to_wave_no_mp(ensembling(MIN_SPEC, specs)).T
    wav_aligned = match_array_shapes(wav_aligned, waves[1], is_swap=True)

    return wav_aligned

def ensemble_inputs(audio_input, algorithm, is_normalization, wav_type_set, save_path, is_wave=False, is_array=False):
    wavs_ = []

    if algorithm == AVERAGE:
        output = average_audio(audio_input)
        samplerate = 44100
    else:
        specs = []

        for i in range(len(audio_input)):
            wave, samplerate = librosa.load(audio_input[i], mono=False, sr=44100)
            wavs_.append(wave)
            specs.append( wave if is_wave else wave_to_spectrogram_no_mp(wave))

        wave_shapes = [w.shape[1] for w in wavs_]
        target_shape = wavs_[wave_shapes.index(max(wave_shapes))]

        output = ensembling(algorithm, specs, is_wavs=True) if is_wave else spectrogram_to_wave_no_mp(ensembling(algorithm, specs))
        output = to_shape(output, target_shape.shape)

    sf.write(save_path, normalize(output.T, is_normalization), samplerate, subtype=wav_type_set)

def to_shape(x, target_shape):
    padding_list = []

    for x_dim, target_dim in zip(x.shape, target_shape):
        padding_list.append((0, target_dim - x_dim))

    return np.pad(x, tuple(padding_list), mode="constant")

def to_shape_minimize(x, target_shape):
    padding_list = []

    for x_dim, target_dim in zip(x.shape, target_shape):
        padding_list.append((0, target_dim - x_dim))

    return np.pad(x, tuple(padding_list), mode="constant")

def detect_leading_silence(audio, sr, silence_threshold=0.007, frame_length=1024):
    if len(audio.shape) == 2:
        channel = np.argmax(np.sum(np.abs(audio), axis=1))
        audio = audio[channel]

    for i in range(0, len(audio), frame_length):
        if np.max(np.abs(audio[i : i + frame_length])) > silence_threshold: return (i / sr) * 1000

    return (len(audio) / sr) * 1000

def adjust_leading_silence(target_audio, reference_audio, silence_threshold=0.01, frame_length=1024):
    def find_silence_end(audio):
        if len(audio.shape) == 2:
            channel = np.argmax(np.sum(np.abs(audio), axis=1))
            audio_mono = audio[channel]
        else: audio_mono = audio

        for i in range(0, len(audio_mono), frame_length):
            if np.max(np.abs(audio_mono[i : i + frame_length])) > silence_threshold: return i

        return len(audio_mono)

    ref_silence_end = find_silence_end(reference_audio)
    target_silence_end = find_silence_end(target_audio)
    silence_difference = ref_silence_end - target_silence_end

    try:
        silence_difference_p = ((ref_silence_end / 44100) * 1000) - ((target_silence_end / 44100) * 1000)
    except Exception as e:
        pass

    if silence_difference > 0: return np.hstack((np.zeros((target_audio.shape[0], silence_difference))if len(target_audio.shape) == 2 else np.zeros(silence_difference), target_audio))
    elif silence_difference < 0: return target_audio[:, -silence_difference:]if len(target_audio.shape) == 2 else target_audio[-silence_difference:]
    else: return target_audio

def match_array_shapes(array_1, array_2, is_swap=False):

    if is_swap: array_1, array_2 = array_1.T, array_2.T

    if array_1.shape[1] > array_2.shape[1]: array_1 = array_1[:, : array_2.shape[1]]
    elif array_1.shape[1] < array_2.shape[1]:
        padding = array_2.shape[1] - array_1.shape[1]
        array_1 = np.pad(array_1, ((0, 0), (0, padding)), "constant", constant_values=0)

    if is_swap: array_1, array_2 = array_1.T, array_2.T

    return array_1

def match_mono_array_shapes(array_1, array_2):
    if len(array_1) > len(array_2): array_1 = array_1[: len(array_2)]
    elif len(array_1) < len(array_2):
        padding = len(array_2) - len(array_1)
        array_1 = np.pad(array_1, (0, padding), "constant", constant_values=0)

    return array_1

def change_pitch_semitones(y, sr, semitone_shift):
    factor = 2 ** (semitone_shift / 12) 
    y_pitch_tuned = []

    for y_channel in y:
        y_pitch_tuned.append(librosa.resample(y_channel, orig_sr=sr, target_sr=sr * factor, res_type="soxr_vhq"))

    y_pitch_tuned = np.array(y_pitch_tuned)
    new_sr = sr * factor

    return y_pitch_tuned, new_sr

def augment_audio(export_path, audio_file, rate, is_normalization, wav_type_set, save_format=None, is_pitch=False, is_time_correction=True):
    wav, sr = librosa.load(audio_file, sr=44100, mono=False)
    if wav.ndim == 1: wav = np.asfortranarray([wav, wav])

    if not is_time_correction: wav_mix = change_pitch_semitones(wav, 44100, semitone_shift=-rate)[0]
    else:
        if is_pitch: wav_1, wav_2 = pitch_shift(wav[0], sr, rate, rbargs=None), pitch_shift(wav[1], sr, rate, rbargs=None)
        else: wav_1, wav_2 = time_stretch(wav[0], sr, rate, rbargs=None), time_stretch(wav[1], sr, rate, rbargs=None)

        if wav_1.shape > wav_2.shape: wav_2 = to_shape(wav_2, wav_1.shape)
        if wav_1.shape < wav_2.shape: wav_1 = to_shape(wav_1, wav_2.shape)

        wav_mix = np.asfortranarray([wav_1, wav_2])

    sf.write(export_path, normalize(wav_mix.T, is_normalization), sr, subtype=wav_type_set)
    save_format(export_path)


def average_audio(audio):
    waves, wave_shapes, final_waves = [], [], []

    for i in range(len(audio)):
        wave = librosa.load(audio[i], sr=44100, mono=False)
        waves.append(wave[0])
        wave_shapes.append(wave[0].shape[1])

    wave_shapes_index = wave_shapes.index(max(wave_shapes))
    target_shape = waves[wave_shapes_index]

    waves.pop(wave_shapes_index)
    final_waves.append(target_shape)

    for n_array in waves:
        wav_target = to_shape(n_array, target_shape.shape)
        final_waves.append(wav_target)

    waves = sum(final_waves)
    return waves / len(audio)

def average_dual_sources(wav_1, wav_2, value):
    if wav_1.shape > wav_2.shape: wav_2 = to_shape(wav_2, wav_1.shape)
    if wav_1.shape < wav_2.shape: wav_1 = to_shape(wav_1, wav_2.shape)

    return (wav_1 * value) + (wav_2 * (1 - value))

def reshape_sources(wav_1, wav_2):
    if wav_1.shape > wav_2.shape: wav_2 = to_shape(wav_2, wav_1.shape)

    if wav_1.shape < wav_2.shape:
        ln = min([wav_1.shape[1], wav_2.shape[1]])
        wav_2 = wav_2[:, :ln]

    ln = min([wav_1.shape[1], wav_2.shape[1]])
    wav_1 = wav_1[:, :ln]
    wav_2 = wav_2[:, :ln]

    return wav_2

def reshape_sources_ref(wav_1_shape, wav_2):
    if wav_1_shape > wav_2.shape: wav_2 = to_shape(wav_2, wav_1_shape)
    return wav_2

def combine_arrarys(audio_sources, is_swap=False):
    source = np.zeros_like(max(audio_sources, key=np.size))

    for v in audio_sources:
        v = match_array_shapes(v, source, is_swap=is_swap)
        source += v

    return source

def combine_audio(paths, audio_file_base=None, wav_type_set="FLOAT", save_format=None):
    source = combine_arrarys([load_audio(i) for i in paths])
    save_path = f"{audio_file_base}_combined.wav"
    sf.write(save_path, source.T, 44100, subtype=wav_type_set)
    save_format(save_path)

def reduce_mix_bv(inst_source, voc_source, reduction_rate=0.9):
    return combine_arrarys([inst_source * (1 - reduction_rate), voc_source], is_swap=True)

def organize_inputs(inputs):
    input_list = {"target": None, "reference": None, "reverb": None, "inst": None}

    for i in inputs:
        if i.endswith("_(Vocals).wav"): input_list["reference"] = i
        elif "_RVC_" in i: input_list["target"] = i
        elif i.endswith("reverbed_stem.wav"): input_list["reverb"] = i
        elif i.endswith("_(Instrumental).wav"): input_list["inst"] = i

    return input_list

def check_if_phase_inverted(wav1, wav2, is_mono=False):
    if not is_mono:
        wav1 = np.mean(wav1, axis=0)
        wav2 = np.mean(wav2, axis=0)

    return np.corrcoef(wav1[:1000], wav2[:1000])[0, 1] < 0

def rerun_mp3(audio_file):
    with audioread.audio_open(audio_file) as f:
        track_length = int(f.duration)

    return track_length

def align_audio(file1, file2, file2_aligned, file_subtracted, wav_type_set, is_save_aligned, command_Text, save_format, align_window, align_intro_val, db_analysis, set_progress_bar, phase_option, phase_shifts, is_match_silence, is_spec_match):
    global progress_value
    progress_value = 0
    is_mono = False

    def get_diff(a, b):
        return np.correlate(a, b, "full").argmax() - (b.shape[0] - 1)

    def progress_bar(length):
        global progress_value
        progress_value += 1

        if (0.90 / length * progress_value) >= 0.9: length = progress_value + 1
        set_progress_bar(0.1, (0.9 / length * progress_value))

    if file1.endswith(".mp3") and is_macos:
        length1 = rerun_mp3(file1)
        wav1, sr1 = librosa.load(file1, duration=length1, sr=44100, mono=False)
    else:
        wav1, sr1 = librosa.load(file1, sr=44100, mono=False)

    if file2.endswith(".mp3") and is_macos:
        length2 = rerun_mp3(file2)
        wav2, sr2 = librosa.load(file2, duration=length2, sr=44100, mono=False)
    else:
        wav2, sr2 = librosa.load(file2, sr=44100, mono=False)

    if wav1.ndim == 1 and wav2.ndim == 1: is_mono = True
    elif wav1.ndim == 1: wav1 = np.asfortranarray([wav1, wav1])
    elif wav2.ndim == 1: wav2 = np.asfortranarray([wav2, wav2])

    if phase_option == AUTO_PHASE:
        if check_if_phase_inverted(wav1, wav2, is_mono=is_mono): wav2 = -wav2
    elif phase_option == POSITIVE_PHASE: wav2 = +wav2
    elif phase_option == NEGATIVE_PHASE: wav2 = -wav2

    if is_match_silence: wav2 = adjust_leading_silence(wav2, wav1)

    wav1_length = int(librosa.get_duration(y=wav1, sr=44100))
    wav2_length = int(librosa.get_duration(y=wav2, sr=44100))

    if not is_mono:
        wav1 = wav1.transpose()
        wav2 = wav2.transpose()

    wav2_org = wav2.copy()

    command_Text(translations["process_file"])
    seconds_length = min(wav1_length, wav2_length)
    wav2_aligned_sources = []

    for sec_len in align_intro_val:
        sec_seg = 1 if sec_len == 1 else int(seconds_length // sec_len)
        index = sr1 * sec_seg 

        if is_mono:
            samp1, samp2 = wav1[index : index + sr1], wav2[index : index + sr1]
            diff = get_diff(samp1, samp2)
        else:
            index = sr1 * sec_seg  
            samp1, samp2 = wav1[index : index + sr1, 0], wav2[index : index + sr1, 0]
            samp1_r, samp2_r = wav1[index : index + sr1, 1], wav2[index : index + sr1, 1]
            diff, _ = get_diff(samp1, samp2), get_diff(samp1_r, samp2_r)

        if diff > 0: wav2_aligned = np.append(np.zeros(diff) if is_mono else np.zeros((diff, 2)), wav2_org, axis=0)
        elif diff < 0: wav2_aligned = wav2_org[-diff:]
        else: wav2_aligned = wav2_org

        if not any(np.array_equal(wav2_aligned, source) for source in wav2_aligned_sources): wav2_aligned_sources.append(wav2_aligned)

    unique_sources = len(wav2_aligned_sources)
    sub_mapper_big_mapper = {}

    for s in wav2_aligned_sources:
        wav2_aligned = match_mono_array_shapes(s, wav1) if is_mono else match_array_shapes(s, wav1, is_swap=True)

        if align_window:
            wav_sub = time_correction(wav1, wav2_aligned, seconds_length, align_window=align_window, db_analysis=db_analysis, progress_bar=progress_bar, unique_sources=unique_sources, phase_shifts=phase_shifts)
            sub_mapper_big_mapper = {**sub_mapper_big_mapper, **{np.abs(wav_sub).mean(): wav_sub}}
        else:
            wav2_aligned = wav2_aligned * np.power(10, db_analysis[0] / 20)

            for db_adjustment in db_analysis[1]:
                sub_mapper_big_mapper = {**sub_mapper_big_mapper, **{np.abs(wav_sub).mean(): wav1 - (wav2_aligned * (10 ** (db_adjustment / 20)))}}

    wav_sub = ensemble_for_align(list(sub_mapper_big_mapper.values())) if is_spec_match and len(list(sub_mapper_big_mapper.values())) >= 2 else ensemble_wav(list(sub_mapper_big_mapper.values()))
    wav_sub = np.clip(wav_sub, -1, +1)

    command_Text(translations["save_instruments"])

    if is_save_aligned or is_spec_match:
        wav1 = match_mono_array_shapes(wav1, wav_sub) if is_mono else match_array_shapes(wav1, wav_sub, is_swap=True)
        wav2_aligned = wav1 - wav_sub

        if is_spec_match:
            if wav1.ndim == 1 and wav2.ndim == 1:
                wav2_aligned = np.asfortranarray([wav2_aligned, wav2_aligned]).T
                wav1 = np.asfortranarray([wav1, wav1]).T

            wav2_aligned = ensemble_for_align([wav2_aligned, wav1])
            wav_sub = wav1 - wav2_aligned

        if is_save_aligned:
            sf.write(file2_aligned, wav2_aligned, sr1, subtype=wav_type_set)
            save_format(file2_aligned)

    sf.write(file_subtracted, wav_sub, sr1, subtype=wav_type_set)
    save_format(file_subtracted)

def phase_shift_hilbert(signal, degree):
    analytic_signal = hilbert(signal)
    return np.cos(np.radians(degree)) * analytic_signal.real - np.sin(np.radians(degree)) * analytic_signal.imag

def get_phase_shifted_tracks(track, phase_shift):
    if phase_shift == 180: return [track, -track]

    step = phase_shift
    end = 180 - (180 % step) if 180 % step == 0 else 181
    phase_range = range(step, end, step)
    flipped_list = [track, -track]

    for i in phase_range:
        flipped_list.extend([phase_shift_hilbert(track, i), phase_shift_hilbert(track, -i)])

    return flipped_list

def time_correction(mix, instrumental, seconds_length, align_window, db_analysis, sr=44100, progress_bar=None, unique_sources=None, phase_shifts=NONE_P):
    def align_tracks(track1, track2):
        shifted_tracks = {}
        track2 = track2 * np.power(10, db_analysis[0] / 20)
        track2_flipped = [track2] if phase_shifts == 190 else get_phase_shifted_tracks(track2, phase_shifts)

        for db_adjustment in db_analysis[1]:
            for t in track2_flipped:
                track2_adjusted = t * (10 ** (db_adjustment / 20))
                track2_shifted = np.roll(track2_adjusted, shift=np.argmax(np.abs(correlate(track1, track2_adjusted))) - (len(track1) - 1))
                shifted_tracks[np.abs(track1 - track2_shifted).mean()] = track2_shifted

        return shifted_tracks[min(shifted_tracks.keys())]

    assert mix.shape == instrumental.shape, translations["assert"].format(mixshape=mix.shape, instrumentalshape=instrumental.shape)
    seconds_length = seconds_length // 2

    sub_mapper = {}
    progress_update_interval, total_iterations = 120, 0

    if len(align_window) > 2: progress_update_interval = 320

    for secs in align_window:
        step = secs / 2
        window_size = int(sr * secs)
        step_size = int(sr * step)

        if len(mix.shape) == 1: total_iterations += ((len(range(0, len(mix) - window_size, step_size)) // progress_update_interval) * unique_sources)
        else: total_iterations += ((len(range(0, len(mix[:, 0]) - window_size, step_size)) * 2 // progress_update_interval) * unique_sources)

    for secs in align_window:
        sub = np.zeros_like(mix)
        divider = np.zeros_like(mix)
        window_size = int(sr * secs)
        step_size = int(sr * secs / 2)
        window = np.hanning(window_size)

        if len(mix.shape) == 1:
            counter = 0

            for i in range(0, len(mix) - window_size, step_size):
                counter += 1
                if counter % progress_update_interval == 0: progress_bar(total_iterations)

                window_mix = mix[i : i + window_size] * window
                window_instrumental = instrumental[i : i + window_size] * window
                window_instrumental_aligned = align_tracks(window_mix, window_instrumental)
                sub[i : i + window_size] += window_mix - window_instrumental_aligned
                divider[i : i + window_size] += window
        else:
            counter = 0

            for ch in range(mix.shape[1]):
                for i in range(0, len(mix[:, ch]) - window_size, step_size):
                    counter += 1

                    if counter % progress_update_interval == 0: progress_bar(total_iterations)

                    window_mix = mix[i : i + window_size, ch] * window
                    window_instrumental = instrumental[i : i + window_size, ch] * window
                    window_instrumental_aligned = align_tracks(window_mix, window_instrumental)
                    sub[i : i + window_size, ch] += window_mix - window_instrumental_aligned
                    divider[i : i + window_size, ch] += window

    return ensemble_wav(list({**sub_mapper, **{np.abs(sub).mean(): np.where(divider > 1e-6, sub / divider, sub)}}.values()), split_size=12)

def ensemble_wav(waveforms, split_size=240):
    waveform_thirds = {i: np.array_split(waveform, split_size) for i, waveform in enumerate(waveforms)}
    final_waveform = []
    for third_idx in range(split_size):
        final_waveform.append(waveform_thirds[np.argmin([np.abs(waveform_thirds[i][third_idx]).mean() for i in range(len(waveforms))])][third_idx])

    return np.concatenate(final_waveform)

def ensemble_wav_min(waveforms):
    for i in range(1, len(waveforms)):
        if i == 1: wave = waveforms[0]
        ln = min(len(wave), len(waveforms[i]))
        wave = wave[:ln]
        waveforms[i] = waveforms[i][:ln]
        wave = np.where(np.abs(waveforms[i]) <= np.abs(wave), waveforms[i], wave)

    return wave

def align_audio_test(wav1, wav2, sr1=44100):
    def get_diff(a, b):
        return np.correlate(a, b, "full").argmax() - (b.shape[0] - 1)

    wav1 = wav1.transpose()
    wav2 = wav2.transpose()
    wav2_org = wav2.copy()
    index = sr1  
    diff = get_diff(wav1[index : index + sr1, 0], wav2[index : index + sr1, 0])

    if diff > 0: wav2_aligned = np.append(np.zeros((diff, 1)), wav2_org, axis=0)
    elif diff < 0: wav2_aligned = wav2_org[-diff:]
    else: wav2_aligned = wav2_org
    return wav2_aligned

def load_audio(audio_file):
    wav, _ = librosa.load(audio_file, sr=44100, mono=False)
    if wav.ndim == 1: wav = np.asfortranarray([wav, wav])
    return wav

def __rubberband(y, sr, **kwargs):
    assert sr > 0
    fd, infile = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    fd, outfile = tempfile.mkstemp(suffix='.wav')
    os.close(fd)

    sf.write(infile, y, sr)

    try:
        arguments = [os.path.join(BASE_PATH_RUB, 'rubberband'), '-q']
        for key, value in six.iteritems(kwargs):
            arguments.append(str(key))
            arguments.append(str(value))

        arguments.extend([infile, outfile])
        subprocess.check_call(arguments, stdout=DEVNULL, stderr=DEVNULL)

        y_out, _ = sf.read(outfile, always_2d=True)
        if y.ndim == 1: y_out = np.squeeze(y_out)
    except OSError as exc:
        six.raise_from(RuntimeError(translations["rubberband"]), exc)
    finally:
        os.unlink(infile)
        os.unlink(outfile)

    return y_out

def time_stretch(y, sr, rate, rbargs=None):
    if rate <= 0: raise ValueError(translations["rate"])
    if rate == 1.0: return y
    if rbargs is None: rbargs = dict()

    rbargs.setdefault('--tempo', rate)
    return __rubberband(y, sr, **rbargs)

def pitch_shift(y, sr, n_steps, rbargs=None):
    if n_steps == 0: return y
    if rbargs is None: rbargs = dict()

    rbargs.setdefault('--pitch', n_steps)
    return __rubberband(y, sr, **rbargs)
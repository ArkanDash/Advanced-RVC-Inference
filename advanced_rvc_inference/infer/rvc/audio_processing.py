import os
import sys
import torch
import librosa

import numpy as np
import scipy.signal as signal

sys.path.append(os.getcwd())

stft = None

def rms(x, eps=1e-9):
    return np.sqrt(np.mean(x ** 2) + eps)

def soft_limiter(x, threshold=0.98):
    return np.tanh(x / threshold) * threshold

def normalize_audio(x, target_rms=0.1):
    cur = rms(x)

    if cur <= 0: return x
    return x * (target_rms / cur)

def compute_mfcc(x, sr, n_mfcc=20, n_fft=1024, hop_length=160):
    mfcc = librosa.feature.mfcc(
        y=x, 
        sr=sr, 
        n_mfcc=n_mfcc, 
        n_fft=n_fft, 
        hop_length=hop_length
    )

    return mfcc

def mix_mfcc_exciter(audio, sr, strength=0.08, n_mfcc=20, n_mels=128):
    mfcc = compute_mfcc(audio, sr, n_mfcc)

    try:
        exc = librosa.feature.inverse.mfcc_to_audio(mfcc, sr=sr, n_mels=n_mels)
    except Exception:
        mel_spec = librosa.feature.inverse.mfcc_to_mel(mfcc)
        exc = librosa.feature.inverse.mel_to_audio(mel_spec, sr=sr)

    if exc.shape[0] < audio.shape[0]:
        exc = np.pad(exc, (0, audio.shape[0] - exc.shape[0]))
    else:
        exc = exc[: audio.shape[0]]

    b, a = signal.butter(2, 300 / (sr / 2), btype="high")
    exc = signal.lfilter(b, a, exc)

    exc = exc / (rms(exc) + 1e-9) * (rms(audio) + 1e-9)
    return audio + strength * exc

def automatic_multiband_eq(audio, sr, n_bands=6, target_slope=0.0, n_fft=1024, hop_length=160):
    S = np.abs(librosa.stft(audio.astype(np.float32), n_fft=n_fft, hop_length=hop_length))
    mean_spec = np.mean(S, axis=1)
    freqs = np.linspace(0, sr // 2, mean_spec.shape[0])

    band_edges = np.geomspace(100, sr / 2, n_bands + 1)
    gains_db = np.zeros(n_bands)

    for i in range(n_bands):
        idx = np.where((freqs >= band_edges[i]) & (freqs < band_edges[i + 1]))[0]
        if idx.size == 0:
            gains_db[i] = 0.0
            continue

        band_power_db = 20 * np.log10(np.mean(mean_spec[idx]) + 1e-9)
        median_db = np.median(20 * np.log10(mean_spec + 1e-9))
        gains_db[i] = median_db - band_power_db

    gains_db = signal.medfilt(gains_db, kernel_size=3)
    gains_db = gains_db + np.linspace(-target_slope, target_slope, n_bands)
    gains = 10 ** (gains_db / 20.0)

    out = np.zeros_like(audio)
    for i in range(n_bands):
        low = band_edges[i]
        high = band_edges[i + 1]

        if low <= 0:
            b, a = signal.butter(2, high / (sr / 2), btype="low")
        elif high >= sr / 2:
            b, a = signal.butter(2, low / (sr / 2), btype="high")
        else:
            b, a = signal.butter(2, [low / (sr / 2), high / (sr / 2)], btype="band")

        band = signal.lfilter(b, a, audio)
        out += gains[i] * band

    out = out / (rms(out) + 1e-9) * (rms(audio) + 1e-9)
    return 0.85 * audio + 0.15 * out

def apply_multiband_eq(audio, sr, bands):
    out = np.zeros_like(audio)

    for low, high, gain_db in bands:
        gain = 10 ** (gain_db / 20.0)

        if low <= 0: b, a = signal.butter(2, high / (sr / 2), btype="low")
        elif high >= sr / 2: b, a = signal.butter(2, low / (sr / 2), btype="high")
        else: b, a = signal.butter(2, [low / (sr / 2), high / (sr / 2)], btype="band")

        band = signal.lfilter(b, a, audio)
        out += gain * band

    return out

def best_multiband_eq(audio, sr, original_audio=None, sr_ref=16000, n_bands=6, target_slope=0.0, n_fft=1024, hop_length=160, strength=0.15):
    if original_audio is not None:
        mf_out = compute_mfcc(audio, sr)
        mf_ref = compute_mfcc(original_audio.astype(np.float32), sr_ref)

        out_mean = np.mean(mf_out, axis=1)
        ref_mean = np.mean(mf_ref, axis=1)
        diff = ref_mean - out_mean

        low_val   = diff[:3].mean()
        mid_val   = diff[3:6].mean()
        upper_val = diff[6:9].mean()
        high_val  = diff[9:13].mean()

        bands = [
            (0, 300, np.clip(low_val * 0.6, -6.0, 6.0)),
            (300, 800, np.clip(mid_val * 0.5, -6.0, 6.0)),
            (800, 2000, np.clip(upper_val * 0.6, -6.0, 6.0)),
            (2000, int(sr / 2 - 1000), np.clip(high_val * 0.6, -6.0, 6.0)),
        ]
        eq_audio = apply_multiband_eq(audio, sr, bands)
    else:
        fft = np.abs(librosa.stft(audio.astype(np.float32), n_fft=n_fft, hop_length=hop_length))
        mean_spec = np.mean(fft, axis=1)
        freqs = np.linspace(0, sr // 2, mean_spec.shape[0])

        band_edges = np.geomspace(100, sr / 2, n_bands + 1)
        gains_db = np.zeros(n_bands)

        for i in range(n_bands):
            idx = np.where((freqs >= band_edges[i]) & (freqs < band_edges[i + 1]))[0]
            if idx.size == 0: continue

            band_power_db = 20 * np.log10(np.mean(mean_spec[idx]) + 1e-9)
            median_db = np.median(20 * np.log10(mean_spec + 1e-9))
            gains_db[i] = median_db - band_power_db

        gains_db = signal.medfilt(gains_db, kernel_size=3)
        gains_db += np.linspace(-target_slope, target_slope, n_bands)
        gains_db = np.clip(gains_db, -6.0, 6.0)

        bands = [(band_edges[i], band_edges[i+1], gains_db[i]) for i in range(n_bands)]
        eq_audio = apply_multiband_eq(audio, sr, bands)

    out = (1 - strength) * audio + strength * eq_audio
    out = out / (rms(out) + 1e-9) * (rms(audio) + 1e-9)

    mx = np.max(np.abs(out)) + 1e-9
    if mx > 0.99: out /= mx * 0.99

    return out

def spectral_subtract_denoise(audio, sr, noise_seconds=0.4, alpha=1.0, n_fft=1024, hop_length=160, device="cpu"):
    global stft

    if stft is None and device.startswith(("ocl", "privateuseone")):
        from main.library.backends.utils import STFT
        stft = STFT(filter_length=n_fft, hop_length=hop_length, win_length=None, window="hann").to(device) 
    else: stft = None

    x = torch.from_numpy(audio.astype(np.float32)).float().unsqueeze(0).to(device)
    window = torch.hann_window(n_fft).to(device)

    if stft is None:
        fft = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window, return_complex=True)
        mag, phase = (fft.real.pow(2) + fft.imag.pow(2)).sqrt(), fft.imag.data.atan2(fft.real.data)
    else:
        mag, phase = stft.transform(x, eps=1e-9, return_phase=True)

    noise_mag = mag[:, :, :max(1, min(int((noise_seconds * sr - n_fft) // hop_length) + 1, mag.shape[-1]))].mean(dim=-1, keepdim=True)
    clean_mag = (mag - alpha * noise_mag).maximum((noise_mag * 1.0) * 0.1)

    xrec = torch.istft(clean_mag * (1j * phase).exp(), n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window, length=x.shape[0]) if stft is None else stft.inverse(clean_mag, phase)
    return xrec.squeeze(0).cpu().numpy()

def repair_bad_frames(audio, sr, frame_ms=20, energy_thresh=0.02):
    frame_len = int(sr * frame_ms / 1000)
    hop = frame_len // 2

    n_frames = 1 + max(0, (len(audio) - frame_len) // hop)
    frames = np.stack([audio[i * hop : i * hop + frame_len] for i in range(n_frames)])

    energies = np.sqrt(np.mean(frames ** 2, axis=1))
    median_e = np.median(energies)
    bad = energies < (energy_thresh * median_e)

    if not np.any(bad): return audio
    out = audio.copy()

    for i, is_bad in enumerate(bad):
        if not is_bad: continue

        start = i * hop
        end = start + frame_len

        left = out[max(0, start - frame_len) : start]
        right = out[end : min(len(out), end + frame_len)]

        if left.size > 0 and right.size > 0: out[start:end] = 0.5 * (np.mean(left) + np.mean(right))
        elif left.size > 0: out[start:end] = left[-1]
        elif right.size > 0: out[start:end] = right[0]
        else: out[start:end] = 0.0

    return out

def harmonic_enrich_and_compress(audio, drive=0.02, comp_ratio=3.0, frame_length=1024, hop_length=160):
    exc = np.abs(audio)
    exc -= np.mean(exc)
    audio2 = audio + drive * exc

    env_rms = librosa.feature.rms(y=audio2.astype(np.float32), frame_length=frame_length, hop_length=hop_length)[0]
    frame_times = np.linspace(0, len(audio2), num=len(env_rms))
    env_s = np.interp(np.arange(len(audio2)), frame_times, env_rms)

    threshold = np.median(env_s) * 1.2
    gain = 1.0 / (1.0 + ((env_s / (threshold + 1e-9)) ** (comp_ratio - 1)))
    out = audio2 * gain

    return out

def fade_in_out(audio, sr, fade_ms=10):
    n = len(audio)

    fade_len = int(sr * fade_ms / 1000)
    if fade_len <= 0: return audio

    win = np.ones(n)
    fade_in = np.linspace(0.0, 1.0, fade_len)
    fade_out = np.linspace(1.0, 0.0, fade_len)

    win[:fade_len] = fade_in
    win[-fade_len:] = fade_out

    return audio * win

def preprocess(audio, sr=16000, target_rms=0.8, device="cpu"):
    x = normalize_audio(audio.astype(np.float32), target_rms=target_rms)
    x -= np.mean(x)

    x = spectral_subtract_denoise(x, sr, device=device)
    x = repair_bad_frames(x, sr)

    x = automatic_multiband_eq(x, sr)
    x = mix_mfcc_exciter(x, sr, strength=0.06)

    x = harmonic_enrich_and_compress(x, drive=0.015, comp_ratio=2.5)
    x = soft_limiter(x, threshold=0.98)

    x = fade_in_out(x, sr, fade_ms=8)
    x /= (np.max(np.abs(x)) + 1e-9) * 0.99

    return x.astype(np.float32)

def postprocess(audio, sr=48000, original_audio=None, sr_ref=16000, device="cpu"):
    x = audio.astype(np.float32)
    x = x - np.mean(x)

    x = fade_in_out(x, sr, fade_ms=6)
    x = spectral_subtract_denoise(x, sr, noise_seconds=0.25, device=device)

    x = best_multiband_eq(x, sr, original_audio=original_audio, sr_ref=sr_ref, n_bands=6, target_slope=0.02, strength=0.15)
    x = soft_limiter(x, threshold=0.995)

    cutoff = min(20000, sr / 2 - 100)
    Wn = cutoff / (sr / 2)

    b, a = signal.butter(2, Wn, btype="low")
    x = signal.filtfilt(b, a, x)

    x /= (np.max(np.abs(x)) + 1e-9) * 0.99
    return x.astype(np.float32)
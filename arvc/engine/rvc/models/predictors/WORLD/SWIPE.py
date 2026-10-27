import math

import numba as nb
import numpy as np

from matplotlib import mlab
from scipy import interpolate
from decimal import Decimal, ROUND_HALF_UP


def swipe(x, fs, f0_floor=50, f0_ceil=1100, frame_period=10, sTHR=0.3):
    plim = np.array([f0_floor, f0_ceil])
    t = np.arange(0, int(1000 * len(x) / fs / (frame_period) + 1)) * (frame_period / 1000)

    log2pc = np.arange(np.log2(plim[0]) * 96, np.log2(plim[-1]) * 96)
    log2pc *= (1 / 96)

    pc = 2 ** log2pc
    S = np.zeros((len(pc), len(t))) 

    logWs = [round_matlab(elm) for elm in np.log2(4 * 2 * fs / plim)]
    ws = 2 ** np.arange(logWs[0], logWs[1] - 1, -1) 
    p0 = 4 * 2 * fs / ws 

    d = 1 + log2pc - np.log2(4 * 2 * fs / ws[0])
    fERBs = erbs2hz(np.arange(hz2erbs(pc[0] / 4), hz2erbs(fs / 2), 0.1))

    for i in range(len(ws)):
        dn = round_matlab(4 * fs / p0[i]) 
        X, f, ti = mlab.specgram(x=np.r_[np.zeros(int(ws[i] / 2)), np.r_[x, np.zeros(int(dn + ws[i] / 2))]], NFFT=ws[i], Fs=fs, window=np.hanning(ws[i] + 2)[1:-1], noverlap=max(0, np.round(ws[i] - dn)), mode='complex')
        ti = np.r_[0, ti[:-1]]
        M = np.maximum(0, interpolate.interp1d(f, np.abs(X.T), kind='cubic')(fERBs)).T

        if i == len(ws) - 1:
            j = np.where(d - (i + 1) > -1)[0]
            k = np.where(d[j] - (i + 1) < 0)[0]
        elif i == 0:
            j = np.where(d - (i + 1) < 1)[0]
            k = np.where(d[j] - (i + 1) > 0)[0]
        else:
            j = np.where(np.abs(d - (i + 1)) < 1)[0]
            k = np.arange(len(j))

        Si = pitchStrengthAllCandidates(fERBs, np.sqrt(M), pc[j])
        Si = interpolate.interp1d(ti, Si, bounds_error=False, fill_value='nan')(t) if Si.shape[1] > 1 else np.full((len(Si), len(t)), np.nan)

        mu = np.ones(j.shape)
        mu[k] = 1 - np.abs(d[j[k]] - i - 1)
        S[j, :] = S[j, :] + np.tile(mu.reshape(-1, 1), (1, Si.shape[1])) * Si


    p = np.full((S.shape[1], 1), np.nan)
    s = np.full((S.shape[1], 1), np.nan)

    for j in range(S.shape[1]):
        s[j] = np.max(S[:, j])
        i = np.argmax(S[:, j])

        if s[j] < sTHR: continue

        if i == 0: p[j] = pc[0]
        elif i == len(pc) - 1: p[j] = pc[0]
        else:
            I = np.arange(i-1, i+2)
            tc = 1 / pc[I]

            ntc = (tc / tc[1] - 1) * 2 * np.pi
            idx = np.isfinite(S[I, j])

            c = np.zeros(len(ntc))
            c += np.nan
            
            I_ = I[idx]

            if len(I_) < 2: c[idx] = (S[I, j])[0] / ntc[0]
            else: c[idx] = np.polyfit(ntc[idx], (S[I_, j]), 2)

            pval = np.polyval(c, ((1 / (2 ** np.arange(np.log2(pc[I[0]]), np.log2(pc[I[2]]) + 1 / 12 / 64, 1 / 12 / 64))) / tc[1] - 1) * 2 * np.pi)
            s[j] = np.max(pval)
            p[j] = 2 ** (np.log2(pc[I[0]]) + (np.argmax(pval)) / 12 / 64)

    p = p.flatten()
    p[np.isnan(p)] = 0

    return np.array(p, dtype=np.float32), np.array(t, dtype=np.float32)

def round_matlab(n):
    return int(Decimal(n).quantize(0, ROUND_HALF_UP))

def pitchStrengthAllCandidates(f, L, pc):
    den = np.sqrt(np.sum(L * L, axis=0))
    den = np.where(den == 0, 2.220446049250313e-16, den)

    L = L / den
    S = np.zeros((len(pc), L.shape[1]))

    for j in range(len(pc)):
        S[j,:] = pitchStrengthOneCandidate(f, L, pc[j])

    return S

def pitchStrengthOneCandidate(f, L, pc):
    k = np.zeros(len(f)) 
    q = f / pc 

    for i in ([1] + sieve(int(np.fix(f[-1] / pc - 0.75)))):
        a = np.abs(q - i)
        p = a < 0.25
        k[p] = np.cos(2 * np.pi * q[p])

        v = np.logical_and((0.25 < a), (a < 0.75))
        k[v] = k[v] + np.cos(2 * np.pi * q[v]) / 2

    k *= np.sqrt(1 / f)
    k /= np.linalg.norm(k[k>0])

    return k @ L

def hz2erbs(hz):
    return 21.4 * np.log10(1 + hz / 229)

def erbs2hz(erbs):
    return (10 ** (erbs / 21.4) - 1) * 229

def sieve(n):
    primes = list(range(2, n + 1))
    num = 2

    while num < math.sqrt(n):
        i = num

        while i <= n:
            i += num

            if i in primes: primes.remove(i)
                
        for j in primes:
            if j > num:
                num = j
                break

    return primes

def stonemask(x, fs, temporal_positions, f0):
    refined_f0 = np.copy(f0)

    for i in range(len(temporal_positions)):
        if f0[i] != 0:
            refined_f0[i] = get_refined_f0(x, fs, temporal_positions[i], f0[i])
            if abs(refined_f0[i] - f0[i]) / f0[i] > 0.2: refined_f0[i] = f0[i]

    return np.array(refined_f0, dtype=np.float32)

def get_refined_f0(x, fs, current_time, current_f0):
    f0_initial = current_f0
    half_window_length = np.ceil(3 * fs / f0_initial / 2)
    window_length_in_time = (2 * half_window_length + 1) / fs

    base_time = np.arange(-half_window_length, half_window_length + 1) / fs
    fft_size = 2 ** math.ceil(math.log((half_window_length * 2 + 1), 2) + 1)

    base_time = np.array([float("{0:.4f}".format(elm)) for elm in base_time])
    index_raw = round_matlab_2((current_time + base_time) * fs)
    
    window_time = ((index_raw - 1) / fs) - current_time
    main_window = 0.42 + 0.5 * np.cos(2 * math.pi * window_time / window_length_in_time) + 0.08 * np.cos(4 * math.pi * window_time / window_length_in_time)
    
    index = np.array(np.maximum(1, np.minimum(len(x), index_raw)), dtype=int)
    spectrum = np.fft.fft(x[index - 1] * main_window, fft_size)

    diff_spectrum = np.fft.fft(x[index - 1] * (-(np.diff(np.r_[0, main_window]) + np.diff(np.r_[main_window, 0])) / 2), fft_size)
    power_spectrum = np.abs(spectrum) ** 2

    from sys import float_info

    power_spectrum[power_spectrum == 0] = float_info.epsilon
    instantaneous_frequency = (np.arange(fft_size) / fft_size * fs) + (np.real(spectrum) * np.imag(diff_spectrum) - np.imag(spectrum) * np.real(diff_spectrum)) / power_spectrum * fs / 2 / math.pi
    
    trim_index = np.array([1, 2])
    index_list_trim = np.array(round_matlab_2(f0_initial * fft_size / fs * trim_index) + 1, int)

    amp_list = np.sqrt(power_spectrum[index_list_trim - 1])
    f0_initial = np.sum(amp_list * instantaneous_frequency[index_list_trim - 1]) / np.sum(amp_list * trim_index)

    if f0_initial < 0: return 0
    
    trim_index = np.array([1, 2, 3, 4, 5, 6])
    index_list_trim = np.array(round_matlab_2(f0_initial * fft_size / fs * trim_index) + 1, int)
    amp_list = np.sqrt(power_spectrum[index_list_trim - 1])

    return np.sum(amp_list * instantaneous_frequency[index_list_trim - 1]) / np.sum(amp_list * trim_index)

@nb.jit((nb.float64[:],), nopython=True, cache=True)
def round_matlab_2(x):
    y = x.copy()
    
    y[x > 0] += 0.5
    y[x <= 0] -= 0.5

    return y
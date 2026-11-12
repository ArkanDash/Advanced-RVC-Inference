import numpy as np

from numpy.lib.stride_tricks import sliding_window_view

def istft(frames, framesize, hopsize):
    frames = np.atleast_2d(frames)
    assert frames.ndim == 2

    analysis_window_size = np.ravel(framesize)[0]
    synthesis_window_size = np.ravel(framesize)[-1]

    assert analysis_window_size >= synthesis_window_size

    A = asymmetric_analysis_window(analysis_window_size, synthesis_window_size) if analysis_window_size != synthesis_window_size else symmetric_window(analysis_window_size)
    S = asymmetric_synthesis_window(analysis_window_size, synthesis_window_size) if analysis_window_size != synthesis_window_size else symmetric_window(synthesis_window_size)

    W = S * hopsize / np.sum(A * S)
    N = frames.shape[0] * hopsize + analysis_window_size

    y = np.zeros((N), float)

    frames[:,  0] = 0
    frames[:, -1] = 0
    frames0 = sliding_window_view(y, analysis_window_size, writeable=True)[::hopsize]
    frames1 = np.fft.irfft(frames, axis=-1, norm='forward') * W

    for i in range(min(len(frames0), len(frames1))):
        frames0[i] += frames1[i]

    return y

def asymmetric_synthesis_window(analysis_window_size, synthesis_window_size):
    n = analysis_window_size
    m = synthesis_window_size // 2

    right = symmetric_window(2 * m)
    window = np.zeros(n)

    window[n-m-m:n-m] = np.square(right[:m]) / symmetric_window(2 * n - 2 * m)[n-m-m:n-m]
    window[-m:] = right[-m:]

    return window

def asymmetric_analysis_window(analysis_window_size, synthesis_window_size):
    n = analysis_window_size
    m = synthesis_window_size // 2

    window = np.zeros(n)
    window[:n-m] = symmetric_window(2 * n - 2 * m)[:n-m]
    window[-m:] = symmetric_window(2 * m)[-m:]

    return window

def symmetric_window(symmetric_window_size):
    n = symmetric_window_size
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)

    return window

def stft(x, framesize, hopsize):
    x = np.atleast_1d(x)
    assert x.ndim == 1

    analysis_window_size = np.ravel(framesize)[0]
    synthesis_window_size = np.ravel(framesize)[-1]

    assert analysis_window_size >= synthesis_window_size

    W = asymmetric_analysis_window(analysis_window_size, synthesis_window_size) if analysis_window_size != synthesis_window_size else symmetric_window(analysis_window_size)

    frames0 = sliding_window_view(x, analysis_window_size, writeable=False)[::hopsize]
    frames1 = np.fft.rfft(frames0 * W, axis=-1, norm='forward')

    return frames1

def normalize(frames, frames0):
    for i in range(len(frames)):
        a = np.real(frames0[i])
        b = np.real(frames[i])
        a = np.dot(a, a)
        b = np.dot(b, b)

        if b == 0: continue
        frames[i] = np.real(frames[i]) * np.sqrt(a / b) + 1j * np.imag(frames[i])

    return frames

def lowpass(cepstrum, quefrency):
    cepstrum[1:quefrency] *= 2
    cepstrum[quefrency+1:] = 0

    return cepstrum

def lifter(frames, quefrency):
    envelopes = np.zeros(frames.shape)

    for i, frame in enumerate(frames):
        with np.errstate(divide='ignore', invalid='ignore'):
            spectrum = np.log10(np.real(frame)) 

        envelopes[i] = np.power(10, np.real(np.fft.rfft(lowpass(np.fft.irfft(spectrum, norm='forward'), quefrency), norm='forward')))

    return envelopes

def resample(x, factor):
    if factor == 1: return x.copy()
    y = np.zeros(x.shape, dtype=x.dtype)
    
    n = len(x)
    m = int(n * factor)

    i = np.arange(min(n, m))
    k = i * (n / m)

    j = np.trunc(k).astype(int)
    k = k - j

    ok = (0 <= j) & (j < n - 1)
    y[i[ok]] = k[ok] * x[j[ok] + 1] + (1 - k[ok]) * x[j[ok]]

    return y

def shiftpitch(frames, factors, samplerate):
    for i in range(len(frames)):
        magnitudes = np.vstack([resample(np.real(frames[i]), factor) for factor in factors])
        frequencies = np.vstack([resample(np.imag(frames[i]), factor) * factor for factor in factors])

        magnitudes[(frequencies <= 0) | (frequencies >= samplerate / 2)] = 0
        mask = np.argmax(magnitudes, axis=0)

        magnitudes = np.take_along_axis(magnitudes, mask[None,:], axis=0)
        frequencies = np.take_along_axis(frequencies, mask[None,:], axis=0)

        frames[i] = magnitudes + 1j * frequencies

    return frames

def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def encode(frames, framesize, hopsize, samplerate):
    M, N = frames.shape
    analysis_framesize = np.ravel(framesize)[0]

    freqinc = samplerate / analysis_framesize
    phaseinc = 2 * np.pi * hopsize / analysis_framesize

    buffer = np.zeros(N)
    data = np.zeros((M, N), complex)

    for m, frame in enumerate(frames):
        arg = np.angle(frame)
        delta = arg - buffer
        
        buffer = arg

        i = np.arange(N)
        data[m] = np.abs(frame) + 1j * ((i + (wrap(delta - i * phaseinc) / phaseinc)) * freqinc)

    return data

def decode(frames, framesize, hopsize, samplerate):
    M, N = frames.shape
    analysis_framesize = np.ravel(framesize)[0]
    synthesis_framesize = np.ravel(framesize)[-1]

    freqinc = samplerate / analysis_framesize
    phaseinc = 2 * np.pi * hopsize / analysis_framesize
    timeshift = 2 * np.pi * synthesis_framesize * np.arange(N) / N if synthesis_framesize != analysis_framesize else 0

    buffer = np.zeros(N)
    data = np.zeros((M, N), complex)

    for m, frame in enumerate(frames):
        i = np.arange(N)
        delta = (i + ((np.imag(frame) - i * freqinc) / freqinc)) * phaseinc
        buffer += delta
        arg = buffer.copy()
        arg -= timeshift 
        data[m] = np.real(frame) * np.exp(1j * arg)

    return data

class StftPitchShift:
    def __init__(self, framesize, hopsize, samplerate):
        self.framesize = framesize
        self.hopsize = hopsize
        self.samplerate = samplerate

    def shiftpitch(self, input, factors = 1, quefrency = 0, distortion = 1, normalization = False):
        input = np.atleast_1d(input)
        dtype = input.dtype
        shape = input.shape

        input = np.squeeze(input)
        if input.ndim != 1: raise ValueError('input.ndim != 1')

        if np.issubdtype(dtype, np.integer):
            a, b = np.iinfo(dtype).min, np.iinfo(dtype).max
            input = ((input.astype(float) - a) / (b - a)) * 2 - 1
        elif not np.issubdtype(dtype, np.floating): raise TypeError('not np.issubdtype(dtype, np.floating)')

        def isnotnormal(x):
            return (np.isinf(x)) | (np.isnan(x)) | (abs(x) < np.finfo(x.dtype).tiny)

        framesize = self.framesize
        hopsize = self.hopsize
        samplerate = self.samplerate

        factors = np.asarray(factors).flatten()
        quefrency = int(quefrency * samplerate)

        frames = encode(stft(input, framesize, hopsize), framesize, hopsize, samplerate)

        if normalization: frames0 = frames.copy()

        if quefrency:
            envelopes = lifter(frames, quefrency)
            mask = isnotnormal(envelopes)

            frames.real /= envelopes
            frames.real[mask] = 0

            if distortion != 1:
                envelopes[mask] = 0

                for i in range(len(envelopes)):
                    envelopes[i] = resample(envelopes[i], distortion)

                mask = isnotnormal(envelopes)

            frames = shiftpitch(frames, factors, samplerate)
            frames.real *= envelopes
            frames.real[mask] = 0
        else: frames = shiftpitch(frames, factors, samplerate)

        if normalization: frames = normalize(frames, frames0)

        output = istft(decode(frames, framesize, hopsize, samplerate), framesize, hopsize)
        output.resize(shape, refcheck=False)

        if np.issubdtype(dtype, np.integer):
            a, b = np.iinfo(dtype).min, np.iinfo(dtype).max
            output = (((output + 1) / 2) * (b - a) + a).clip(a, b).astype(dtype)
        elif output.dtype != dtype: output = output.astype(dtype)

        assert output.dtype == dtype
        assert output.shape == shape

        return output
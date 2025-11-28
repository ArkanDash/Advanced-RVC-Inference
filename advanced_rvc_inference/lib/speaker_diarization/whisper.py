import os
import sys
import gzip
import zlib
import tqdm
import torch
import base64
import string
import tiktoken
import itertools

import numba as nb
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import replace
from torch.distributions import Categorical
from functools import cached_property, lru_cache

sys.path.append(os.getcwd())

from main.app.variables import configs, logger
from main.library.backends import directml, opencl

LANGUAGES = {"en": "english", "zh": "chinese", "de": "german", "es": "spanish", "ru": "russian", "ko": "korean", "fr": "french", "ja": "japanese", "pt": "portuguese", "tr": "turkish", "pl": "polish", "ca": "catalan", "nl": "dutch", "ar": "arabic", "sv": "swedish", "it": "italian", "id": "indonesian", "hi": "hindi", "fi": "finnish", "vi": "vietnamese", "he": "hebrew", "uk": "ukrainian", "el": "greek", "ms": "malay", "cs": "czech", "ro": "romanian", "da": "danish", "hu": "hungarian", "ta": "tamil", "no": "norwegian", "th": "thai", "ur": "urdu", "hr": "croatian", "bg": "bulgarian", "lt": "lithuanian", "la": "latin", "mi": "maori", "ml": "malayalam", "cy": "welsh", "sk": "slovak", "te": "telugu", "fa": "persian", "lv": "latvian", "bn": "bengali", "sr": "serbian", "az": "azerbaijani", "sl": "slovenian", "kn": "kannada", "et": "estonian", "mk": "macedonian", "br": "breton", "eu": "basque", "is": "icelandic", "hy": "armenian", "ne": "nepali", "mn": "mongolian", "bs": "bosnian", "kk": "kazakh", "sq": "albanian", "sw": "swahili", "gl": "galician", "mr": "marathi", "pa": "punjabi", "si": "sinhala", "km": "khmer", "sn": "shona", "yo": "yoruba", "so": "somali", "af": "afrikaans", "oc": "occitan", "ka": "georgian", "be": "belarusian", "tg": "tajik", "sd": "sindhi", "gu": "gujarati", "am": "amharic", "yi": "yiddish", "lo": "lao", "uz": "uzbek", "fo": "faroese", "ht": "haitian creole", "ps": "pashto", "tk": "turkmen", "nn": "nynorsk", "mt": "maltese", "sa": "sanskrit", "lb": "luxembourgish", "my": "myanmar", "bo": "tibetan", "tl": "tagalog", "mg": "malagasy", "as": "assamese", "tt": "tatar", "haw": "hawaiian", "ln": "lingala", "ha": "hausa", "ba": "bashkir", "jw": "javanese", "su": "sundanese", "yue": "cantonese"}
TO_LANGUAGE_CODE = {**{language: code for code, language in LANGUAGES.items()}, "burmese": "my", "valencian": "ca", "flemish": "nl", "haitian": "ht", "letzeburgesch": "lb", "pushto": "ps", "panjabi": "pa", "moldavian": "ro", "moldovan": "ro", "sinhalese": "si", "castilian": "es", "mandarin": "zh"}
_ALIGNMENT_HEADS = {"tiny.en": b"ABzY8J1N>@0{>%R00Bk>$p{7v037`oCl~+#00", "tiny": b"ABzY8bu8Lr0{>%RKn9Fp%m@SkK7Kt=7ytkO", "base.en": b"ABzY8;40c<0{>%RzzG;p*o+Vo09|#PsxSZm00", "base": b"ABzY8KQ!870{>%RzyTQH3`Q^yNP!>##QT-<FaQ7m", "small.en": b"ABzY8>?_)10{>%RpeA61k&I|OI3I$65C{;;pbCHh0B{qLQ;+}v00", "small": b"ABzY8DmU6=0{>%Rpa?J`kvJ6qF(V^F86#Xh7JUGMK}P<N0000", "medium.en": b"ABzY8usPae0{>%R7<zz_OvQ{)4kMa0BMw6u5rT}kRKX;$NfYBv00*Hl@qhsU00", "medium": b"ABzY8B0Jh+0{>%R7}kK1fFL7w6%<-Pf*t^=N)Qr&0RR9", "large-v1": b"ABzY8r9j$a0{>%R7#4sLmoOs{s)o3~84-RPdcFk!JR<kSfC2yj", "large-v2": b"ABzY8zd+h!0{>%R7=D0pU<_bnWW*tkYAhobTNnu$jnkEkXqp)j;w1Tzk)UH3X%SZd&fFZ2fC2yj", "large-v3": b"ABzY8gWO1E0{>%R7(9S+Kn!D~%ngiGaR?*L!iJG9p-nab0JQ=-{D1-g00", "large": b"ABzY8gWO1E0{>%R7(9S+Kn!D~%ngiGaR?*L!iJG9p-nab0JQ=-{D1-g00", "large-v3-turbo": b"ABzY8j^C+e0{>%RARaKHP%t(lGR*)0g!tONPyhe`"}

SAMPLE_RATE, N_FFT, HOP_LENGTH, CHUNK_LENGTH = 16000, 400, 160, 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE 
N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  
stft = None

def exact_div(x, y):
    assert x % y == 0
    return x // y

N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH) 
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN) 

def load_model(name = "base", device = "cpu"):
    checkpoint_file = os.path.join(configs["speaker_diarization_path"], "models", name + ".pt")
    alignment_heads = _ALIGNMENT_HEADS[name]

    with open(checkpoint_file, "rb") as fp:
        checkpoint = torch.load(fp, map_location="cpu", weights_only=True)

    del checkpoint_file

    model = Whisper(ModelDimensions(**checkpoint["dims"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.set_alignment_heads(alignment_heads)

    return model.to(device)

def merge_punctuations(alignment, prepended, appended):
    i = len(alignment) - 2
    j = len(alignment) - 1

    while i >= 0:
        previous = alignment[i]
        following = alignment[j]
        
        if previous.word.startswith(" ") and previous.word.strip() in prepended:
            following.word = previous.word + following.word
            following.tokens = previous.tokens + following.tokens

            previous.word = ""
            previous.tokens = []
        else: j = i

        i -= 1

    i = 0
    j = 1

    while j < len(alignment):
        previous = alignment[i]
        following = alignment[j]

        if not previous.word.endswith(" ") and following.word in appended:
            previous.word = previous.word + following.word
            previous.tokens = previous.tokens + following.tokens

            following.word = ""
            following.tokens = []
        else: i = j

        j += 1

class WordTiming:
    def __init__(self, word, tokens, start, end, probability):
        self.word = word
        self.tokens = tokens
        self.start = start
        self.end = end
        self.probability = probability

def median_filter(x, filter_width):
    pad_width = filter_width // 2

    if x.shape[-1] <= pad_width: return x
    if (ndim := x.ndim) <= 2: x = x[None, None, :]

    assert (filter_width > 0 and filter_width % 2 == 1)

    result = None
    x = F.pad(x, (filter_width // 2, filter_width // 2, 0, 0), mode="reflect")

    if result is None: result = x.unfold(-1, filter_width, 1).sort()[0][..., filter_width // 2]
    if ndim <= 2: result = result[0, 0]

    return result

@nb.jit(nopython=True)
def backtrace(trace):
    i = trace.shape[0] - 1
    j = trace.shape[1] - 1

    trace[0, :] = 2
    trace[:, 0] = 1

    result = []
    while i > 0 or j > 0:
        result.append((i - 1, j - 1))

        if trace[i, j] == 0:
            i -= 1
            j -= 1
        elif trace[i, j] == 1: i -= 1
        elif trace[i, j] == 2: j -= 1
        else: raise ValueError

    return np.array(result)[::-1, :].T


@nb.jit(nopython=True, parallel=True)
def dtw_cpu(x):
    N, M = x.shape

    cost = np.ones((N + 1, M + 1), dtype=np.float32) * np.inf
    trace = -np.ones((N + 1, M + 1), dtype=np.float32)
    cost[0, 0] = 0

    for j in range(1, M + 1):
        for i in range(1, N + 1):
            c0 = cost[i - 1, j - 1]
            c1 = cost[i - 1, j]
            c2 = cost[i, j - 1]

            if c0 < c1 and c0 < c2: c, t = c0, 0
            elif c1 < c0 and c1 < c2: c, t = c1, 1
            else: c, t = c2, 2

            cost[i, j] = x[i - 1, j - 1] + c
            trace[i, j] = t

    return backtrace(trace)

def dtw(x):
    return dtw_cpu(x.double().cpu().numpy())

def find_alignment(model, tokenizer, text_tokens, mel, num_frames, *, medfilt_width = 7, qk_scale = 1.0):
    if len(text_tokens) == 0: return []

    tokens = torch.tensor([*tokenizer.sot_sequence, tokenizer.no_timestamps, *text_tokens, tokenizer.eot]).to(model.device)

    QKs = [None] * model.dims.n_text_layer
    hooks = [block.cross_attn.register_forward_hook(lambda _, ins, outs, index=i: QKs.__setitem__(index, outs[-1][0])) for i, block in enumerate(model.decoder.blocks)]

    with torch.no_grad():
        token_probs = model(mel.unsqueeze(0), tokens.unsqueeze(0))[0][len(tokenizer.sot_sequence) :, : tokenizer.eot].softmax(dim=-1)
        text_token_probs = token_probs[np.arange(len(text_tokens)), text_tokens].tolist()

    for hook in hooks:
        hook.remove()

    if not (opencl.is_available() or directml.is_available()):
        alignment_indices = model.alignment_heads.indices().T
    else:
        alignment_indices = [(l, h) for l in range(model.alignment_heads.size(0)) for h in range(model.alignment_heads.size(1)) if model.alignment_heads[l, h]]
        
    weights = (torch.stack([QKs[_l][_h] for _l, _h in alignment_indices])[:, :, : num_frames // 2] * qk_scale).softmax(dim=-1)
    std, mean = torch.std_mean(weights, dim=-2, keepdim=True, unbiased=False)

    if directml.is_available():
        weights = median_filter(((weights - mean) / std).cpu(), medfilt_width).to(weights.device)
    else:
        weights = median_filter((weights - mean) / std, medfilt_width)

    text_indices, time_indices = dtw(-weights.mean(axis=0)[len(tokenizer.sot_sequence) : -1])

    words, word_tokens = tokenizer.split_to_word_tokens(text_tokens + [tokenizer.eot])
    if len(word_tokens) <= 1: return []

    word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))
    jump_times = time_indices[np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)] / TOKENS_PER_SECOND

    return [WordTiming(word, tokens, start, end, probability) for word, tokens, start, end, probability in zip(words, word_tokens, jump_times[word_boundaries[:-1]], jump_times[word_boundaries[1:]], [np.mean(text_token_probs[i:j]) for i, j in zip(word_boundaries[:-1], word_boundaries[1:])])]

def add_word_timestamps(*, segments, model, tokenizer, mel, num_frames, prepend_punctuations = "\"'“¿([{-", append_punctuations = "\"'.。,，!！?？:：”)]}、", last_speech_timestamp, **kwargs):
    if len(segments) == 0: return

    text_tokens_per_segment = [[token for token in segment["tokens"] if token < tokenizer.eot] for segment in segments]

    text_tokens = list(itertools.chain.from_iterable(text_tokens_per_segment))
    alignment = find_alignment(model, tokenizer, text_tokens, mel, num_frames, **kwargs)

    word_durations = np.array([t.end - t.start for t in alignment])
    word_durations = word_durations[word_durations.nonzero()]

    median_duration = min(0.7, float(np.median(word_durations) if len(word_durations) > 0 else 0.0))
    max_duration = median_duration * 2

    if len(word_durations) > 0:
        sentence_end_marks = ".。!！?？"
        for i in range(1, len(alignment)):
            if alignment[i].end - alignment[i].start > max_duration:
                if alignment[i].word in sentence_end_marks: alignment[i].end = alignment[i].start + max_duration
                elif alignment[i - 1].word in sentence_end_marks: alignment[i].start = alignment[i].end - max_duration

    merge_punctuations(alignment, prepend_punctuations, append_punctuations)

    time_offset = segments[0]["seek"] * HOP_LENGTH / SAMPLE_RATE
    word_index = 0

    for segment, text_tokens in zip(segments, text_tokens_per_segment):
        saved_tokens = 0
        words = []

        while word_index < len(alignment) and saved_tokens < len(text_tokens):
            timing = alignment[word_index]

            if timing.word: words.append(dict(word=timing.word, start=round(time_offset + timing.start, 2), end=round(time_offset + timing.end, 2), probability=timing.probability))

            saved_tokens += len(timing.tokens)
            word_index += 1

        if len(words) > 0:
            if words[0]["end"] - last_speech_timestamp > median_duration * 4 and (words[0]["end"] - words[0]["start"] > max_duration or (len(words) > 1 and words[1]["end"] - words[0]["start"] > max_duration * 2)):
                if (len(words) > 1 and words[1]["end"] - words[1]["start"] > max_duration): words[0]["end"] = words[1]["start"] = max(words[1]["end"] / 2, words[1]["end"] - max_duration)
                words[0]["start"] = max(0, words[0]["end"] - max_duration)

            if (segment["start"] < words[0]["end"] and segment["start"] - 0.5 > words[0]["start"]): words[0]["start"] = max(0, min(words[0]["end"] - median_duration, segment["start"]))
            else: segment["start"] = words[0]["start"]

            if (segment["end"] > words[-1]["start"] and segment["end"] + 0.5 < words[-1]["end"]): words[-1]["end"] = max(words[-1]["start"] + median_duration, segment["end"])
            else: segment["end"] = words[-1]["end"]

            last_speech_timestamp = segment["end"]

        segment["words"] = words

@lru_cache(maxsize=None)
def mel_filters(device, n_mels):
    assert n_mels in {80, 128}

    with np.load(os.path.join(configs["speaker_diarization_path"], "assets", "mel_filters.npz"), allow_pickle=False) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)

def log_mel_spectrogram(audio, n_mels = 80, padding = 0, device = None):
    global stft

    if not torch.is_tensor(audio):
        if isinstance(audio, str): 
            from main.library.utils import load_audio
            audio = load_audio(audio, sample_rate=SAMPLE_RATE).astype(np.float32)
        audio = torch.from_numpy(audio)

    if device is not None: audio = audio.to(device)
    if padding > 0: audio = F.pad(audio, (0, padding))

    if str(audio.device).startswith(("ocl", "privateuseone")):
        if stft is None: 
            from main.library.backends.utils import STFT
            stft = STFT(N_FFT, HOP_LENGTH, N_FFT).to(audio.device)
        fft = stft.transform(audio.unsqueeze(0), eps=1e-9).squeeze(0)
    else:
        fft = torch.stft(audio, N_FFT, HOP_LENGTH, window=torch.hann_window(N_FFT).to(audio.device), return_complex=True)

    log_spec = (mel_filters(audio.device, n_mels) @ fft[..., :-1].abs() ** 2).clamp(min=1e-10).log10()
    return (log_spec.maximum(log_spec.max() - 8.0) + 4.0) / 4.0

def pad_or_trim(array, length = N_SAMPLES, *, axis = -1):
    if torch.is_tensor(array):
        if array.shape[axis] > length: array = array.index_select(dim=axis, index=torch.arange(length, device=array.device))

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length: array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array

def get_end(segments):
    return next((w["end"] for s in reversed(segments) for w in reversed(s["words"])), segments[-1]["end"] if segments else None)

def transcribe_function(model, audio, *, verbose = None, temperature = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0), compression_ratio_threshold = 2.4, logprob_threshold = -1.0, no_speech_threshold = 0.6, condition_on_previous_text = True, initial_prompt = None, carry_initial_prompt = False, word_timestamps = False, prepend_punctuations = "\"'“¿([{-", append_punctuations = "\"'.。,，!！?？:：”)]}、", clip_timestamps = "0", hallucination_silence_threshold = None, fp16 = False, **decode_options):
    dtype = torch.float16 if fp16 else torch.float32
    decode_options["fp16"] = fp16

    mel = log_mel_spectrogram(audio, model.dims.n_mels, padding=N_SAMPLES)
    content_frames = mel.shape[-1] - N_FRAMES
    content_duration = float(content_frames * HOP_LENGTH / SAMPLE_RATE)

    if decode_options.get("language", None) is None:
        if not model.is_multilingual: decode_options["language"] = "vi"
        else:
            mel_segment = pad_or_trim(mel, N_FRAMES).to(model.device).to(dtype)
            _, probs = model.detect_language(mel_segment)
            decode_options["language"] = max(probs, key=probs.get)

            if verbose is not None: logger.info(f"{LANGUAGES[decode_options['language']].title()}")

    language = decode_options["language"]
    task = decode_options.get("task", "transcribe")
    tokenizer = get_tokenizer(model.is_multilingual, num_languages=model.num_languages, language=language, task=task)

    if isinstance(clip_timestamps, str): clip_timestamps = [float(ts) for ts in (clip_timestamps.split(",") if clip_timestamps else [])]
    seek_points = [round(ts * FRAMES_PER_SECOND) for ts in clip_timestamps]

    if len(seek_points) == 0: seek_points.append(0)
    if len(seek_points) % 2 == 1: seek_points.append(content_frames)

    seek_clips = list(zip(seek_points[::2], seek_points[1::2]))
    punctuation = "\"'“¿([{-\"'.。,，!！?？:：”)]}、"

    def decode_with_fallback(segment):
        temperatures = ([temperature] if isinstance(temperature, (int, float)) else temperature)
        decode_result = None

        for t in temperatures:
            kwargs = {**decode_options}

            if t > 0:
                kwargs.pop("beam_size", None)
                kwargs.pop("patience", None)
            else: kwargs.pop("best_of", None)

            decode_result = model.decode(segment, DecodingOptions(**kwargs, temperature=t))
            needs_fallback = False

            if (compression_ratio_threshold is not None and decode_result.compression_ratio > compression_ratio_threshold): needs_fallback = True  
            if (logprob_threshold is not None and decode_result.avg_logprob < logprob_threshold): needs_fallback = True  
            if (no_speech_threshold is not None and decode_result.no_speech_prob > no_speech_threshold and logprob_threshold is not None and decode_result.avg_logprob < logprob_threshold): needs_fallback = False 
            if not needs_fallback: break

        return decode_result

    clip_idx = 0
    seek = seek_clips[clip_idx][0]

    input_stride = exact_div(N_FRAMES, model.dims.n_audio_ctx)  
    time_precision = (input_stride * HOP_LENGTH / SAMPLE_RATE) 

    all_tokens, all_segments = [], []
    prompt_reset_since = 0

    remaining_prompt_length = model.dims.n_text_ctx // 2 - 1

    if initial_prompt is not None:
        initial_prompt_tokens = tokenizer.encode(" " + initial_prompt.strip())
        all_tokens.extend(initial_prompt_tokens)
        remaining_prompt_length -= len(initial_prompt_tokens)
    else: initial_prompt_tokens = []

    def new_segment(*, start, end, tokens, result):
        tokens = tokens.tolist()
        return {"seek": seek, "start": start, "end": end, "text": tokenizer.decode([token for token in tokens if token < tokenizer.eot]), "tokens": tokens, "temperature": result.temperature, "avg_logprob": result.avg_logprob, "compression_ratio": result.compression_ratio, "no_speech_prob": result.no_speech_prob}

    with tqdm.tqdm(total=content_frames, unit="frames", disable=verbose is not False) as pbar:
        last_speech_timestamp = 0.0
        while clip_idx < len(seek_clips):
            seek_clip_start, seek_clip_end = seek_clips[clip_idx]
            if seek < seek_clip_start: seek = seek_clip_start

            if seek >= seek_clip_end:
                clip_idx += 1
                if clip_idx < len(seek_clips): seek = seek_clips[clip_idx][0]
                continue

            time_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
            window_end_time = float((seek + N_FRAMES) * HOP_LENGTH / SAMPLE_RATE)

            segment_size = min(N_FRAMES, content_frames - seek, seek_clip_end - seek)
            mel_segment = mel[:, seek : seek + segment_size]

            segment_duration = segment_size * HOP_LENGTH / SAMPLE_RATE
            mel_segment = pad_or_trim(mel_segment, N_FRAMES).to(model.device).to(dtype)

            if carry_initial_prompt: decode_options["prompt"] = initial_prompt_tokens + all_tokens[max(len(initial_prompt_tokens), prompt_reset_since):][-remaining_prompt_length:]
            else: decode_options["prompt"] = all_tokens[prompt_reset_since:]

            result = decode_with_fallback(mel_segment)
            tokens = torch.tensor(result.tokens)

            if no_speech_threshold is not None:
                should_skip = result.no_speech_prob > no_speech_threshold
                if (logprob_threshold is not None and result.avg_logprob > logprob_threshold):
                    should_skip = False

                if should_skip:
                    seek += segment_size  
                    continue

            previous_seek = seek
            current_segments = []

            def word_anomaly_score(word):
                probability = word.get("probability", 0.0)
                duration = word["end"] - word["start"]
                score = 0.0

                if probability < 0.15: score += 1.0
                if duration < 0.133: score += (0.133 - duration) * 15
                if duration > 2.0: score += duration - 2.0

                return score

            def is_segment_anomaly(segment):
                if segment is None or not segment["words"]: return False
                
                words = [w for w in segment["words"] if w["word"] not in punctuation]
                words = words[:8]

                score = sum(word_anomaly_score(w) for w in words)

                return score >= 3 or score + 0.01 >= len(words)

            def next_words_segment(segments):
                return next((s for s in segments if s["words"]), None)

            timestamp_tokens = tokens.ge(tokenizer.timestamp_begin)
            single_timestamp_ending = timestamp_tokens[-2:].tolist() == [False, True]

            consecutive = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0]
            consecutive.add_(1)

            if len(consecutive) > 0:
                slices = consecutive.tolist()
                if single_timestamp_ending:
                    slices.append(len(tokens))

                last_slice = 0
                for current_slice in slices:
                    sliced_tokens = tokens[last_slice:current_slice]
                    current_segments.append(new_segment(start=time_offset + (sliced_tokens[0].item() - tokenizer.timestamp_begin) * time_precision, end=time_offset + (sliced_tokens[-1].item() - tokenizer.timestamp_begin) * time_precision, tokens=sliced_tokens, result=result))
                    last_slice = current_slice

                if single_timestamp_ending: seek += segment_size
                else: seek += (tokens[last_slice - 1].item() - tokenizer.timestamp_begin) * input_stride
            else:
                duration = segment_duration

                timestamps = tokens[timestamp_tokens.nonzero().flatten()]
                if (len(timestamps) > 0 and timestamps[-1].item() != tokenizer.timestamp_begin): duration = (timestamps[-1].item() - tokenizer.timestamp_begin) * time_precision

                current_segments.append(new_segment(start=time_offset, end=time_offset + duration, tokens=tokens, result=result))
                seek += segment_size

            if word_timestamps:
                add_word_timestamps(segments=current_segments, model=model, tokenizer=tokenizer, mel=mel_segment, num_frames=segment_size, prepend_punctuations=prepend_punctuations, append_punctuations=append_punctuations, last_speech_timestamp=last_speech_timestamp)

                if not single_timestamp_ending:
                    last_word_end = get_end(current_segments)
                    if last_word_end is not None and last_word_end > time_offset: seek = round(last_word_end * FRAMES_PER_SECOND)

                if hallucination_silence_threshold is not None:
                    threshold = hallucination_silence_threshold

                    if not single_timestamp_ending:
                        last_word_end = get_end(current_segments)
                        if last_word_end is not None and last_word_end > time_offset: seek = round(last_word_end * FRAMES_PER_SECOND) if (window_end_time - last_word_end) > threshold else (previous_seek + segment_size)

                    first_segment = next_words_segment(current_segments)

                    if first_segment is not None and is_segment_anomaly(first_segment):
                        gap = first_segment["start"] - time_offset

                        if gap > threshold:
                            seek = previous_seek + round(gap * FRAMES_PER_SECOND)
                            continue

                    hal_last_end = last_speech_timestamp

                    for si in range(len(current_segments)):
                        segment = current_segments[si]
                        if not segment["words"]: continue

                        if is_segment_anomaly(segment):
                            next_segment = next_words_segment(current_segments[si + 1 :])
                            hal_next_start = next_segment["words"][0]["start"] if next_segment is not None else (time_offset + segment_duration)

                            if (segment["start"] - hal_last_end > threshold or segment["start"] < threshold or segment["start"] - time_offset < 2.0) and (hal_next_start - segment["end"] > threshold or is_segment_anomaly(next_segment) or window_end_time - segment["end"] < 2.0):
                                seek = round(max(time_offset + 1, segment["start"]) * FRAMES_PER_SECOND)
                                if content_duration - segment["end"] < threshold: seek = content_frames

                                current_segments[si:] = []
                                break

                        hal_last_end = segment["end"]

                last_word_end = get_end(current_segments)
                if last_word_end is not None: last_speech_timestamp = last_word_end

            for _, segment in enumerate(current_segments):
                if segment["start"] == segment["end"] or segment["text"].strip() == "":
                    segment["text"] = ""
                    segment["tokens"] = []
                    segment["words"] = []

            all_segments.extend([{"id": i, **segment} for i, segment in enumerate(current_segments, start=len(all_segments))])
            all_tokens.extend([token for segment in current_segments for token in segment["tokens"]])

            if not condition_on_previous_text or result.temperature > 0.5: prompt_reset_since = len(all_tokens)
            pbar.update(min(content_frames, seek) - previous_seek)

    return dict(text=tokenizer.decode(all_tokens[len(initial_prompt_tokens) :]), segments=all_segments, language=language)

def compression_ratio(text):
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))

def sinusoids(length, channels, max_timescale=10000):
    assert channels % 2 == 0

    scaled_time = torch.arange(length)[:, np.newaxis] * (-(np.log(max_timescale) / (channels // 2 - 1)) * torch.arange(channels // 2)).exp()[np.newaxis, :]
    return torch.cat([scaled_time.sin(), scaled_time.cos()], dim=1)

@torch.no_grad()
def detect_language_function(model, mel, tokenizer = None):
    if tokenizer is None: tokenizer = get_tokenizer(model.is_multilingual, num_languages=model.num_languages)
    if (tokenizer.language is None or tokenizer.language_token not in tokenizer.sot_sequence): raise ValueError

    single = mel.ndim == 2

    if single: mel = mel.unsqueeze(0)
    if mel.shape[-2:] != (model.dims.n_audio_ctx, model.dims.n_audio_state): mel = model.encoder(mel)

    n_audio = mel.shape[0]
    logits = model.logits(torch.tensor([[tokenizer.sot]] * n_audio).to(mel.device), mel)[:, 0]

    mask = torch.ones(logits.shape[-1], dtype=torch.bool)
    mask[list(tokenizer.all_language_tokens)] = False

    logits[:, mask] = -np.inf

    language_tokens = logits.argmax(dim=-1)
    language_probs = [{c: logits.softmax(dim=-1).cpu()[i, j].item() for j, c in zip(tokenizer.all_language_tokens, tokenizer.all_language_codes)} for i in range(n_audio)]

    if single:
        language_tokens = language_tokens[0]
        language_probs = language_probs[0]

    return language_tokens, language_probs

@lru_cache(maxsize=None)
def get_tokenizer(multilingual, *, num_languages = 99, language = None, task = None):
    if language is not None:
        language = language.lower()
        if language not in LANGUAGES:
            if language in TO_LANGUAGE_CODE: language = TO_LANGUAGE_CODE[language]
            else: raise ValueError

    if multilingual:
        encoding_name = "multilingual"
        language = language or "en"
        task = task or "transcribe"
    else:
        encoding_name = "gpt2"
        language = None
        task = None

    return Tokenizer(encoding_name=encoding_name, num_languages=num_languages, language=language, task=task)

@lru_cache(maxsize=None)
def get_encoding(name = "gpt2", num_languages = 99):
    vocab_path = os.path.join(configs["speaker_diarization_path"], "assets", f"{name}.tiktoken")
    ranks = {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in open(vocab_path) if line)}

    n_vocab = len(ranks)
    special_tokens = {}

    specials = ["<|endoftext|>", "<|startoftranscript|>", *[f"<|{lang}|>" for lang in list(LANGUAGES.keys())[:num_languages]], "<|translate|>", "<|transcribe|>", "<|startoflm|>", "<|startofprev|>", "<|nospeech|>", "<|notimestamps|>", *[f"<|{i * 0.02:.2f}|>" for i in range(1501)]]

    for token in specials:
        special_tokens[token] = n_vocab
        n_vocab += 1

    return tiktoken.Encoding(name=os.path.basename(vocab_path), explicit_n_vocab=n_vocab, pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", mergeable_ranks=ranks, special_tokens=special_tokens)

class DecodingOptions:
    def __init__(self, task = "transcribe", language = None, temperature = 0.0, sample_len = None, best_of = None, beam_size = None, patience = None, length_penalty = None, prompt = None, prefix = None, suppress_tokens = "-1", suppress_blank = True, without_timestamps = False, max_initial_timestamp = 1.0, fp16 = False):
        self.task = task
        self.language = language
        self.temperature = temperature
        self.sample_len = sample_len
        self.best_of = best_of
        self.beam_size = beam_size
        self.patience = patience
        self.length_penalty = length_penalty
        self.prompt = prompt
        self.prefix = prefix
        self.suppress_tokens = suppress_tokens
        self.suppress_blank = suppress_blank
        self.without_timestamps = without_timestamps
        self.max_initial_timestamp = max_initial_timestamp
        self.fp16 = fp16

@torch.no_grad()
def decode_function(model, mel, options = DecodingOptions(), **kwargs):
    if single := mel.ndim == 2: mel = mel.unsqueeze(0)
    if kwargs: options = replace(options, **kwargs)

    result = DecodingTask(model, options).run(mel)
    return result[0] if single else result

class ModelDimensions:
    def __init__(self, n_mels, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer, n_vocab, n_text_ctx, n_text_state, n_text_head, n_text_layer):
        self.n_mels = n_mels
        self.n_audio_ctx = n_audio_ctx
        self.n_audio_state = n_audio_state
        self.n_audio_head = n_audio_head
        self.n_audio_layer = n_audio_layer
        self.n_vocab = n_vocab
        self.n_text_ctx = n_text_ctx
        self.n_text_state = n_text_state
        self.n_text_head = n_text_head
        self.n_text_layer = n_text_layer

class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)
    
class Linear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))

class Conv1d(nn.Conv1d):
    def _conv_forward(self, x, weight, bias):
        return super()._conv_forward(x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype))

class TextDecoder(nn.Module):
    def __init__(self, n_vocab, n_ctx, n_state, n_head, n_layer):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks = nn.ModuleList([ResidualAttentionBlock(n_state, n_head, cross_attention=True) for _ in range(n_layer)])
        self.ln = LayerNorm(n_state)
        self.register_buffer("mask", torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1), persistent=False)

    def forward(self, x, xa, kv_cache = None):
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]).to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        return x @ self.token_embedding.weight.to(x.dtype).transpose(0, 1).float()

class AudioEncoder(nn.Module):
    def __init__(self, n_mels, n_ctx, n_state, n_head, n_layer):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks = nn.ModuleList([ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)])
        self.ln_post = LayerNorm(n_state)

    def forward(self, x):
        x = F.gelu(self.conv2(F.gelu(self.conv1(x)))).permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        return self.ln_post(x)

class Whisper(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(self.dims.n_mels, self.dims.n_audio_ctx, self.dims.n_audio_state, self.dims.n_audio_head, self.dims.n_audio_layer)
        self.decoder = TextDecoder(self.dims.n_vocab, self.dims.n_text_ctx, self.dims.n_text_state, self.dims.n_text_head, self.dims.n_text_layer)

        all_heads = torch.zeros(self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool)
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.register_buffer("alignment_heads", all_heads if opencl.is_available() or directml.is_available() else all_heads.to_sparse(), persistent=False)

    def set_alignment_heads(self, dump):
        alignment = torch.from_numpy(np.frombuffer(gzip.decompress(base64.b85decode(dump)), dtype=bool).copy()).reshape(self.dims.n_text_layer, self.dims.n_text_head)
        if not (opencl.is_available() or directml.is_available()): alignment = alignment.to_sparse()

        self.register_buffer("alignment_heads", alignment, persistent=False)

    def embed_audio(self, mel):
        return self.encoder(mel)

    def logits(self, tokens, audio_features):
        return self.decoder(tokens, audio_features)

    def forward(self, mel, tokens):
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab >= 51865

    @property
    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)

    def install_kv_cache_hooks(self, cache = None):
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            cache[module] = output if module not in cache or output.shape[1] > self.dims.n_text_ctx else torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function

class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state, n_head, cross_attention = False):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)
        self.cross_attn = (MultiHeadAttention(n_state, n_head) if cross_attention else None)
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(self, x, xa = None, mask = None, kv_cache = None):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn: x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]

        return x + self.mlp(self.mlp_ln(x))
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_state, n_head):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(self, x, xa = None, mask = None, kv_cache = None):
        k, v = (self.key(x if xa is None else xa), self.value(x if xa is None else xa)) if kv_cache is None or xa is None or self.key not in kv_cache else (kv_cache[self.key], kv_cache[self.value])
        wv, qk = self.qkv_attention(self.query(x), k, v, mask)

        return self.out(wv), qk

    def qkv_attention(self, q, k, v, mask = None):
        _, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q, k, v = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3), k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3), v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = (q * scale) @ (k * scale).transpose(-1, -2)
        if mask is not None: qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        return (F.softmax(qk, dim=-1).to(q.dtype) @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()
    
class LogitFilter:
    def apply(self, logits, tokens):
        pass

class SuppressBlank(LogitFilter):
    def __init__(self, tokenizer, sample_begin):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin

    def apply(self, logits, tokens):
        if tokens.shape[1] == self.sample_begin: logits[:, self.tokenizer.encode(" ") + [self.tokenizer.eot]] = -np.inf

class SuppressTokens(LogitFilter):
    def __init__(self, suppress_tokens):
        self.suppress_tokens = list(suppress_tokens)

    def apply(self, logits, tokens):
        logits[:, self.suppress_tokens] = -np.inf

class Inference:
    def logits(self, tokens, audio_features):
        pass

    def rearrange_kv_cache(self, source_indices):
        pass

    def cleanup_caching(self):
        pass

class PyTorchInference(Inference):
    def __init__(self, model, initial_token_length):
        self.model = model
        self.initial_token_length = initial_token_length
        self.kv_cache = {}
        self.hooks = []

        self.kv_modules = [block.attn.key for block in self.model.decoder.blocks] + [block.attn.value for block in self.model.decoder.blocks]

    def logits(self, tokens, audio_features):
        if not self.kv_cache: self.kv_cache, self.hooks = self.model.install_kv_cache_hooks()
        if tokens.shape[-1] > self.initial_token_length: tokens = tokens[:, -1:]

        return self.model.decoder(tokens, audio_features, kv_cache=self.kv_cache)

    def cleanup_caching(self):
        for hook in self.hooks:
            hook.remove()

        self.kv_cache = {}
        self.hooks = []

    def rearrange_kv_cache(self, source_indices):
        if source_indices != list(range(len(source_indices))):
            for module in self.kv_modules:
                self.kv_cache[module] = self.kv_cache[module][source_indices].detach()

class SequenceRanker:
    def rank(self, tokens, sum_logprobs):
        pass

class MaximumLikelihoodRanker(SequenceRanker):
    def __init__(self, length_penalty):
        self.length_penalty = length_penalty

    def rank(self, tokens, sum_logprobs):
        def scores(logprobs, lengths):
            result = []
            for logprob, length in zip(logprobs, lengths):
                result.append(logprob / (length if self.length_penalty is None else ((5 + length) / 6) ** self.length_penalty))
            return result

        return [np.argmax(scores(p, l)) for p, l in zip(sum_logprobs, [[len(t) for t in s] for s in tokens])]
    
class TokenDecoder:
    def reset(self):
        pass

    def update(self, tokens, logits, sum_logprobs):
        pass

    def finalize(self, tokens, sum_logprobs):
        pass

class GreedyDecoder(TokenDecoder):
    def __init__(self, temperature, eot):
        self.temperature = temperature
        self.eot = eot

    def update(self, tokens, logits, sum_logprobs):
        next_tokens = logits.argmax(dim=-1) if self.temperature == 0 else (
            Categorical(logits=(logits / self.temperature).cpu() if opencl.is_available() else (logits / self.temperature))
        ).sample().to(logits.device)

        logprobs = F.log_softmax(logits.float(), dim=-1)
        sum_logprobs += logprobs[torch.arange(logprobs.shape[0]), next_tokens] * (tokens[:, -1] != self.eot)

        next_tokens[tokens[:, -1] == self.eot] = self.eot
        tokens = torch.cat([tokens, next_tokens[:, None]], dim=-1)

        return tokens, (tokens[:, -1] == self.eot).all()

    def finalize(self, tokens, sum_logprobs):
        return F.pad(tokens, (0, 1), value=self.eot), sum_logprobs.tolist()

class BeamSearchDecoder(TokenDecoder):
    def __init__(self, beam_size, eot, inference, patience = None):
        self.beam_size = beam_size
        self.eot = eot
        self.inference = inference
        self.patience = patience or 1.0
        self.max_candidates = round(beam_size * self.patience)
        self.finished_sequences = None

        assert (self.max_candidates > 0)

    def reset(self):
        self.finished_sequences = None

    def update(self, tokens, logits, sum_logprobs):
        if tokens.shape[0] % self.beam_size != 0: raise ValueError(f"{tokens.shape}[0] % {self.beam_size} != 0")

        n_audio = tokens.shape[0] // self.beam_size
        if self.finished_sequences is None: self.finished_sequences = [{} for _ in range(n_audio)]

        logprobs = F.log_softmax(logits.float(), dim=-1)
        next_tokens, source_indices, finished_sequences = [], [], []

        for i in range(n_audio):
            scores, sources, finished = {}, {}, {}

            for j in range(self.beam_size):
                idx = i * self.beam_size + j
                prefix = tokens[idx].tolist()
                for logprob, token in zip(*logprobs[idx].topk(self.beam_size + 1)):
                    sequence = tuple(prefix + [token.item()])
                    scores[sequence] = (sum_logprobs[idx] + logprob).item()
                    sources[sequence] = idx

            saved = 0

            for sequence in sorted(scores, key=scores.get, reverse=True):
                if sequence[-1] == self.eot: finished[sequence] = scores[sequence]
                else:
                    sum_logprobs[len(next_tokens)] = scores[sequence]
                    next_tokens.append(sequence)
                    source_indices.append(sources[sequence])

                    saved += 1
                    if saved == self.beam_size: break

            finished_sequences.append(finished)

        self.inference.rearrange_kv_cache(source_indices)
        assert len(self.finished_sequences) == len(finished_sequences)

        for previously_finished, newly_finished in zip(self.finished_sequences, finished_sequences):
            for seq in sorted(newly_finished, key=newly_finished.get, reverse=True):
                if len(previously_finished) >= self.max_candidates: break  
                previously_finished[seq] = newly_finished[seq]

        return torch.tensor(next_tokens, device=tokens.device), all(len(sequences) >= self.max_candidates for sequences in self.finished_sequences)

    def finalize(self, preceding_tokens, sum_logprobs):
        sum_logprobs = sum_logprobs.cpu()

        for i, sequences in enumerate(self.finished_sequences):
            if (len(sequences) < self.beam_size):  
                for j in list(np.argsort(sum_logprobs[i]))[::-1]:
                    sequence = preceding_tokens[i, j].tolist() + [self.eot]
                    sequences[tuple(sequence)] = sum_logprobs[i][j].item()
                    if len(sequences) >= self.beam_size: break

        return [[torch.tensor(seq) for seq in sequences.keys()] for sequences in self.finished_sequences], [list(sequences.values()) for sequences in self.finished_sequences]

class ApplyTimestampRules(LogitFilter):
    def __init__(self, tokenizer, sample_begin, max_initial_timestamp_index):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin
        self.max_initial_timestamp_index = max_initial_timestamp_index

    def apply(self, logits, tokens):
        if self.tokenizer.no_timestamps is not None: logits[:, self.tokenizer.no_timestamps] = -np.inf

        for k in range(tokens.shape[0]):
            sampled_tokens = tokens[k, self.sample_begin :]
            seq = [t for t in sampled_tokens.tolist()]

            last_was_timestamp = (len(seq) >= 1 and seq[-1] >= self.tokenizer.timestamp_begin)
            penultimate_was_timestamp = (len(seq) < 2 or seq[-2] >= self.tokenizer.timestamp_begin)

            if last_was_timestamp:
                if penultimate_was_timestamp: logits[k, self.tokenizer.timestamp_begin :] = -np.inf
                else: logits[k, : self.tokenizer.eot] = -np.inf

            timestamps = sampled_tokens[sampled_tokens.ge(self.tokenizer.timestamp_begin)]

            if timestamps.numel() > 0: logits[k, self.tokenizer.timestamp_begin : timestamps[-1] if last_was_timestamp and not penultimate_was_timestamp else (timestamps[-1] + 1)] = -np.inf

        if tokens.shape[1] == self.sample_begin:
            logits[:, : self.tokenizer.timestamp_begin] = -np.inf

            if self.max_initial_timestamp_index is not None:
                last_allowed = (self.tokenizer.timestamp_begin + self.max_initial_timestamp_index)
                logits[:, last_allowed + 1 :] = -np.inf

        logprobs = F.log_softmax(logits.float(), dim=-1)
        for k in range(tokens.shape[0]):
            if logprobs[k, self.tokenizer.timestamp_begin :].logsumexp(dim=-1) > logprobs[k, : self.tokenizer.timestamp_begin].max(): logits[k, : self.tokenizer.timestamp_begin] = -np.inf

class DecodingTask:
    def __init__(self, model, options):
        self.model = model

        language = options.language or "en"
        tokenizer = get_tokenizer(model.is_multilingual, num_languages=model.num_languages, language=language, task=options.task)

        self.tokenizer = tokenizer
        self.options = self._verify_options(options)

        self.n_group = options.beam_size or options.best_of or 1
        self.n_ctx = model.dims.n_text_ctx
        self.sample_len = options.sample_len or model.dims.n_text_ctx // 2

        self.sot_sequence = tokenizer.sot_sequence
        if self.options.without_timestamps: self.sot_sequence = tokenizer.sot_sequence_including_notimestamps

        self.initial_tokens = self._get_initial_tokens()
        self.sample_begin = len(self.initial_tokens)
        self.sot_index = self.initial_tokens.index(tokenizer.sot)
        self.inference = PyTorchInference(model, len(self.initial_tokens))
        self.sequence_ranker = MaximumLikelihoodRanker(options.length_penalty)
        self.decoder = BeamSearchDecoder(options.beam_size, tokenizer.eot, self.inference, options.patience) if options.beam_size is not None else GreedyDecoder(options.temperature, tokenizer.eot)

        self.logit_filters = []

        if self.options.suppress_blank: self.logit_filters.append(SuppressBlank(self.tokenizer, self.sample_begin))
        if self.options.suppress_tokens: self.logit_filters.append(SuppressTokens(self._get_suppress_tokens()))

        if not options.without_timestamps:
            max_initial_timestamp_index = None
            if options.max_initial_timestamp: max_initial_timestamp_index = round(self.options.max_initial_timestamp / (CHUNK_LENGTH / model.dims.n_audio_ctx))
            self.logit_filters.append(ApplyTimestampRules(tokenizer, self.sample_begin, max_initial_timestamp_index))

    def _verify_options(self, options):
        if options.beam_size is not None and options.best_of is not None: raise ValueError
        if options.temperature == 0 and options.best_of is not None: raise ValueError
        if options.patience is not None and options.beam_size is None: raise ValueError
        if options.length_penalty is not None and not (0 <= options.length_penalty <= 1): raise ValueError

        return options

    def _get_initial_tokens(self):
        tokens = list(self.sot_sequence)

        if prefix := self.options.prefix:
            prefix_tokens = (self.tokenizer.encode(" " + prefix.strip()) if isinstance(prefix, str) else prefix)
            if self.sample_len is not None: prefix_tokens = prefix_tokens[-(self.n_ctx // 2 - self.sample_len):]
            tokens = tokens + prefix_tokens

        if prompt := self.options.prompt: tokens = ([self.tokenizer.sot_prev] + (self.tokenizer.encode(" " + prompt.strip()) if isinstance(prompt, str) else prompt)[-(self.n_ctx // 2 - 1) :] + tokens)

        return tuple(tokens)

    def _get_suppress_tokens(self):
        suppress_tokens = self.options.suppress_tokens
        if isinstance(suppress_tokens, str): suppress_tokens = [int(t) for t in suppress_tokens.split(",")]

        if -1 in suppress_tokens:
            suppress_tokens = [t for t in suppress_tokens if t >= 0]
            suppress_tokens.extend(self.tokenizer.non_speech_tokens)
        elif suppress_tokens is None or len(suppress_tokens) == 0: suppress_tokens = [] 
        else: assert isinstance(suppress_tokens, list)

        suppress_tokens.extend([self.tokenizer.transcribe, self.tokenizer.translate, self.tokenizer.sot, self.tokenizer.sot_prev, self.tokenizer.sot_lm])

        if self.tokenizer.no_speech is not None: suppress_tokens.append(self.tokenizer.no_speech)
        return tuple(sorted(set(suppress_tokens)))

    def _get_audio_features(self, mel):
        if self.options.fp16: mel = mel.half()

        audio_features = mel if mel.shape[-2:] == (self.model.dims.n_audio_ctx, self.model.dims.n_audio_state) else self.model.encoder(mel)
        if audio_features.dtype != (torch.float16 if self.options.fp16 else torch.float32): return TypeError

        return audio_features

    def _detect_language(self, audio_features, tokens):
        languages = [self.options.language] * audio_features.shape[0]
        lang_probs = None

        if self.options.language is None or self.options.task == "lang_id":
            lang_tokens, lang_probs = self.model.detect_language(audio_features, self.tokenizer)
            languages = [max(probs, key=probs.get) for probs in lang_probs]

            if self.options.language is None: tokens[:, self.sot_index + 1] = lang_tokens

        return languages, lang_probs

    def _main_loop(self, audio_features, tokens):
        n_batch = tokens.shape[0]
        sum_logprobs = torch.zeros(n_batch, device=audio_features.device)
        no_speech_probs = [np.nan] * n_batch

        try:
            for i in range(self.sample_len):
                logits = self.inference.logits(tokens, audio_features)

                if (i == 0 and self.tokenizer.no_speech is not None):  
                    probs_at_sot = logits[:, self.sot_index].float().softmax(dim=-1)
                    no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech].tolist()

                logits = logits[:, -1]
                for logit_filter in self.logit_filters:
                    logit_filter.apply(logits.to("cpu") if opencl.is_available() else logits, tokens)

                tokens, completed = self.decoder.update(tokens, logits, sum_logprobs)
                if completed or tokens.shape[-1] > self.n_ctx: break
        finally:
            self.inference.cleanup_caching()

        return tokens, sum_logprobs, no_speech_probs

    @torch.no_grad()
    def run(self, mel):
        self.decoder.reset()
        tokenizer = self.tokenizer
        n_audio = mel.shape[0]

        audio_features = self._get_audio_features(mel)  
        tokens = torch.tensor([self.initial_tokens]).repeat(n_audio, 1)

        languages, language_probs = self._detect_language(audio_features, tokens)
        if self.options.task == "lang_id": return [DecodingResult(audio_features=features, language=language, language_probs=probs) for features, language, probs in zip(audio_features, languages, language_probs)]

        tokens = tokens.repeat_interleave(self.n_group, dim=0).to(audio_features.device)
        tokens, sum_logprobs, no_speech_probs = self._main_loop(audio_features, tokens)

        audio_features = audio_features[:: self.n_group]
        no_speech_probs = no_speech_probs[:: self.n_group]

        assert audio_features.shape[0] == len(no_speech_probs) == n_audio

        tokens = tokens.reshape(n_audio, self.n_group, -1)
        sum_logprobs = sum_logprobs.reshape(n_audio, self.n_group)

        tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)
        tokens = [[t[self.sample_begin : (t == tokenizer.eot).nonzero()[0, 0]] for t in s] for s in tokens]

        selected = self.sequence_ranker.rank(tokens, sum_logprobs)
        tokens = [t[i].tolist() for i, t in zip(selected, tokens)]

        fields = ([tokenizer.decode(t).strip() for t in tokens], languages, tokens, audio_features, [lp / (len(t) + 1) for t, lp in zip(tokens, [lp[i] for i, lp in zip(selected, sum_logprobs)])], no_speech_probs)
        if len(set(map(len, fields))) != 1: raise RuntimeError

        return [DecodingResult(audio_features=features, language=language, tokens=tokens, text=text, avg_logprob=avg_logprob, no_speech_prob=no_speech_prob, temperature=self.options.temperature, compression_ratio=compression_ratio(text)) for text, language, tokens, features, avg_logprob, no_speech_prob in zip(*fields)]
    
class DecodingResult:
    def __init__(self, audio_features, language, language_probs = None, tokens = None, text = "", avg_logprob = np.nan, no_speech_prob = np.nan, temperature = np.nan, compression_ratio = np.nan):
        self.audio_features = audio_features
        self.language = language
        self.language_probs = language_probs if language_probs is not None else {}
        self.tokens = tokens if tokens is not None else []
        self.text = text
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob
        self.temperature = temperature
        self.compression_ratio = compression_ratio

class Tokenizer:
    def __init__(self, encoding_name, num_languages = 2, language = None, task = None, sot_sequence = ()):
        self.encoding = get_encoding(name=encoding_name, num_languages=num_languages)
        self.num_languages = num_languages
        self.language = language
        self.task = task
        self.sot_sequence = sot_sequence 
        self.special_tokens = {}

        for special in self.encoding.special_tokens_set:
            special_token = self.encoding.encode_single_token(special)
            self.special_tokens[special] = special_token

        sot = self.special_tokens["<|startoftranscript|>"]
        langs = tuple(LANGUAGES.keys())[: self.num_languages]
        sot_sequence = [sot]

        if self.language is not None: sot_sequence.append(sot + 1 + langs.index(self.language))
        if self.task is not None: sot_sequence.append(self.special_tokens["<|transcribe|>"] if self.task == "transcribe" else self.special_tokens["<|translate|>"])

        self.sot_sequence = tuple(sot_sequence)

    def encode(self, text, **kwargs):
        return self.encoding.encode(text, **kwargs)

    def decode(self, token_ids, **kwargs):
        return self.encoding.decode([t for t in token_ids if t < self.timestamp_begin], **kwargs)

    def decode_with_timestamps(self, token_ids, **kwargs):
        return self.encoding.decode(token_ids, **kwargs)

    @cached_property
    def eot(self):
        return self.encoding.eot_token

    @cached_property
    def transcribe(self):
        return self.special_tokens["<|transcribe|>"]

    @cached_property
    def translate(self):
        return self.special_tokens["<|translate|>"]

    @cached_property
    def sot(self):
        return self.special_tokens["<|startoftranscript|>"]

    @cached_property
    def sot_lm(self):
        return self.special_tokens["<|startoflm|>"]

    @cached_property
    def sot_prev(self):
        return self.special_tokens["<|startofprev|>"]

    @cached_property
    def no_speech(self):
        return self.special_tokens["<|nospeech|>"]

    @cached_property
    def no_timestamps(self):
        return self.special_tokens["<|notimestamps|>"]

    @cached_property
    def timestamp_begin(self):
        return self.special_tokens["<|0.00|>"]

    @cached_property
    def language_token(self):
        if self.language is None: raise ValueError
        return self.to_language_token(self.language)

    def to_language_token(self, language):
        if token := self.special_tokens.get(f"<|{language}|>", None): return token
        raise KeyError

    @cached_property
    def all_language_tokens(self):
        result = []
        for token, token_id in self.special_tokens.items():
            if token.strip("<|>") in LANGUAGES: result.append(token_id)

        return tuple(result)[: self.num_languages]

    @cached_property
    def all_language_codes(self):
        return tuple(self.decode([_l]).strip("<|>") for _l in self.all_language_tokens)

    @cached_property
    def sot_sequence_including_notimestamps(self):
        return tuple(list(self.sot_sequence) + [self.no_timestamps])

    @cached_property
    def non_speech_tokens(self):
        symbols = list('"#()*+/:;<=>@[\\]^_`{|}~「」『』')
        symbols += ("<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪".split())

        miscellaneous = set("♩♪♫♬♭♮♯")
        assert all(0x2640 <= ord(c) <= 0x267F for c in miscellaneous)

        result = {self.encoding.encode(" -")[0], self.encoding.encode(" '")[0]}
        for symbol in symbols + list(miscellaneous):
            for tokens in [self.encoding.encode(symbol), self.encoding.encode(" " + symbol)]:
                if len(tokens) == 1 or symbol in miscellaneous: result.add(tokens[0])

        return tuple(sorted(result))

    def split_to_word_tokens(self, tokens):
        if self.language in {"zh", "ja", "th", "lo", "my", "yue"}: return self.split_tokens_on_unicode(tokens)
        return self.split_tokens_on_spaces(tokens)

    def split_tokens_on_unicode(self, tokens):
        replacement_char = "\ufffd"

        words, word_tokens, current_tokens = [], [], []
        unicode_offset = 0

        for token in tokens:
            current_tokens.append(token)
            decoded = self.decode_with_timestamps(current_tokens)

            if (replacement_char not in decoded or self.decode_with_timestamps(tokens)[unicode_offset + decoded.index(replacement_char)] == replacement_char):
                words.append(decoded)
                word_tokens.append(current_tokens)
                current_tokens = []
                unicode_offset += len(decoded)

        return words, word_tokens

    def split_tokens_on_spaces(self, tokens):
        subwords, subword_tokens_list = self.split_tokens_on_unicode(tokens)
        words, word_tokens = [], []

        for subword, subword_tokens in zip(subwords, subword_tokens_list):
            if (subword_tokens[0] >= self.eot) or (subword.startswith(" ")) or (subword.strip() in string.punctuation) or len(words) == 0:
                words.append(subword)
                word_tokens.append(subword_tokens)
            else:
                words[-1] = words[-1] + subword
                word_tokens[-1].extend(subword_tokens)

        return words, word_tokens
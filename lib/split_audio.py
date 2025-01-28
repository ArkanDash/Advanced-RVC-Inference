import os
from pydub import AudioSegment
from pydub.silence import detect_silence, detect_nonsilent

SEPERATE_DIR = os.path.join(os.getcwd(), "seperate")
TEMP_DIR = os.path.join(SEPERATE_DIR, "temp")
cache = {}

os.makedirs(SEPERATE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

def cache_result(func):
    def wrapper(*args, **kwargs):
        key = (args, frozenset(kwargs.items()))
        if key in cache:
            return cache[key]
        else:
            result = func(*args, **kwargs)
            cache[key] = result
            return result
    return wrapper   

def get_non_silent(audio_name, audio, min_silence, silence_thresh, seek_step, keep_silence):
    """
    Function to get non-silent parts of the audio.
    """
    nonsilent_ranges = detect_nonsilent(audio, min_silence_len=min_silence, silence_thresh=silence_thresh, seek_step=seek_step)
    nonsilent_files = []
    for index, range in enumerate(nonsilent_ranges):
        nonsilent_name = os.path.join(SEPERATE_DIR, f"{audio_name}_min{min_silence}_t{silence_thresh}_ss{seek_step}_ks{keep_silence}", f"nonsilent{index}-{audio_name}.wav")
        start, end = range[0] - keep_silence, range[1] + keep_silence
        audio[start:end].export(nonsilent_name, format="wav")
        nonsilent_files.append(nonsilent_name)
    return nonsilent_files

def get_silence(audio_name, audio, min_silence, silence_thresh, seek_step, keep_silence):
    """
    Function to get silent parts of the audio.
    """
    silence_ranges = detect_silence(audio, min_silence_len=min_silence, silence_thresh=silence_thresh, seek_step=seek_step)
    silence_files = []
    for index, range in enumerate(silence_ranges):
        silence_name = os.path.join(SEPERATE_DIR, f"{audio_name}_min{min_silence}_t{silence_thresh}_ss{seek_step}_ks{keep_silence}", f"silence{index}-{audio_name}.wav")
        start, end = range[0] + keep_silence, range[1] - keep_silence
        audio[start:end].export(silence_name, format="wav")
        silence_files.append(silence_name)
    return silence_files

@cache_result
def split_silence_nonsilent(input_path, min_silence=500, silence_thresh=-40, seek_step=1, keep_silence=100):
    """
    Function to split the audio into silent and non-silent parts.
    """
    audio_name = os.path.splitext(os.path.basename(input_path))[0]
    os.makedirs(os.path.join(SEPERATE_DIR, f"{audio_name}_min{min_silence}_t{silence_thresh}_ss{seek_step}_ks{keep_silence}"), exist_ok=True)
    audio = AudioSegment.silent(duration=1000) + AudioSegment.from_file(input_path) + AudioSegment.silent(duration=1000)
    silence_files = get_silence(audio_name, audio, min_silence, silence_thresh, seek_step, keep_silence)
    nonsilent_files = get_non_silent(audio_name, audio, min_silence, silence_thresh, seek_step, keep_silence)
    return silence_files, nonsilent_files

def adjust_audio_lengths(original_audios, inferred_audios):
    """
    Function to adjust the lengths of the inferred audio files list to match the original audio files length.
    """
    adjusted_audios = []
    for original_audio, inferred_audio in zip(original_audios, inferred_audios):
        audio_1 = AudioSegment.from_file(original_audio)
        audio_2 = AudioSegment.from_file(inferred_audio)
        
        if len(audio_1) > len(audio_2):
            audio_2 += AudioSegment.silent(duration=len(audio_1) - len(audio_2))
        else:
            audio_2 = audio_2[:len(audio_1)]
        
        adjusted_file = os.path.join(TEMP_DIR, f"adjusted-{os.path.basename(inferred_audio)}")
        audio_2.export(adjusted_file, format="wav")
        adjusted_audios.append(adjusted_file)
    
    return adjusted_audios

def combine_silence_nonsilent(silence_files, nonsilent_files, keep_silence, output):
    """
    Function to combine the silent and non-silent parts of the audio.
    """
    combined = AudioSegment.empty()
    for silence, nonsilent in zip(silence_files, nonsilent_files):
        combined += AudioSegment.from_wav(silence) + AudioSegment.from_wav(nonsilent)
    combined += AudioSegment.from_wav(silence_files[-1])
    combined = AudioSegment.silent(duration=keep_silence) + combined[1000:-1000] + AudioSegment.silent(duration=keep_silence)
    combined.export(output, format="wav")
    return output
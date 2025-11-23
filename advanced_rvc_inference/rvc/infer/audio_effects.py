import os
import sys
import librosa
import argparse

import numpy as np
import soundfile as sf

from distutils.util import strtobool
from scipy.signal import butter, filtfilt
from pedalboard import Pedalboard, Chorus, Distortion, Reverb, PitchShift, Delay, Limiter, Gain, Bitcrush, Clipping, Compressor, Phaser, HighpassFilter

sys.path.append(os.getcwd())

from main.library.utils import pydub_load
from main.app.core.ui import replace_export_format
from main.app.variables import translations, logger

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_effects", action='store_true')
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./audios/apply_effects.wav")
    parser.add_argument("--export_format", type=str, default="wav")
    parser.add_argument("--resample", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--resample_sr", type=int, default=0)
    parser.add_argument("--chorus", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--chorus_depth", type=float, default=0.5)
    parser.add_argument("--chorus_rate", type=float, default=1.5)
    parser.add_argument("--chorus_mix", type=float, default=0.5)
    parser.add_argument("--chorus_delay", type=int, default=10)
    parser.add_argument("--chorus_feedback", type=float, default=0)
    parser.add_argument("--distortion", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--drive_db", type=int, default=20)
    parser.add_argument("--reverb", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--reverb_room_size", type=float, default=0.5)
    parser.add_argument("--reverb_damping", type=float, default=0.5)
    parser.add_argument("--reverb_wet_level", type=float, default=0.33)
    parser.add_argument("--reverb_dry_level", type=float, default=0.67)
    parser.add_argument("--reverb_width", type=float, default=1)
    parser.add_argument("--reverb_freeze_mode", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--pitchshift", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--pitch_shift", type=int, default=0)
    parser.add_argument("--delay", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--delay_seconds", type=float, default=0.5)
    parser.add_argument("--delay_feedback", type=float, default=0.5)
    parser.add_argument("--delay_mix", type=float, default=0.5)
    parser.add_argument("--compressor", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--compressor_threshold", type=int, default=-20)
    parser.add_argument("--compressor_ratio", type=float, default=4)
    parser.add_argument("--compressor_attack_ms", type=float, default=10)
    parser.add_argument("--compressor_release_ms", type=int, default=200)
    parser.add_argument("--limiter", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--limiter_threshold", type=int, default=0)
    parser.add_argument("--limiter_release", type=int, default=100)
    parser.add_argument("--gain", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--gain_db", type=int, default=0)
    parser.add_argument("--bitcrush", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--bitcrush_bit_depth", type=int, default=16)
    parser.add_argument("--clipping", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--clipping_threshold", type=int, default=-10)
    parser.add_argument("--phaser", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--phaser_rate_hz", type=float, default=0.5)
    parser.add_argument("--phaser_depth", type=float, default=0.5)
    parser.add_argument("--phaser_centre_frequency_hz", type=int, default=1000)
    parser.add_argument("--phaser_feedback", type=float, default=0)
    parser.add_argument("--phaser_mix", type=float, default=0.5)
    parser.add_argument("--treble_bass_boost", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--bass_boost_db", type=int, default=0)
    parser.add_argument("--bass_boost_frequency", type=int, default=100)
    parser.add_argument("--treble_boost_db", type=int, default=0)
    parser.add_argument("--treble_boost_frequency", type=int, default=3000)
    parser.add_argument("--fade_in_out", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--fade_in_duration", type=float, default=2000)
    parser.add_argument("--fade_out_duration", type=float, default=2000)
    parser.add_argument("--audio_combination", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--audio_combination_input", type=str)
    parser.add_argument("--main_volume", type=int, default=0)
    parser.add_argument("--combination_volume", type=int, default=-7)
    
    return parser.parse_args()

def process_audio(input_path, output_path, resample, resample_sr, chorus_depth, chorus_rate, chorus_mix, chorus_delay, chorus_feedback, distortion_drive, reverb_room_size, reverb_damping, reverb_wet_level, reverb_dry_level, reverb_width, reverb_freeze_mode, pitch_shift, delay_seconds, delay_feedback, delay_mix, compressor_threshold, compressor_ratio, compressor_attack_ms, compressor_release_ms, limiter_threshold, limiter_release, gain_db, bitcrush_bit_depth, clipping_threshold, phaser_rate_hz, phaser_depth, phaser_centre_frequency_hz, phaser_feedback, phaser_mix, bass_boost_db, bass_boost_frequency, treble_boost_db, treble_boost_frequency, fade_in_duration, fade_out_duration, export_format, chorus, distortion, reverb, pitchshift, delay, compressor, limiter, gain, bitcrush, clipping, phaser, treble_bass_boost, fade_in_out, audio_combination, audio_combination_input, main_volume, combination_volume):
    def _filtfilt(b, a, audio):
        padlen = 3 * max(len(a), len(b))
        original_len = len(audio)

        if original_len <= padlen:
            pad_width = padlen - original_len + 1
            audio = np.pad(audio, (pad_width, 0), mode='reflect')

        filtered = filtfilt(b, a, audio, padlen=0)
        return filtered[-original_len:]
    
    def bass_boost(audio, gain_db, frequency, sample_rate):
        if gain_db >= 1:
            b, a = butter(4, frequency / (0.5 * sample_rate), btype='low')
            boosted = _filtfilt(b, a, audio)
            return boosted * (10 ** (gain_db / 20))
        return audio

    def treble_boost(audio, gain_db, frequency, sample_rate):
        if gain_db >= 1:
            b, a = butter(4, frequency / (0.5 * sample_rate), btype='high')
            boosted = _filtfilt(b, a, audio)
            return boosted * (10 ** (gain_db / 20))
        return audio

    def fade_out_effect(audio, sr, duration=3.0):
        length = int(duration * sr)
        end = audio.shape[0]
        if length > end: length = end  
        start = end - length
        audio[start:end] = audio[start:end] * np.linspace(1.0, 0.0, length)
        return audio

    def fade_in_effect(audio, sr, duration=3.0):
        length = int(duration * sr)
        start = 0
        if length > audio.shape[0]: length = audio.shape[0]  
        end = length
        audio[start:end] = audio[start:end] * np.linspace(0.0, 1.0, length)
        return audio

    if not input_path or not os.path.exists(input_path): 
        logger.warning(translations["input_not_valid"])
        sys.exit(1)

    if not output_path: 
        logger.warning(translations["output_not_valid"])
        sys.exit(1)
    
    if os.path.exists(output_path): os.remove(output_path)
    
    try:
        input_path = input_path.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        try:
            audio, sample_rate = sf.read(input_path, dtype=np.float32)
        except:
            audio, sample_rate = librosa.load(input_path, sr=None)
    except Exception as e:
        logger.debug(f"{translations['errors_loading_audio']}: {e}")
        raise RuntimeError(f"{translations['errors_loading_audio']}: {e}")

    try:
        board = Pedalboard([HighpassFilter()])

        if chorus: board.append(Chorus(depth=chorus_depth, rate_hz=chorus_rate, mix=chorus_mix, centre_delay_ms=chorus_delay, feedback=chorus_feedback))
        if distortion: board.append(Distortion(drive_db=distortion_drive))
        if reverb: board.append(Reverb(room_size=reverb_room_size, damping=reverb_damping, wet_level=reverb_wet_level, dry_level=reverb_dry_level, width=reverb_width, freeze_mode=int(reverb_freeze_mode)))
        if pitchshift: board.append(PitchShift(semitones=pitch_shift))
        if delay: board.append(Delay(delay_seconds=delay_seconds, feedback=delay_feedback, mix=delay_mix))
        if compressor: board.append(Compressor(threshold_db=compressor_threshold, ratio=compressor_ratio, attack_ms=compressor_attack_ms, release_ms=compressor_release_ms))
        if limiter: board.append(Limiter(threshold_db=limiter_threshold, release_ms=limiter_release))
        if gain: board.append(Gain(gain_db=gain_db))
        if bitcrush: board.append(Bitcrush(bit_depth=bitcrush_bit_depth))
        if clipping: board.append(Clipping(threshold_db=clipping_threshold)) 
        if phaser: board.append(Phaser(rate_hz=phaser_rate_hz, depth=phaser_depth, centre_frequency_hz=phaser_centre_frequency_hz, feedback=phaser_feedback, mix=phaser_mix))

        processed_audio = board(audio, sample_rate)

        if treble_bass_boost:
            processed_audio = bass_boost(processed_audio, bass_boost_db, bass_boost_frequency, sample_rate)
            processed_audio = treble_boost(processed_audio, treble_boost_db, treble_boost_frequency, sample_rate)

        if fade_in_out:
            processed_audio = fade_in_effect(processed_audio, sample_rate, fade_in_duration)
            processed_audio = fade_out_effect(processed_audio, sample_rate, fade_out_duration)
            
        if resample and resample_sr != sample_rate and resample_sr > 0:
            processed_audio = librosa.resample(processed_audio, orig_sr=sample_rate, target_sr=resample_sr, res_type="soxr_vhq")
            sample_rate = resample_sr

        sf.write(replace_export_format(output_path, export_format), processed_audio, sample_rate, format=export_format)
        if audio_combination: pydub_load(audio_combination_input, combination_volume).overlay(pydub_load(replace_export_format(output_path, export_format), main_volume)).export(replace_export_format(output_path, export_format), format=export_format)
    except Exception as e:
        import traceback
        logger.debug(traceback.format_exc())
        raise RuntimeError(translations["apply_error"].format(e=e))
    return output_path

def main():
    args = parse_arguments()
    process_audio(input_path=args.input_path, output_path=args.output_path, resample=args.resample, resample_sr=args.resample_sr, chorus_depth=args.chorus_depth, chorus_rate=args.chorus_rate, chorus_mix=args.chorus_mix, chorus_delay=args.chorus_delay, chorus_feedback=args.chorus_feedback, distortion_drive=args.drive_db, reverb_room_size=args.reverb_room_size, reverb_damping=args.reverb_damping, reverb_wet_level=args.reverb_wet_level, reverb_dry_level=args.reverb_dry_level, reverb_width=args.reverb_width, reverb_freeze_mode=args.reverb_freeze_mode, pitch_shift=args.pitch_shift, delay_seconds=args.delay_seconds, delay_feedback=args.delay_feedback, delay_mix=args.delay_mix, compressor_threshold=args.compressor_threshold, compressor_ratio=args.compressor_ratio, compressor_attack_ms=args.compressor_attack_ms, compressor_release_ms=args.compressor_release_ms, limiter_threshold=args.limiter_threshold, limiter_release=args.limiter_release, gain_db=args.gain_db, bitcrush_bit_depth=args.bitcrush_bit_depth, clipping_threshold=args.clipping_threshold, phaser_rate_hz=args.phaser_rate_hz, phaser_depth=args.phaser_depth, phaser_centre_frequency_hz=args.phaser_centre_frequency_hz, phaser_feedback=args.phaser_feedback, phaser_mix=args.phaser_mix, bass_boost_db=args.bass_boost_db, bass_boost_frequency=args.bass_boost_frequency, treble_boost_db=args.treble_boost_db, treble_boost_frequency=args.treble_boost_frequency, fade_in_duration=args.fade_in_duration, fade_out_duration=args.fade_out_duration, export_format=args.export_format, chorus=args.chorus, distortion=args.distortion, reverb=args.reverb, pitchshift=args.pitchshift, delay=args.delay, compressor=args.compressor, limiter=args.limiter, gain=args.gain, bitcrush=args.bitcrush, clipping=args.clipping, phaser=args.phaser, treble_bass_boost=args.treble_bass_boost, fade_in_out=args.fade_in_out, audio_combination=args.audio_combination, audio_combination_input=args.audio_combination_input, main_volume=args.main_volume, combination_volume=args.combination_volume)

if __name__ == "__main__": main()
import os
import sys
import random
import subprocess

sys.path.append(os.getcwd())

from main.app.variables import python, translations, configs
from main.app.core.ui import gr_info, gr_warning, process_output, replace_export_format

def audio_effects(input_path, output_path, resample, resample_sr, chorus_depth, chorus_rate, chorus_mix, chorus_delay, chorus_feedback, distortion_drive, reverb_room_size, reverb_damping, reverb_wet_level, reverb_dry_level, reverb_width, reverb_freeze_mode, pitch_shift, delay_seconds, delay_feedback, delay_mix, compressor_threshold, compressor_ratio, compressor_attack_ms, compressor_release_ms, limiter_threshold, limiter_release, gain_db, bitcrush_bit_depth, clipping_threshold, phaser_rate_hz, phaser_depth, phaser_centre_frequency_hz, phaser_feedback, phaser_mix, bass_boost_db, bass_boost_frequency, treble_boost_db, treble_boost_frequency, fade_in_duration, fade_out_duration, export_format, chorus, distortion, reverb, delay, compressor, limiter, gain, bitcrush, clipping, phaser, treble_bass_boost, fade_in_out, audio_combination, audio_combination_input, main_vol, combine_vol):
    if not input_path or not os.path.exists(input_path) or os.path.isdir(input_path): 
        gr_warning(translations["input_not_valid"])
        return None
        
    if not output_path:
        gr_warning(translations["output_not_valid"])
        return None
    
    if os.path.isdir(output_path): output_path = os.path.join(output_path, f"audio_effects.{export_format}")
    output_dir = os.path.dirname(output_path) or output_path

    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    output_path = process_output(output_path)
    
    gr_info(translations["start"].format(start=translations["apply_effect"]))

    subprocess.run([python, configs["audio_effects_path"], "--input_path", input_path, "--output_path", output_path, "--resample", str(resample), "--resample_sr", str(resample_sr), "--chorus_depth", str(chorus_depth), "--chorus_rate", str(chorus_rate), "--chorus_mix", str(chorus_mix), "--chorus_delay", str(chorus_delay), "--chorus_feedback", str(chorus_feedback), "--drive_db", str(distortion_drive), "--reverb_room_size", str(reverb_room_size), "--reverb_damping", str(reverb_damping), "--reverb_wet_level", str(reverb_wet_level), "--reverb_dry_level", str(reverb_dry_level), "--reverb_width", str(reverb_width), "--reverb_freeze_mode", str(reverb_freeze_mode), "--pitch_shift", str(pitch_shift), "--delay_seconds", str(delay_seconds), "--delay_feedback", str(delay_feedback), "--delay_mix", str(delay_mix), "--compressor_threshold", str(compressor_threshold), "--compressor_ratio", str(compressor_ratio), "--compressor_attack_ms", str(compressor_attack_ms), "--compressor_release_ms", str(compressor_release_ms), "--limiter_threshold", str(limiter_threshold), "--limiter_release", str(limiter_release), "--gain_db", str(gain_db), "--bitcrush_bit_depth", str(bitcrush_bit_depth), "--clipping_threshold", str(clipping_threshold), "--phaser_rate_hz", str(phaser_rate_hz), "--phaser_depth", str(phaser_depth), "--phaser_centre_frequency_hz", str(phaser_centre_frequency_hz), "--phaser_feedback", str(phaser_feedback), "--phaser_mix", str(phaser_mix), "--bass_boost_db", str(bass_boost_db), "--bass_boost_frequency", str(bass_boost_frequency), "--treble_boost_db", str(treble_boost_db), "--treble_boost_frequency", str(treble_boost_frequency), "--fade_in_duration", str(fade_in_duration), "--fade_out_duration", str(fade_out_duration), "--export_format", export_format, "--chorus", str(chorus), "--distortion", str(distortion), "--reverb", str(reverb), "--pitchshift", str(pitch_shift != 0), "--delay", str(delay), "--compressor", str(compressor), "--limiter", str(limiter), "--gain", str(gain), "--bitcrush", str(bitcrush), "--clipping", str(clipping), "--phaser", str(phaser), "--treble_bass_boost", str(treble_bass_boost), "--fade_in_out", str(fade_in_out), "--audio_combination", str(audio_combination), "--audio_combination_input", audio_combination_input, "--main_volume", str(main_vol), "--combination_volume", str(combine_vol)])

    gr_info(translations["success"])
    return replace_export_format(output_path, export_format)

def apply_voice_quirk(audio_path, mode, output_path, export_format):
    if not audio_path or not os.path.exists(audio_path) or os.path.isdir(audio_path): 
        gr_warning(translations["input_not_valid"])
        return None
        
    if not output_path:
        gr_warning(translations["output_not_valid"])
        return None
    
    if os.path.isdir(output_path): output_path = os.path.join(output_path, f"audio_quirk.{export_format}")
    output_dir = os.path.dirname(output_path) or output_path

    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    output_path = process_output(output_path)
    
    gr_info(translations["start"].format(start=translations["apply_effect"]))

    import librosa
    import numpy as np
    import soundfile as sf

    def vibrato(y, sr, freq=5, depth=0.003):
        return y[np.clip((np.arange(len(y)) + (depth * np.sin(2 * np.pi * freq * (np.arange(len(y)) / sr))) * sr).astype(int), 0, len(y) - 1)]

    y, sr = librosa.load(audio_path, sr=None)
    output_path = replace_export_format(output_path, export_format)

    mode = translations["quirk_choice"][mode]
    if mode == 0: mode = random.randint(1, 16)

    if mode == 1: y *= np.random.uniform(0.5, 0.8, size=len(y))
    elif mode == 2: y = librosa.effects.pitch_shift(y=y + np.random.normal(0, 0.01, y.shape), sr=sr, n_steps=np.random.uniform(-1.5, -3.5))
    elif mode == 3: y = librosa.effects.time_stretch(librosa.effects.pitch_shift(y=y, sr=sr, n_steps=3), rate=1.2)
    elif mode == 4: y = librosa.effects.time_stretch(librosa.effects.pitch_shift(y=y, sr=sr, n_steps=8), rate=1.3)
    elif mode == 5: y = librosa.effects.time_stretch(librosa.effects.pitch_shift(y=y, sr=sr, n_steps=-3), rate=0.75)
    elif mode == 6: y *= np.sin(np.linspace(0, np.pi * 20, len(y))) * 0.5 + 0.5
    elif mode == 7: y = librosa.effects.time_stretch(vibrato(librosa.effects.pitch_shift(y=y, sr=sr, n_steps=-4), sr, freq=3, depth=0.004), rate=0.85)
    elif mode == 8: y *= 0.6 + np.pad(y, (sr // 2, 0), mode='constant')[:len(y)] * 0.4
    elif mode == 9: y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=2) + np.sin(np.linspace(0, np.pi * 20, len(y))) * 0.02
    elif mode == 10: y = vibrato(y, sr, freq=8, depth=0.005)
    elif mode == 11: y = librosa.effects.time_stretch(librosa.effects.pitch_shift(y=y, sr=sr, n_steps=4), rate=1.25)
    elif mode == 12: y = np.hstack([np.pad(f, (0, int(len(f)*0.3)), mode='edge') for f in librosa.util.frame(y, frame_length=2048, hop_length=512).T])
    elif mode == 13: y = np.concatenate([y, np.sin(2 * np.pi * np.linspace(0, 1, int(0.05 * sr))) * 0.02])
    elif mode == 14: y += np.random.normal(0, 0.005, len(y))
    elif mode == 15:
        frame = int(sr * 0.2)
        chunks = [y[i:i + frame] for i in range(0, len(y), frame)]

        np.random.shuffle(chunks)
        y = np.concatenate(chunks)
    elif mode == 16:
        frame = int(sr * 0.3)

        for i in range(0, len(y), frame * 2):
            y[i:i+frame] = y[i:i+frame][::-1]

    sf.write(output_path, y, sr, format=export_format)
    gr_info(translations["success"])

    return output_path
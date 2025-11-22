import os
import sys
import json

sys.path.append(os.getcwd())

from main.app.variables import translations, configs
from main.app.core.ui import gr_info, gr_warning, change_preset_choices, change_effect_preset_choices

def load_presets(presets, cleaner, autotune, pitch, clean_strength, index_strength, resample_sr, filter_radius, rms_mix_rate, protect, split_audio, f0_autotune_strength, formant_shifting, formant_qfrency, formant_timbre, proposal_pitch, proposal_pitch_threshold):
    if not presets: gr_warning(translations["provide_file_settings"])
    
    file = {}
    if presets:
        with open(os.path.join(configs["presets_path"], presets)) as f:
            file = json.load(f)

        gr_info(translations["load_presets"].format(presets=presets))

    return [file.get("cleaner", cleaner), file.get("autotune", autotune), file.get("pitch", pitch), file.get("clean_strength", clean_strength), file.get("index_strength", index_strength), file.get("resample_sr", resample_sr), file.get("filter_radius", filter_radius), file.get("rms_mix_rate", rms_mix_rate), file.get("protect", protect), file.get("split_audio", split_audio), file.get("f0_autotune_strength", f0_autotune_strength), file.get("formant_shifting", formant_shifting), file.get("formant_qfrency", formant_qfrency), file.get("formant_timbre", formant_timbre), file.get("proposal_pitch", proposal_pitch), file.get("proposal_pitch_threshold", proposal_pitch_threshold)]

def save_presets(name, cleaner, autotune, pitch, clean_strength, index_strength, resample_sr, filter_radius, rms_mix_rate, protect, split_audio, f0_autotune_strength, cleaner_chbox, autotune_chbox, pitch_chbox, index_strength_chbox, resample_sr_chbox, filter_radius_chbox, rms_mix_rate_chbox, protect_chbox, split_audio_chbox, formant_shifting_chbox, formant_shifting, formant_qfrency, formant_timbre, proposal_pitch, proposal_pitch_threshold):  
    if not name: return gr_warning(translations["provide_filename_settings"])
    if not any([cleaner_chbox, autotune_chbox, pitch_chbox, index_strength_chbox, resample_sr_chbox, filter_radius_chbox, rms_mix_rate_chbox, protect_chbox, split_audio_chbox, formant_shifting_chbox]): return gr_warning(translations["choose1"])

    settings = {}

    for checkbox, data in [(cleaner_chbox, {"cleaner": cleaner, "clean_strength": clean_strength}), (autotune_chbox, {"autotune": autotune, "f0_autotune_strength": f0_autotune_strength}), (pitch_chbox, {"pitch": pitch}), (index_strength_chbox, {"index_strength": index_strength}), (resample_sr_chbox, {"resample_sr": resample_sr}), (filter_radius_chbox, {"filter_radius": filter_radius}), (rms_mix_rate_chbox, {"rms_mix_rate": rms_mix_rate}), (protect_chbox, {"protect": protect}), (split_audio_chbox, {"split_audio": split_audio}), (formant_shifting_chbox, {"formant_shifting": formant_shifting, "formant_qfrency": formant_qfrency, "formant_timbre": formant_timbre}), (proposal_pitch, {"proposal_pitch": proposal_pitch, "proposal_pitch_threshold": proposal_pitch_threshold})]:
        if checkbox: settings.update(data)

    with open(os.path.join(configs["presets_path"], name + ".conversion.json"), "w") as f:
        json.dump(settings, f, indent=4)

    gr_info(translations["export_settings"].format(name=name))
    return change_preset_choices()

def audio_effect_load_presets(presets, resample_checkbox, audio_effect_resample_sr, chorus_depth, chorus_rate_hz, chorus_mix, chorus_centre_delay_ms, chorus_feedback, distortion_drive_db, reverb_room_size, reverb_damping, reverb_wet_level, reverb_dry_level, reverb_width, reverb_freeze_mode, pitch_shift_semitones, delay_second, delay_feedback, delay_mix, compressor_threshold_db, compressor_ratio, compressor_attack_ms, compressor_release_ms, limiter_threshold_db, limiter_release_ms, gain_db, bitcrush_bit_depth, clipping_threshold_db, phaser_rate_hz, phaser_depth, phaser_centre_frequency_hz, phaser_feedback, phaser_mix, bass_boost, bass_frequency, treble_boost, treble_frequency, fade_in, fade_out, chorus_check_box, distortion_checkbox, reverb_check_box, delay_check_box, compressor_check_box, limiter, gain_checkbox, bitcrush_checkbox, clipping_checkbox, phaser_check_box, bass_or_treble, fade):
    if not presets: gr_warning(translations["provide_file_settings"])
    
    file = {}
    if presets:
        with open(os.path.join(configs["presets_path"], presets)) as f:
            file = json.load(f)

        gr_info(translations["load_presets"].format(presets=presets))
    return [
        file.get("resample_checkbox", resample_checkbox), file.get("audio_effect_resample_sr", audio_effect_resample_sr), 
        file.get("chorus_depth", chorus_depth), file.get("chorus_rate_hz", chorus_rate_hz), 
        file.get("chorus_mix", chorus_mix), file.get("chorus_centre_delay_ms", chorus_centre_delay_ms), 
        file.get("chorus_feedback", chorus_feedback), file.get("distortion_drive_db", distortion_drive_db), 
        file.get("reverb_room_size", reverb_room_size), file.get("reverb_damping", reverb_damping), 
        file.get("reverb_wet_level", reverb_wet_level), file.get("reverb_dry_level", reverb_dry_level), 
        file.get("reverb_width", reverb_width), file.get("reverb_freeze_mode", reverb_freeze_mode), 
        file.get("pitch_shift_semitones", pitch_shift_semitones), file.get("delay_second", delay_second), 
        file.get("delay_feedback", delay_feedback), file.get("delay_mix", delay_mix), 
        file.get("compressor_threshold_db", compressor_threshold_db), file.get("compressor_ratio", compressor_ratio), 
        file.get("compressor_attack_ms", compressor_attack_ms), file.get("compressor_release_ms", compressor_release_ms), 
        file.get("limiter_threshold_db", limiter_threshold_db), file.get("limiter_release_ms", limiter_release_ms), 
        file.get("gain_db", gain_db), file.get("bitcrush_bit_depth", bitcrush_bit_depth), 
        file.get("clipping_threshold_db", clipping_threshold_db), file.get("phaser_rate_hz", phaser_rate_hz), 
        file.get("phaser_depth", phaser_depth), file.get("phaser_centre_frequency_hz", phaser_centre_frequency_hz), 
        file.get("phaser_feedback", phaser_feedback), file.get("phaser_mix", phaser_mix), 
        file.get("bass_boost", bass_boost), file.get("bass_frequency", bass_frequency), 
        file.get("treble_boost", treble_boost), file.get("treble_frequency", treble_frequency), 
        file.get("fade_in", fade_in), file.get("fade_out", fade_out), 
        file.get("chorus_check_box", chorus_check_box), file.get("distortion_checkbox", distortion_checkbox),
        file.get("reverb_check_box", reverb_check_box), file.get("delay_check_box", delay_check_box),
        file.get("compressor_check_box", compressor_check_box), file.get("limiter", limiter),
        file.get("gain_checkbox", gain_checkbox), file.get("bitcrush_checkbox", bitcrush_checkbox),
        file.get("clipping_checkbox", clipping_checkbox), file.get("phaser_check_box", phaser_check_box),
        file.get("bass_or_treble", bass_or_treble), file.get("fade", fade)
    ]

def audio_effect_save_presets(name, resample_checkbox, audio_effect_resample_sr, chorus_depth, chorus_rate_hz, chorus_mix, chorus_centre_delay_ms, chorus_feedback, distortion_drive_db, reverb_room_size, reverb_damping, reverb_wet_level, reverb_dry_level, reverb_width, reverb_freeze_mode, pitch_shift_semitones, delay_second, delay_feedback, delay_mix, compressor_threshold_db, compressor_ratio, compressor_attack_ms, compressor_release_ms, limiter_threshold_db, limiter_release_ms, gain_db, bitcrush_bit_depth, clipping_threshold_db, phaser_rate_hz, phaser_depth, phaser_centre_frequency_hz, phaser_feedback, phaser_mix, bass_boost, bass_frequency, treble_boost, treble_frequency, fade_in, fade_out, chorus_check_box, distortion_checkbox, reverb_check_box, delay_check_box, compressor_check_box, limiter, gain_checkbox, bitcrush_checkbox, clipping_checkbox, phaser_check_box, bass_or_treble, fade):
    if not name: return gr_warning(translations["provide_filename_settings"])
    if not any([resample_checkbox, chorus_check_box, distortion_checkbox, reverb_check_box, delay_check_box, compressor_check_box, limiter, gain_checkbox, bitcrush_checkbox, clipping_checkbox, phaser_check_box, bass_or_treble, fade, pitch_shift_semitones != 0]): return gr_warning(translations["choose1"])

    settings = {}

    for checkbox, data in [
        (resample_checkbox, {
            "resample_checkbox": resample_checkbox, 
            "audio_effect_resample_sr": audio_effect_resample_sr
        }), 
        (chorus_check_box, {
            "chorus_check_box": chorus_check_box, 
            "chorus_depth": chorus_depth,
            "chorus_rate_hz": chorus_rate_hz,
            "chorus_mix": chorus_mix,
            "chorus_centre_delay_ms": chorus_centre_delay_ms,
            "chorus_feedback": chorus_feedback
        }), 
        (distortion_checkbox, {
            "distortion_checkbox": distortion_checkbox, 
            "distortion_drive_db": distortion_drive_db
        }), 
        (reverb_check_box, {
            "reverb_check_box": reverb_check_box,
            "reverb_room_size": reverb_room_size,
            "reverb_damping": reverb_damping,
            "reverb_wet_level": reverb_wet_level,
            "reverb_dry_level": reverb_dry_level,
            "reverb_width": reverb_width,
            "reverb_freeze_mode": reverb_freeze_mode
        }), 
        (pitch_shift_semitones != 0, {
            "pitch_shift_semitones": pitch_shift_semitones
        }), 
        (delay_check_box, {
            "delay_check_box": delay_check_box,
            "delay_second": delay_second,
            "delay_feedback": delay_feedback,
            "delay_mix": delay_mix
        }), 
        (compressor_check_box, {
            "compressor_check_box": compressor_check_box,
            "compressor_threshold_db": compressor_threshold_db,
            "compressor_ratio": compressor_ratio,
            "compressor_attack_ms": compressor_attack_ms,
            "compressor_release_ms": compressor_release_ms
        }), 
        (limiter, {
            "limiter": limiter,
            "limiter_threshold_db": limiter_threshold_db,
            "limiter_release_ms": limiter_release_ms
        }), 
        (gain_checkbox, {
            "gain_checkbox": gain_checkbox,
            "gain_db": gain_db
        }), 
        (bitcrush_checkbox, {
            "bitcrush_checkbox": bitcrush_checkbox, 
            "bitcrush_bit_depth": bitcrush_bit_depth
        }),
        (clipping_checkbox, {
            "clipping_checkbox": clipping_checkbox,
            "clipping_threshold_db": clipping_threshold_db
        }),
        (phaser_check_box, {
            "phaser_check_box": phaser_check_box,
            "phaser_rate_hz": phaser_rate_hz,
            "phaser_depth": phaser_depth,
            "phaser_centre_frequency_hz": phaser_centre_frequency_hz,
            "phaser_feedback": phaser_feedback,
            "phaser_mix": phaser_mix
        }),
        (bass_or_treble, {
            "bass_or_treble": bass_or_treble,
            "bass_boost": bass_boost,
            "bass_frequency": bass_frequency,
            "treble_boost": treble_boost,
            "treble_frequency": treble_frequency
        }),
        (fade, {
            "fade": fade,
            "fade_in": fade_in,
            "fade_out": fade_out
        })
    ]:
        if checkbox: settings.update(data)

    with open(os.path.join(configs["presets_path"], name + ".effect.json"), "w") as f:
        json.dump(settings, f, indent=4)

    gr_info(translations["export_settings"].format(name=name))
    return change_effect_preset_choices()
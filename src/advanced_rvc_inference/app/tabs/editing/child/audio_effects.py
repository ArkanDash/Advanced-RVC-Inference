import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from main.app.core.editing import audio_effects
from main.app.core.presets import audio_effect_load_presets, audio_effect_save_presets
from main.app.core.ui import visible, change_audios_choices, change_effect_preset_choices, shutil_move
from main.app.variables import translations, paths_for_files, sample_rate_choice, audio_effect_presets_file, configs, file_types, export_format_choices

def audio_effects_tab():
    with gr.Row():
        gr.Markdown(translations["audio_effects_edit"])
    with gr.Row():
        with gr.Column():
            with gr.Row():
                reverb_check_box = gr.Checkbox(label=translations["reverb"], value=False, interactive=True)
                chorus_check_box = gr.Checkbox(label=translations["chorus"], value=False, interactive=True)
                delay_check_box = gr.Checkbox(label=translations["delay"], value=False, interactive=True)
                phaser_check_box = gr.Checkbox(label=translations["phaser"], value=False, interactive=True)
                compressor_check_box = gr.Checkbox(label=translations["compressor"], value=False, interactive=True)
                more_options = gr.Checkbox(label=translations["more_option"], value=False, interactive=True)    
    with gr.Row():
        with gr.Accordion(translations["input_output"], open=False):
            with gr.Row():
                upload_audio = gr.Files(label=translations["drop_audio"], file_types=file_types)
            with gr.Row():
                audio_in_path = gr.Dropdown(label=translations["input_audio"], value="", choices=paths_for_files, info=translations["provide_audio"], interactive=True, allow_custom_value=True)
                audio_out_path = gr.Textbox(label=translations["output_audio"], value="audios/audio_effects.wav", placeholder="audios/audio_effects.wav", info=translations["provide_output"], interactive=True)
            with gr.Row():
                with gr.Column():
                    audio_combination = gr.Checkbox(label=translations["merge_instruments"], value=False, interactive=True)
                    audio_combination_input = gr.Dropdown(label=translations["input_audio"], value="", choices=paths_for_files, info=translations["provide_audio"], interactive=True, allow_custom_value=True, visible=audio_combination.value)
            with gr.Row():
                main_vol = gr.Slider(minimum=-80, maximum=80, label=translations["main_volume"], info=translations["main_volume_info"], value=-4, step=1, interactive=True, visible=audio_combination.value)
                combine_vol = gr.Slider(minimum=-80, maximum=80, label=translations["combination_volume"], info=translations["combination_volume_info"], value=-7, step=1, interactive=True, visible=audio_combination.value)
            with gr.Row():
                audio_effects_refresh = gr.Button(translations["refresh"])
            with gr.Row():
                audio_output_format = gr.Radio(label=translations["export_format"], info=translations["export_info"], choices=export_format_choices, value="wav", interactive=True)
    with gr.Row():
        with gr.Accordion(translations["use_presets"], open=False):
            with gr.Row():
                presets_name = gr.Dropdown(label=translations["file_preset"], choices=audio_effect_presets_file, value=audio_effect_presets_file[0] if len(audio_effect_presets_file) > 0 else '', interactive=True, allow_custom_value=True)
            with gr.Row():
                load_click = gr.Button(translations["load_file"], variant="primary")
                refresh_click = gr.Button(translations["refresh"])
            with gr.Accordion(translations["export_file"], open=False):
                with gr.Row():
                    with gr.Column():
                        name_to_save_file = gr.Textbox(label=translations["filename_to_save"])
                        save_file_button = gr.Button(translations["export_file"])
            with gr.Row():
                upload_presets = gr.Files(label=translations["upload_presets"], file_types=[".effect.json"]) 
    with gr.Row():
        apply_effects_button = gr.Button(translations["apply"], variant="primary", scale=2)
    with gr.Row():
        with gr.Column():
            with gr.Row():
                with gr.Accordion(translations["reverb"], open=False, visible=reverb_check_box.value) as reverb_accordion:
                    reverb_freeze_mode = gr.Checkbox(label=translations["reverb_freeze"], info=translations["reverb_freeze_info"], value=False, interactive=True)
                    reverb_room_size = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.15, label=translations["room_size"], info=translations["room_size_info"], interactive=True)
                    reverb_damping = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.7, label=translations["damping"], info=translations["damping_info"], interactive=True)
                    reverb_wet_level = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.2, label=translations["wet_level"], info=translations["wet_level_info"], interactive=True)
                    reverb_dry_level = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.8, label=translations["dry_level"], info=translations["dry_level_info"], interactive=True)
                    reverb_width = gr.Slider(minimum=0, maximum=1, step=0.01, value=1, label=translations["width"], info=translations["width_info"], interactive=True)
            with gr.Row():
                with gr.Accordion(translations["chorus"], open=False, visible=chorus_check_box.value) as chorus_accordion:
                    chorus_depth = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label=translations["chorus_depth"], info=translations["chorus_depth_info"], interactive=True)
                    chorus_rate_hz = gr.Slider(minimum=0.1, maximum=10, step=0.1, value=1.5, label=translations["chorus_rate_hz"], info=translations["chorus_rate_hz_info"], interactive=True)
                    chorus_mix = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label=translations["chorus_mix"], info=translations["chorus_mix_info"], interactive=True)
                    chorus_centre_delay_ms = gr.Slider(minimum=0, maximum=50, step=1, value=10, label=translations["chorus_centre_delay_ms"], info=translations["chorus_centre_delay_ms_info"], interactive=True)
                    chorus_feedback = gr.Slider(minimum=-1, maximum=1, step=0.01, value=0, label=translations["chorus_feedback"], info=translations["chorus_feedback_info"], interactive=True)
            with gr.Row():
                with gr.Accordion(translations["delay"], open=False, visible=delay_check_box.value) as delay_accordion:
                    delay_second = gr.Slider(minimum=0, maximum=5, step=0.01, value=0.5, label=translations["delay_seconds"], info=translations["delay_seconds_info"], interactive=True)
                    delay_feedback = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label=translations["delay_feedback"], info=translations["delay_feedback_info"], interactive=True)
                    delay_mix = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label=translations["delay_mix"], info=translations["delay_mix_info"], interactive=True)
        with gr.Column():
            with gr.Row():
                with gr.Accordion(translations["more_option"], open=False, visible=more_options.value) as more_accordion:
                    with gr.Row():
                        fade = gr.Checkbox(label=translations["fade"], value=False, interactive=True)
                        bass_or_treble = gr.Checkbox(label=translations["bass_or_treble"], value=False, interactive=True)
                        limiter = gr.Checkbox(label=translations["limiter"], value=False, interactive=True)
                        resample_checkbox = gr.Checkbox(label=translations["resample"], value=False, interactive=True)
                    with gr.Row():
                        distortion_checkbox = gr.Checkbox(label=translations["distortion"], value=False, interactive=True)
                        gain_checkbox = gr.Checkbox(label=translations["gain"], value=False, interactive=True)
                        bitcrush_checkbox = gr.Checkbox(label=translations["bitcrush"], value=False, interactive=True)
                        clipping_checkbox = gr.Checkbox(label=translations["clipping"], value=False, interactive=True)
                    with gr.Accordion(translations["fade"], open=True, visible=fade.value) as fade_accordion:
                        with gr.Row():
                            fade_in = gr.Slider(minimum=0, maximum=10000, step=100, value=0, label=translations["fade_in"], info=translations["fade_in_info"], interactive=True)
                            fade_out = gr.Slider(minimum=0, maximum=10000, step=100, value=0, label=translations["fade_out"], info=translations["fade_out_info"], interactive=True)
                    with gr.Accordion(translations["bass_or_treble"], open=True, visible=bass_or_treble.value) as bass_treble_accordion:
                        with gr.Row():
                            bass_boost = gr.Slider(minimum=0, maximum=20, step=1, value=0, label=translations["bass_boost"], info=translations["bass_boost_info"], interactive=True)
                            bass_frequency = gr.Slider(minimum=20, maximum=200, step=10, value=100, label=translations["bass_frequency"], info=translations["bass_frequency_info"], interactive=True)
                        with gr.Row():
                            treble_boost = gr.Slider(minimum=0, maximum=20, step=1, value=0, label=translations["treble_boost"], info=translations["treble_boost_info"], interactive=True)
                            treble_frequency = gr.Slider(minimum=1000, maximum=10000, step=500, value=3000, label=translations["treble_frequency"], info=translations["treble_frequency_info"], interactive=True)
                    with gr.Accordion(translations["limiter"], open=True, visible=limiter.value) as limiter_accordion:
                        with gr.Row():
                            limiter_threshold_db = gr.Slider(minimum=-60, maximum=0, step=1, value=-1, label=translations["limiter_threshold_db"], info=translations["limiter_threshold_db_info"], interactive=True)
                            limiter_release_ms = gr.Slider(minimum=10, maximum=1000, step=1, value=100, label=translations["limiter_release_ms"], info=translations["limiter_release_ms_info"], interactive=True)
                    with gr.Column():
                        pitch_shift_semitones = gr.Slider(minimum=-20, maximum=20, step=1, value=0, label=translations["pitch"], info=translations["pitch_info"], interactive=True)
                        audio_effect_resample_sr = gr.Radio(choices=[0]+sample_rate_choice, value=0, label=translations["resample"], info=translations["resample_info"], interactive=True, visible=resample_checkbox.value)
                        distortion_drive_db = gr.Slider(minimum=0, maximum=50, step=1, value=20, label=translations["distortion"], info=translations["distortion_info"], interactive=True, visible=distortion_checkbox.value)
                        gain_db = gr.Slider(minimum=-60, maximum=60, step=1, value=0, label=translations["gain"], info=translations["gain_info"], interactive=True, visible=gain_checkbox.value)
                        clipping_threshold_db = gr.Slider(minimum=-60, maximum=0, step=1, value=-1, label=translations["clipping_threshold_db"], info=translations["clipping_threshold_db_info"], interactive=True, visible=clipping_checkbox.value)
                        bitcrush_bit_depth = gr.Slider(minimum=1, maximum=24, step=1, value=16, label=translations["bitcrush_bit_depth"], info=translations["bitcrush_bit_depth_info"], interactive=True, visible=bitcrush_checkbox.value)
            with gr.Row():
                with gr.Accordion(translations["phaser"], open=False, visible=phaser_check_box.value) as phaser_accordion:
                    phaser_depth = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label=translations["phaser_depth"], info=translations["phaser_depth_info"], interactive=True)
                    phaser_rate_hz = gr.Slider(minimum=0.1, maximum=10, step=0.1, value=1, label=translations["phaser_rate_hz"], info=translations["phaser_rate_hz_info"], interactive=True)
                    phaser_mix = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label=translations["phaser_mix"], info=translations["phaser_mix_info"], interactive=True)
                    phaser_centre_frequency_hz = gr.Slider(minimum=50, maximum=5000, step=10, value=1000, label=translations["phaser_centre_frequency_hz"], info=translations["phaser_centre_frequency_hz_info"], interactive=True)
                    phaser_feedback = gr.Slider(minimum=-1, maximum=1, step=0.01, value=0, label=translations["phaser_feedback"], info=translations["phaser_feedback_info"], interactive=True)
            with gr.Row():
                with gr.Accordion(translations["compressor"], open=False, visible=compressor_check_box.value) as compressor_accordion:
                    compressor_threshold_db = gr.Slider(minimum=-60, maximum=0, step=1, value=-20, label=translations["compressor_threshold_db"], info=translations["compressor_threshold_db_info"], interactive=True)
                    compressor_ratio = gr.Slider(minimum=1, maximum=20, step=0.1, value=1, label=translations["compressor_ratio"], info=translations["compressor_ratio_info"], interactive=True)
                    compressor_attack_ms = gr.Slider(minimum=0.1, maximum=100, step=0.1, value=10, label=translations["compressor_attack_ms"], info=translations["compressor_attack_ms_info"], interactive=True)
                    compressor_release_ms = gr.Slider(minimum=10, maximum=1000, step=1, value=100, label=translations["compressor_release_ms"], info=translations["compressor_release_ms_info"], interactive=True)   
    with gr.Row():
        gr.Markdown(translations["output_audio"])
    with gr.Row():
        audio_play_input = gr.Audio(show_download_button=True, interactive=False, label=translations["input_audio"])
        audio_play_output = gr.Audio(show_download_button=True, interactive=False, label=translations["output_audio"])
    with gr.Row():
        reverb_check_box.change(fn=visible, inputs=[reverb_check_box], outputs=[reverb_accordion])
        chorus_check_box.change(fn=visible, inputs=[chorus_check_box], outputs=[chorus_accordion])
        delay_check_box.change(fn=visible, inputs=[delay_check_box], outputs=[delay_accordion])
    with gr.Row():
        compressor_check_box.change(fn=visible, inputs=[compressor_check_box], outputs=[compressor_accordion])
        phaser_check_box.change(fn=visible, inputs=[phaser_check_box], outputs=[phaser_accordion])
        more_options.change(fn=visible, inputs=[more_options], outputs=[more_accordion])
    with gr.Row():
        fade.change(fn=visible, inputs=[fade], outputs=[fade_accordion])
        bass_or_treble.change(fn=visible, inputs=[bass_or_treble], outputs=[bass_treble_accordion])
        limiter.change(fn=visible, inputs=[limiter], outputs=[limiter_accordion])
        resample_checkbox.change(fn=visible, inputs=[resample_checkbox], outputs=[audio_effect_resample_sr])
    with gr.Row():
        distortion_checkbox.change(fn=visible, inputs=[distortion_checkbox], outputs=[distortion_drive_db])
        gain_checkbox.change(fn=visible, inputs=[gain_checkbox], outputs=[gain_db])
        clipping_checkbox.change(fn=visible, inputs=[clipping_checkbox], outputs=[clipping_threshold_db])
        bitcrush_checkbox.change(fn=visible, inputs=[bitcrush_checkbox], outputs=[bitcrush_bit_depth])
    with gr.Row():
        upload_audio.upload(fn=lambda audio_in: [shutil_move(audio.name, configs["audios_path"]) for audio in audio_in][0], inputs=[upload_audio], outputs=[audio_in_path])
        audio_in_path.change(fn=lambda audio: audio if audio else None, inputs=[audio_in_path], outputs=[audio_play_input])
        audio_effects_refresh.click(fn=lambda a, b: [change_audios_choices(a), change_audios_choices(b)], inputs=[audio_in_path, audio_combination_input], outputs=[audio_in_path, audio_combination_input])
    with gr.Row():
        more_options.change(fn=lambda: [False]*8, inputs=[], outputs=[fade, bass_or_treble, limiter, resample_checkbox, distortion_checkbox, gain_checkbox, clipping_checkbox, bitcrush_checkbox])
        audio_combination.change(fn=visible, inputs=[audio_combination], outputs=[audio_combination_input])
        audio_combination.change(fn=lambda a: [visible(a) for _ in range(2)], inputs=[audio_combination], outputs=[main_vol, combine_vol])
    with gr.Row():
        upload_presets.upload(fn=lambda presets_in: [shutil_move(preset.name, configs["presets_path"]) for preset in presets_in][0], inputs=[upload_presets], outputs=[presets_name])
        refresh_click.click(fn=change_effect_preset_choices, inputs=[], outputs=[presets_name])
    with gr.Row():
        load_click.click(
            fn=audio_effect_load_presets,
            inputs=[
                presets_name, 
                resample_checkbox, 
                audio_effect_resample_sr, 
                chorus_depth, 
                chorus_rate_hz, 
                chorus_mix, 
                chorus_centre_delay_ms, 
                chorus_feedback, 
                distortion_drive_db, 
                reverb_room_size, 
                reverb_damping, 
                reverb_wet_level, 
                reverb_dry_level, 
                reverb_width, 
                reverb_freeze_mode, 
                pitch_shift_semitones, 
                delay_second, 
                delay_feedback, 
                delay_mix, 
                compressor_threshold_db, 
                compressor_ratio, 
                compressor_attack_ms, 
                compressor_release_ms, 
                limiter_threshold_db, 
                limiter_release_ms, 
                gain_db, 
                bitcrush_bit_depth, 
                clipping_threshold_db, 
                phaser_rate_hz, 
                phaser_depth, 
                phaser_centre_frequency_hz, 
                phaser_feedback, 
                phaser_mix, 
                bass_boost, 
                bass_frequency, 
                treble_boost, 
                treble_frequency, 
                fade_in, 
                fade_out, 
                chorus_check_box, 
                distortion_checkbox, 
                reverb_check_box, 
                delay_check_box, 
                compressor_check_box, 
                limiter, 
                gain_checkbox, 
                bitcrush_checkbox, 
                clipping_checkbox, 
                phaser_check_box, 
                bass_or_treble, 
                fade
            ],
            outputs=[
                resample_checkbox, 
                audio_effect_resample_sr, 
                chorus_depth, 
                chorus_rate_hz, 
                chorus_mix, 
                chorus_centre_delay_ms, 
                chorus_feedback, 
                distortion_drive_db, 
                reverb_room_size, 
                reverb_damping, 
                reverb_wet_level, 
                reverb_dry_level, 
                reverb_width, 
                reverb_freeze_mode, 
                pitch_shift_semitones, 
                delay_second, 
                delay_feedback, 
                delay_mix, 
                compressor_threshold_db, 
                compressor_ratio, 
                compressor_attack_ms, 
                compressor_release_ms, 
                limiter_threshold_db, 
                limiter_release_ms, 
                gain_db, 
                bitcrush_bit_depth, 
                clipping_threshold_db, 
                phaser_rate_hz, 
                phaser_depth, 
                phaser_centre_frequency_hz, 
                phaser_feedback, 
                phaser_mix, 
                bass_boost, 
                bass_frequency, 
                treble_boost, 
                treble_frequency, 
                fade_in, 
                fade_out, 
                chorus_check_box, 
                distortion_checkbox, 
                reverb_check_box, 
                delay_check_box, 
                compressor_check_box, 
                limiter, 
                gain_checkbox, 
                bitcrush_checkbox, 
                clipping_checkbox, 
                phaser_check_box, 
                bass_or_treble, 
                fade
            ],
        )
        save_file_button.click(
            fn=audio_effect_save_presets,
            inputs=[
                name_to_save_file, 
                resample_checkbox, 
                audio_effect_resample_sr, 
                chorus_depth, 
                chorus_rate_hz, 
                chorus_mix, 
                chorus_centre_delay_ms, 
                chorus_feedback, 
                distortion_drive_db, 
                reverb_room_size, 
                reverb_damping, 
                reverb_wet_level, 
                reverb_dry_level, 
                reverb_width, 
                reverb_freeze_mode, 
                pitch_shift_semitones, 
                delay_second, 
                delay_feedback, 
                delay_mix, 
                compressor_threshold_db, 
                compressor_ratio, 
                compressor_attack_ms, 
                compressor_release_ms, 
                limiter_threshold_db, 
                limiter_release_ms, 
                gain_db, 
                bitcrush_bit_depth, 
                clipping_threshold_db, 
                phaser_rate_hz, 
                phaser_depth, 
                phaser_centre_frequency_hz, 
                phaser_feedback, 
                phaser_mix, 
                bass_boost, 
                bass_frequency, 
                treble_boost, 
                treble_frequency, 
                fade_in, 
                fade_out, 
                chorus_check_box, 
                distortion_checkbox, 
                reverb_check_box, 
                delay_check_box, 
                compressor_check_box, 
                limiter, 
                gain_checkbox, 
                bitcrush_checkbox, 
                clipping_checkbox, 
                phaser_check_box, 
                bass_or_treble, 
                fade
            ],
            outputs=[presets_name]
        )
    with gr.Row():
        apply_effects_button.click(
            fn=audio_effects,
            inputs=[
                audio_in_path, 
                audio_out_path, 
                resample_checkbox, 
                audio_effect_resample_sr, 
                chorus_depth, 
                chorus_rate_hz, 
                chorus_mix, 
                chorus_centre_delay_ms, 
                chorus_feedback, 
                distortion_drive_db, 
                reverb_room_size, 
                reverb_damping, 
                reverb_wet_level, 
                reverb_dry_level, 
                reverb_width, 
                reverb_freeze_mode, 
                pitch_shift_semitones, 
                delay_second, 
                delay_feedback, 
                delay_mix, 
                compressor_threshold_db, 
                compressor_ratio, 
                compressor_attack_ms, 
                compressor_release_ms, 
                limiter_threshold_db, 
                limiter_release_ms, 
                gain_db, 
                bitcrush_bit_depth, 
                clipping_threshold_db, 
                phaser_rate_hz, 
                phaser_depth, 
                phaser_centre_frequency_hz, 
                phaser_feedback, 
                phaser_mix, 
                bass_boost, 
                bass_frequency, 
                treble_boost, 
                treble_frequency, 
                fade_in, 
                fade_out, 
                audio_output_format, 
                chorus_check_box, 
                distortion_checkbox, 
                reverb_check_box, 
                delay_check_box, 
                compressor_check_box, 
                limiter, 
                gain_checkbox, 
                bitcrush_checkbox, 
                clipping_checkbox, 
                phaser_check_box, 
                bass_or_treble, 
                fade,
                audio_combination,
                audio_combination_input,
                main_vol,
                combine_vol
            ],
            outputs=[audio_play_output],
            api_name="audio_effects"
        )
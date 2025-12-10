import os
import sys
import subprocess

sys.path.append(os.getcwd())

from main.app.core.ui import gr_info, gr_warning
from main.app.variables import python, translations, configs

def separate_music(
    input_path,
    output_dirs,
    export_format, 
    model_name, 
    karaoke_model,
    reverb_model,
    denoise_model,
    sample_rate,
    shifts, 
    batch_size, 
    overlap, 
    aggression,
    hop_length, 
    window_size,
    segments_size, 
    post_process_threshold,
    enable_tta,
    enable_denoise, 
    high_end_process,
    enable_post_process,
    separate_backing,
    separate_reverb
):
    output_dirs = os.path.dirname(output_dirs) or output_dirs

    if not input_path or not os.path.exists(input_path) or os.path.isdir(input_path): 
        gr_warning(translations["input_not_valid"])
        return [None]*4
    
    if not os.path.exists(output_dirs): 
        gr_warning(translations["output_not_valid"])
        return [None]*4

    if not os.path.exists(output_dirs): os.makedirs(output_dirs)
    gr_info(translations["start"].format(start=translations["separator_music"]))

    subprocess.run([
        python, configs["separate_path"], 
        "--input_path", input_path,
        "--output_dirs", output_dirs,
        "--export_format", export_format,
        "--model_name", model_name,
        "--karaoke_model", karaoke_model,
        "--reverb_model", reverb_model,
        "--denoise_model", denoise_model,
        "--sample_rate", str(sample_rate),
        "--shifts", str(shifts),
        "--batch_size", str(batch_size),
        "--overlap", str(overlap),
        "--aggression", str(aggression),
        "--hop_length", str(hop_length),
        "--window_size", str(window_size),
        "--segments_size", str(segments_size),
        "--post_process_threshold", str(post_process_threshold),
        "--enable_tta", str(enable_tta),
        "--enable_denoise", str(enable_denoise),
        "--high_end_process", str(high_end_process),
        "--enable_post_process", str(enable_post_process),
        "--separate_backing", str(separate_backing),
        "--separate_reverb", str(separate_reverb),
    ])

    gr_info(translations["success"])

    filename, _ = os.path.splitext(os.path.basename(input_path))
    output_dirs = os.path.join(output_dirs, filename)

    return [
        os.path.join(
            output_dirs, 
            f"Original_Vocals_No_Reverb.{export_format}" if separate_reverb else f"Original_Vocals.{export_format}"
        ), 
        os.path.join(
            output_dirs, 
            f"Instruments.{export_format}"
        ), 
        os.path.join(
            output_dirs, 
            f"Main_Vocals_No_Reverb.{export_format}" if separate_reverb else f"Main_Vocals.{export_format}"
        ) if separate_backing else None,
        os.path.join(
            output_dirs, 
            f"Backing_Vocals.{export_format}"
        ) if separate_backing else None
    ] if os.path.isfile(input_path) else [None]*4
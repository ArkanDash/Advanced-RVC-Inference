import os, sys
import subprocess

sys.path.append(os.getcwd())

from arvc.utils.feedback import gr_info, gr_warning, gr_error
from arvc.utils.variables import python, translations, configs

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
    # Force output to the specific UVR directory
    output_dirs = configs.get("uvr_path", os.path.join("arvc", "assets", "audios", "uvr"))
    
    # Ensure the directory exists
    os.makedirs(output_dirs, exist_ok=True)

    if not input_path or not os.path.exists(input_path) or os.path.isdir(input_path): 
        gr_warning(translations["input_not_valid"])
        return [None]*4
    
    gr_info(translations["start"].format(start=translations["separator_music"]))

    try:
        result = subprocess.run([
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
        ], timeout=3600)  # 1 hour timeout

        if result.returncode != 0:
            gr_error(translations["error_occurred"].format(e=f"Separation process exited with code {result.returncode}"))
            return [None] * 4
    except subprocess.TimeoutExpired:
        gr_error(translations.get("timeout_error", "Separation process timed out after 1 hour"))
        return [None] * 4
    except FileNotFoundError:
        gr_error(translations.get("separate_not_found", "Separation script not found"))
        return [None] * 4

    gr_info(translations["success"])
    
    # _separate() creates a subdirectory named after the input filename
    filename = os.path.splitext(os.path.basename(input_path))[0]
    file_output_dir = os.path.join(output_dirs, filename)
    
    return [
        os.path.join(
            file_output_dir, 
            f"Original_Vocals_No_Reverb.{export_format}" if separate_reverb else f"Original_Vocals.{export_format}"
        ), 
        os.path.join(
            file_output_dir, 
            f"Instruments.{export_format}"
        ), 
        os.path.join(
            file_output_dir, 
            f"Main_Vocals_No_Reverb.{export_format}" if separate_reverb else f"Main_Vocals.{export_format}"
        ) if separate_backing else None,
        os.path.join(
            file_output_dir, 
            f"Backing_Vocals.{export_format}"
        ) if separate_backing else None
    ] if os.path.isfile(input_path) else [None]*4

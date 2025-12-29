"""
Optimized inference module for RVC voice conversion.

This module provides efficient voice conversion with support for
batch processing, speaker diarization, and text-to-speech integration.
"""

import os
import re
import gc
import sys
import shutil
import datetime
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

# Import heavy dependencies lazily
try:
    import numpy as np
except ImportError:
    np = None

# Path setup
PACKAGE_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))


def _get_translations():
    """Lazy load translations."""
    from advanced_rvc_inference.variables import translations

    return translations


def _get_logger():
    """Lazy load logger."""
    from advanced_rvc_inference.variables import logger

    return logger


def _get_configs():
    """Lazy load configs."""
    from advanced_rvc_inference.variables import configs, config

    return configs, config


def _get_ui_helpers():
    """Lazy load UI helper functions."""
    from advanced_rvc_inference.core.ui import gr_info, gr_warning, gr_error, process_output, replace_export_format

    return gr_info, gr_warning, gr_error, process_output, replace_export_format


def search_separated_audio_folders(search_path: Optional[str] = None) -> List[str]:
    """
    Search for folders containing separated audio files.

    Args:
        search_path: Base path to search (uses UVR path if None)

    Returns:
        List of folder names containing audio files
    """
    configs, _ = _get_configs()

    translations = _get_translations()

    # Default to UVR directory if no path specified
    if search_path is None:
        uvr_base = os.path.join("advanced_rvc_inference", "assets", "audios", "uvr")
        # Check if custom uvr_path is configured
        search_path = configs.get("uvr_path", uvr_base)

    if not os.path.exists(search_path):
        return []

    valid_dirs = []
    audio_extensions = (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4",
                       ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3")

    for dir_name in os.listdir(search_path):
        dir_path = os.path.join(search_path, dir_name)
        if not os.path.isdir(dir_path):
            continue

        has_audio = any(
            f.lower().endswith(audio_extensions)
            for f in os.listdir(dir_path)
        )
        if has_audio:
            valid_dirs.append(dir_name)

    return sorted(valid_dirs)


def get_uvr_output_path(input_audio_name: Optional[str] = None) -> str:
    """
    Get the path to UVR output audio.

    Args:
        input_audio_name: Name of the separated audio folder

    Returns:
        Path to the UVR output directory
    """
    configs, _ = _get_configs()

    uvr_base = os.path.join("advanced_rvc_inference", "assets", "audios", "uvr")
    uvr_path = configs.get("uvr_path", uvr_base)

    if input_audio_name:
        return os.path.join(uvr_path, input_audio_name)
    return uvr_path


def convert(
    pitch: int,
    filter_radius: int,
    index_rate: float,
    rms_mix_rate: float,
    protect: float,
    hop_length: int,
    f0_method: str,
    input_path: str,
    output_path: str,
    pth_path: str,
    index_path: Optional[str],
    f0_autotune: bool,
    clean_audio: bool,
    clean_strength: float,
    export_format: str,
    embedder_model: str,
    resample_sr: int,
    split_audio: bool,
    f0_autotune_strength: float,
    checkpointing: bool,
    f0_onnx: bool,
    embedders_mode: str,
    formant_shifting: bool,
    formant_qfrency: float,
    formant_timbre: float,
    f0_file: str,
    proposal_pitch: bool,
    proposal_pitch_threshold: float,
    audio_processing: bool = False,
    alpha: float = 0.5,
):
    """
    Run voice conversion on an audio file.

    Args:
        pitch: Pitch shift in semitones
        filter_radius: Filter radius for pitch extraction
        index_rate: Rate for index-based retrieval
        rms_mix_rate: Rate for RMS mixing
        protect: Protection parameter for vocals
        hop_length: Hop length for pitch extraction
        f0_method: Pitch extraction method
        input_path: Path to input audio
        output_path: Path to output audio
        pth_path: Path to model file
        index_path: Path to index file
        f0_autotune: Whether to apply autotune
        clean_audio: Whether to clean audio
        clean_strength: Strength of audio cleaning
        export_format: Output format
        embedder_model: Embedder model name
        resample_sr: Target sample rate (0 for original)
        split_audio: Whether to split audio
        f0_autotune_strength: Autotune strength
        checkpointing: Whether to use checkpointing
        f0_onnx: Whether to use ONNX for F0
        embedders_mode: Embedder mode
        formant_shifting: Whether to shift formants
        formant_qfrency: Formant frequency shift
        formant_timbre: Formant timbre shift
        f0_file: Path to F0 file
        proposal_pitch: Whether to use proposal pitch
        proposal_pitch_threshold: Proposal pitch threshold
        audio_processing: Whether to use audio processing
        alpha: Alpha value for mixing
    """
    from advanced_rvc_inference.variables import python

    cmd = [
        python,
        configs["convert_path"],
        "--pitch", str(pitch),
        "--filter_radius", str(filter_radius),
        "--index_rate", str(index_rate),
        "--rms_mix_rate", str(rms_mix_rate),
        "--protect", str(protect),
        "--hop_length", str(hop_length),
        "--f0_method", f0_method,
        "--input_path", input_path,
        "--output_path", output_path,
        "--pth_path", pth_path,
        "--f0_autotune", str(f0_autotune),
        "--clean_audio", str(clean_audio),
        "--clean_strength", str(clean_strength),
        "--export_format", export_format,
        "--embedder_model", embedder_model,
        "--resample_sr", str(resample_sr),
        "--split_audio", str(split_audio),
        "--f0_autotune_strength", str(f0_autotune_strength),
        "--checkpointing", str(checkpointing),
        "--f0_onnx", str(f0_onnx),
        "--embedders_mode", embedders_mode,
        "--formant_shifting", str(formant_shifting),
        "--formant_qfrency", str(formant_qfrency),
        "--formant_timbre", str(formant_timbre),
        "--f0_file", f0_file,
        "--proposal_pitch", str(proposal_pitch),
        "--proposal_pitch_threshold", str(proposal_pitch_threshold),
        "--audio_processing", str(audio_processing),
        "--alpha", str(alpha),
    ]

    if index_path:
        cmd.extend(["--index_path", index_path])

    subprocess.run(cmd, check=True)


def convert_audio(
    clean: bool,
    autotune: bool,
    use_audio: bool,
    use_original: bool,
    convert_backing: bool,
    not_merge_backing: bool,
    merge_instrument: bool,
    pitch: int,
    clean_strength: float,
    model: str,
    index: Optional[str],
    index_rate: float,
    input_path: str,
    output_path: str,
    format: str,
    method: str,
    hybrid_method: str,
    hop_length: int,
    embedders: str,
    custom_embedders: str,
    resample_sr: int,
    filter_radius: int,
    rms_mix_rate: float,
    protect: float,
    split_audio: bool,
    f0_autotune_strength: float,
    input_audio_name: Optional[str],
    checkpointing: bool,
    onnx_f0_mode: bool,
    formant_shifting: bool,
    formant_qfrency: float,
    formant_timbre: float,
    f0_file: str,
    embedders_mode: str,
    proposal_pitch: bool,
    proposal_pitch_threshold: float,
    audio_processing: bool = False,
    alpha: float = 0.5,
) -> Tuple:
    """
    Convert audio with various options.

    Args:
        clean: Whether to clean audio
        autotune: Whether to apply autotune
        use_audio: Whether to use separated audio
        use_original: Whether to use original vocals
        convert_backing: Whether to convert backing
        not_merge_backing: Whether to skip merging backing
        merge_instrument: Whether to merge with instruments
        pitch: Pitch shift
        clean_strength: Cleaning strength
        model: Model name/path
        index: Index path
        index_rate: Index rate
        input_path: Input path
        output_path: Output path
        format: Output format
        method: F0 method
        hybrid_method: Hybrid F0 method
        hop_length: Hop length
        embedders: Embedder model
        custom_embedders: Custom embedder path
        resample_sr: Resample rate
        filter_radius: Filter radius
        rms_mix_rate: RMS mix rate
        protect: Protection
        split_audio: Whether to split audio
        f0_autotune_strength: Autotune strength
        input_audio_name: Name of separated audio folder
        checkpointing: Use checkpointing
        onnx_f0_mode: Use ONNX for F0
        formant_shifting: Shift formants
        formant_qfrency: Formant frequency
        formant_timbre: Formant timbre
        f0_file: F0 file path
        embedders_mode: Embedder mode
        proposal_pitch: Use proposal pitch
        proposal_pitch_threshold: Proposal pitch threshold
        audio_processing: Use audio processing
        alpha: Alpha value

    Returns:
        Tuple of output values
    """
    global configs

    translations = _get_translations()
    gr_warning_func, _, _, process_output_func, replace_export_format_func = _get_ui_helpers()

    model_path = (
        os.path.join(configs["weights_path"], model)
        if not os.path.exists(model)
        else model
    )

    return_none = [None] * 6
    return_none[5] = {"visible": True, "__type__": "update"}

    if not use_audio:
        if merge_instrument or not_merge_backing or convert_backing or use_original:
            gr_warning_func(translations["turn_on_use_audio"])
            return return_none

    if use_original:
        if convert_backing:
            gr_warning_func(translations["turn_off_convert_backup"])
            return return_none
        elif not_merge_backing:
            gr_warning_func(translations["turn_off_merge_backup"])
            return return_none

    if not model or not os.path.exists(model_path) or os.path.isdir(model_path) or not model.endswith((".pth", ".onnx")):
        gr_warning_func(translations["provide_file"].format(filename=translations["model"]))
        return return_none

    f0method = method if method != "hybrid" else hybrid_method
    embedder_model = embedders if embedders != "custom" else custom_embedders

    if use_audio:
        # Use UVR output path instead of audios_path
        output_audio = get_uvr_output_path(input_audio_name)

        from advanced_rvc_inference.library.utils import pydub_load

        def get_audio_file(label: str) -> str:
            matching_files = [f for f in os.listdir(output_audio) if label in f]
            if not matching_files:
                return translations["notfound"]
            return os.path.join(output_audio, matching_files[0])

        output_path = os.path.join(output_audio, f"Convert_Vocals.{format}")
        output_backing = os.path.join(output_audio, f"Convert_Backing.{format}")
        output_merge_backup = os.path.join(output_audio, f"Vocals+Backing.{format}")
        output_merge_instrument = os.path.join(output_audio, f"Vocals+Instruments.{format}")

        if os.path.exists(output_audio):
            os.makedirs(output_audio, exist_ok=True)
        output_path = process_output_func(output_path)

        if use_original:
            original_vocal = get_audio_file("Original_Vocals_No_Reverb.")
            if original_vocal == translations["notfound"]:
                original_vocal = get_audio_file("Original_Vocals.")
            if original_vocal == translations["notfound"]:
                gr_warning_func(translations["not_found_original_vocal"])
                return return_none
            input_path = original_vocal
        else:
            main_vocal = get_audio_file("main_Vocals_No_Reverb.")
            backing_vocal = get_audio_file("Backing_Vocals.")
            if main_vocal == translations["notfound"]:
                main_vocal = get_audio_file("main_Vocals.")
            if main_vocal == translations["notfound"]:
                gr_warning_func(translations["not_found_main_vocal"])
                return return_none
            if not not_merge_backing and backing_vocal == translations["notfound"]:
                gr_warning_func(translations["not_found_backing_vocal"])
                return return_none
            input_path = main_vocal
            backing_path = backing_vocal

        # Convert vocals
        convert(
            pitch, filter_radius, index_rate, rms_mix_rate, protect, hop_length,
            f0method, input_path, output_path, model_path, index, autotune,
            clean, clean_strength, format, embedder_model, resample_sr,
            split_audio, f0_autotune_strength, checkpointing, onnx_f0_mode,
            embedders_mode, formant_shifting, formant_qfrency, formant_timbre,
            f0_file, proposal_pitch, proposal_pitch_threshold, audio_processing, alpha
        )

        # Convert backing if requested
        if convert_backing:
            output_backing = process_output_func(output_backing)
            convert(
                pitch, filter_radius, index_rate, rms_mix_rate, protect, hop_length,
                f0method, backing_path, output_backing, model_path, index, autotune,
                clean, clean_strength, format, embedder_model, resample_sr,
                split_audio, f0_autotune_strength, checkpointing, onnx_f0_mode,
                embedders_mode, formant_shifting, formant_qfrency, formant_timbre,
                f0_file, proposal_pitch, proposal_pitch_threshold, audio_processing, alpha
            )

        # Merge audio
        try:
            if not not_merge_backing and not use_original:
                backing_source = output_backing if convert_backing else backing_vocal
                output_merge_backup = process_output_func(output_merge_backup)
                pydub_load(output_path, volume=-4).overlay(
                    pydub_load(backing_source, volume=-6)
                ).export(output_merge_backup, format=format)

            if merge_instrument:
                vocals = output_merge_backup if not not_merge_backing and not use_original else output_path
                output_merge_instrument = process_output_func(output_merge_instrument)
                instruments = get_audio_file("Instruments.")
                if instruments == translations["notfound"]:
                    output_merge_instrument = None
                else:
                    pydub_load(instruments, volume=-7).overlay(
                        pydub_load(vocals, volume=-4 if use_original else None)
                    ).export(output_merge_instrument, format=format)
        except Exception:
            return return_none

        return [
            None if use_original else output_path,
            output_backing,
            None if not_merge_backing and use_original else output_merge_backup,
            output_path if use_original else None,
            output_merge_instrument if merge_instrument else None,
            {"visible": True, "__type__": "update"}
        ]
    else:
        if not input_path or not os.path.exists(input_path):
            gr_warning_func(translations["input_not_valid"])
            return return_none

        if not output_path:
            gr_warning_func(translations["output_not_valid"])
            return return_none

        output_path = replace_export_format_func(output_path, format)

        if os.path.isdir(input_path):
            if not any(
                f.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a",
                                   ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3"))
                for f in os.listdir(input_path)
            ):
                gr_warning_func(translations["not_found_in_folder"])
                return return_none

            output_dir = os.path.dirname(output_path) or output_path
            convert(
                pitch, filter_radius, index_rate, rms_mix_rate, protect, hop_length,
                f0method, input_path, output_dir, model_path, index, autotune,
                clean, clean_strength, format, embedder_model, resample_sr,
                split_audio, f0_autotune_strength, checkpointing, onnx_f0_mode,
                embedders_mode, formant_shifting, formant_qfrency, formant_timbre,
                f0_file, proposal_pitch, proposal_pitch_threshold, audio_processing, alpha
            )
            return return_none
        else:
            output_dir = os.path.dirname(output_path) or output_path
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            output_path = process_output_func(output_path)

            convert(
                pitch, filter_radius, index_rate, rms_mix_rate, protect, hop_length,
                f0method, input_path, output_path, model_path, index, autotune,
                clean, clean_strength, format, embedder_model, resample_sr,
                split_audio, f0_autotune_strength, checkpointing, onnx_f0_mode,
                embedders_mode, formant_shifting, formant_qfrency, formant_timbre,
                f0_file, proposal_pitch, proposal_pitch_threshold, audio_processing, alpha
            )

            return_none[0] = output_path
            return return_none


def convert_selection(
    clean: bool,
    autotune: bool,
    use_audio: bool,
    use_original: bool,
    convert_backing: bool,
    not_merge_backing: bool,
    merge_instrument: bool,
    pitch: int,
    clean_strength: float,
    model: str,
    index: Optional[str],
    index_rate: float,
    input_path: Optional[str],
    output_path: Optional[str],
    format: str,
    method: str,
    hybrid_method: str,
    hop_length: int,
    embedders: str,
    custom_embedders: str,
    resample_sr: int,
    filter_radius: int,
    rms_mix_rate: float,
    protect: float,
    split_audio: bool,
    f0_autotune_strength: float,
    checkpointing: bool,
    onnx_f0_mode: bool,
    formant_shifting: bool,
    formant_qfrency: float,
    formant_timbre: float,
    f0_file: str,
    embedders_mode: str,
    proposal_pitch: bool,
    proposal_pitch_threshold: float,
    audio_processing: bool = False,
    alpha: float = 0.5,
) -> Dict[str, Any]:
    """
    Convert audio with selection for separated tracks.

    Args:
        clean: Whether to clean audio
        autotune: Whether to apply autotune
        use_audio: Whether to use separated audio
        use_original: Whether to use original vocals
        convert_backing: Whether to convert backing
        not_merge_backing: Whether to skip merging backing
        merge_instrument: Whether to merge with instruments
        pitch: Pitch shift
        clean_strength: Cleaning strength
        model: Model name/path
        index: Index path
        index_rate: Index rate
        input_path: Input path
        output_path: Output path
        format: Output format
        method: F0 method
        hybrid_method: Hybrid F0 method
        hop_length: Hop length
        embedders: Embedder model
        custom_embedders: Custom embedder path
        resample_sr: Resample rate
        filter_radius: Filter radius
        rms_mix_rate: RMS mix rate
        protect: Protection
        split_audio: Whether to split audio
        f0_autotune_strength: Autotune strength
        checkpointing: Use checkpointing
        onnx_f0_mode: Use ONNX for F0
        formant_shifting: Shift formants
        formant_qfrency: Formant frequency
        formant_timbre: Formant timbre
        f0_file: F0 file path
        embedders_mode: Embedder mode
        proposal_pitch: Use proposal pitch
        proposal_pitch_threshold: Proposal pitch threshold
        audio_processing: Use audio processing
        alpha: Alpha value

    Returns:
        Dictionary with UI update values
    """
    global configs

    translations = _get_translations()
    gr_info_func = _get_ui_helpers()[0]

    if use_audio:
        gr_info_func(translations["search_separate"])
        choice = search_separated_audio_folders()

        gr_info_func(translations["found_choice"].format(choice=len(choice)))

        if len(choice) == 0:
            gr_info_func(translations["separator==0"])
            return {
                "choices": [],
                "value": "",
                "interactive": False,
                "visible": False,
                "__type__": "update"
            }, None, None, None, None, None, {"visible": True, "__type__": "update"}, {"visible": False, "__type__": "update"}
        elif len(choice) == 1:
            convert_output = convert_audio(
                clean, autotune, use_audio, use_original, convert_backing,
                not_merge_backing, merge_instrument, pitch, clean_strength,
                model, index, index_rate, None, None, format, method,
                hybrid_method, hop_length, embedders, custom_embedders,
                resample_sr, filter_radius, rms_mix_rate, protect,
                split_audio, f0_autotune_strength, choice[0], checkpointing,
                onnx_f0_mode, formant_shifting, formant_qfrency, formant_timbre,
                f0_file, embedders_mode, proposal_pitch, proposal_pitch_threshold,
                audio_processing, alpha
            )
            return {
                "choices": [],
                "value": "",
                "interactive": False,
                "visible": False,
                "__type__": "update"
            }, convert_output[0], convert_output[1], convert_output[2], convert_output[3], convert_output[4], {"visible": True, "__type__": "update"}, {"visible": False, "__type__": "update"}
        else:
            return {
                "choices": choice,
                "value": choice[0],
                "interactive": True,
                "visible": True,
                "__type__": "update"
            }, None, None, None, None, None, {"visible": False, "__type__": "update"}, {"visible": True, "__type__": "update"}
    else:
        advanced_rvc_inference_convert = convert_audio(
            clean, autotune, use_audio, use_original, convert_backing,
            not_merge_backing, merge_instrument, pitch, clean_strength,
            model, index, index_rate, input_path, output_path, format,
            method, hybrid_method, hop_length, embedders, custom_embedders,
            resample_sr, filter_radius, rms_mix_rate, protect,
            split_audio, f0_autotune_strength, None, checkpointing,
            onnx_f0_mode, formant_shifting, formant_qfrency, formant_timbre,
            f0_file, embedders_mode, proposal_pitch, proposal_pitch_threshold,
            audio_processing, alpha
        )
        return {
            "choices": [],
            "value": "",
            "interactive": False,
            "visible": False,
            "__type__": "update"
        }, advanced_rvc_inference_convert[0], None, None, None, None, {"visible": True, "__type__": "update"}, {"visible": False, "__type__": "update"}


def whisper_process(
    model_size: str,
    input_audio: str,
    configs: Dict[str, Any],
    device: str,
    out_queue,
    word_timestamps: bool = True,
):
    """Process audio with Whisper for speaker diarization."""
    from advanced_rvc_inference.library.speaker_diarization.whisper import load_model

    try:
        model = load_model(model_size, device=device)
        segments = model.transcribe(
            input_audio,
            fp16=configs.get("fp16", False),
            word_timestamps=word_timestamps
        )
        out_queue.put(segments["segments"])
    except Exception as e:
        out_queue.put(e)
    finally:
        gc.collect()


def convert_with_whisper(
    num_spk: int,
    model_size: str,
    cleaner: bool,
    clean_strength: float,
    autotune: bool,
    f0_autotune_strength: float,
    checkpointing: bool,
    model_1: str,
    model_2: str,
    model_index_1: Optional[str],
    model_index_2: Optional[str],
    pitch_1: int,
    pitch_2: int,
    index_strength_1: float,
    index_strength_2: float,
    export_format: str,
    input_audio: str,
    output_audio: str,
    onnx_f0_mode: bool,
    method: str,
    hybrid_method: str,
    hop_length: int,
    embed_mode: str,
    embedders: str,
    custom_embedders: str,
    resample_sr: int,
    filter_radius: int,
    rms_mix_rate: float,
    protect: float,
    formant_shifting: bool,
    formant_qfrency_1: float,
    formant_timbre_1: float,
    formant_qfrency_2: float,
    formant_timbre_2: float,
    proposal_pitch: bool,
    proposal_pitch_threshold: float,
    audio_processing: bool = False,
    alpha: float = 0.5,
) -> Optional[str]:
    """
    Convert audio with speaker diarization using Whisper.

    Args:
        num_spk: Number of speakers
        model_size: Whisper model size
        cleaner: Whether to clean audio
        clean_strength: Cleaning strength
        autotune: Whether to apply autotune
        f0_autotune_strength: Autotune strength
        checkpointing: Use checkpointing
        model_1: First model path
        model_2: Second model path
        model_index_1: First model index
        model_index_2: Second model index
        pitch_1: First pitch shift
        pitch_2: Second pitch shift
        index_strength_1: First index strength
        index_strength_2: Second index strength
        export_format: Output format
        input_audio: Input audio path
        output_audio: Output audio path
        onnx_f0_mode: Use ONNX for F0
        method: F0 method
        hybrid_method: Hybrid F0 method
        hop_length: Hop length
        embed_mode: Embedder mode
        embedders: Embedder model
        custom_embedders: Custom embedder path
        resample_sr: Resample rate
        filter_radius: Filter radius
        rms_mix_rate: RMS mix rate
        protect: Protection
        formant_shifting: Shift formants
        formant_qfrency_1: First formant frequency
        formant_timbre_1: First formant timbre
        formant_qfrency_2: Second formant frequency
        formant_timbre_2: Second formant timbre
        proposal_pitch: Use proposal pitch
        proposal_pitch_threshold: Proposal pitch threshold
        audio_processing: Use audio processing
        alpha: Alpha value

    Returns:
        Path to output audio or None if failed
    """
    global configs, config

    translations = _get_translations()
    gr_info_func = _get_ui_helpers()[0]
    gr_error_func = _get_ui_helpers()[2]

    try:
        import multiprocessing as mp
        import librosa
        import numpy as np
        from pydub import AudioSegment
        from sklearn.cluster import AgglomerativeClustering

        from advanced_rvc_inference.library.utils import clear_gpu_cache, pydub_load
        from advanced_rvc_inference.library.speaker_diarization.audio import Audio
        from advanced_rvc_inference.library.speaker_diarization.segment import Segment
        from advanced_rvc_inference.library.speaker_diarization.embedding import SpeechBrainPretrainedSpeakerEmbedding

        # Validate models
        model_pth_1 = (
            os.path.join(configs["weights_path"], model_1)
            if not os.path.exists(model_1)
            else model_1
        )
        model_pth_2 = (
            os.path.join(configs["weights_path"], model_2)
            if not os.path.exists(model_2)
            else model_2
        )

        if (not model_1 or not os.path.exists(model_pth_1) or os.path.isdir(model_pth_1) or not model_pth_1.endswith((".pth", ".onnx"))) and \
           (not model_2 or not os.path.exists(model_pth_2) or os.path.isdir(model_pth_2) or not model_pth_2.endswith((".pth", ".onnx"))):
            gr_warning_func = _get_ui_helpers()[1]
            gr_warning_func(translations["provide_file"].format(filename=translations["model"]))
            return None

        if not model_1:
            model_pth_1 = model_pth_2
        if not model_2:
            model_pth_2 = model_pth_1

        if not input_audio or not os.path.exists(input_audio) or os.path.isdir(input_audio):
            gr_warning_func = _get_ui_helpers()[1]
            gr_warning_func(translations["input_not_valid"])
            return None

        if not output_audio:
            gr_warning_func = _get_ui_helpers()[1]
            gr_warning_func(translations["output_not_valid"])
            return None

        process_output_func = _get_ui_helpers()[3]
        output_audio = process_output_func(output_audio)
        gr_info_func(translations["start_whisper"])

        # Setup multiprocessing
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass

        whisper_queue = mp.Queue()
        whisperprocess = mp.Process(
            target=whisper_process,
            args=(model_size, input_audio, configs, config.device, whisper_queue, True)
        )
        whisperprocess.start()

        segments = whisper_queue.get()
        audio = Audio()

        embedding_model = SpeechBrainPretrainedSpeakerEmbedding(
            embedding=os.path.join(configs["speaker_diarization_path"], "models", "speechbrain"),
            device=config.device
        )
        y, sr = librosa.load(input_audio, sr=None)
        duration = len(y) / sr

        def segment_embedding(segment):
            waveform, _ = audio.crop(input_audio, Segment(segment["start"], min(duration, segment["end"])))
            return embedding_model(waveform.mean(dim=0, keepdim=True)[None] if waveform.shape[0] == 2 else waveform[None])

        def time_format(secs):
            return datetime.timedelta(seconds=round(secs))

        def merge_audio(files_list: List[str], time_stamps: List[Tuple[int, int]], original_file_path: str, output_path: str, fmt: str) -> str:
            def extract_number(filename: str) -> int:
                match = re.search(r"_(\d+)", filename)
                return int(match.group(1)) if match else 0

            total_duration = len(pydub_load(original_file_path))
            combined = AudioSegment.empty()
            current_position = 0

            for file, (start_i, end_i) in zip(sorted(files_list, key=extract_number), time_stamps):
                if start_i > current_position:
                    combined += AudioSegment.silent(duration=start_i - current_position)
                combined += pydub_load(file)
                current_position = end_i

            if current_position < total_duration:
                combined += AudioSegment.silent(duration=total_duration - current_position)
            combined.export(output_path, format=fmt)
            return output_path

        # Generate embeddings
        embeddings = np.zeros(shape=(len(segments), 192))
        for i, segment in enumerate(segments):
            embeddings[i] = segment_embedding(segment)

        # Cluster speakers
        labels = AgglomerativeClustering(num_spk).fit(np.nan_to_num(embeddings)).labels_
        for i in range(len(segments)):
            segments[i]["speaker"] = "SPEAKER " + str(labels[i] + 1)

        # Merge segments
        merged_segments = []
        current_text = []
        current_speaker = None
        current_start = None
        end_time = 0

        for i, segment in enumerate(segments):
            speaker = segment["speaker"]
            start_time = segment["start"]
            text = segment["text"][1:]

            if speaker == current_speaker:
                current_text.append(text)
                end_time = segment["end"]
            else:
                if current_speaker is not None:
                    merged_segments.append({
                        "speaker": current_speaker,
                        "start": current_start,
                        "end": end_time,
                        "text": " ".join(current_text)
                    })
                current_speaker = speaker
                current_start = start_time
                current_text = [text]
                end_time = segment["end"]

        if current_speaker is not None:
            merged_segments.append({
                "speaker": current_speaker,
                "start": current_start,
                "end": end_time,
                "text": " ".join(current_text)
            })

        gr_info_func(translations["whisper_done"])

        # Log transcription
        x = ""
        for segment in merged_segments:
            x += f"\n{segment['speaker']} {str(time_format(segment['start']))} - {str(time_format(segment['end']))}\n"
            x += segment["text"] + "\n"

        _get_logger().info(x)

        # Cleanup
        del audio, embedding_model, segments, labels
        clear_gpu_cache()
        gc.collect()

        gr_info_func(translations["process_audio"])

        # Process segments
        audio = pydub_load(input_audio)
        output_folder = "audios_temp"

        if os.path.exists(output_folder):
            shutil.rmtree(output_folder, ignore_errors=True)
        for f in [output_folder, os.path.join(output_folder, "1"), os.path.join(output_folder, "2")]:
            os.makedirs(f, exist_ok=True)

        time_stamps = []
        processed_segments = []
        for i, segment in enumerate(merged_segments):
            start_ms = int(segment["start"] * 1000)
            end_ms = int(segment["end"] * 1000)
            index = i + 1

            segment_filename = os.path.join(output_folder, "1" if i % 2 == 1 else "2", f"segment_{index}.wav")
            audio[start_ms:end_ms].export(segment_filename, format="wav")

            processed_segments.append(
                os.path.join(output_folder, "1" if i % 2 == 1 else "2", f"segment_{index}_output.wav")
            )
            time_stamps.append((start_ms, end_ms))

        f0method = method if method != "hybrid" else hybrid_method
        embedder_model = embedders if embedders != "custom" else custom_embedders

        gr_info_func(translations["process_done_start_convert"])

        # Convert both speaker tracks
        convert(
            pitch_1, filter_radius, index_strength_1, rms_mix_rate, protect, hop_length,
            f0method, os.path.join(output_folder, "1"), output_folder, model_pth_1,
            model_index_1, autotune, cleaner, clean_strength, "wav", embedder_model,
            resample_sr, False, f0_autotune_strength, checkpointing, onnx_f0_mode,
            embed_mode, formant_shifting, formant_qfrency_1, formant_timbre_1, "",
            proposal_pitch, proposal_pitch_threshold, audio_processing, alpha
        )
        convert(
            pitch_2, filter_radius, index_strength_2, rms_mix_rate, protect, hop_length,
            f0method, os.path.join(output_folder, "2"), output_folder, model_pth_2,
            model_index_2, autotune, cleaner, clean_strength, "wav", embedder_model,
            resample_sr, False, f0_autotune_strength, checkpointing, onnx_f0_mode,
            embed_mode, formant_shifting, formant_qfrency_2, formant_timbre_2, "",
            proposal_pitch, proposal_pitch_threshold, audio_processing, alpha
        )

        gr_info_func(translations["convert_success"])
        replace_export_format_func = _get_ui_helpers()[4]
        return merge_audio(
            processed_segments, time_stamps, input_audio,
            replace_export_format_func(output_audio, export_format),
            export_format
        )

    except Exception as e:
        gr_error_func(translations["error_occurred"].format(e=e))
        import traceback

        _get_logger().debug(traceback.format_exc())
        return None

    finally:
        if os.path.exists("audios_temp"):
            shutil.rmtree("audios_temp", ignore_errors=True)


def convert_tts(
    clean: bool,
    autotune: bool,
    pitch: int,
    clean_strength: float,
    model: str,
    index: Optional[str],
    index_rate: float,
    input_path: str,
    output_path: str,
    format: str,
    method: str,
    hybrid_method: str,
    hop_length: int,
    embedders: str,
    custom_embedders: str,
    resample_sr: int,
    filter_radius: int,
    rms_mix_rate: float,
    protect: float,
    split_audio: bool,
    f0_autotune_strength: float,
    checkpointing: bool,
    onnx_f0_mode: bool,
    formant_shifting: bool,
    formant_qfrency: float,
    formant_timbre: float,
    f0_file: str,
    embedders_mode: str,
    proposal_pitch: bool,
    proposal_pitch_threshold: float,
    audio_processing: bool = False,
    alpha: float = 0.5,
) -> Optional[str]:
    """
    Convert TTS audio with RVC model.

    Args:
        clean: Whether to clean audio
        autotune: Whether to apply autotune
        pitch: Pitch shift
        clean_strength: Cleaning strength
        model: Model name/path
        index: Index path
        index_rate: Index rate
        input_path: Input path
        output_path: Output path
        format: Output format
        method: F0 method
        hybrid_method: Hybrid F0 method
        hop_length: Hop length
        embedders: Embedder model
        custom_embedders: Custom embedder path
        resample_sr: Resample rate
        filter_radius: Filter radius
        rms_mix_rate: RMS mix rate
        protect: Protection
        split_audio: Whether to split audio
        f0_autotune_strength: Autotune strength
        checkpointing: Use checkpointing
        onnx_f0_mode: Use ONNX for F0
        formant_shifting: Shift formants
        formant_qfrency: Formant frequency
        formant_timbre: Formant timbre
        f0_file: F0 file path
        embedders_mode: Embedder mode
        proposal_pitch: Use proposal pitch
        proposal_pitch_threshold: Proposal pitch threshold
        audio_processing: Use audio processing
        alpha: Alpha value

    Returns:
        Path to output audio or None if failed
    """
    global configs

    translations = _get_translations()
    gr_warning_func = _get_ui_helpers()[1]
    gr_info_func = _get_ui_helpers()[0]
    replace_export_format_func = _get_ui_helpers()[4]

    model_path = (
        os.path.join(configs["weights_path"], model)
        if not os.path.exists(model)
        else model
    )

    if not model_path or not os.path.exists(model_path) or os.path.isdir(model_path) or not model.endswith((".pth", ".onnx")):
        gr_warning_func(translations["provide_file"].format(filename=translations["model"]))
        return None

    if not input_path or not os.path.exists(input_path):
        gr_warning_func(translations["input_not_valid"])
        return None

    if os.path.isdir(input_path):
        input_audio = [f for f in os.listdir(input_path) if "tts" in f and f.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3"))]
        if not input_audio:
            gr_warning_func(translations["not_found_in_folder"])
            return None
        input_path = os.path.join(input_path, input_audio[0])

    if not output_path:
        gr_warning_func(translations["output_not_valid"])
        return None

    output_path = replace_export_format_func(output_path, format)
    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, f"tts.{format}")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    process_output_func = _get_ui_helpers()[3]
    output_path = process_output_func(output_path)

    f0method = method if method != "hybrid" else hybrid_method
    embedder_model = embedders if embedders != "custom" else custom_embedders

    gr_info_func(translations["convert_vocal"])

    convert(
        pitch, filter_radius, index_rate, rms_mix_rate, protect, hop_length,
        f0method, input_path, output_path, model_path, index, autotune,
        clean, clean_strength, format, embedder_model, resample_sr,
        split_audio, f0_autotune_strength, checkpointing, onnx_f0_mode,
        embedders_mode, formant_shifting, formant_qfrency, formant_timbre,
        f0_file, proposal_pitch, proposal_pitch_threshold, audio_processing, alpha
    )

    gr_info_func(translations["convert_success"])
    return output_path

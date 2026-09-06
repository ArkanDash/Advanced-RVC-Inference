"""
Lightweight feedback and UI helper functions for Advanced RVC Inference.

This module provides:
1. Logging-only feedback functions (gr_info, gr_warning, gr_error)
2. Common utility functions (process_output, shutil_move, replace_*, etc.)
3. Gradio UI helper functions (visible, change_models_choices, get_index, etc.)

These functions can be safely imported in headless/CLI/Colab-no-UI mode
without requiring Gradio or any UI dependencies. The Gradio helpers return
plain dict updates ({"__type__": "update", ...}) that work with both
Gradio 4.x and 5.x.
"""

import os
import re
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Lazy reference to config - only loaded when needed
_config = None
_configs = None

# Cache for variables that may not always be available
_variables_cache = {}


def _get_config():
    """Lazily get the config singleton."""
    global _config
    if _config is None:
        from arvc.utils.variables import config
        _config = config
    return _config


def _get_configs():
    """Lazily get the configs dict."""
    global _configs
    if _configs is None:
        from arvc.utils.variables import configs
        _configs = configs
    return _configs


def _get_var(name, default=None):
    """Lazily get a variable from arvc.utils.variables."""
    if name not in _variables_cache:
        try:
            from arvc.utils import variables as _vars_mod
            val = getattr(_vars_mod, name, default)
            _variables_cache[name] = val
        except Exception:
            _variables_cache[name] = default
    return _variables_cache[name]


# ============================================================
# Feedback functions (logging-only, no Gradio dependency)
# ============================================================

def gr_info(message: str) -> None:
    """Display info message in log. In UI mode, also shows a Gradio toast."""
    logger.info(message)


def gr_warning(message: str) -> None:
    """Display warning message in log. In UI mode, also shows a Gradio toast."""
    logger.warning(message)


def gr_error(message: str, **kwargs) -> None:
    """Display error message in log. In UI mode, also shows a Gradio toast."""
    logger.error(message)


# ============================================================
# Utility functions (no UI dependency)
# ============================================================

def process_output(file_path: str) -> str:
    """Process output file path to avoid overwriting existing files."""
    try:
        config = _get_config()
        if config.configs.get("delete_exists_file", True):
            if os.path.exists(file_path) and os.path.isfile(file_path):
                os.remove(file_path)
            return file_path
        else:
            if not os.path.exists(file_path):
                return file_path

            # Generate a new filename to avoid overwriting
            base, ext = os.path.splitext(os.path.basename(file_path))
            directory = os.path.dirname(file_path)

            counter = 1
            while True:
                new_file_path = os.path.join(directory, f"{base}_{counter}{ext}")
                if not os.path.exists(new_file_path):
                    return new_file_path
                counter += 1
    except Exception as e:
        logger.error(f"Error processing output path {file_path}: {str(e)}")
        return file_path


def shutil_move(input_path: str, output_path: str) -> str:
    """Safely move a file to a new location."""
    try:
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, os.path.basename(input_path))

        processed_path = process_output(output_path)
        return shutil.move(input_path, processed_path)
    except Exception as e:
        logger.error(f"Error moving file from {input_path} to {output_path}: {str(e)}")
        raise


def replace_punctuation(filename: str) -> str:
    """Sanitize filename by removing/replacing problematic characters."""
    try:
        result = filename
        result = result.replace("-_-", "_").replace("_-_", "_")
        for ch in ["(", ")", "[", "]", ",", '"', "'", "{", "}"]:
            result = result.replace(ch, "")
        result = result.replace(" ", "_").replace("|", "_")
        result = re.sub(r'[-_]+', '_', result)
        return result.strip('_').strip()
    except Exception as e:
        logger.error(f"Error replacing punctuation in {filename}: {str(e)}")
        return filename


def replace_url(url: str) -> str:
    """Sanitize URL for downloading."""
    try:
        return url.replace("/blob/", "/resolve/").replace("/tree/", "/resolve/").replace("?download=true", "").strip()
    except Exception as e:
        logger.error(f"Error replacing URL in {url}: {str(e)}")
        return url


def replace_modelname(modelname: str) -> str:
    """Sanitize model name by removing extensions and problematic characters."""
    try:
        clean_name = modelname.replace(".onnx", "").replace(".pth", "").replace(".index", "").replace(".zip", "")
        return replace_punctuation(clean_name)
    except Exception as e:
        logger.error(f"Error replacing model name in {modelname}: {str(e)}")
        return modelname


def replace_export_format(audio_path: str, export_format: str = "wav") -> str:
    """Change the export format of an audio file path."""
    try:
        export_format = f".{export_format}"
        if audio_path.endswith(export_format):
            return audio_path

        base_path = os.path.splitext(audio_path)[0]
        return f"{base_path}{export_format}"
    except Exception as e:
        logger.error(f"Error replacing export format in {audio_path}: {str(e)}")
        return audio_path


# ============================================================
# Gradio UI helper functions
# These return plain dict updates that work with Gradio 4.x/5.x
# ============================================================

def _update_dict(**kwargs):
    """Build a Gradio-compatible update dict."""
    out = {"__type__": "update"}
    out.update(kwargs)
    return out


def visible(value):
    """Return a Gradio update that sets visibility based on truthiness of `value`.

    Used as a callback: `checkbox.change(fn=visible, inputs=[checkbox], outputs=[row])`.
    """
    return _update_dict(visible=bool(value))


def valueFalse_interactive(value):
    """Return a Gradio update that sets interactive=False with the same value.

    Used to lock a control after a value is selected.
    """
    return _update_dict(interactive=False, value=value)


def change_models_choices():
    """Refresh model and index dropdowns.

    Returns a list of two update dicts: [model_pth_update, model_index_update].
    """
    model_name = _get_var("model_name", [])
    index_path = _get_var("index_path", [])
    return [
        _update_dict(choices=list(model_name), value=None),
        _update_dict(choices=list(index_path), value=None),
    ]


def change_pretrained_choices():
    """Refresh pretrained G/D dropdowns.

    Returns [pretrained_D_update, pretrained_G_update].
    """
    pretrainedD = _get_var("pretrainedD", [])
    pretrainedG = _get_var("pretrainedG", [])
    return [
        _update_dict(choices=list(pretrainedD), value=None),
        _update_dict(choices=list(pretrainedG), value=None),
    ]


def change_download_choices(*args, **kwargs):
    """Refresh download model dropdown (list of available model_options keys)."""
    model_options = _get_var("model_options", {}) or {}
    choices = list(model_options.keys()) if isinstance(model_options, dict) else []
    return _update_dict(choices=choices, value=None)


def change_download_pretrained_choices(*args, **kwargs):
    """Refresh download pretrained dropdown based on selected RVC version."""
    pretrainedD = _get_var("pretrainedD", [])
    pretrainedG = _get_var("pretrainedG", [])
    return [
        _update_dict(choices=list(pretrainedD), value=None),
        _update_dict(choices=list(pretrainedG), value=None),
    ]


def get_index(model_pth):
    """Return index file choices that match the selected model name.

    When the user picks a model, this updates the index dropdown with the
    index files found in the corresponding model log directory.
    """
    try:
        if not model_pth:
            return _update_dict(choices=[], value=None)

        # Get the model name without extension and look for matching index files
        from arvc.utils.variables import configs, LOGS_PATH
        base_name = os.path.splitext(os.path.basename(model_pth))[0]
        logs_path = Path(configs.get("logs_path", str(LOGS_PATH)))

        candidate_dirs = [logs_path / base_name, logs_path]
        indices = []
        for d in candidate_dirs:
            if d.exists():
                for f in d.rglob("*.index"):
                    if "trained" not in f.name and f.name not in indices:
                        indices.append(str(f))

        return _update_dict(choices=indices, value=indices[0] if indices else None)
    except Exception:
        index_path = _get_var("index_path", [])
        return _update_dict(choices=list(index_path), value=None)


def index_strength_show(model_index):
    """Show/hide the index strength slider based on whether an index is selected."""
    return _update_dict(visible=bool(model_index))


def unlock_f0(value):
    """Toggle the f0 method dropdown between short and full method lists.

    When the user enables "unlock full method", returns the full list,
    otherwise returns the short list.
    """
    method_f0 = _get_var("method_f0", [])
    method_f0_full = _get_var("method_f0_full", [])
    if value:
        return _update_dict(choices=list(method_f0_full), value=method_f0_full[0] if method_f0_full else None)
    return _update_dict(choices=list(method_f0), value=method_f0[0] if method_f0 else None)


def hoplength_show(method, hybrid_method=None):
    """Show hop_length slider only for methods that use it (crepe/rmvpe/hybrid)."""
    hop_methods = {
        "mangio-crepe-tiny", "mangio-crepe-small", "mangio-crepe-medium",
        "mangio-crepe-large", "mangio-crepe-full",
        "crepe-tiny", "crepe-small", "crepe-medium", "crepe-large", "crepe-full",
        "rmvpe", "rmvpe-clipping", "rmvpe-medfilt", "rmvpe-clipping-medfilt",
        "hpa-rmvpe", "hpa-rmvpe-medfilt",
        "hybrid",
        "penn", "mangio-penn",
        "djcm", "djcm-clipping", "djcm-medfilt", "djcm-clipping-medfilt",
    }
    show = method in hop_methods
    return _update_dict(visible=show)


def change_embedders_mode(mode):
    """Return embedders choices based on the selected mode (fairseq/onnx/transformers/spin/whisper)."""
    embedders_model = _get_var("embedders_model", [])
    spin_model = _get_var("spin_model", [])
    whisper_model = _get_var("whisper_model", [])

    if mode == "spin":
        choices = list(spin_model)
    elif mode == "whisper":
        choices = list(whisper_model)
    else:
        # fairseq, onnx, transformers all use embedders_model
        choices = list(embedders_model)

    return _update_dict(choices=choices, value=choices[0] if choices else None)


def change_audios_choices(audio_path=None):
    """Refresh audio file dropdown with all audio files in the audios folder."""
    paths_for_files = _get_var("paths_for_files", [])
    return _update_dict(choices=list(paths_for_files), value=None)


def change_f0_choices():
    """Refresh f0 file dropdown with all f0 .txt files."""
    f0_file = _get_var("f0_file", [])
    return _update_dict(choices=list(f0_file), value=None)


def change_preset_choices():
    """Refresh conversion preset dropdown with all .conversion.json files."""
    presets_file = _get_var("presets_file", [])
    return _update_dict(choices=list(presets_file), value=None)


def change_effect_preset_choices():
    """Refresh audio effect preset dropdown with all .effect.json files."""
    audio_effect_presets_file = _get_var("audio_effect_presets_file", [])
    return _update_dict(choices=list(audio_effect_presets_file), value=None)


def change_tts_voice_choices(use_google):
    """Return TTS voice choices based on whether Google TTS is selected."""
    try:
        if use_google:
            google_tts_voice = _get_var("google_tts_voice", ["vi", "en"])
            return _update_dict(choices=list(google_tts_voice), value=google_tts_voice[0] if google_tts_voice else None)
        edgetts = _get_var("edgetts", ["vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"])
        return _update_dict(choices=list(edgetts), value=edgetts[0] if edgetts else None)
    except Exception:
        return _update_dict(choices=[], value=None)


def change_backing_choices(*args, **kwargs):
    """Return backing track choices (placeholder - returns empty update)."""
    return _update_dict(choices=[], value=None)


def separate_change(*args, **kwargs):
    """Update UI when separation tab changes (placeholder)."""
    return _update_dict()


def create_dataset_change(*args, **kwargs):
    """Update UI when create-dataset tab changes (placeholder)."""
    return _update_dict()


def change_fp(value):
    """Toggle FP precision dropdown (placeholder)."""
    return _update_dict(value=value)


def get_gpu_info():
    """Return GPU information string for display."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            return f"{gpu_name} ({gpu_mem} GB)"
        return "CPU only"
    except Exception:
        return "GPU info unavailable"


def gpu_number_str():
    """Return the number of available GPUs as a string."""
    try:
        import torch
        if torch.cuda.is_available():
            return str(torch.cuda.device_count())
        return "0"
    except Exception:
        return "0"


def vocoders_lock(pitch_guidance):
    """Lock vocoder dropdown based on whether pitch guidance is enabled.

    Some vocoders (e.g. HiFi-GAN NSF) require pitch guidance. When pitch guidance
    is off, only vocoders that don't require pitch are selectable.
    """
    vocoder_choices = ["Default", "BigVGAN", "MRF-HiFi-GAN", "RefineGAN"]
    if not pitch_guidance:
        # Only keep vocoders that don't require pitch
        vocoder_choices = ["BigVGAN", "RefineGAN"]
    return _update_dict(
        choices=vocoder_choices,
        value=vocoder_choices[0] if vocoder_choices else None,
    )


def unlock_ver(rvc_version, vocoders):
    """Lock RVC version based on selected vocoder (some vocoders are v2-only)."""
    # Vocoder "Default" works for both v1 and v2; others are v2-only
    if vocoders in ("BigVGAN", "MRF-HiFi-GAN", "RefineGAN"):
        return _update_dict(choices=["v2"], value="v2")
    return _update_dict(choices=["v1", "v2"], value=rvc_version or "v2")


def unlock_vocoder(rvc_version, vocoders):
    """Lock vocoder dropdown based on RVC version (v1 only supports Default)."""
    if rvc_version == "v1":
        return _update_dict(choices=["Default"], value="Default")
    return _update_dict(
        choices=["Default", "BigVGAN", "MRF-HiFi-GAN", "RefineGAN"],
        value=vocoders or "Default",
    )


def pitch_guidance_lock(vocoders):
    """Lock pitch guidance checkbox based on selected vocoder.

    Default (HiFi-GAN NSF) requires pitch guidance; others are optional.
    """
    if vocoders == "Default":
        return _update_dict(value=True, interactive=False)
    return _update_dict(value=False, interactive=False)


def change_reference_choices():
    """Refresh reference set dropdown with available reference folders."""
    reference_list = _get_var("reference_list", [])
    return _update_dict(choices=list(reference_list), value=None)


# ============================================================
# Audio device helpers (used by realtime tab)
# ============================================================

def audio_device(*args, **kwargs):
    """Return (input_devices, output_devices) where each is a dict
    mapping device name -> [device_id, max_channels].

    Used by the realtime tab to populate the input/output device
    dropdowns and to retrieve the device ID + channel count when
    starting realtime conversion.

    Returns:
        (dict, dict)
    """
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        # Normalize to a list (newer sounddevice returns a dict-like)
        if not isinstance(devices, list):
            devices = [devices[i] for i in range(len(devices))]

        input_devices = {}
        output_devices = {}
        for idx, d in enumerate(devices):
            name = d.get("name", f"Device {idx}")
            max_inputs = d.get("max_input_channels", 0) or 0
            max_outputs = d.get("max_output_channels", 0) or 0

            if max_inputs > 0:
                # Avoid clobbering when the same name appears twice
                key = name if name not in input_devices else f"{name} (in:{idx})"
                input_devices[key] = [idx, max_inputs]
            if max_outputs > 0:
                key = name if name not in output_devices else f"{name} (out:{idx})"
                output_devices[key] = [idx, max_outputs]

        return input_devices, output_devices
    except Exception:
        return {}, {}


def change_audio_device_choices(*args, **kwargs):
    """Refresh audio device dropdown choices."""
    input_devices, output_devices = audio_device()
    return _update_dict(choices=list(input_devices.keys()), value=None)


def update_audio_device(*args, **kwargs):
    """Update audio device selection (placeholder)."""
    return _update_dict()


def update_dropdowns_from_json(*args, **kwargs):
    """Update multiple dropdowns from a JSON config file (placeholder)."""
    return _update_dict()


def update_button_from_json(*args, **kwargs):
    """Update button state from a JSON config file (placeholder)."""
    return _update_dict()

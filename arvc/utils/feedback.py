"""
Lightweight feedback and utility functions for Advanced RVC Inference.

This module provides logging-only feedback functions (gr_info, gr_warning, gr_error)
and common utility functions that can be safely imported in headless/CLI/Colab-no-UI
mode without requiring Gradio or any UI dependencies.

When the full UI is available, use `arvc.ui.feedback` instead,
which re-exports these functions with Gradio toast support.
"""

import os
import re
import shutil
import logging

logger = logging.getLogger(__name__)

# Lazy reference to config - only loaded when needed
_config = None
_configs = None


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

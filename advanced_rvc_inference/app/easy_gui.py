"""
Easy GUI Module for Advanced RVC Inference.

A simplified Gradio interface inspired by the "EasierGUI" from
Mangio-Kalo-Tweaks. Provides a streamlined experience with:
- Quick Convert: One-tab voice conversion
- One-Click Train: Full training pipeline in a single click
- Download: Quick model download
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_easy_app(theme=None):
    """Create the Easy GUI Gradio Blocks app.

    A simplified, user-friendly interface for Advanced RVC Inference
    with three tabs: Quick Convert, One-Click Train, and Download.

    Args:
        theme: Optional Gradio theme to apply.

    Returns:
        gr.Blocks: The Gradio app instance.
    """
    import gradio as gr

    # Ensure CWD is in path for imports
    cwd = Path.cwd()
    if str(cwd) not in sys.path:
        sys.path.insert(0, str(cwd))

    from advanced_rvc_inference.utils.variables import (
        python,
        translations,
        configs,
        model_name as model_names,
        index_path as index_paths,
        allow_disk,
    )
    from advanced_rvc_inference.core.training import one_click_train
    from advanced_rvc_inference.core.downloads import download_model
    from advanced_rvc_inference.core.ui import (
        change_models_choices,
        get_index,
        index_strength_show,
        shutil_move,
    )
    from advanced_rvc_inference.library.optimizers import get_optimizer_choices
    from advanced_rvc_inference.library.generators import get_vocoder_choices

    # F0 methods for the easy UI (simplified subset)
    easy_f0_methods = ["rmvpe", "crepe-full", "harvest", "fcpe"]

    def easy_convert(
        input_audio, model, pitch, index_rate, f0_method, index_file, export_format
    ):
        """Run voice conversion via the convert CLI script."""
        import gradio as _gr

        if not input_audio:
            _gr.Warning("Please provide an input audio file.")
            return None

        if not model:
            _gr.Warning("Please select a model.")
            return None

        model_path = os.path.join(configs["weights_path"], model)
        if not os.path.exists(model_path):
            _gr.Warning(f"Model file not found: {model}")
            return None

        output_path = os.path.join(
            configs["audios_path"], "easy_convert_output." + export_format
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        index_path_arg = ""
        if index_file and os.path.exists(index_file):
            index_path_arg = f' --index "{index_file}"'

        cmd = (
            f'{python} {configs["convert_path"]}'
            f' -i "{input_audio}"'
            f' -m "{model_path}"'
            f' -p {pitch}'
            f' --f0_method {f0_method}'
            f' --index_rate {index_rate}'
            f'{index_path_arg}'
            f' -f {export_format}'
            f' -o "{output_path}"'
        )

        _gr.Info("Starting voice conversion...")

        import subprocess

        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0 and os.path.exists(output_path):
                _gr.Info("Conversion completed successfully!")
                return output_path
            else:
                err = result.stderr.strip() or result.stdout.strip() or "Unknown error"
                _gr.Warning(f"Conversion failed: {err}")
                return None
        except subprocess.TimeoutExpired:
            _gr.Warning("Conversion timed out (300s).")
            return None
        except Exception as e:
            _gr.Warning(f"Conversion error: {e}")
            return None

    def easy_download(url, name):
        """Download a model from URL."""
        import gradio as _gr

        if not url:
            _gr.Warning("Please provide a download URL.")
            return "No URL provided."

        _gr.Info("Starting download...")
        try:
            result = download_model(url=url, model=name or None)
            if result:
                return str(result)
            return "Download failed."
        except Exception as e:
            return f"Download error: {e}"

    # ── Build the UI ──
    with gr.Blocks(
        title="Advanced RVC - Easy GUI", theme=theme
    ) as app:
        gr.HTML(
            "<h1 style='text-align: center;'>Advanced RVC Inference - Easy Mode</h1>"
        )

        with gr.Tabs():
            # ════════════════════════════════════════════════════════════
            # Tab 1: Quick Convert
            # ════════════════════════════════════════════════════════════
            with gr.TabItem("Quick Convert"):
                gr.Markdown(
                    "Convert audio using a trained RVC voice model. "
                    "Select a model, provide input audio, adjust settings, and convert."
                )

                with gr.Row():
                    with gr.Column(scale=2):
                        model_pth = gr.Dropdown(
                            label="Model",
                            choices=model_names,
                            value=model_names[0] if len(model_names) >= 1 else "",
                            interactive=True,
                            allow_custom_value=True,
                        )
                        model_index = gr.Dropdown(
                            label="Index File",
                            choices=index_paths,
                            value=index_paths[0] if len(index_paths) >= 1 else "",
                            interactive=True,
                            allow_custom_value=True,
                        )

                    refresh_btn = gr.Button("Refresh Models", scale=1)

                with gr.Row():
                    pitch = gr.Slider(
                        minimum=-12,
                        maximum=12,
                        step=1,
                        label="Pitch Shift",
                        info="Shift pitch in semitones (0 = no change)",
                        value=0,
                        interactive=True,
                    )
                    index_rate = gr.Slider(
                        minimum=0,
                        maximum=1,
                        step=0.01,
                        label="Index Rate",
                        info="How much the index influences the output (0-1)",
                        value=0.5,
                        interactive=True,
                    )
                    f0_method = gr.Dropdown(
                        label="F0 Method",
                        choices=easy_f0_methods,
                        value="rmvpe",
                        interactive=True,
                    )
                    export_format = gr.Dropdown(
                        label="Export Format",
                        choices=["wav", "mp3", "flac", "ogg"],
                        value="wav",
                        interactive=True,
                    )

                with gr.Row():
                    with gr.Column():
                        input_audio_upload = gr.Audio(
                            label="Upload Audio",
                            type="filepath",
                            interactive=True,
                        )
                    with gr.Column():
                        output_audio = gr.Audio(
                            label="Output Audio",
                            type="filepath",
                            interactive=False,
                        )

                with gr.Row():
                    convert_btn = gr.Button(
                        "Convert", variant="primary"
                    )

                # Event bindings
                refresh_btn.click(
                    fn=change_models_choices,
                    inputs=[],
                    outputs=[model_pth, model_index],
                )
                model_pth.change(
                    fn=get_index,
                    inputs=[model_pth],
                    outputs=[model_index],
                )
                convert_btn.click(
                    fn=easy_convert,
                    inputs=[
                        input_audio_upload,
                        model_pth,
                        pitch,
                        index_rate,
                        f0_method,
                        model_index,
                        export_format,
                    ],
                    outputs=[output_audio],
                )

            # ════════════════════════════════════════════════════════════
            # Tab 2: One-Click Train
            # ════════════════════════════════════════════════════════════
            with gr.TabItem("One-Click Train"):
                gr.Markdown(
                    "Train a new RVC voice model in one click! "
                    "This pipeline automatically runs: **Preprocess → Extract Features → Train → Create Index**.\n\n"
                    "Just fill in the settings below and hit the big button. "
                    "Make sure your dataset folder contains audio files (wav, mp3, flac, etc.)."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        train_name = gr.Textbox(
                            label="Model Name",
                            info="Name for the new model (no spaces)",
                            value="",
                            placeholder="my_voice_model",
                            interactive=True,
                        )
                        train_version = gr.Radio(
                            label="RVC Version",
                            info="v2 is recommended for best quality",
                            choices=["v1", "v2"],
                            value="v2",
                            interactive=True,
                        )
                        train_sr = gr.Radio(
                            label="Sample Rate",
                            info="Higher = better quality but slower training",
                            choices=["32k", "40k", "48k"],
                            value="40k",
                            interactive=True,
                        )
                        train_dataset = gr.Textbox(
                            label="Dataset Folder",
                            info="Path to folder containing audio files",
                            value="advanced_rvc_inference/assets/dataset",
                            interactive=True,
                        )

                    with gr.Column(scale=1):
                        train_pitch = gr.Checkbox(
                            label="Pitch Guidance",
                            info="Use pitch extraction for better quality",
                            value=True,
                            interactive=True,
                        )
                        train_f0 = gr.Dropdown(
                            label="F0 Method",
                            info="Pitch extraction algorithm",
                            choices=easy_f0_methods,
                            value="rmvpe",
                            interactive=True,
                        )
                        train_gpu = gr.Textbox(
                            label="GPU",
                            info="GPU index (e.g. '0' for first GPU, '-' for CPU)",
                            value="0",
                            interactive=True,
                        )
                        train_author = gr.Textbox(
                            label="Model Author",
                            info="Optional author name to embed in the model",
                            value="",
                            placeholder="Your name",
                            interactive=True,
                        )

                with gr.Row():
                    with gr.Column(scale=1):
                        train_epochs = gr.Slider(
                            minimum=1,
                            maximum=10000,
                            step=1,
                            label="Total Epochs",
                            info="Total training iterations (300 is a good default)",
                            value=300,
                            interactive=True,
                        )
                        train_batch = gr.Slider(
                            minimum=1,
                            maximum=64,
                            step=1,
                            label="Batch Size",
                            info="Higher = faster but uses more VRAM",
                            value=8,
                            interactive=True,
                        )
                        train_save_every = gr.Slider(
                            minimum=1,
                            maximum=1000,
                            step=1,
                            label="Save Every N Epochs",
                            info="How often to save checkpoints",
                            value=50,
                            interactive=True,
                        )

                    with gr.Column(scale=1):
                        train_vocoder = gr.Dropdown(
                            label="Vocoder",
                            info="Audio synthesis vocoder",
                            choices=get_vocoder_choices(),
                            value=get_vocoder_choices()[0],
                            interactive=True,
                            allow_custom_value=True,
                        )
                        train_optimizer = gr.Dropdown(
                            label="Optimizer",
                            info="Training optimizer algorithm",
                            choices=get_optimizer_choices(),
                            value="AdamW",
                            interactive=True,
                        )

                with gr.Row():
                    train_btn = gr.Button(
                        "One-Click Train", variant="primary", size="lg"
                    )

                with gr.Row():
                    train_logs = gr.Textbox(
                        label="Training Progress",
                        value="",
                        interactive=False,
                        lines=15,
                        max_lines=50,
                        autoscroll=True,
                    )

                # Event binding
                train_btn.click(
                    fn=one_click_train,
                    inputs=[
                        train_name,
                        train_version,
                        train_sr,
                        train_dataset,
                        train_pitch,
                        train_f0,
                        train_epochs,
                        train_batch,
                        train_save_every,
                        train_gpu,
                        train_vocoder,
                        train_optimizer,
                        train_author,
                    ],
                    outputs=[train_logs],
                )

            # ════════════════════════════════════════════════════════════
            # Tab 3: Download Model
            # ════════════════════════════════════════════════════════════
            with gr.TabItem("Download"):
                gr.Markdown(
                    "Download RVC voice models from URLs. "
                    "Supports HuggingFace, Google Drive, MediaFire, PixelDrain, and Mega links."
                )

                with gr.Row():
                    with gr.Column(scale=3):
                        dl_url = gr.Textbox(
                            label="Download URL",
                            info="Direct link to model file (.pth, .onnx, .index, or .zip)",
                            value="",
                            placeholder="https://huggingface.co/...",
                            interactive=True,
                        )
                    with gr.Column(scale=1):
                        dl_name = gr.Textbox(
                            label="Model Name",
                            info="Optional custom name (uses original filename if empty)",
                            value="",
                            placeholder="my_model",
                            interactive=True,
                        )

                with gr.Row():
                    dl_btn = gr.Button("Download Model", variant="primary")

                with gr.Row():
                    dl_output = gr.Textbox(
                        label="Download Status",
                        value="",
                        interactive=False,
                        lines=3,
                    )

                # Event binding
                dl_btn.click(
                    fn=easy_download,
                    inputs=[dl_url, dl_name],
                    outputs=[dl_output],
                )

    return app

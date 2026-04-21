"""
Easy GUI Module for Advanced RVC Inference.

A simplified Gradio interface inspired by Rejekts EasyGUI (EVC).
Provides a streamlined experience with:
- Model Inference: Quick voice conversion with file upload, mic, or audio dropdown
- Download Model: Quick model download from URLs
- Train: 3-column training pipeline (Preprocess → Extract → Train+Index)

Design based on RejektsAI/EVC EasyGUI v2.9.
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
    styled after Rejekts EasyGUI with three tabs.

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
        paths_for_files,
    )
    from advanced_rvc_inference.core.training import (
        one_click_train,
        preprocess,
        extract,
        training,
        create_index,
    )
    from advanced_rvc_inference.core.downloads import download_model
    from advanced_rvc_inference.core.ui import (
        change_models_choices,
        get_index,
        gr_info,
        gr_warning,
    )
    from advanced_rvc_inference.library.optimizers import get_optimizer_choices
    from advanced_rvc_inference.library.generators import get_vocoder_choices

    # F0 methods for the easy UI (simplified subset)
    easy_f0_methods = ["rmvpe", "crepe", "crepe-full", "harvest", "fcpe"]

    # ── Build the UI ──
    with gr.Blocks(title="EasyGUI", theme=theme) as app:
        gr.HTML('<h1 style="text-align: center;"> EasyGUI </h1>')

        with gr.Tabs():
            # ════════════════════════════════════════════════════════════
            # Tab 1: Model Inference
            # ════════════════════════════════════════════════════════════
            with gr.TabItem("Model Inference"):
                # Row 1: Top controls bar
                with gr.Row():
                    model_pth = gr.Dropdown(
                        label="Model",
                        choices=model_names,
                        value=model_names[0] if len(model_names) >= 1 else "",
                        interactive=True,
                        allow_custom_value=True,
                    )
                    refresh_btn = gr.Button(
                        "Refresh Models", variant="primary"
                    )
                    pitch_shift = gr.Number(
                        label="Pitch (semitones, +12 up, -12 down)",
                        value=0,
                        precision=0,
                        interactive=True,
                    )
                    convert_btn = gr.Button(
                        "Convert", variant="primary"
                    )

                # Row 2: Two-column layout (input left, output+settings right)
                with gr.Row():
                    # Left column: audio input
                    with gr.Column(scale=1):
                        gr.Markdown("**1. Choose your audio**")
                        input_audio_upload = gr.File(
                            label="Drop your audio here & hit Convert.",
                            file_types=["audio"],
                            interactive=True,
                        )
                        input_audio_mic = gr.Audio(
                            label="OR Record audio.",
                            sources="microphone",
                            type="filepath",
                            interactive=True,
                        )
                        audio_dropdown = gr.Dropdown(
                            label="Or select from uploaded audios",
                            choices=[os.path.basename(p) for p in paths_for_files],
                            interactive=True,
                            allow_custom_value=True,
                        )

                    # Right column: settings, index, output
                    with gr.Column(scale=1):
                        gr.Markdown("**2. Select your index file**")
                        with gr.Accordion("Index Settings", open=False):
                            model_index = gr.Dropdown(
                                label="Index File",
                                choices=index_paths,
                                value=index_paths[0] if len(index_paths) >= 1 else "",
                                interactive=True,
                                allow_custom_value=True,
                            )
                            index_rate = gr.Slider(
                                minimum=0,
                                maximum=1,
                                step=0.01,
                                label="Search Feature Ratio",
                                value=0.66,
                                interactive=True,
                            )

                        gr.Markdown("**3. Output**")
                        output_audio = gr.Audio(
                            label="Output Audio",
                            type="filepath",
                            interactive=False,
                        )

                        with gr.Accordion("Advanced Settings", open=False):
                            f0_method = gr.Dropdown(
                                label="F0 Method",
                                choices=easy_f0_methods,
                                value="rmvpe",
                                interactive=True,
                            )
                            filter_radius = gr.Slider(
                                minimum=0,
                                maximum=7,
                                step=1,
                                label="Filter Radius",
                                value=3,
                                interactive=True,
                            )
                            rms_mix_rate = gr.Slider(
                                minimum=0,
                                maximum=1,
                                step=0.01,
                                label="Volume Envelope Mix",
                                value=0.21,
                                interactive=True,
                            )
                            protect = gr.Slider(
                                minimum=0,
                                maximum=0.5,
                                step=0.01,
                                label="Protect Voiceless Consonants",
                                value=0.33,
                                interactive=True,
                            )
                            export_format = gr.Dropdown(
                                label="Export Format",
                                choices=["wav", "mp3", "flac", "ogg"],
                                value="wav",
                                interactive=True,
                            )

                # Status output
                convert_status = gr.Textbox(
                    label="Status",
                    value="",
                    interactive=False,
                    lines=2,
                )

                # ── Convert logic ──
                def easy_convert(
                    input_file, input_mic, audio_name,
                    model, pitch, index_rate_val, f0_method_val,
                    index_file, filter_radius_val, rms_mix_val,
                    protect_val, export_format_val,
                ):
                    import gradio as _gr

                    # Determine input audio source
                    input_audio = None
                    if input_file:
                        input_audio = input_file if isinstance(input_file, str) else input_file.name
                    elif input_mic:
                        input_audio = input_mic
                    elif audio_name:
                        # Find the selected audio from paths
                        for p in paths_for_files:
                            if os.path.basename(p) == audio_name:
                                input_audio = p
                                break

                    if not input_audio:
                        _gr.Warning("Please provide an input audio file.")
                        return None, "No input audio provided."

                    if not model:
                        _gr.Warning("Please select a model.")
                        return None, "No model selected."

                    model_path = os.path.join(configs["weights_path"], model)
                    if not os.path.exists(model_path):
                        _gr.Warning(f"Model file not found: {model}")
                        return None, f"Model not found: {model}"

                    # Build output path
                    base_name = os.path.splitext(os.path.basename(input_audio))[0]
                    output_path = os.path.join(
                        configs["audios_path"], "easy_convert",
                        f"{base_name}_output.{export_format_val}"
                    )
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)

                    index_path_arg = ""
                    if index_file and os.path.exists(index_file):
                        index_path_arg = f' --index "{index_file}"'

                    cmd = (
                        f'{python} {configs["convert_path"]}'
                        f' -i "{input_audio}"'
                        f' -m "{model_path}"'
                        f' -p {int(pitch)}'
                        f' --f0_method {f0_method_val}'
                        f' --index_rate {index_rate_val}'
                        f' --filter_radius {filter_radius_val}'
                        f' --rms_mix_rate {rms_mix_val}'
                        f' --protect {protect_val}'
                        f'{index_path_arg}'
                        f' -f {export_format_val}'
                        f' -o "{output_path}"'
                    )

                    _gr.Info("Starting voice conversion...")

                    import subprocess

                    try:
                        result = subprocess.run(
                            cmd, shell=True, capture_output=True, text=True, timeout=300
                        )
                        if result.returncode == 0 and os.path.exists(output_path):
                            return output_path, "Conversion completed successfully!"
                        else:
                            err = result.stderr.strip() or result.stdout.strip() or "Unknown error"
                            return None, f"Conversion failed: {err}"
                    except subprocess.TimeoutExpired:
                        return None, "Conversion timed out (300s)."
                    except Exception as e:
                        return None, f"Conversion error: {e}"

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
                        input_audio_mic,
                        audio_dropdown,
                        model_pth,
                        pitch_shift,
                        index_rate,
                        f0_method,
                        model_index,
                        filter_radius,
                        rms_mix_rate,
                        protect,
                        export_format,
                    ],
                    outputs=[output_audio, convert_status],
                )

                # ── Batch conversion (collapsible) ──
                with gr.Accordion("Batch Conversion", open=False):
                    with gr.Row():
                        batch_input = gr.Textbox(
                            label="Input Folder",
                            value="",
                            interactive=True,
                        )
                        batch_pitch = gr.Number(
                            label="Pitch", value=0, precision=0, interactive=True,
                        )
                        batch_f0 = gr.Dropdown(
                            label="F0 Method",
                            choices=easy_f0_methods,
                            value="rmvpe",
                            interactive=True,
                        )
                    with gr.Row():
                        batch_index = gr.Dropdown(
                            label="Index File",
                            choices=index_paths,
                            value=index_paths[0] if len(index_paths) >= 1 else "",
                            interactive=True,
                            allow_custom_value=True,
                        )
                        batch_index_rate = gr.Slider(
                            label="Search Feature Ratio",
                            minimum=0, maximum=1, step=0.01,
                            value=0.66, interactive=True,
                        )
                        batch_filter = gr.Slider(
                            label="Filter Radius",
                            minimum=0, maximum=7, step=1,
                            value=3, interactive=True,
                        )
                        batch_format = gr.Dropdown(
                            label="Export Format",
                            choices=["wav", "mp3", "flac"],
                            value="wav",
                            interactive=True,
                        )
                    with gr.Row():
                        batch_convert_btn = gr.Button(
                            "Convert", variant="primary"
                        )
                    batch_status = gr.Textbox(
                        label="Batch Status",
                        value="",
                        interactive=False,
                        lines=3,
                    )

                    def easy_batch_convert(
                        input_folder, batch_model, pitch, f0_method_val,
                        index_file, index_rate_val, filter_radius_val, export_format_val,
                    ):
                        import gradio as _gr
                        import subprocess
                        import glob

                        if not input_folder or not os.path.isdir(input_folder):
                            return "Please provide a valid input folder path."
                        if not batch_model:
                            return "Please select a model."

                        model_path = os.path.join(configs["weights_path"], batch_model)
                        if not os.path.exists(model_path):
                            return f"Model not found: {batch_model}"

                        audio_exts = (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a")
                        audio_files = glob.glob(os.path.join(input_folder, "*"))
                        audio_files = [f for f in audio_files
                                       if f.lower().endswith(audio_exts)]

                        if not audio_files:
                            return f"No audio files found in {input_folder}"

                        out_dir = os.path.join(configs["audios_path"], "easy_batch")
                        os.makedirs(out_dir, exist_ok=True)

                        index_path_arg = ""
                        if index_file and os.path.exists(index_file):
                            index_path_arg = f' --index "{index_file}"'

                        results = []
                        for i, audio_file in enumerate(audio_files, 1):
                            base_name = os.path.splitext(os.path.basename(audio_file))[0]
                            output_path = os.path.join(out_dir, f"{base_name}.{export_format_val}")

                            cmd = (
                                f'{python} {configs["convert_path"]}'
                                f' -i "{audio_file}"'
                                f' -m "{model_path}"'
                                f' -p {int(pitch)}'
                                f' --f0_method {f0_method_val}'
                                f' --index_rate {index_rate_val}'
                                f' --filter_radius {filter_radius_val}'
                                f'{index_path_arg}'
                                f' -f {export_format_val}'
                                f' -o "{output_path}"'
                            )

                            result = subprocess.run(
                                cmd, shell=True, capture_output=True, text=True, timeout=300
                            )

                            status = "OK" if result.returncode == 0 else "FAIL"
                            results.append(f"[{i}/{len(audio_files)}] {status}: {base_name}")

                        return "\n".join(results)

                    batch_convert_btn.click(
                        fn=lambda *args: easy_batch_convert(*args),
                        inputs=[
                            batch_input, model_pth, batch_pitch, batch_f0,
                            batch_index, batch_index_rate, batch_filter, batch_format,
                        ],
                        outputs=[batch_status],
                    )

            # ════════════════════════════════════════════════════════════
            # Tab 2: Download Model
            # ════════════════════════════════════════════════════════════
            with gr.TabItem("Download Model"):
                gr.Markdown(
                    "Download RVC voice models from URLs. "
                    "Supports HuggingFace, Google Drive, MediaFire, PixelDrain, and Mega links."
                )

                dl_url = gr.Textbox(
                    label="Enter the URL to the Model:",
                    value="",
                    placeholder="https://huggingface.co/...",
                    interactive=True,
                )
                dl_name = gr.Textbox(
                    label="Name your model:",
                    value="",
                    placeholder="my_model",
                    interactive=True,
                )
                dl_btn = gr.Button("Download", variant="primary")
                dl_output = gr.Textbox(
                    label="Status",
                    value="",
                    interactive=False,
                    lines=3,
                )

                # ── Download logic ──
                def easy_download(url, name):
                    import gradio as _gr

                    if not url:
                        _gr.Warning("Please provide a download URL.")
                        return "No URL provided."

                    _gr.Info("Starting download...")
                    try:
                        result = download_model(url=url, model=name or None)
                        if result:
                            return f"Downloaded successfully: {result}"
                        return "Download failed."
                    except Exception as e:
                        return f"Download error: {e}"

                dl_btn.click(
                    fn=easy_download,
                    inputs=[dl_url, dl_name],
                    outputs=[dl_output],
                )

                gr.Markdown(
                    "\n---\n"
                    "Credits: Based on [Rejekts EasyGUI](https://github.com/RejektsAI/EVC) | "
                    "[Advanced RVC Inference](https://github.com/ArkanDash/Advanced-RVC-Inference)"
                )

            # ════════════════════════════════════════════════════════════
            # Tab 3: Train (3-column layout like Rejekts EasyGUI)
            # ════════════════════════════════════════════════════════════
            with gr.TabItem("Train"):
                gr.Markdown(
                    "Train a new RVC voice model. Use the left column to preprocess data, "
                    "the middle column to extract features, and the right column to train and create an index."
                )

                with gr.Row():
                    # ── Left Column: Data Preparation ──
                    with gr.Column():
                        gr.Markdown("**Data Preparation**")
                        train_name = gr.Textbox(
                            label="Experiment Name",
                            value="My-Voice",
                            interactive=True,
                        )
                        train_sr = gr.Radio(
                            label="Sample Rate",
                            choices=["32k", "40k", "48k"],
                            value="40k",
                            visible=False,
                        )
                        train_version = gr.Radio(
                            label="Version",
                            choices=["v1", "v2"],
                            value="v2",
                            visible=False,
                        )
                        train_pitch = gr.Radio(
                            label="Pitch Guidance",
                            choices=[True, False],
                            value=True,
                            visible=False,
                        )
                        train_dataset = gr.Textbox(
                            label="Training Folder",
                            value="advanced_rvc_inference/assets/dataset",
                            interactive=True,
                        )
                        train_audio_upload = gr.Files(
                            label="Or batch upload audio files",
                            file_types=["audio"],
                            interactive=True,
                        )
                        cpu_cores = gr.Slider(
                            minimum=1,
                            maximum=16,
                            step=1,
                            label="CPU Processes",
                            value=max(1, (os.cpu_count() or 4) * 2 // 3),
                            interactive=True,
                        )
                        preprocess_btn = gr.Button(
                            "Process Data", variant="primary"
                        )
                        preprocess_status = gr.Textbox(
                            label="Status",
                            value="",
                            interactive=False,
                            lines=6,
                            max_lines=20,
                            autoscroll=True,
                        )

                    # ── Middle Column: Feature Extraction ──
                    with gr.Column():
                        gr.Markdown("**Feature Extraction**")
                        train_f0 = gr.Radio(
                            label="F0 Method",
                            choices=["rmvpe", "crepe", "crepe-full", "harvest", "fcpe"],
                            value="rmvpe",
                            interactive=True,
                        )
                        train_gpu = gr.Textbox(
                            label="GPU IDs (comma separated)",
                            value="0",
                            interactive=True,
                        )
                        gpu_info = gr.Textbox(
                            label="GPU Info",
                            value="",
                            interactive=False,
                            lines=2,
                        )
                        extract_btn = gr.Button(
                            "Extract Features", variant="primary"
                        )
                        extract_status = gr.Textbox(
                            label="Status",
                            value="",
                            interactive=False,
                            lines=6,
                            max_lines=15,
                            autoscroll=True,
                        )

                    # ── Right Column: Train & Index ──
                    with gr.Column():
                        gr.Markdown("**Train & Index**")
                        train_epochs = gr.Slider(
                            minimum=2,
                            maximum=10000,
                            step=1,
                            label="Total Epochs",
                            value=300,
                            interactive=True,
                        )
                        train_batch = gr.Slider(
                            minimum=1,
                            maximum=64,
                            step=1,
                            label="Batch Size",
                            value=8,
                            interactive=True,
                        )
                        train_save_every = gr.Slider(
                            minimum=1,
                            maximum=500,
                            step=1,
                            label="Save Every N Epochs",
                            value=25,
                            interactive=True,
                        )
                        train_vocoder = gr.Dropdown(
                            label="Vocoder",
                            choices=get_vocoder_choices(),
                            value=get_vocoder_choices()[0],
                            interactive=True,
                            allow_custom_value=True,
                        )
                        train_optimizer = gr.Dropdown(
                            label="Optimizer",
                            choices=get_optimizer_choices(),
                            value="AdamW",
                            interactive=True,
                        )
                        train_author = gr.Textbox(
                            label="Model Author",
                            value="",
                            interactive=True,
                        )

                        with gr.Row():
                            train_btn = gr.Button(
                                "Train Model", variant="primary"
                            )
                            index_btn = gr.Button(
                                "Train Index", variant="primary"
                            )

                        train_status = gr.Textbox(
                            label="Status",
                            value="",
                            interactive=False,
                            lines=8,
                            max_lines=30,
                            autoscroll=True,
                        )

                # ── GPU info auto-detect ──
                def get_gpu_info():
                    import torch
                    if torch.cuda.is_available():
                        name = torch.cuda.get_device_name(0)
                        mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        return f"{name} ({mem:.1f} GB)"
                    return "No GPU detected"

                gpu_info.value = get_gpu_info()

                # ── Auto-fill dataset from uploaded files ──
                def handle_audio_upload(files):
                    if not files:
                        return "advanced_rvc_inference/assets/dataset"
                    # If user uploaded files, note the paths
                    file_list = [f.name for f in files] if hasattr(files[0], 'name') else files
                    return f"{len(file_list)} files selected"

                train_audio_upload.change(
                    fn=handle_audio_upload,
                    inputs=[train_audio_upload],
                    outputs=[train_dataset],
                )

                # ── Step-by-step event bindings ──
                preprocess_btn.click(
                    fn=preprocess,
                    inputs=[
                        train_name, train_sr, cpu_cores,
                        "Automatic", False, train_dataset,
                        False, 0.7,
                        3.0, 0.3, "none",
                    ],
                    outputs=[preprocess_status],
                )

                extract_btn.click(
                    fn=extract,
                    inputs=[
                        train_name, train_version, train_f0,
                        train_pitch, 128, cpu_cores, train_gpu,
                        train_sr, "hubert_base", "hubert_base",
                        False, "fairseq", False, 1.0,
                        "hybrid[pm+crepe-tiny]", False, 0.5,
                    ],
                    outputs=[extract_status],
                )

                train_btn.click(
                    fn=training,
                    inputs=[
                        train_name, train_version, train_save_every,
                        True, True, train_epochs, train_sr,
                        train_batch, train_gpu, train_pitch,
                        False, False, "", "",
                        False, 50, False, True,
                        train_author, train_vocoder,
                        False, False, False, train_optimizer,
                        False,
                    ],
                    outputs=[train_status],
                )

                index_btn.click(
                    fn=create_index,
                    inputs=[train_name, train_version, "Auto"],
                    outputs=[train_status],
                )

                # ── One-click train (collapsible) ──
                with gr.Accordion("One-Click Train (run all steps automatically)", open=False):
                    gr.Markdown(
                        "Automatically runs: **Process Data → Extract Features → Train Model → Train Index**. "
                        "Configure all settings above first, then click the button."
                    )
                    one_click_btn = gr.Button(
                        "One-Click Train", variant="primary"
                    )
                    one_click_status = gr.Textbox(
                        label="Training Progress",
                        value="",
                        interactive=False,
                        lines=12,
                        max_lines=50,
                        autoscroll=True,
                    )

                    one_click_btn.click(
                        fn=one_click_train,
                        inputs=[
                            train_name, train_version, train_sr,
                            train_dataset, train_pitch, train_f0,
                            train_epochs, train_batch, train_save_every,
                            train_gpu, train_vocoder, train_optimizer,
                            train_author,
                        ],
                        outputs=[one_click_status],
                    )

    return app

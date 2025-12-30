import os
import signal

process_pids = []

import shutil
import sys
import json
from multiprocessing import cpu_count

import gradio as gr

from core import (
    run_extract_script,
    run_index_script,
    run_preprocess_script,
    run_prerequisites_script,
    run_train_script,
    stop_train_script,
)
from rvc.configs.config import get_gpu_info, get_number_of_gpus, max_vram_gpu, microarchitecture_capability_checker, check_if_fp16
from rvc.lib.utils import format_title

now_dir = os.getcwd()
sys.path.append(now_dir)

supported_audio_ext = { "wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3", }

saved_components = [] # List of components that should have their states saved ~ For presets


# Custom Pretraineds
pretraineds_custom_path = os.path.join(now_dir, "rvc", "models", "pretraineds", "custom")
pretraineds_custom_path_relative = os.path.relpath(pretraineds_custom_path, now_dir)
# Custom embedders
custom_embedder_root = os.path.join(now_dir, "rvc", "models", "embedders", "embedders_custom")
custom_embedder_root_relative = os.path.relpath(custom_embedder_root, now_dir)
# Training presets
presets_path = os.path.join(now_dir, 'assets', 'training_presets')
presets_path_relative = os.path.relpath(presets_path, now_dir)

# Ensure dirs existence
os.makedirs(pretraineds_custom_path_relative, exist_ok=True)
os.makedirs(custom_embedder_root, exist_ok=True)
os.makedirs(presets_path, exist_ok=True)


def get_pretrained_list(suffix):
    return [
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(pretraineds_custom_path_relative)
        for filename in filenames
        if filename.endswith(".pth") and suffix in filename
    ]

pretraineds_list_d = get_pretrained_list("D")
pretraineds_list_g = get_pretrained_list("G")

def refresh_custom_pretraineds():
    return (
        {"choices": sorted(get_pretrained_list("G")), "__type__": "update"},
        {"choices": sorted(get_pretrained_list("D")), "__type__": "update"},
    )

# Dataset Creator
datasets_path = os.path.join(now_dir, "assets", "datasets")

if not os.path.exists(datasets_path):
    os.makedirs(datasets_path)

datasets_path_relative = os.path.relpath(datasets_path, now_dir)

def get_datasets_list():
    return [
        dirpath
        for dirpath, _, filenames in os.walk(datasets_path_relative)
        if any(filename.endswith(tuple(supported_audio_ext)) for filename in filenames)
    ]

def refresh_datasets():
    return {"choices": sorted(get_datasets_list()), "__type__": "update"}

# Model Names
models_path = os.path.join(now_dir, "logs")

def get_models_list():
    return [
        os.path.basename(dirpath)
        for dirpath in os.listdir(models_path)
        if os.path.isdir(os.path.join(models_path, dirpath))
        and all(excluded not in dirpath for excluded in ["zips", "mute", "reference"])
    ]

def refresh_models():
    return {"choices": sorted(get_models_list()), "__type__": "update"}

# Refresh Models and Datasets
def refresh_models_and_datasets():
    return (
        {"choices": sorted(get_models_list()), "__type__": "update"},
        {"choices": sorted(get_datasets_list()), "__type__": "update"},
    )

# Refresh Custom Embedders
def get_embedder_custom_list():
    return [
        os.path.join(dirpath, dirname)
        for dirpath, dirnames, _ in os.walk(custom_embedder_root_relative)
        for dirname in dirnames
    ]

def refresh_custom_embedder_list():
    return {"choices": sorted(get_embedder_custom_list()), "__type__": "update"}

# Retrieve presets
def get_presets_list():
    return [os.path.splitext(s)[0] for s in os.listdir(presets_path) if s.endswith('.json')]

# Drop Model
def save_drop_model(dropbox):
    if ".pth" not in dropbox:
        gr.Info("The file you dropped is not a valid pretrained file. Please try again.")
    else:
        file_name = os.path.basename(dropbox)
        pretrained_path = os.path.join(pretraineds_custom_path_relative, file_name)
        if os.path.exists(pretrained_path):
            os.remove(pretrained_path)
        shutil.copy(dropbox, pretrained_path)
        gr.Info("Click the refresh button to see the pretrained file in the dropdown menu.")
    return None

# Drop Dataset
def save_drop_dataset_audio(dropbox, dataset_name):
    if not dataset_name:
        gr.Info("Please enter a valid dataset name. Please try again.")
        return None, None
    else:
        file_extension = os.path.splitext(dropbox)[1][1:].lower()
        if file_extension not in supported_audio_ext:
            gr.Info("The file you dropped is not a valid audio file. Please try again.")
        else:
            dataset_name = format_title(dataset_name)
            audio_file = format_title(os.path.basename(dropbox))
            dataset_path = os.path.join(now_dir, "assets", "datasets", dataset_name)
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)
            destination_path = os.path.join(dataset_path, audio_file)
            if os.path.exists(destination_path):
                os.remove(destination_path)
            shutil.copy(dropbox, destination_path)
            gr.Info(
                "The audio file has been successfully added to the dataset. Please click the preprocess button."
            )
            dataset_path = os.path.dirname(destination_path)
            relative_dataset_path = os.path.relpath(dataset_path, now_dir)

            return None, relative_dataset_path

# Drop Custom Embedder
def create_folder_and_move_files(folder_name, bin_file, config_file):
    if not folder_name:
        return "Folder name must not be empty."

    folder_name = os.path.basename(folder_name)
    target_folder = os.path.join(custom_embedder_root, folder_name)
    normalized_target_folder = os.path.abspath(target_folder)
    normalized_custom_embedder_root = os.path.abspath(custom_embedder_root)

    if not normalized_target_folder.startswith(normalized_custom_embedder_root):
        return "Invalid folder name. Folder must be within the custom embedder root directory."

    os.makedirs(target_folder, exist_ok=True)

    if bin_file:
        shutil.copy(bin_file, os.path.join(target_folder, os.path.basename(bin_file)))

    if config_file:
        shutil.copy(config_file, os.path.join(target_folder, os.path.basename(config_file)))

    return f"Files moved to folder {target_folder}"

def refresh_embedders_folders():
    custom_embedders = [
        os.path.join(dirpath, dirname)
        for dirpath, dirnames, _ in os.walk(custom_embedder_root_relative)
        for dirname in dirnames
    ]
    return custom_embedders

# Export
def get_pth_list():
    return [
        os.path.relpath(os.path.join(dirpath, filename), now_dir)
        for dirpath, _, filenames in os.walk(models_path)
        for filename in filenames
        if filename.endswith(".pth")
    ]

def get_index_list():
    return [
        os.path.relpath(os.path.join(dirpath, filename), now_dir)
        for dirpath, _, filenames in os.walk(models_path)
        for filename in filenames
        if filename.endswith(".index") and "trained" not in filename
    ]

def refresh_pth_and_index_list():
    return (
        {"choices": sorted(get_pth_list()), "__type__": "update"},
        {"choices": sorted(get_index_list()), "__type__": "update"},
    )

# Export Pth and Index Files
def export_pth(pth_path):
    allowed_paths = get_pth_list()
    normalized_allowed_paths = [os.path.abspath(os.path.join(now_dir, p)) for p in allowed_paths]
    normalized_pth_path = os.path.abspath(os.path.join(now_dir, pth_path))

    if normalized_pth_path in normalized_allowed_paths:
        return pth_path
    else:
        print(f"Attempted to export invalid pth path: {pth_path}")
        return None

def export_index(index_path):
    allowed_paths = get_index_list()
    normalized_allowed_paths = [os.path.abspath(os.path.join(now_dir, p)) for p in allowed_paths]
    normalized_index_path = os.path.abspath(os.path.join(now_dir, index_path))

    if normalized_index_path in normalized_allowed_paths:
        return index_path
    else:
        print(f"Attempted to export invalid index path: {index_path}")
        return None

# Upload to Google Drive
def upload_to_google_drive(pth_path, index_path):
    def upload_file(file_path):
        if file_path:
            try:
                gr.Info(f"Uploading {pth_path} to Google Drive...")
                google_drive_folder = "/content/drive/MyDrive/Codename-RVC-Fork-Exported"
                if not os.path.exists(google_drive_folder):
                    os.makedirs(google_drive_folder)
                google_drive_file_path = os.path.join(
                    google_drive_folder, os.path.basename(file_path)
                )
                if os.path.exists(google_drive_file_path):
                    os.remove(google_drive_file_path)
                shutil.copy2(file_path, google_drive_file_path)
                gr.Info("File uploaded successfully.")
            except Exception as error:
                print(f"An error occurred uploading to Google Drive: {error}")
                gr.Info("Error uploading to Google Drive")

    upload_file(pth_path)
    upload_file(index_path)

# Enable checkpointing for gpus with memory 
def auto_enable_checkpointing():
    try:
        return max_vram_gpu(0) < 6
    except:
        return False

# Init state for certain options.
initial_sample_rate_choices = ["32000", "40000", "48000"]
initial_sample_rate = "48000"

# Microarch. dependent features, options, functionalities etc.. Might expand in future.
fp16_check = None

if microarchitecture_capability_checker():
    # Ampere-Microarchitecture and higher viable:
    initial_optimizer = "AdamW BF16"
    initial_optimizer_choices = ["AdamW BF16", "AdamW", "RAdam", "AdamSPD", "Ranger21", "DiffGrad", "Prodigy"]
    architecture_choices = ["RVC", "Fork/Applio", "Fork"]
    fp16_check = True
else:
    # Below Ampere-Microarchitecture viable:
    initial_optimizer = "AdamW"
    initial_optimizer_choices = ["AdamW", "RAdam", "AdamSPD", "Ranger21", "DiffGrad", "Prodigy"]
    architecture_choices = ["RVC", "Fork/Applio"]
    fp16_check = True

# FP16 checker
if fp16_check:
    if check_if_fp16():
        initial_optimizer = "AdamW"
        initial_optimizer_choices = ["AdamW", "RAdam", "AdamSPD", "Ranger21", "DiffGrad", "Prodigy"]


# Train Tab
def train_tab():
    # Training presets section
    with gr.Accordion("Training Presets", open=False):
        with gr.Row():
            refresh_presets_button = gr.Button("Refresh Presets")
        with gr.Row():
            with gr.Column():
                preset_dropdown = gr.Dropdown(
                    choices=get_presets_list(),
                    label="Preset Name",
                    allow_custom_value=True,
                    interactive=True
                )
            with gr.Column():
                save_preset_button = gr.Button("Save to preset")
                load_preset_button = gr.Button("Load from preset")

    # Model settings section
    with gr.Accordion("Model Settings"):
        with gr.Row():
            with gr.Column():
                model_name = gr.Dropdown(
                    label="Model Name",
                    info="Name of the new model.",
                    choices=get_models_list(),
                    value="example-model-name",
                    interactive=True,
                    allow_custom_value=True,
                    key='model_name'
                )
                architecture = gr.Radio(
                    label="Architecture",
                    info="Choose the model architecture:\n- **RVC (V2):ㅤDefault/OG-Architecture - Compatible with all clients.**\n- **Fork/Applio:ㅤOG-Arch's discs + RefineGAN/MRF-HiFi-GAN** - Only for this Fork or Applio **( Experimental. )** \n- **Fork:ㅤCodename-selected custom archs** - Only for this Fork **( Exclusive. )**",
                    choices=architecture_choices,
                    value="RVC",
                    interactive=True,
                    visible=True,
                    key='architecture'
                )
                vits2_mode = gr.Checkbox(
                    label="Enables VITS2 mode",
                    value=False,
                    interactive=True,
                    visible=True,
                )
                vocoder_arch = gr.State("hifi_mrf_refine")
                optimizer = gr.Radio(
                    label="Optimizer",
                    info="Choose an optimizer used in training: \n ( If unsure, just leave it as it is or try these in this order: AdamW -> AdamSPD -> RAdam. ) \n- **AdamW BF16:** Good and reliable. ( BF16 ver. )  \n- **AdamW:** Normal AdamW. ( **Use the BF16 version unless you train in FP32-only or FP16** ) \n- **RAdam:** Rectified Adam. ( **Can help** with early instability - **Most likely slower convergence** ) \n- **AdamSPD:** Adam with SPD. ( SPD: New weight-decay technique **tailored for fine-tuning.** ) \n- **Ranger21:** AdamW + LookAhead and few more extras. ( **Most likely unstable** ) \n- **DiffGrad:** An optimizer with CNN in mind. ( **Probs** a good AdamW alternative - **For finetuning** ) \n- **Prodigy:** A self-tuning optimizer. Lr will adapt automatically ( **Don't touch the lr** )",
                    choices=initial_optimizer_choices,
                    value=initial_optimizer,
                    interactive=True,
                    visible=True,
                    key='optimizer'
                )
            with gr.Column():
                sampling_rate = gr.Radio(
                    label="Sampling Rate",
                    info="The sampling rate of the model you wanna train. \n**( If possible, should match your dataset. Small deviations are allowed. )**",
                    choices=initial_sample_rate_choices,
                    value=initial_sample_rate,
                    interactive=True,
                    key='sampling_rate'
                )
                vocoder = gr.Radio(
                    label="Vocoder",
                    info="**Vocoder for audio synthesis:** \n \n **HiFi-GAN:** \n- **Arch overview:ㅤHiFi-GAN + Hn-NSF for f0 handling. ( RVC's og vocoder )** \n- **COMPATIBILITY:ㅤAll clients incl. Mainline RVC / W-okada etc.** \n\n**MRF HiFi-GAN:** \n - **Arch overview:ㅤHiFi-Gan + Hn-NSF + MRF block** \n- **COMPATIBILITY:ㅤThis Fork or Applio ( afaik, no rt-vc clients support it. )** \n\n**RefineGAN:** \n - **Arch overview:ㅤHiFi-Gan + Hn-NSF + ParallelResBlock + AdaIN** \n- **COMPATIBILITY:ㅤThis Fork or Applio ( As for rt-vc, vonovox beta supports it. )** \n\n**RingFormer:** \n- **Arch overview:ㅤA hybrid Conformer-Based Vocoder + Snake-Beta act. + RingAttention + Hn-NSF** \n- **COMPATIBILITY:ㅤThis Fork ( As for rt-vc, 'Vonovox' supports it. )**  \n\n**PCPH-GAN:** \n- **Arch overview: HiFi-Gan + PCPH prior + SnakeBeta & Silu ** \n- **COMPATIBILITY:ㅤThis Fork ( No rt-vc clients support it atm. )** \n\n **NOTES:** \n **( RingFormer Requires min. RTX 30xx [ At least Ampere microarchitecture ] )** \n **( Each Vocoder and it's supported sample rates require appropriate pretrained models. )**",
                    choices=["HiFi-GAN"],
                    value="HiFi-GAN",
                    interactive=False,
                    visible=True,
                    key='vocoder'
                )
        with gr.Accordion(
            "CPU / GPU settings for ' f0 ' and ' features ' extraction.",
            open=False,
        ):
            with gr.Row():
                with gr.Column():
                    cpu_threads = gr.Slider(
                        1,
                        min(cpu_count(), 192),  # max 192 parallel processes
                        min(cpu_count(), 192),
                        step=1,
                        label="CPU Threads",
                        info="The number of CPU threads used in the extraction process. \n By default, it is set to the maximum number of threads available on your CPU. \n ( Which is recommended in most cases. )",
                        interactive=True,
                        key='cpu_threads'
                    )
                with gr.Column():
                    extract_gpu = gr.Textbox(
                        label="GPU ID",
                        info="Specify the number of GPUs you wish to utilize for extracting by entering their ID separated by hyphens (-). \n i.e.: 0-1-2  ( for 3 gpus)",
                        placeholder="0 to ∞ separated by -",
                        value=str(get_number_of_gpus()),
                        interactive=True,
                        key='extract_gpu'
                    )
                    gr.Textbox(
                        label="GPU Information",
                        info="The GPU information will be displayed here.",
                        value=get_gpu_info(),
                        interactive=False,
                    )

    # Preprocess section
    with gr.Accordion("Preprocess"):
        dataset_path = gr.Dropdown(
            label="Dataset Path",
            info="Path to the dataset folder. ( Or you can use the dropbox to browse the folders. )",
            choices=get_datasets_list(),
            allow_custom_value=True,
            interactive=True,
            key='dataset_path'
        )
        dataset_creator = gr.Checkbox(
            label="Dataset Creator",
            value=False,
            interactive=True,
            visible=True,
        )
        with gr.Column(visible=False) as dataset_creator_settings:
            with gr.Accordion("Dataset Creator"):
                dataset_name = gr.Textbox(
                    label="Dataset Name",
                    info="Name of the new dataset.",
                    placeholder="Enter dataset name",
                    interactive=True,
                )
                upload_audio_dataset = gr.File(
                    label="Upload Audio Dataset",
                    type="filepath",
                    interactive=True,
                )
        refresh = gr.Button("Refresh")

        with gr.Accordion("Advanced Settings for the preprocessing step", open=True):
            gr.Markdown(
            """
             
             
            **The provided default settings are optimal for anyone as long:**
             
            + Your dataset is a " 1 file " type ( Say, fused all smaller samples / chunks into 1 .wav file )
            + Your dataset doesn't have peaks going crazy haywire ( tl;dr - You don't operate on source/dynamics-inconsistent samples )
            + You performed silence-truncation the right way
             
             
            ( If your set has major peak / consistency issues, I recommend learning about " Peak taming compression ". )
            <br>
            (( Generally.. you shouldn't tweak these unless you know what and why you're doing it. ))
            <br>
            ((( The only exception would be for " DC / high-pass filtering " and " Noise Reduction " ~ Read their description. )))
            """
            )
            with gr.Row():
                loading_resampling = gr.Radio(
                    label="Resampling & Loading Handler",
                    info="- **librosa** - Uses SoX resampler \n ( SoXr set to VHQ by default. ).\n- **ffmpeg** -  Uses SW resampler \n ( Windowed Sinc filter with Blackman-Nuttall window ) \n\n **Both are viable choices!** \n **( But I'd actually go with Sinc / FFmpeg. )**",
                    choices=["librosa", "ffmpeg"],
                    value="ffmpeg",
                    interactive=True,
                    scale=1.45,
                    key='loading_resampling'
                )
                normalization_mode = gr.Radio(
                    label="Loudness Normalization",
                    info="- **none:** Disabled \n ( Select this if the files are already normalized. ) \n- **post_lufs:** PyLoud Post-Normalization \n ( Loudness norm. of each sliced segment. ) \n- **post_peak:** Peak Post-Normalization \n ( Peak [ 0.95] norm of each sliced segment. ) \n ",
                    choices=["none", "post_lufs", "post_peak"],
                    value="post_lufs",
                    interactive=True,
                    visible=True,
                    key='normalization_mode'
                )
                target_lufs = gr.Number(
                    label="Target LUFS",
                    info="Specify **target LUFS** for: \n 'pyloudnorm' loudness normalization. \n \n **If unsure what it does:** \n - **1. Keep LUFS finder enabled.** \n - **2. Forget 'bout this box :>**",
                    value=-20.0,
                    interactive=True,
                    scale=0.9,
                    key='target_lufs'
                )
                lufs_range_finder = gr.Checkbox(
                    label="LUFS range finder",
                    info="Enable to automatically: \n - Find the LUFS **for your dataset.** \n( Just sit back and relax :> ) \n \n Disable ONLY IF: \n - **You know** what LUFS to use. \n ",
                    value=True,
                    interactive=True,
                    visible=True,
                    scale=0.8,
                    key='lufs_range_finder'
                )
            with gr.Row():
                cut_preprocess = gr.Radio(
                    label="Audio cutting",
                    info="Audio file slicing-method selection:\n - **Skip** - if the files are already pre-sliced and properly processed. \n- **Simple** - If your dataset is already silence-truncated. \n- **Automatic** - for automatic silence detection and slicing around it. \n\n **It is advised to have the dataset properly silence-truncated and to use the 'Simple' method.** \n **(PS. Automatic is crap ~ I advise against it. )**",
                    choices=["Skip", "Simple", "Automatic"],
                    value="Simple",
                    interactive=True,
                    key='cut_preprocess'
                )
                chunk_len = gr.Slider(
                    0.5,
                    30.0,
                    3.0,
                    step=0.1,
                    label="Chunk length (sec)",
                    info="Length of the audio slice for 'Simple' method.",
                    interactive=True,
                    scale=0.46,
                    key='chunk_len'
                )
                overlap_len = gr.Slider(
                    0.0,
                    0.4,
                    0.3,
                    step=0.1,
                    label="Overlap length (sec)",
                    info="Length of the overlap between slices for 'Simple' method.",
                    interactive=True,
                    scale=0.57,
                    key='overlap_len'
                )
            with gr.Column():
                process_effects = gr.Checkbox(
                    label="DC / high-pass filtering",
                    info="**Applies high-pass filtering to get rid of low-freq. noise, DC offset and some Rumble. ( Disable if your dataset is already high-pass filtered. )**",
                    value=True,
                    interactive=True,
                    visible=True,
                    key='process_effects'
                )
            with gr.Column():
                noise_reduction = gr.Checkbox(
                    label="Noise Reduction",
                    info="**Spectral-Gating-Based noise reduction. ( Keep it disabled if your dataset is already Denoised or Noise-Free. )**",
                    value=False,
                    interactive=True,
                    visible=True,
                    key='noise_reduction'
                )
                clean_strength = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Noise Reduction Strength",
                    info="Set the desired level for clean-up level. Higher values result in more aggressive cleaning, but can negatively impact the audio.",
                    visible=False,
                    value=0.5,
                    interactive=True,
                    key='clean_strength'
                )
        preprocess_output_info = gr.Textbox(
            label="Output Information",
            info="The output information will be displayed here.",
            value="",
            max_lines=8,
            interactive=False,
        )

        with gr.Row():
            preprocess_button = gr.Button("Preprocess Dataset")
            preprocess_button.click(
                fn=run_preprocess_script,
                inputs=[
                    model_name,
                    dataset_path,
                    sampling_rate,
                    cpu_threads,
                    cut_preprocess,
                    process_effects,
                    noise_reduction,
                    clean_strength,
                    chunk_len,
                    overlap_len,
                    normalization_mode,
                    loading_resampling,
                    target_lufs,
                    lufs_range_finder,
                ],
                outputs=[preprocess_output_info],
            )

    # Extract section
    with gr.Accordion("Extract"):
        with gr.Row():
            f0_method = gr.Radio(
                label="Pitch extraction algorithm",
                info="Pitch extraction algorithm to use for the audio conversion: \n\n**RMVPE:** The default algorithm, recommended for most cases. \n- The fastest, very robust to noise. Can tolerate harmonies / layered vocals to some degree.  \n\n**CREPE:** Better suited for truly clean audio. \n- Is slower and way worse in handling noise. Can provide different / softer-ish results. \n\n**CREPE-TINY:** Smaller / lighter variant of CREPE. \n- Performs worse than 'full' ( standard crepe ) but is way lighter on hardware.",
                choices=["crepe", "crepe-tiny", "rmvpe", "fcpe"],
                value="rmvpe",
                interactive=True,
                key='f0_method'
            )

            embedder_model = gr.Radio(
                label="Embedder Model",
                info="Model used for learning speaker embedding and features extraction.",
                choices=[
                    "contentvec",
                    "spin_v1",
                    "spin_v2",
                    "custom",
                ],
                value="contentvec",
                interactive=True,
                key='embedder_model'
            )
        include_mutes = gr.Slider(
            0,
            10,
            2,
            step=1,
            label="Silent ( 'mute' ) files for training.",
            info="**Adding several silent files to the training set enables the model to handle pure silence in inferred audio files. Select '0' ( zero ) if your dataset is clean and already contains segments of pure silence.**",
            value=True,
            interactive=True,
            key='include_mutes'
        )
        with gr.Row(visible=False) as embedder_custom:
            with gr.Accordion("Custom Embedder", open=True):
                with gr.Row():
                    embedder_model_custom = gr.Dropdown(
                        label="Select Custom Embedder",
                        choices=refresh_embedders_folders(),
                        interactive=True,
                        allow_custom_value=True,
                        key='embedder_model_custom'
                    )
                    refresh_embedders_button = gr.Button("Refresh embedders")
                folder_name_input = gr.Textbox(label="Folder Name", interactive=True)
                with gr.Row():
                    bin_file_upload = gr.File(
                        label="Upload .bin", type="filepath", interactive=True
                    )
                    config_file_upload = gr.File(
                        label="Upload .json", type="filepath", interactive=True
                    )
                move_files_button = gr.Button("Move files to custom embedder folder")

        extract_output_info = gr.Textbox(
            label="Output Information",
            info="The output information will be displayed here.",
            value="",
            max_lines=8,
            interactive=False,
        )
        extract_button = gr.Button("Extract Features")
        extract_button.click(
            fn=run_extract_script,
            inputs=[
                model_name,
                f0_method,
                cpu_threads,
                extract_gpu,
                sampling_rate,
                vocoder_arch,
                embedder_model,
                embedder_model_custom,
                include_mutes,
            ],
            outputs=[extract_output_info],
        )

    # Training section
    with gr.Accordion("Training"):
        with gr.Row():
            batch_size = gr.Slider(
                1,
                128,
                max_vram_gpu(0),
                step=1,
                label="Batch Size",
                info="[ TOO BIG BATCH SIZE CAN LEAD TO VRAM 'OOM' ISSUES. ]\n\n Bigger batch size: \n- Promotes smoother, more stable gradients. \n- Can beneficial in cases where your dataset is big and diverse. \n- Can lead to early overtraining or flat / ' stuck ' graphs. \n- Generalization might be worsened. \n- **Favors slower learning rate.** \n\n Smaller batch size: \n- Promotes noisier, less stable gradients. \n- More suitable when your dataset is small, less diverse or repetitive. \n- Can lead to instability / divergence or noisy as hell graphs. \n- Generalization might be improved. \n- **Favors faster learning rate.**",
                interactive=True,
                key='batch_size'
            )
            epoch_save_frequency = gr.Slider(
                1,
                5000,
                1,
                step=1,
                label="Saving frequency",
                info="Determines the saving frequency of epochs. \n For example: Saving every 5th epoch.",
                interactive=True,
                key='epoch_save_frequency'
            )
            total_epoch_count = gr.Slider(
                1,
                10000,
                500,
                step=1,
                label="Total Epochs",
                info="Specifies the overall quantity of epochs for the model training process.",
                interactive=True,
                key='total_epoch_count'
            )
        with gr.Accordion("Advanced Settings for training", open=False):
            with gr.Row():
                with gr.Column():
                    save_only_latest_net_models = gr.Checkbox(
                        label="Save Only Latest G/D",
                        info="Don't disable it unless you need each 'G' and 'D' model saved every epoch. \n( It has it's use for pretrains creation, but not for finetuning. )",
                        value=True,
                        interactive=True,
                        key='save_only_latest_net_models'
                    )
                    save_weight_models = gr.Checkbox(
                        label="Save weight models",
                        info="Keep it enabled, else the small ' weight models '( actual voice models ) won't be saved.",
                        value=True,
                        interactive=True,
                        key='save_weight_models'
                    )
                    pretrained = gr.Checkbox(
                        label="Pretrained",
                        info="Utilize pretrained models for fine-tuning. \nKeep it enabled unless you're training from-scratch",
                        value=True,
                        interactive=True,
                        key='pretrained'
                    )
                    cleanup = gr.Checkbox(
                        label="Fresh Training",
                        info="Enable this setting only if you are training a new model from scratch or restarting the training. \nWhat it does is essentially deleting all previously generated weights and tensorboard logs.",
                        value=False,
                        interactive=True,
                        key='cleanup'
                    )
                    cache_dataset_in_gpu = gr.Checkbox(
                        label="Cache Dataset in GPU",
                        info="Cache the dataset in GPU memory to speed up the training process. \n NOTE: It is advised to have it turned off! ",
                        value=False,
                        interactive=True,
                        key='cache_dataset_in_gpu'
                    )
                    use_checkpointing = gr.Checkbox(
                        label="Checkpointing",
                        info="Enables memory-efficient training. \n This reduces the vram usage in exchange for slower training speed.",
                        value=auto_enable_checkpointing,
                        interactive=True,
                        key='use_checkpointing'
                    )
                    use_validation = gr.Checkbox(
                        label="Enable hold-out validation",
                        info="**Requires carefully handled dataset!**",
                        value=False,
                        interactive=True,
                        key='use_validation'
                    )
                with gr.Column():
                    use_tf32 = gr.Checkbox(
                        label="use 'TF32' precision",
                        info="Uses TF32 precision instead of FP32, typically resulting in 30% to 100% faster training. \n**Requires min. RTX 30xx ( At least Ampere microarchitecture )**",
                        value=microarchitecture_capability_checker(),
                        interactive=microarchitecture_capability_checker(),
                        key='use_tf32'
                    )
                    use_benchmark = gr.Checkbox(
                        label="Use 'cuDNN benchmark' mode",
                        info="Enable cuDNN benchmark mode **for potential speedup.**",
                        value=True,
                        interactive=True,
                        key='use_benchmark'
                    )
                    use_deterministic = gr.Checkbox(
                        label="Use 'cuDNN deterministic' mode",
                        info="Toggle deterministic mode for reproducibility **at possible performance cost.**",
                        value=False,
                        interactive=True,
                        key='use_deterministic'
                    )
                    spectral_loss = gr.Radio(
                        label="Spectral loss",
                        info="- **L1 Mel Loss:** Standard L1 mel spectrogram loss - **Safe default.** \n- **Multi-Scale Mel Loss:** Mel spectrogram loss that utilizes multiple-scales - **Results vary.** \n- **Multi-Res STFT Loss:** STFT Spec. based loss that utilizes multiple-resolutions - **EXPERIMENTAL.** ",
                        choices=["L1 Mel Loss", "Multi-Scale Mel Loss", "Multi-Res STFT Loss"],
                        value="L1 Mel Loss",
                        interactive=True,
                        key='spectral_loss'
                    )
                    lr_scheduler = gr.Radio(
                        label="Learning rate scheduler",
                        info="- **exp decay:** Decays the lr exponentially - **Safe default.** \n **( 'step' variant decays per step, 'epoch' per epoch. For finetuning per-step is recommended. )** \n- **cosine annealing:** Cosine annealing schedule - **Optional alternative.** \n- **none:** No scheduler - **For debugging or developing.**",
                        choices=["exp decay step", "exp decay epoch", "cosine annealing", "none"],
                        value="exp decay step",
                        interactive=True,
                        key='lr_scheduler'
                    )
                    exp_decay_gamma = gr.Radio(
                        label="Exponential decay gamma",
                        info="Gamma / decay factor for exponential lr scheduler",
                        choices=["0.9999996", "0.999875", "0.999", "0.9975", "0.995"],
                        value="0.999875",
                        interactive=True,
                        visible=True,
                        key='exp_decay_gamma'
                    )
                    use_kl_annealing = gr.Checkbox(
                        label="Use KL loss annealing",
                        info="Enables cyclic KL loss annealing for training. \n **Might potentially** mitigate overfitting on smaller datasets and generally should help with convergence. \n **Experimental**",
                        value=False,
                        interactive=True,
                        key='use_kl_annealing'
                    )
                    kl_annealing_cycle_duration = gr.Slider(
                        1,
                        100,
                        3,
                        step=1,
                        label="KL annealing cycle duration",
                        info="Determines the duration of each repeating annealing cycle. \n Limited testing showed 3 epochs is the most optimal, **but you can experiment for yourself**. \n **( Duration in epochs )**",
                        interactive=True,
                        visible=False,
                        key='kl_annealing_cycle_duration'
                    )
            with gr.Column():
                custom_pretrained = gr.Checkbox(
                    label="Custom Pretrained",
                    info="Utilizing custom pretrained models can lead to superior results, as selecting the most suitable pretrained models tailored to the specific use case can significantly enhance performance.",
                    value=False,
                    interactive=True,
                    key='custom_pretrained'
                )
                with gr.Column(visible=False) as pretrained_custom_settings:
                    with gr.Accordion("Pretrained Custom Settings"):
                        upload_pretrained = gr.File(
                            label="Upload Pretrained Model",
                            type="filepath",
                            interactive=True,
                        )
                        refresh_custom_pretaineds_button = gr.Button("Refresh Custom Pretraineds")
                        g_pretrained_path = gr.Dropdown(
                            label="Custom Pretrained G",
                            info="Select the custom pretrained model for the generator.",
                            choices=sorted(pretraineds_list_g),
                            interactive=True,
                            allow_custom_value=True,
                            key='g_pretrained_path'
                        )
                        d_pretrained_path = gr.Dropdown(
                            label="Custom Pretrained D",
                            info="Select the custom pretrained model for the discriminator.",
                            choices=sorted(pretraineds_list_d),
                            interactive=True,
                            allow_custom_value=True,
                            key='d_pretrained_path'
                        )
                multiple_gpu = gr.Checkbox(
                    label="GPU Settings",
                    info=(
                        "Lets you set / configure which GPUs you wanna utilize for training the model. ( In case you wanna use more than 1 GPU, that is. )"
                    ),
                    value=False,
                    interactive=True,
                    key='multiple_gpu'
                )
                with gr.Column(visible=False) as gpu_custom_settings:
                    with gr.Accordion("GPU ID override / Multi-gpu-training configuration"):
                        training_gpu = gr.Textbox(
                            label="GPU Number",
                            info="Specify the number of GPUs you wish to utilize for training by entering their ID and have them separated by hyphens. (These symbols: -)",
                            placeholder="0 to ∞ separated by -",
                            value=str(get_number_of_gpus()),
                            interactive=True,
                            key="training_gpu"
                        )
                        gr.Textbox(
                            label="GPU Information",
                            info="The GPU information will be displayed here.",
                            value=get_gpu_info(),
                            interactive=False,
                        )
                use_warmup = gr.Checkbox(
                    label="Warmup phase for training",
                    info="Enables usage of warmup for training. ( Currently supports only ' linear lr warmup ' )",
                    value=False,
                    interactive=True,
                    key='use_warmup'
                )
                with gr.Column(visible=False) as warmup_settings:
                    with gr.Accordion("Warmup settings"):
                        warmup_duration = gr.Slider(
                            1,
                            100,
                            5,
                            step=1,
                            label="Duration of the warmup phase",
                            info="Set the maximum number of epochs you want the warmup phase to last for. For small datasets you can try anywhere from 2 to 10. Alternatively, follow the ' 5–10% of the total epochs ' rule ",
                            interactive=True,
                            key='warmup_duration'
                        )

                use_custom_lr = gr.Checkbox(
                    label="Custom lr for gen and disc",
                    info="Enables customization of learning rate for Generator and Discriminator.",
                    value=False,
                    interactive=True,
                    key='use_custom_lr'
                )
                with gr.Column(visible=False) as custom_lr_settings:
                    with gr.Accordion("Custom lr settings"):
                        custom_lr_g = gr.Textbox(
                            label="Learning rate for Generator",
                            placeholder="e.g. 0.0001 or 1e-4",
                            info="Define the lr for generator. Accepts both decimals and scientific notation i.e. '1e-4'. ",
                            interactive=True,
                            key='custom_lr_g'
                        )
                        custom_lr_d = gr.Textbox(
                            label="Learning rate for Discriminator",
                            placeholder="e.g. 0.0001 or 1e-4",
                            info="Define the lr for discriminator. Accepts both decimals and scientific notation i.e. '1e-4'. ",
                            interactive=True,
                            key='custom_lr_d'
                        )

                index_algorithm = gr.Radio(
                    label="Index Algorithm",
                    info="KMeans is a clustering algorithm that divides the dataset into K clusters. This setting is particularly useful for large datasets.",
                    choices=["Auto", "Faiss", "KMeans"],
                    value="Auto",
                    interactive=True,
                    key='index_algorithm'
                )

        def enforce_terms(terms_accepted, *args):
            if not terms_accepted:
                message = "You must agree to the Terms of Use to proceed."
                gr.Info(message)
                return message
            return run_train_script(*args)

        terms_checkbox = gr.Checkbox(
            label="I agree to the terms of use",
            info="Please ensure compliance with the terms and conditions detailed in [this document](https://github.com/codename0og/codename-rvc-fork-3/blob/main/TERMS_OF_USE.md) before proceeding with your training.",
            value=False,
            interactive=True,
        )
        train_output_info = gr.Textbox(
            label="Output Information",
            info="The output information will be displayed here.",
            value="",
            max_lines=8,
            interactive=False,
        )

        with gr.Row():
            train_button = gr.Button("Start Training")
            train_button.click(
                fn=enforce_terms,
                inputs=[
                    terms_checkbox,
                    model_name,
                    epoch_save_frequency,
                    save_only_latest_net_models,
                    save_weight_models,
                    total_epoch_count,
                    sampling_rate,
                    batch_size,
                    training_gpu,
                    use_warmup,
                    warmup_duration,
                    pretrained,
                    cleanup,
                    index_algorithm,
                    cache_dataset_in_gpu,
                    custom_pretrained,
                    g_pretrained_path,
                    d_pretrained_path,
                    vocoder,
                    architecture,
                    optimizer,
                    use_checkpointing,
                    use_tf32,
                    use_benchmark,
                    use_deterministic,
                    spectral_loss,
                    lr_scheduler,
                    exp_decay_gamma,
                    use_validation,
                    use_kl_annealing,
                    kl_annealing_cycle_duration,
                    vits2_mode,
                    use_custom_lr,
                    custom_lr_g,
                    custom_lr_d,
                ],
                outputs=[train_output_info],
            )

            stop_train_button = gr.Button("Stop Training", visible=True)
            stop_train_button.click(
                fn=stop_train_script,
                inputs=[],
                outputs=[train_output_info],
            )

            index_button = gr.Button("Generate Index")
            index_button.click(
                fn=run_index_script,
                inputs=[model_name, index_algorithm],
                outputs=[train_output_info],
            )

    # Export Model section
    with gr.Accordion("Export Model", open=False):
        if not os.name == "nt":
            gr.Markdown(
                "The button 'Upload' is only for google colab: Uploads the exported files to the ForkExported folder in your Google Drive."
            )
        with gr.Row():
            with gr.Column():
                pth_file_export = gr.File(
                    label="Exported Pth file",
                    type="filepath",
                    value=None,
                    interactive=False,
                )
                pth_dropdown_export = gr.Dropdown(
                    label="Pth file",
                    info="Select the pth file to be exported",
                    choices=get_pth_list(),
                    value=None,
                    interactive=True,
                    allow_custom_value=True,
                )
            with gr.Column():
                index_file_export = gr.File(
                    label="Exported Index File",
                    type="filepath",
                    value=None,
                    interactive=False,
                )
                index_dropdown_export = gr.Dropdown(
                    label="Index File",
                    info="Select the index file to be exported",
                    choices=get_index_list(),
                    value=None,
                    interactive=True,
                    allow_custom_value=True,
                )
        with gr.Row():
            with gr.Column():
                refresh_export = gr.Button("Refresh")
                if not os.name == "nt":
                    upload_exported = gr.Button("Upload")
                    upload_exported.click(
                        fn=upload_to_google_drive,
                        inputs=[pth_dropdown_export, index_dropdown_export],
                        outputs=[],
                    )

            def toggle_visible(checkbox):
                return {"visible": checkbox, "__type__": "update"}

            def toggle_visible_gamma(lr_scheduler):
                if lr_scheduler in ["exp decay step", "exp decay epoch"]:
                    return {"visible": True, "__type__": "update"}
                return {"visible": False, "__type__": "update"}

            def toggle_pretrained(pretrained, custom_pretrained):
                if custom_pretrained == False:
                    return {"visible": pretrained, "__type__": "update"}, {
                        "visible": False,
                        "__type__": "update",
                    }
                else:
                    return {"visible": pretrained, "__type__": "update"}, {
                        "visible": pretrained,
                        "__type__": "update",
                    }

            def download_prerequisites():
                    gr.Info(
                        "Checking for prerequisites with pitch guidance... Missing files will be downloaded. If you already have them, this step will be skipped."
                    )
                    run_prerequisites_script(
                        pretraineds_hifigan=True,
                        models=False,
                        exe=False,
                    )
                    gr.Info(
                        "Prerequisites check complete. Missing files were downloaded, and you may now start preprocessing."
                    )

            def toggle_visible_embedder_custom(embedder_model):
                if embedder_model == "custom":
                    return {"visible": True, "__type__": "update"}
                return {"visible": False, "__type__": "update"}

            def toggle_architecture(architecture, vocoder_arch):
                if architecture == "Fork/Applio":
                    vocoder_arch_value = "hifi_mrf_refine"
                    return (
                        {
                            "choices": ["32000", "40000", "48000"],
                            "__type__": "update",
                        },
                        {
                            "choices": ["RefineGAN", "MRF HiFi-GAN"],
                            "__type__": "update",
                            "interactive": True,
                            "value": "RefineGAN",
                        },
                        vocoder_arch_value,
                    )
                elif architecture == "Fork":
                    vocoder_arch_value = "ringformer_v2"
                    return (
                        {
                            "choices": ["24000", "32000", "40000", "48000"],
                            "__type__": "update",
                            "value": "48000",
                        },
                        {
                            "choices": ["RingFormer_v1", "RingFormer_v2", "PCPH-GAN"],
                            "__type__": "update",
                            "interactive": True,
                            "value": "RingFormer_v2",
                        },
                        vocoder_arch_value,
                    )
                else:
                    vocoder_arch_value = "hifi_mrf_refine"
                    return (
                        {
                            "choices": ["32000", "40000", "48000"],
                            "__type__": "update",
                            "value": "48000",
                        },
                        {
                            "choices": ["HiFi-GAN"],
                            "__type__": "update",
                            "value": "HiFi-GAN",
                            "interactive": False,
                        },
                        vocoder_arch_value,
                    )
            def fork_vocoder_handler(architecture, vocoder_arch, vocoder):
                if architecture == "Fork" and vocoder == "RingFormer_v1":
                    vocoder_arch_value = "ringformer_v1"
                    return (
                        {
                            "choices": ["24000", "32000", "40000", "48000"],
                            "__type__": "update",
                            "value": "48000",
                        },
                        vocoder_arch_value,
                    )
                elif architecture == "Fork" and vocoder == "RingFormer_v2":
                    vocoder_arch_value = "ringformer_v2"
                    return (
                        {
                            "choices": ["24000", "32000", "40000", "48000"],
                            "__type__": "update",
                            "value": "48000",
                        },
                        vocoder_arch_value,
                    )
                elif architecture == "Fork" and vocoder == "PCPH-GAN":
                    vocoder_arch_value = "pcph_gan"
                    return (
                        {
                            "choices": ["24000", "32000", "40000", "48000"],
                            "__type__": "update",
                            "value": "48000",
                        },
                        vocoder_arch_value,
                    )
                else:
                    return gr.skip()

            def update_noise_reduce_slider_visibility(noise_reduction):
                return gr.update(visible=noise_reduction)
            
            saved_components.extend([
                # Model settings
                architecture, optimizer, vocoder, sampling_rate, cpu_threads, extract_gpu,

                # Preprocessing
                dataset_path, loading_resampling, normalization_mode,
                target_lufs, lufs_range_finder, cut_preprocess,
                chunk_len, overlap_len, process_effects,
                noise_reduction, clean_strength,

                # Feature extract
                f0_method, embedder_model, include_mutes,
                embedder_model_custom,

                # Training
                batch_size, epoch_save_frequency, total_epoch_count,
                save_only_latest_net_models, save_weight_models, pretrained,
                cleanup, cache_dataset_in_gpu, use_checkpointing,
                use_tf32, use_benchmark, use_deterministic, spectral_loss,
                lr_scheduler, exp_decay_gamma, use_validation,
                custom_pretrained, g_pretrained_path,
                d_pretrained_path, multiple_gpu, training_gpu, use_warmup,
                warmup_duration, use_custom_lr, custom_lr_g,
                custom_lr_d, use_kl_annealing, kl_annealing_cycle_duration,
                index_algorithm
            ])

            def save_training_preset(inputs):
                settings = {}
                for component in saved_components:
                    settings[component.key] = inputs[component]

                preset_path = os.path.normpath(os.path.abspath(os.path.join(presets_path, inputs[preset_dropdown] + '.json')))

                if not preset_path.startswith(presets_path):
                    raise gr.Error(f"Invalid training preset name: {inputs[preset_dropdown]}", duration=5)

                with open(preset_path, 'w', encoding='utf-8') as of:
                    json.dump(settings, of, indent=4, ensure_ascii=False)

            def load_training_preset(preset_name):
                if preset_name not in get_presets_list():
                    raise gr.Error(f'Preset does not exist: {preset_name}')

                preset_path = os.path.normpath(os.path.abspath(os.path.join(presets_path, preset_name + '.json')))

                with open(preset_path, 'r', encoding='utf-8') as ifile:
                    settings = json.loads(ifile.read())

                return [
                    settings[component.key] if component.key in settings else gr.skip()
                    for component in saved_components
                ]

            refresh_presets_button.click(
                fn=lambda: gr.Dropdown(choices=get_presets_list()), 
                outputs=[preset_dropdown]
            )

            save_preset_button.click(
                fn=save_training_preset,
                inputs=set(saved_components) | {preset_dropdown}
            ).then(
                fn=lambda: gr.Dropdown(choices=get_presets_list()), 
                outputs=[preset_dropdown]
            )

            load_preset_button.click(
                fn=load_training_preset,
                inputs=[preset_dropdown],
                outputs=saved_components
            ).then(  # update twice so components depending on "change" events get updated
                fn=load_training_preset,
                inputs=[preset_dropdown],
                outputs=saved_components
            )

            noise_reduction.change(
                fn=update_noise_reduce_slider_visibility,
                inputs=noise_reduction,
                outputs=clean_strength,
            )
            architecture.change(
                fn=toggle_architecture,
                inputs=[architecture],
                outputs=[sampling_rate, vocoder, vocoder_arch],
            )
            vocoder.change( 
                fn=fork_vocoder_handler,
                inputs=[architecture, vocoder_arch, vocoder],
                outputs=[sampling_rate, vocoder_arch],
            )
            refresh.click(
                fn=refresh_models_and_datasets,
                inputs=[],
                outputs=[model_name, dataset_path],
            )
            dataset_creator.change(
                fn=toggle_visible,
                inputs=[dataset_creator],
                outputs=[dataset_creator_settings],
            )
            upload_audio_dataset.upload(
                fn=save_drop_dataset_audio,
                inputs=[upload_audio_dataset, dataset_name],
                outputs=[upload_audio_dataset, dataset_path],
            )
            embedder_model.change(
                fn=toggle_visible_embedder_custom,
                inputs=[embedder_model],
                outputs=[embedder_custom],
            )
            embedder_model.change(
                fn=toggle_visible_embedder_custom,
                inputs=[embedder_model],
                outputs=[embedder_custom],
            )
            move_files_button.click(
                fn=create_folder_and_move_files,
                inputs=[folder_name_input, bin_file_upload, config_file_upload],
                outputs=[],
            )
            refresh_embedders_button.click(
                fn=refresh_embedders_folders, inputs=[], outputs=[embedder_model_custom]
            )
            pretrained.change(
                fn=toggle_pretrained,
                inputs=[pretrained, custom_pretrained],
                outputs=[custom_pretrained, pretrained_custom_settings],
            )
            custom_pretrained.change(
                fn=toggle_visible,
                inputs=[custom_pretrained],
                outputs=[pretrained_custom_settings],
            )
            refresh_custom_pretaineds_button.click(
                fn=refresh_custom_pretraineds,
                inputs=[],
                outputs=[g_pretrained_path, d_pretrained_path],
            )
            upload_pretrained.upload(
                fn=save_drop_model,
                inputs=[upload_pretrained],
                outputs=[upload_pretrained],
            )
            use_warmup.change(
                fn=toggle_visible,
                inputs=[use_warmup],
                outputs=[warmup_settings],
            )
            use_custom_lr.change(
                fn=toggle_visible,
                inputs=[use_custom_lr],
                outputs=[custom_lr_settings],
            )
            use_kl_annealing.change(
                fn=toggle_visible,
                inputs=[use_kl_annealing],
                outputs=[kl_annealing_cycle_duration]
            )
            lr_scheduler.change(
                fn=toggle_visible_gamma,
                inputs=[lr_scheduler],
                outputs=[exp_decay_gamma],
            )
            multiple_gpu.change(
                fn=toggle_visible,
                inputs=[multiple_gpu],
                outputs=[gpu_custom_settings],
            )
            pth_dropdown_export.change(
                fn=export_pth,
                inputs=[pth_dropdown_export],
                outputs=[pth_file_export],
            )
            index_dropdown_export.change(
                fn=export_index,
                inputs=[index_dropdown_export],
                outputs=[index_file_export],
            )
            refresh_export.click(
                fn=refresh_pth_and_index_list,
                inputs=[],
                outputs=[pth_dropdown_export, index_dropdown_export],
            )

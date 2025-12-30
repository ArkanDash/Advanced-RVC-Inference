import os, sys
import gradio as gr
import regex as re
import shutil
import datetime
import json
import torch
import io
import zstandard as zstd


from core import (
    run_infer_script,
    run_batch_infer_script,
)

from rvc.lib.utils import format_title
from tabs.settings.sections.restart import stop_infer

now_dir = os.getcwd()
sys.path.append(now_dir)

model_root = os.path.join(now_dir, "logs")
audio_root = os.path.join(now_dir, "assets", "audios")
custom_embedder_root = os.path.join(
    now_dir, "rvc", "models", "embedders", "embedders_custom"
)

PRESETS_DIR = os.path.join(now_dir, "assets", "presets")
FORMANTSHIFT_DIR = os.path.join(now_dir, "assets", "formant_shift")

os.makedirs(custom_embedder_root, exist_ok=True)

custom_embedder_root_relative = os.path.relpath(custom_embedder_root, now_dir)
model_root_relative = os.path.relpath(model_root, now_dir)
audio_root_relative = os.path.relpath(audio_root, now_dir)

sup_audioext = {
    "wav",
    "mp3",
    "flac",
    "ogg",
    "opus",
    "m4a",
    "mp4",
    "aac",
    "alac",
    "wma",
    "aiff",
    "webm",
    "ac3",
}

names = [
    os.path.join(root, file)
    for root, _, files in os.walk(model_root_relative, topdown=False)
    for file in files
    if (
        file.endswith((".pth", ".uvmp"))
        and not (file.startswith("G_") or file.startswith("D_"))
    )
]

default_weight = names[0] if names else None

indexes_list = [
    os.path.join(root, name)
    for root, _, files in os.walk(model_root_relative, topdown=False)
    for name in files
    if name.endswith(".index") and "trained" not in name
]

audio_paths = [
    os.path.join(root, name)
    for root, _, files in os.walk(audio_root_relative, topdown=False)
    for name in files
    if name.endswith(tuple(sup_audioext))
    and root == audio_root_relative
    and "_output" not in name
]

custom_embedders = [
    os.path.join(dirpath, dirname)
    for dirpath, dirnames, _ in os.walk(custom_embedder_root_relative)
    for dirname in dirnames
]


def update_sliders(preset):
    with open(
        os.path.join(PRESETS_DIR, f"{preset}.json"), "r", encoding="utf-8"
    ) as json_file:
        values = json.load(json_file)
    return (
        values["pitch"],
        values["filter_radius"],
        values["index_rate"],
        values["rms_mix_rate"],
        values["protect"],
    )


def update_sliders_formant(preset):
    with open(
        os.path.join(FORMANTSHIFT_DIR, f"{preset}.json"), "r", encoding="utf-8"
    ) as json_file:
        values = json.load(json_file)
    return (
        values["formant_qfrency"],
        values["formant_timbre"],
    )


def export_presets(presets, file_path):
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(presets, json_file, ensure_ascii=False, indent=4)


def import_presets(file_path):
    with open(file_path, "r", encoding="utf-8") as json_file:
        presets = json.load(json_file)
    return presets


def get_presets_data(pitch, filter_radius, index_rate, rms_mix_rate, protect):
    return {
        "pitch": pitch,
        "filter_radius": filter_radius,
        "index_rate": index_rate,
        "rms_mix_rate": rms_mix_rate,
        "protect": protect,
    }


def export_presets_button(
    preset_name, pitch, filter_radius, index_rate, rms_mix_rate, protect
):
    if preset_name:
        file_path = os.path.join(PRESETS_DIR, f"{preset_name}.json")
        presets_data = get_presets_data(
            pitch, filter_radius, index_rate, rms_mix_rate, protect
        )
        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(presets_data, json_file, ensure_ascii=False, indent=4)
        return "Export successful"
    return "Export cancelled"


def import_presets_button(file_path):
    if file_path:
        imported_presets = import_presets(file_path.name)
        return (
            list(imported_presets.keys()),
            imported_presets,
            "Presets imported successfully!",
        )
    return [], {}, "No file selected for import."


def list_json_files(directory):
    return [f.rsplit(".", 1)[0] for f in os.listdir(directory) if f.endswith(".json")]


def refresh_presets():
    json_files = list_json_files(PRESETS_DIR)
    return gr.update(choices=json_files)


def output_path_fn(input_audio_path):
    original_name_without_extension = os.path.basename(input_audio_path).rsplit(".", 1)[
        0
    ]
    new_name = original_name_without_extension + "_output.wav"
    output_path = os.path.join(os.path.dirname(input_audio_path), new_name)
    return output_path


def change_choices(model):
    if model:
        speakers = get_speakers_id(model)
    else:
        speakers = [0]
    names = [
        os.path.join(root, file)
        for root, _, files in os.walk(model_root_relative, topdown=False)
        for file in files
        if (
            file.endswith((".pth", ".uvmp"))
            and not (file.startswith("G_") or file.startswith("D_"))
        )
    ]

    indexes_list = [
        os.path.join(root, name)
        for root, _, files in os.walk(model_root_relative, topdown=False)
        for name in files
        if name.endswith(".index") and "trained" not in name
    ]

    audio_paths = [
        os.path.join(root, name)
        for root, _, files in os.walk(audio_root_relative, topdown=False)
        for name in files
        if name.endswith(tuple(sup_audioext))
        and root == audio_root_relative
        and "_output" not in name
    ]

    return (
        {"choices": sorted(names), "__type__": "update"},
        {"choices": sorted(indexes_list), "__type__": "update"},
        {"choices": sorted(audio_paths), "__type__": "update"},
        {
            "choices": (
                sorted(speakers)
                if speakers is not None and isinstance(speakers, (list, tuple))
                else [0]
            ),
            "__type__": "update",
        },
        {
            "choices": (
                sorted(speakers)
                if speakers is not None and isinstance(speakers, (list, tuple))
                else [0]
            ),
            "__type__": "update",
        },
    )


def get_indexes():
    indexes_list = [
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(model_root_relative)
        for filename in filenames
        if filename.endswith(".index") and "trained" not in filename
    ]

    return indexes_list if indexes_list else ""


def extract_model_and_epoch(path):
    base_name = os.path.basename(path)
    match = re.match(r"(.+?)_(\d+)e_", base_name)
    if match:
        model, epoch = match.groups()
        return model, int(epoch)
    return "", 0


def save_to_wav(record_button):
    if record_button is None:
        pass
    else:
        path_to_file = record_button
        new_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".wav"
        target_path = os.path.join(audio_root_relative, os.path.basename(new_name))

        shutil.move(path_to_file, target_path)
        return target_path, output_path_fn(target_path)


def save_to_wav2(upload_audio):
    file_path = upload_audio
    formated_name = format_title(os.path.basename(file_path))
    target_path = os.path.join(audio_root_relative, formated_name)

    if os.path.exists(target_path):
        os.remove(target_path)

    shutil.copy(file_path, target_path)
    return target_path, output_path_fn(target_path)


def delete_outputs():
    gr.Info(f"Inference outputs cleared!")
    for root, _, files in os.walk(audio_root_relative, topdown=False):
        for name in files:
            if name.endswith(tuple(sup_audioext)) and name.__contains__("_output"):
                os.remove(os.path.join(root, name))


def match_index(model_file_value):
    if model_file_value and model_file_value.endswith(".uvmp"):
        return ""
    if model_file_value:
        model_folder = os.path.dirname(model_file_value)
        model_name = os.path.basename(model_file_value)
        index_files = get_indexes()
        pattern = r"^(.*?)_"
        match = re.match(pattern, model_name)
        for index_file in index_files:
            if os.path.dirname(index_file) == model_folder:
                return index_file
            elif match and match.group(1) in os.path.basename(index_file):
                return index_file
            elif model_name in os.path.basename(index_file):
                return index_file
    return ""


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


def refresh_formant():
    json_files = list_json_files(FORMANTSHIFT_DIR)
    return gr.update(choices=json_files)


def refresh_embedders_folders():
    custom_embedders = [
        os.path.join(dirpath, dirname)
        for dirpath, dirnames, _ in os.walk(custom_embedder_root_relative)
        for dirname in dirnames
    ]
    return custom_embedders

def get_speakers_id(model):
    if not model or not os.path.exists(os.path.join(now_dir, model)):
        return [0]
    try:
        if model.endswith(".uvmp"):
            with open(os.path.join(now_dir, model), 'rb') as f_comp:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(f_comp) as reader:
                    decompressed_data = reader.read()
            buffer = io.BytesIO(decompressed_data)
            model_data = torch.load(buffer, map_location="cpu", weights_only=False)
            
            if "models" in model_data:
                return sorted(list(model_data["models"].keys()))
            else: # Backwards compatibility for single-model .uvmp
                return [0]
        else:
            model_data = torch.load(os.path.join(now_dir, model), map_location="cpu", weights_only=True)
        
        speakers_id = model_data.get("speakers_id")
        if speakers_id:
            return list(range(speakers_id))
        else:
            return [0]
    except Exception as e:
        print(f"Error loading model to get speaker IDs: {e}")
        return [0]


# Inference tab
def inference_tab():
    with gr.Column():
        with gr.Row():
            model_file = gr.Dropdown(
                label="Voice Model",
                info="Select the voice model (.pth or .uvmp) used for inference.",
                choices=sorted(names, key=lambda x: extract_model_and_epoch(x)),
                interactive=True,
                value=default_weight,
                allow_custom_value=True,
            )

            index_file = gr.Dropdown(
                label="Index File",
                info="Select the index file. Disabled if a .uvmp model is selected.",
                choices=get_indexes(),
                value=match_index(default_weight) if default_weight else "",
                interactive=True,
                allow_custom_value=True,
            )
        with gr.Row():
            unload_button = gr.Button("Unload the voice model")
            refresh_button = gr.Button("Refresh models, indexes and audios")

            unload_button.click(
                fn=lambda: (
                    {"value": "", "__type__": "update"},
                    {"value": "", "__type__": "update"},
                ),
                inputs=[],
                outputs=[model_file, index_file],
            )

        def on_model_change(model_path):
            """
            Handles UI changes when a .uvmp voice model is selected.
            """
            is_uvmp = model_path and model_path.endswith(".uvmp")
            speakers = get_speakers_id(model_path)
            speaker_val = speakers[0] if speakers else 0

            if is_uvmp:
                # .uvmp selected: Repurpose index_file for speaker ID, hide original sid
                return (
                    gr.update(
                        label="Speaker ID",
                        info="Select the speaker ID for the .uvmp model.",
                        choices=speakers,
                        value=speaker_val,
                        interactive=True,
                        visible=True,
                    ),
                    gr.update(visible=False, value=speaker_val, choices=speakers),
                )
            else:
                # .pth selected: Restore index_file and show original sid
                return (
                    gr.update(
                        label="Index File",
                        info="Select the index file. Disabled if a .uvmp model is selected.",
                        choices=get_indexes(),
                        value=match_index(model_path),
                        interactive=True,
                        visible=True,
                    ),
                    gr.update(visible=True, choices=speakers, value=speaker_val, interactive=True),
                )
            
        def sync_speaker_id(model_path, repurposed_index_value):
            if model_path and model_path.endswith(".uvmp"):
                return gr.update(value=repurposed_index_value)
            return gr.update()

    # Single inference tab
    with gr.Tab("Single input infer"):
        with gr.Column():
            upload_audio = gr.Audio(
                label="Upload Audio", type="filepath", editable=False
            )
            with gr.Row():
                audio = gr.Dropdown(
                    label="Select Audio Input",
                    info="Select the audio for inference.",
                    choices=sorted(audio_paths),
                    value=audio_paths[0] if audio_paths else "",
                    interactive=True,
                    allow_custom_value=True,
                )

        with gr.Accordion("Advanced Settings for inference", open=False):
            with gr.Column():
                clear_outputs_infer = gr.Button("Clear '_output' audio files ( infer outputs ) from 'assets/audios' ")
                output_path = gr.Textbox(
                    label="Path for infer outputs",
                    placeholder="Provide the path for inference outputs",
                    info="The path where inference outputs will be saved. \nBy default they land in 'assets/audios' ",
                    value=(
                        output_path_fn(audio_paths[0])
                        if audio_paths
                        else os.path.join(now_dir, "assets", "audios", "output.wav")
                    ),
                    interactive=True,
                )
                export_format = gr.Radio(
                    label="Export Format",
                    info="Choose the audio export format.",
                    choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
                    value="WAV",
                    interactive=True,
                )
                seed = gr.Number(
                    label="Inference Seed",
                    info="Specify any seed to be used for inference or leave at '0' for random outputs. ( Classic RVC behavior. )",
                    value=0,
                    interactive=True,
                )
                sid = gr.Dropdown(
                    label="Speaker ID",
                    info="Select the speaker ID used for inference. \nApplicable only for multi-speaker models.",
                    choices=get_speakers_id(model_file.value),
                    value=0,
                    interactive=True,
                )
                split_audio = gr.Checkbox(
                    label="Audio splitting",
                    info="Splits the audio into chunks ( based on **silence** regions! ). \nCan potentially improve the results.",
                    visible=True,
                    value=False,
                    interactive=True,
                )
                autotune = gr.Checkbox(
                    label="Autotuning",
                    info="Applies the Autotune effect.",
                    visible=True,
                    value=False,
                    interactive=True,
                )
                autotune_strength = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Strength of autotuning",
                    info="Autotune effect's strength. \nHigher values snap the pitch more tightly to the chromatic grid.",
                    visible=False,
                    value=1,
                    interactive=True,
                )
                clean_audio = gr.Checkbox(
                    label="Audio cleanup",
                    info="Cleans your audio using noise detection algorithms, preferable for talking / speech audios.",
                    visible=True,
                    value=False,
                    interactive=True,
                )
                clean_strength = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Strength of cleaning",
                    info="Set the strenght of cleaning. If you set it too high, the audio might come out muffly or degraded in quality.",
                    visible=False,
                    value=0.3,
                    interactive=True,
                )
                formant_shifting = gr.Checkbox(
                    label="Formant Shifting",
                    info="Enables formant shifting. Useful in situations where your model is a female but input is a male ( and vice-versa ).",
                    value=False,
                    visible=True,
                    interactive=True,
                )
                post_process = gr.Checkbox(
                    label="Post-Processing",
                    info="Various audio effects and processing for the audio output.",
                    value=False,
                    interactive=True,
                )
                with gr.Row(visible=False) as formant_row:
                    formant_preset = gr.Dropdown(
                        label="Browse presets for formant shifting",
                        info="Presets are located in '/assets/formant_shift' folder.",
                        choices=list_json_files(FORMANTSHIFT_DIR),
                        visible=False,
                        interactive=True,
                    )
                    formant_refresh_button = gr.Button(
                        value="Refresh",
                        visible=False,
                    )
                formant_qfrency = gr.Slider(
                    value=1.0,
                    info="Controls the quefrency used for formant shifting. Default is 1.0.",
                    label="Formant Quefrency.",
                    minimum=0.0,
                    maximum=16.0,
                    step=0.1,
                    visible=False,
                    interactive=True,
                )
                formant_timbre = gr.Slider(
                    value=1.0,
                    info="Adjusts timbre characteristics during formant shifting. Default is 1.0.",
                    label="Formant Timbre",
                    minimum=0.0,
                    maximum=16.0,
                    step=0.1,
                    visible=False,
                    interactive=True,
                )
                reverb = gr.Checkbox(
                    label="Reverb",
                    info="Applies reverb to the audio output",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                reverb_room_size = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Reverb; Room Size",
                    info="Set the room size of the reverb.",
                    value=0.5,
                    interactive=True,
                    visible=False,
                )

                reverb_damping = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Reverb Damping",
                    info="Set the damping of the reverb.",
                    value=0.5,
                    interactive=True,
                    visible=False,
                )

                reverb_wet_gain = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Reverb; Wet Gain",
                    info="Set the wet gain of the reverb.",
                    value=0.33,
                    interactive=True,
                    visible=False,
                )

                reverb_dry_gain = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Reverb; Dry Gain",
                    info="Set the dry gain of the reverb.",
                    value=0.4,
                    interactive=True,
                    visible=False,
                )

                reverb_width = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Reverb; Width",
                    info="Set the width of the reverb.",
                    value=1.0,
                    interactive=True,
                    visible=False,
                )

                reverb_freeze_mode = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Reverb; Freeze Mode",
                    info="Set the freeze mode of the reverb.",
                    value=0.0,
                    interactive=True,
                    visible=False,
                )
                pitch_shift = gr.Checkbox(
                    label="Pitch Shift",
                    info="Enable pitch shifting for the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                pitch_shift_semitones = gr.Slider(
                    minimum=-12,
                    maximum=12,
                    label="Pitch Shift ( Semitones )",
                    info="Set how many semitones to shift the pitch (up or down).",
                    value=0,
                    interactive=True,
                    visible=False,
                )
                limiter = gr.Checkbox(
                    label="Limiter",
                    info="Apply limiter to the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                limiter_threshold = gr.Slider(
                    minimum=-60,
                    maximum=0,
                    label="Limiter Threshold dB",
                    info="Set the limiter's threshold ( decibels ).",
                    value=-6,
                    interactive=True,
                    visible=False,
                )
                limiter_release_time = gr.Slider(
                    minimum=0.01,
                    maximum=1,
                    label="Limiter Release Time",
                    info="Set the limiter release time.",
                    value=0.05,
                    interactive=True,
                    visible=False,
                )
                gain = gr.Checkbox(
                    label="Gain",
                    info="Apply gain to the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                gain_db = gr.Slider(
                    minimum=-60,
                    maximum=60,
                    label="Gain dB",
                    info="Set the gain ( decibels ).",
                    value=0,
                    interactive=True,
                    visible=False,
                )
                distortion = gr.Checkbox(
                    label="Distortion",
                    info="Apply distortion to the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                distortion_gain = gr.Slider(
                    minimum=-60,
                    maximum=60,
                    label="Distortion Gain",
                    info="Set the distortion gain.",
                    value=25,
                    interactive=True,
                    visible=False,
                )
                chorus = gr.Checkbox(
                    label="chorus",
                    info="Apply chorus to the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                chorus_rate = gr.Slider(
                    minimum=0,
                    maximum=100,
                    label="Chorus Rate Hz",
                    info="Set the chorus rate ( Hertz ).",
                    value=1.0,
                    interactive=True,
                    visible=False,
                )

                chorus_depth = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="chorus Depth",
                    info="Set the chorus depth.",
                    value=0.25,
                    interactive=True,
                    visible=False,
                )

                chorus_center_delay = gr.Slider(
                    minimum=7,
                    maximum=8,
                    label="chorus Center Delay ms",
                    info="Set the chorus center delay ms.",
                    value=7,
                    interactive=True,
                    visible=False,
                )

                chorus_feedback = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="chorus Feedback",
                    info="Set the chorus feedback.",
                    value=0.0,
                    interactive=True,
                    visible=False,
                )

                chorus_mix = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Chorus Mix",
                    info="Set the chorus mix.",
                    value=0.5,
                    interactive=True,
                    visible=False,
                )
                bitcrush = gr.Checkbox(
                    label="Bitcrush",
                    info="Apply bitcrush to the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                bitcrush_bit_depth = gr.Slider(
                    minimum=1,
                    maximum=32,
                    label="Bitcrush Bit Depth",
                    info="Set the bitcrush bit depth.",
                    value=8,
                    interactive=True,
                    visible=False,
                )
                clipping = gr.Checkbox(
                    label="Clipping",
                    info="Apply clipping to the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                clipping_threshold = gr.Slider(
                    minimum=-60,
                    maximum=0,
                    label="Clipping Threshold",
                    info="Set the clipping threshold.",
                    value=-6,
                    interactive=True,
                    visible=False,
                )
                compressor = gr.Checkbox(
                    label="Compressor",
                    info="Apply compressor to the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                compressor_threshold = gr.Slider(
                    minimum=-60,
                    maximum=0,
                    label="Compressor Threshold dB",
                    info="Set the compressor threshold dB.",
                    value=0,
                    interactive=True,
                    visible=False,
                )

                compressor_ratio = gr.Slider(
                    minimum=1,
                    maximum=20,
                    label="Compressor Ratio",
                    info="Set the compressor ratio.",
                    value=1,
                    interactive=True,
                    visible=False,
                )

                compressor_attack = gr.Slider(
                    minimum=0.0,
                    maximum=100,
                    label="Compressor Attack ms",
                    info="Set the compressor attack ms.",
                    value=1.0,
                    interactive=True,
                    visible=False,
                )

                compressor_release = gr.Slider(
                    minimum=0.01,
                    maximum=100,
                    label="Compressor Release ms",
                    info="Set the compressor release ms.",
                    value=100,
                    interactive=True,
                    visible=False,
                )
                delay = gr.Checkbox(
                    label="Delay",
                    info="Apply delay to the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                delay_seconds = gr.Slider(
                    minimum=0.0,
                    maximum=5.0,
                    label="Delay Seconds",
                    info="Set the delay seconds.",
                    value=0.5,
                    interactive=True,
                    visible=False,
                )

                delay_feedback = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    label="Delay Feedback",
                    info="Set the delay feedback.",
                    value=0.0,
                    interactive=True,
                    visible=False,
                )

                delay_mix = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    label="Delay Mix",
                    info="Set the delay mix.",
                    value=0.5,
                    interactive=True,
                    visible=False,
                )
                with gr.Accordion("Preset Settings", open=False):
                    with gr.Row():
                        preset_dropdown = gr.Dropdown(
                            label="Select Custom Preset",
                            choices=list_json_files(PRESETS_DIR),
                            interactive=True,
                        )
                        presets_refresh_button = gr.Button("Refresh Presets")
                    import_file = gr.File(
                        label="Select file to import",
                        file_count="single",
                        type="filepath",
                        interactive=True,
                    )
                    import_file.change(
                        import_presets_button,
                        inputs=import_file,
                        outputs=[preset_dropdown],
                    )
                    presets_refresh_button.click(
                        refresh_presets, outputs=preset_dropdown
                    )
                    with gr.Row():
                        preset_name_input = gr.Textbox(
                            label="Preset Name",
                            placeholder="Enter preset name",
                        )
                        export_button = gr.Button("Export Preset")
                pitch = gr.Slider(
                    minimum=-24,
                    maximum=24,
                    step=1,
                    label="Pitch",
                    info="Set the pitch of the audio, the higher the value, the higher the pitch. \nCheat-sheet: 0 = 1:1 as input, 12 = 1 octave higher, -12 = 1 octave lower. \n ***You can also try: 6, 3, -3, -6. Some singers do this if they're uncomfortable with the song's tonality or vocal range.***",
                    value=0,
                    interactive=True,
                )
                filter_radius = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Filter Radius",
                    info="f0 smoothing / pitch contour filtering \n-Lower values preserve more of the natural variations in the pitch, including subtle pitch shifts and fluctuations. \n( More dynamic, expressive pitch that might better capture natural pitch variation but could also introduce more 'noise' or instability. ) \n \n-Higher values remove more of the fine details and fluctuations in the pitch, resulting in a smoother and more stable pitch curve. \n ( Yet, potentially flatter and innatural sounding sound + loss of fine-details. ) \n \n ( Best to leave it set to the default '0.006', especially if you're unsure of how it works. ",
                    value=0.006,
                    step=0.001,
                    interactive=False,
                    visible=False,
                )
                index_rate = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Search Feature Ratio",
                    info="Influence exerted by the index file; a higher value corresponds to greater influence. However, opting for lower values can help mitigate artifacts present in the audio. \n ***Basically, worse models can't afford to have it too high else you'll get potential artifacts.***",
                    value=0.5,
                    interactive=True,
                )
                rms_mix_rate = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="RMS Volume Envelope",
                    info="Adjust the loudness (RMS) of the converted voice to match the original/input voice. \n At 1, the output stays the same; values closer to 0 make the output match the input's loudness more closely. \n ***Recommended to leave it at 1, it's a pretty crap functionality.***",
                    value=1,
                    interactive=True,
                )
                protect = gr.Slider(
                    minimum=0,
                    maximum=0.5,
                    label="Protect Voiceless Consonants",
                    info="Safeguard distinct consonants and breathing sounds to prevent electric / buzz, tearing and other artifacts. \n Setting it to its max value of 0.5 offers comprehensive protection.\n ***Generally speaking, higher it is potentially lower the index accuracy.***",
                    value=0.33,
                    interactive=True,
                )
                preset_dropdown.change(
                    update_sliders,
                    inputs=preset_dropdown,
                    outputs=[
                        pitch,
                        filter_radius,
                        index_rate,
                        rms_mix_rate,
                        protect,
                    ],
                )
                export_button.click(
                    export_presets_button,
                    inputs=[
                        preset_name_input,
                        pitch,
                        filter_radius,
                        index_rate,
                        rms_mix_rate,
                        protect,
                    ],
                )
                f0_method = gr.Radio(
                    label="Pitch extraction algorithm",
                    info="Pitch extraction algorithm to use for the audio conversion. The default algorithm is rmvpe, which is recommended for most cases.",
                    choices=[
                        "crepe",
                        "crepe-tiny",
                        "rmvpe",
                        "fcpe",
                    ],
                    value="rmvpe",
                    interactive=True,
                )
                embedder_model = gr.Radio(
                    label="Embedder Model",
                    info="Model used for learning speaker embedding.",
                    choices=[
                        "contentvec",
                        "spin_v1",
                        "spin_v2",
                        "custom",
                    ],
                    value="contentvec",
                    interactive=True,
                )
                with gr.Column(visible=False) as embedder_custom:
                    with gr.Accordion("Custom Embedder", open=True):
                        with gr.Row():
                            embedder_model_custom = gr.Dropdown(
                                label="Select Custom Embedder",
                                choices=refresh_embedders_folders(),
                                interactive=True,
                                allow_custom_value=True,
                            )
                            refresh_embedders_button = gr.Button("Refresh embedders")
                        folder_name_input = gr.Textbox(label="Folder Name", interactive=True)
                        with gr.Row():
                            bin_file_upload = gr.File(
                                label="Upload .bin",
                                type="filepath",
                                interactive=True,
                            )
                            config_file_upload = gr.File(
                                label="Upload .json",
                                type="filepath",
                                interactive=True,
                            )
                        move_files_button = gr.Button("Move files to custom embedder folder")

                f0_file = gr.File(
                    label="The f0 curve represents the variations in the base frequency of a voice over time, showing how pitch rises and falls.",
                    visible=True,
                )

        def enforce_terms(terms_accepted, *args):
            if not terms_accepted:
                message = "You must agree to the Terms of Use to proceed."
                gr.Info(message)
                return message, None
            return run_infer_script(*args)

        def enforce_terms_batch(terms_accepted, *args):
            if not terms_accepted:
                message = "You must agree to the Terms of Use to proceed."
                gr.Info(message)
                return message, None
            return run_batch_infer_script(*args)

        terms_checkbox = gr.Checkbox(
            label="I agree to the terms of use",
            info="Please ensure compliance with the terms and conditions detailed in [this document](https://github.com/codename0og/codename-rvc-fork-3/blob/main/TERMS_OF_USE.md) before proceeding with your inference.",
            value=False,
            interactive=True,
        )

        convert_button1 = gr.Button("Convert")

        with gr.Row():
            vc_output1 = gr.Textbox(
                label="Output Information",
                info="The output information will be displayed here.",
            )
            vc_output2 = gr.Audio("Export Audio")

    # Batch inference tab
    with gr.Tab("Batch"):
        with gr.Row():
            with gr.Column():
                input_folder_batch = gr.Textbox(
                    label="Input Folder",
                    info="Select the folder containing the audios to convert.",
                    placeholder="Enter input path",
                    value=os.path.join(now_dir, "assets", "audios"),
                    interactive=True,
                )
                output_folder_batch = gr.Textbox(
                    label="Output Folder",
                    info="Select the folder where the output audios will be saved.",
                    placeholder="Enter output path",
                    value=os.path.join(now_dir, "assets", "audios"),
                    interactive=True,
                )
        with gr.Accordion("Advanced Settings", open=False):
            with gr.Column():
                clear_outputs_batch = gr.Button("Clear Outputs (Deletes all audios in assets/audios)")
                export_format_batch = gr.Radio(
                    label="Export Format",
                    info="Select the format to export the audio.",
                    choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
                    value="WAV",
                    interactive=True,
                )
                sid_batch = gr.Dropdown(
                    label="Speaker ID",
                    info="Select the speaker ID to use for the conversion.",
                    choices=get_speakers_id(model_file.value),
                    value=0,
                    interactive=True,
                )
                split_audio_batch = gr.Checkbox(
                    label="Split Audio",
                    info="Split the audio into chunks for inference to obtain better results in some cases.",
                    visible=True,
                    value=False,
                    interactive=True,
                )
                autotune_batch = gr.Checkbox(
                    label="Autotune",
                    info="Apply a soft autotune to your inferences, recommended for singing conversions.",
                    visible=True,
                    value=False,
                    interactive=True,
                )
                autotune_strength_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Autotune Strength",
                    info="Set the autotune strength - the more you increase it the more it will snap to the chromatic grid.",
                    visible=False,
                    value=1,
                    interactive=True,
                )
                clean_audio_batch = gr.Checkbox(
                    label="Clean Audio",
                    info="Clean your audio output using noise detection algorithms, recommended for speaking audios.",
                    visible=True,
                    value=False,
                    interactive=True,
                )
                clean_strength_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Clean Strength",
                    info="Set the clean-up level to the audio you want, the more you increase it the more it will clean up, but it is possible that the audio will be more compressed.",
                    visible=False,
                    value=0.5,
                    interactive=True,
                )
                formant_shifting_batch = gr.Checkbox(
                    label="Formant Shifting",
                    info="Enable formant shifting. Used for male to female and vice-versa convertions.",
                    value=False,
                    visible=True,
                    interactive=True,
                )
                post_process_batch = gr.Checkbox(
                    label="Post-Process",
                    info="Post-process the audio to apply effects to the output.",
                    value=False,
                    interactive=True,
                )
                with gr.Row(visible=False) as formant_row_batch:
                    formant_preset_batch = gr.Dropdown(
                        label="Browse presets for formanting",
                        info="Presets are located in /assets/formant_shift folder",
                        choices=list_json_files(FORMANTSHIFT_DIR),
                        visible=False,
                        interactive=True,
                    )
                    formant_refresh_button_batch = gr.Button(
                        value="Refresh",
                        visible=False,
                    )
                formant_qfrency_batch = gr.Slider(
                    value=1.0,
                    info="Default value is 1.0",
                    label="Quefrency for formant shifting",
                    minimum=0.0,
                    maximum=16.0,
                    step=0.1,
                    visible=False,
                    interactive=True,
                )
                formant_timbre_batch = gr.Slider(
                    value=1.0,
                    info="Default value is 1.0",
                    label="Timbre for formant shifting",
                    minimum=0.0,
                    maximum=16.0,
                    step=0.1,
                    visible=False,
                    interactive=True,
                )
                reverb_batch = gr.Checkbox(
                    label="Reverb",
                    info="Apply reverb to the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                reverb_room_size_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Reverb Room Size",
                    info="Set the room size of the reverb.",
                    value=0.5,
                    interactive=True,
                    visible=False,
                )

                reverb_damping_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Reverb Damping",
                    info="Set the damping of the reverb.",
                    value=0.5,
                    interactive=True,
                    visible=False,
                )

                reverb_wet_gain_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Reverb Wet Gain",
                    info="Set the wet gain of the reverb.",
                    value=0.33,
                    interactive=True,
                    visible=False,
                )

                reverb_dry_gain_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Reverb Dry Gain",
                    info="Set the dry gain of the reverb.",
                    value=0.4,
                    interactive=True,
                    visible=False,
                )

                reverb_width_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Reverb Width",
                    info="Set the width of the reverb.",
                    value=1.0,
                    interactive=True,
                    visible=False,
                )

                reverb_freeze_mode_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Reverb Freeze Mode",
                    info="Set the freeze mode of the reverb.",
                    value=0.0,
                    interactive=True,
                    visible=False,
                )
                pitch_shift_batch = gr.Checkbox(
                    label="Pitch Shift",
                    info="Apply pitch shift to the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                pitch_shift_semitones_batch = gr.Slider(
                    minimum=-12,
                    maximum=12,
                    label="Pitch Shift Semitones",
                    info="Set the pitch shift semitones.",
                    value=0,
                    interactive=True,
                    visible=False,
                )
                limiter_batch = gr.Checkbox(
                    label="Limiter",
                    info="Apply limiter to the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                limiter_threshold_batch = gr.Slider(
                    minimum=-60,
                    maximum=0,
                    label="Limiter Threshold dB",
                    info="Set the limiter threshold dB.",
                    value=-6,
                    interactive=True,
                    visible=False,
                )

                limiter_release_time_batch = gr.Slider(
                    minimum=0.01,
                    maximum=1,
                    label="Limiter Release Time",
                    info="Set the limiter release time.",
                    value=0.05,
                    interactive=True,
                    visible=False,
                )
                gain_batch = gr.Checkbox(
                    label="Gain",
                    info="Apply gain to the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                gain_db_batch = gr.Slider(
                    minimum=-60,
                    maximum=60,
                    label="Gain dB",
                    info="Set the gain dB.",
                    value=0,
                    interactive=True,
                    visible=False,
                )
                distortion_batch = gr.Checkbox(
                    label="Distortion",
                    info="Apply distortion to the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                distortion_gain_batch = gr.Slider(
                    minimum=-60,
                    maximum=60,
                    label="Distortion Gain",
                    info="Set the distortion gain.",
                    value=25,
                    interactive=True,
                    visible=False,
                )
                chorus_batch = gr.Checkbox(
                    label="chorus",
                    info="Apply chorus to the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                chorus_rate_batch = gr.Slider(
                    minimum=0,
                    maximum=100,
                    label="Chorus Rate Hz",
                    info="Set the chorus rate Hz.",
                    value=1.0,
                    interactive=True,
                    visible=False,
                )

                chorus_depth_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="chorus Depth",
                    info="Set the chorus depth.",
                    value=0.25,
                    interactive=True,
                    visible=False,
                )

                chorus_center_delay_batch = gr.Slider(
                    minimum=7,
                    maximum=8,
                    label="chorus Center Delay ms",
                    info="Set the chorus center delay ms.",
                    value=7,
                    interactive=True,
                    visible=False,
                )

                chorus_feedback_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="chorus Feedback",
                    info="Set the chorus feedback.",
                    value=0.0,
                    interactive=True,
                    visible=False,
                )

                chorus_mix_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Chorus Mix",
                    info="Set the chorus mix.",
                    value=0.5,
                    interactive=True,
                    visible=False,
                )
                bitcrush_batch = gr.Checkbox(
                    label="Bitcrush",
                    info="Apply bitcrush to the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                bitcrush_bit_depth_batch = gr.Slider(
                    minimum=1,
                    maximum=32,
                    label="Bitcrush Bit Depth",
                    info="Set the bitcrush bit depth.",
                    value=8,
                    interactive=True,
                    visible=False,
                )
                clipping_batch = gr.Checkbox(
                    label="Clipping",
                    info="Apply clipping to the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                clipping_threshold_batch = gr.Slider(
                    minimum=-60,
                    maximum=0,
                    label="Clipping Threshold",
                    info="Set the clipping threshold.",
                    value=-6,
                    interactive=True,
                    visible=False,
                )
                compressor_batch = gr.Checkbox(
                    label="Compressor",
                    info="Apply compressor to the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                compressor_threshold_batch = gr.Slider(
                    minimum=-60,
                    maximum=0,
                    label="Compressor Threshold dB",
                    info="Set the compressor threshold dB.",
                    value=0,
                    interactive=True,
                    visible=False,
                )

                compressor_ratio_batch = gr.Slider(
                    minimum=1,
                    maximum=20,
                    label="Compressor Ratio",
                    info="Set the compressor ratio.",
                    value=1,
                    interactive=True,
                    visible=False,
                )

                compressor_attack_batch = gr.Slider(
                    minimum=0.0,
                    maximum=100,
                    label="Compressor Attack ms",
                    info="Set the compressor attack ms.",
                    value=1.0,
                    interactive=True,
                    visible=False,
                )

                compressor_release_batch = gr.Slider(
                    minimum=0.01,
                    maximum=100,
                    label="Compressor Release ms",
                    info="Set the compressor release ms.",
                    value=100,
                    interactive=True,
                    visible=False,
                )
                delay_batch = gr.Checkbox(
                    label="Delay",
                    info="Apply delay to the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                delay_seconds_batch = gr.Slider(
                    minimum=0.0,
                    maximum=5.0,
                    label="Delay Seconds",
                    info="Set the delay seconds.",
                    value=0.5,
                    interactive=True,
                    visible=False,
                )

                delay_feedback_batch = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    label="Delay Feedback",
                    info="Set the delay feedback.",
                    value=0.0,
                    interactive=True,
                    visible=False,
                )

                delay_mix_batch = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    label="Delay Mix",
                    info="Set the delay mix.",
                    value=0.5,
                    interactive=True,
                    visible=False,
                )
                pitch_batch = gr.Slider(
                    minimum=-24,
                    maximum=24,
                    step=1,
                    label="Pitch",
                    info="Set the pitch of the audio, the higher the value, the higher the pitch.",
                    value=0,
                    interactive=True,
                )
                filter_radius_batch = gr.Slider(
                    minimum=0,
                    maximum=7,
                    label="Filter Radius",
                    info="If the number is greater than or equal to three, employing median filtering on the collected tone results has the potential to decrease respiration.",
                    value=3,
                    step=1,
                    interactive=False,
                    visible=False,
                )
                index_rate_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Search Feature Ratio",
                    info="Influence exerted by the index file; a higher value corresponds to greater influence. However, opting for lower values can help mitigate artifacts present in the audio.",
                    value=0.5,
                    interactive=True,
                )
                rms_mix_rate_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Volume Envelope",
                    info="Substitute or blend with the volume envelope of the output. The closer the ratio is to 1, the more the output envelope is employed.",
                    value=1,
                    interactive=True,
                )
                protect_batch = gr.Slider(
                    minimum=0,
                    maximum=0.5,
                    label="Protect Voiceless Consonants",
                    info="Safeguard distinct consonants and breathing sounds to prevent electro-acoustic tearing and other artifacts. Pulling the parameter to its maximum value of 0.5 offers comprehensive protection. However, reducing this value might decrease the extent of protection while potentially mitigating the indexing effect.",
                    value=0.3,
                    interactive=True,
                )
                preset_dropdown.change(
                    update_sliders,
                    inputs=preset_dropdown,
                    outputs=[
                        pitch_batch,
                        filter_radius_batch,
                        index_rate_batch,
                        rms_mix_rate_batch,
                        protect_batch,
                    ],
                )
                export_button.click(
                    export_presets_button,
                    inputs=[
                        preset_name_input,
                        pitch,
                        filter_radius,
                        index_rate,
                        rms_mix_rate,
                        protect,
                    ],
                    outputs=[],
                )
                f0_method_batch = gr.Radio(
                    label="Pitch extraction algorithm",
                    info="Pitch extraction algorithm to use for the audio conversion. The default algorithm is rmvpe, which is ***recommended for most cases.***",
                    choices=[
                        "crepe",
                        "crepe-tiny",
                        "rmvpe",
                        "fcpe",
                    ],
                    value="rmvpe",
                    interactive=True,
                )
                embedder_model_batch = gr.Radio(
                    label="Embedder Model",
                    info="Model used for learning speaker embedding.",
                    choices=[
                        "contentvec",
                        "spin_v1",
                        "spin_v2",
                        "chinese-hubert-base",
                        "japanese-hubert-base",
                        "korean-hubert-base",
                        "custom",
                    ],
                    value="contentvec",
                    interactive=True,
                )
                f0_file_batch = gr.File(
                    label="The f0 curve represents the variations in the base frequency of a voice over time, showing how pitch rises and falls.",
                    visible=True,
                )
                with gr.Column(visible=False) as embedder_custom_batch:
                    with gr.Accordion("Custom Embedder", open=True):
                        with gr.Row():
                            embedder_model_custom_batch = gr.Dropdown(
                                label="Select Custom Embedder",
                                choices=refresh_embedders_folders(),
                                interactive=True,
                                allow_custom_value=True,
                            )
                            refresh_embedders_button_batch = gr.Button("Refresh embedders")
                        folder_name_input_batch = gr.Textbox(
                            label="Folder Name", interactive=True
                        )
                        with gr.Row():
                            bin_file_upload_batch = gr.File(
                                label="Upload .bin",
                                type="filepath",
                                interactive=True,
                            )
                            config_file_upload_batch = gr.File(
                                label="Upload .json",
                                type="filepath",
                                interactive=True,
                            )
                        move_files_button_batch = gr.Button("Move files to custom embedder folder")

        terms_checkbox_batch = gr.Checkbox(
            label="I agree to the terms of use",
            info="Please ensure compliance with the terms and conditions detailed in [this document](https://github.com/IAHispano/Applio/blob/main/TERMS_OF_USE.md) before proceeding with your inference.",
            value=False,
            interactive=True,
        )
        convert_button_batch = gr.Button("Convert")
        stop_button = gr.Button("Stop convert", visible=False)
        stop_button.click(fn=stop_infer, inputs=[], outputs=[])

        with gr.Row():
            vc_output3 = gr.Textbox(
                label="Output Information",
                info="The output information will be displayed here.",
            )

    def toggle_visible(checkbox):
        return {"visible": checkbox, "__type__": "update"}

    def toggle_visible_embedder_custom(embedder_model):
        if embedder_model == "custom":
            return {"visible": True, "__type__": "update"}
        return {"visible": False, "__type__": "update"}

    def enable_stop_convert_button():
        return {"visible": False, "__type__": "update"}, {
            "visible": True,
            "__type__": "update",
        }

    def disable_stop_convert_button():
        return {"visible": True, "__type__": "update"}, {
            "visible": False,
            "__type__": "update",
        }

    def toggle_visible_formant_shifting(checkbox):
        if checkbox:
            return (
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
            )
        else:
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )

    def update_visibility(checkbox, count):
        return [gr.update(visible=checkbox) for _ in range(count)]

    def post_process_visible(checkbox):
        return update_visibility(checkbox, 10)

    def reverb_visible(checkbox):
        return update_visibility(checkbox, 6)

    def limiter_visible(checkbox):
        return update_visibility(checkbox, 2)

    def chorus_visible(checkbox):
        return update_visibility(checkbox, 6)

    def bitcrush_visible(checkbox):
        return update_visibility(checkbox, 1)

    def compress_visible(checkbox):
        return update_visibility(checkbox, 4)

    def delay_visible(checkbox):
        return update_visibility(checkbox, 3)

    model_file.change(
        fn=on_model_change,
        inputs=[model_file],
        outputs=[index_file, sid],
    )

    index_file.change(
        fn=sync_speaker_id,
        inputs=[model_file, index_file],
        outputs=[sid],
    )
    autotune.change(
        fn=toggle_visible,
        inputs=[autotune],
        outputs=[autotune_strength],
    )
    clean_audio.change(
        fn=toggle_visible,
        inputs=[clean_audio],
        outputs=[clean_strength],
    )
    formant_shifting.change(
        fn=toggle_visible_formant_shifting,
        inputs=[formant_shifting],
        outputs=[
            formant_row,
            formant_preset,
            formant_refresh_button,
            formant_qfrency,
            formant_timbre,
        ],
    )
    formant_shifting_batch.change(
        fn=toggle_visible_formant_shifting,
        inputs=[formant_shifting],
        outputs=[
            formant_row_batch,
            formant_preset_batch,
            formant_refresh_button_batch,
            formant_qfrency_batch,
            formant_timbre_batch,
        ],
    )
    formant_refresh_button.click(
        fn=refresh_formant,
        inputs=[],
        outputs=[formant_preset],
    )
    formant_preset.change(
        fn=update_sliders_formant,
        inputs=[formant_preset],
        outputs=[
            formant_qfrency,
            formant_timbre,
        ],
    )
    formant_preset_batch.change(
        fn=update_sliders_formant,
        inputs=[formant_preset_batch],
        outputs=[
            formant_qfrency,
            formant_timbre,
        ],
    )
    post_process.change(
        fn=post_process_visible,
        inputs=[post_process],
        outputs=[
            reverb,
            pitch_shift,
            limiter,
            gain,
            distortion,
            chorus,
            bitcrush,
            clipping,
            compressor,
            delay,
        ],
    )

    reverb.change(
        fn=reverb_visible,
        inputs=[reverb],
        outputs=[
            reverb_room_size,
            reverb_damping,
            reverb_wet_gain,
            reverb_dry_gain,
            reverb_width,
            reverb_freeze_mode,
        ],
    )
    pitch_shift.change(
        fn=toggle_visible,
        inputs=[pitch_shift],
        outputs=[pitch_shift_semitones],
    )
    limiter.change(
        fn=limiter_visible,
        inputs=[limiter],
        outputs=[limiter_threshold, limiter_release_time],
    )
    gain.change(
        fn=toggle_visible,
        inputs=[gain],
        outputs=[gain_db],
    )
    distortion.change(
        fn=toggle_visible,
        inputs=[distortion],
        outputs=[distortion_gain],
    )
    chorus.change(
        fn=chorus_visible,
        inputs=[chorus],
        outputs=[
            chorus_rate,
            chorus_depth,
            chorus_center_delay,
            chorus_feedback,
            chorus_mix,
        ],
    )
    bitcrush.change(
        fn=bitcrush_visible,
        inputs=[bitcrush],
        outputs=[bitcrush_bit_depth],
    )
    clipping.change(
        fn=toggle_visible,
        inputs=[clipping],
        outputs=[clipping_threshold],
    )
    compressor.change(
        fn=compress_visible,
        inputs=[compressor],
        outputs=[
            compressor_threshold,
            compressor_ratio,
            compressor_attack,
            compressor_release,
        ],
    )
    delay.change(
        fn=delay_visible,
        inputs=[delay],
        outputs=[delay_seconds, delay_feedback, delay_mix],
    )
    post_process_batch.change(
        fn=post_process_visible,
        inputs=[post_process_batch],
        outputs=[
            reverb_batch,
            pitch_shift_batch,
            limiter_batch,
            gain_batch,
            distortion_batch,
            chorus_batch,
            bitcrush_batch,
            clipping_batch,
            compressor_batch,
            delay_batch,
        ],
    )

    reverb_batch.change(
        fn=reverb_visible,
        inputs=[reverb_batch],
        outputs=[
            reverb_room_size_batch,
            reverb_damping_batch,
            reverb_wet_gain_batch,
            reverb_dry_gain_batch,
            reverb_width_batch,
            reverb_freeze_mode_batch,
        ],
    )
    pitch_shift_batch.change(
        fn=toggle_visible,
        inputs=[pitch_shift_batch],
        outputs=[pitch_shift_semitones_batch],
    )
    limiter_batch.change(
        fn=limiter_visible,
        inputs=[limiter_batch],
        outputs=[limiter_threshold_batch, limiter_release_time_batch],
    )
    gain_batch.change(
        fn=toggle_visible,
        inputs=[gain_batch],
        outputs=[gain_db_batch],
    )
    distortion_batch.change(
        fn=toggle_visible,
        inputs=[distortion_batch],
        outputs=[distortion_gain_batch],
    )
    chorus_batch.change(
        fn=chorus_visible,
        inputs=[chorus_batch],
        outputs=[
            chorus_rate_batch,
            chorus_depth_batch,
            chorus_center_delay_batch,
            chorus_feedback_batch,
            chorus_mix_batch,
        ],
    )
    bitcrush_batch.change(
        fn=bitcrush_visible,
        inputs=[bitcrush_batch],
        outputs=[bitcrush_bit_depth_batch],
    )
    clipping_batch.change(
        fn=toggle_visible,
        inputs=[clipping_batch],
        outputs=[clipping_threshold_batch],
    )
    compressor_batch.change(
        fn=compress_visible,
        inputs=[compressor_batch],
        outputs=[
            compressor_threshold_batch,
            compressor_ratio_batch,
            compressor_attack_batch,
            compressor_release_batch,
        ],
    )
    delay_batch.change(
        fn=delay_visible,
        inputs=[delay_batch],
        outputs=[delay_seconds_batch, delay_feedback_batch, delay_mix_batch],
    )
    autotune_batch.change(
        fn=toggle_visible,
        inputs=[autotune_batch],
        outputs=[autotune_strength_batch],
    )
    clean_audio_batch.change(
        fn=toggle_visible,
        inputs=[clean_audio_batch],
        outputs=[clean_strength_batch],
    )
    refresh_button.click(
        fn=change_choices,
        inputs=[model_file],
        outputs=[model_file, index_file, audio, sid, sid_batch],
    )
    audio.change(
        fn=output_path_fn,
        inputs=[audio],
        outputs=[output_path],
    )
    upload_audio.upload(
        fn=save_to_wav2,
        inputs=[upload_audio],
        outputs=[audio, output_path],
    )
    upload_audio.stop_recording(
        fn=save_to_wav,
        inputs=[upload_audio],
        outputs=[audio, output_path],
    )
    clear_outputs_infer.click(
        fn=delete_outputs,
        inputs=[],
        outputs=[],
    )
    clear_outputs_batch.click(
        fn=delete_outputs,
        inputs=[],
        outputs=[],
    )
    embedder_model.change(
        fn=toggle_visible_embedder_custom,
        inputs=[embedder_model],
        outputs=[embedder_custom],
    )
    embedder_model_batch.change(
        fn=toggle_visible_embedder_custom,
        inputs=[embedder_model_batch],
        outputs=[embedder_custom_batch],
    )
    move_files_button.click(
        fn=create_folder_and_move_files,
        inputs=[folder_name_input, bin_file_upload, config_file_upload],
        outputs=[],
    )
    refresh_embedders_button.click(
        fn=lambda: gr.update(choices=refresh_embedders_folders()),
        inputs=[],
        outputs=[embedder_model_custom],
    )
    move_files_button_batch.click(
        fn=create_folder_and_move_files,
        inputs=[
            folder_name_input_batch,
            bin_file_upload_batch,
            config_file_upload_batch,
        ],
        outputs=[],
    )
    refresh_embedders_button_batch.click(
        fn=lambda: gr.update(choices=refresh_embedders_folders()),
        inputs=[],
        outputs=[embedder_model_custom_batch],
    )
    convert_button1.click(
        fn=enforce_terms,
        inputs=[
            terms_checkbox,
            pitch,
            filter_radius,
            index_rate,
            rms_mix_rate,
            protect,
            f0_method,
            audio,
            output_path,
            model_file,
            index_file,
            split_audio,
            autotune,
            autotune_strength,
            clean_audio,
            clean_strength,
            export_format,
            f0_file,
            embedder_model,
            embedder_model_custom,
            formant_shifting,
            formant_qfrency,
            formant_timbre,
            post_process,
            reverb,
            pitch_shift,
            limiter,
            gain,
            distortion,
            chorus,
            bitcrush,
            clipping,
            compressor,
            delay,
            reverb_room_size,
            reverb_damping,
            reverb_wet_gain,
            reverb_dry_gain,
            reverb_width,
            reverb_freeze_mode,
            pitch_shift_semitones,
            limiter_threshold,
            limiter_release_time,
            gain_db,
            distortion_gain,
            chorus_rate,
            chorus_depth,
            chorus_center_delay,
            chorus_feedback,
            chorus_mix,
            bitcrush_bit_depth,
            clipping_threshold,
            compressor_threshold,
            compressor_ratio,
            compressor_attack,
            compressor_release,
            delay_seconds,
            delay_feedback,
            delay_mix,
            sid,
            seed,
        ],
        outputs=[vc_output1, vc_output2],
    )
    convert_button_batch.click(
        fn=enforce_terms_batch,
        inputs=[
            terms_checkbox_batch,
            pitch_batch,
            filter_radius_batch,
            index_rate_batch,
            rms_mix_rate_batch,
            protect_batch,
            f0_method_batch,
            input_folder_batch,
            output_folder_batch,
            model_file,
            index_file,
            split_audio_batch,
            autotune_batch,
            autotune_strength_batch,
            clean_audio_batch,
            clean_strength_batch,
            export_format_batch,
            f0_file_batch,
            embedder_model_batch,
            embedder_model_custom_batch,
            formant_shifting_batch,
            formant_qfrency_batch,
            formant_timbre_batch,
            post_process_batch,
            reverb_batch,
            pitch_shift_batch,
            limiter_batch,
            gain_batch,
            distortion_batch,
            chorus_batch,
            bitcrush_batch,
            clipping_batch,
            compressor_batch,
            delay_batch,
            reverb_room_size_batch,
            reverb_damping_batch,
            reverb_wet_gain_batch,
            reverb_dry_gain_batch,
            reverb_width_batch,
            reverb_freeze_mode_batch,
            pitch_shift_semitones_batch,
            limiter_threshold_batch,
            limiter_release_time_batch,
            gain_db_batch,
            distortion_gain_batch,
            chorus_rate_batch,
            chorus_depth_batch,
            chorus_center_delay_batch,
            chorus_feedback_batch,
            chorus_mix_batch,
            bitcrush_bit_depth_batch,
            clipping_threshold_batch,
            compressor_threshold_batch,
            compressor_ratio_batch,
            compressor_attack_batch,
            compressor_release_batch,
            delay_seconds_batch,
            delay_feedback_batch,
            delay_mix_batch,
            sid_batch,
            seed,
        ],
        outputs=[vc_output3],
    )
    convert_button_batch.click(
        fn=enable_stop_convert_button,
        inputs=[],
        outputs=[convert_button_batch, stop_button],
    )
    stop_button.click(
        fn=disable_stop_convert_button,
        inputs=[],
        outputs=[convert_button_batch, stop_button],
    )
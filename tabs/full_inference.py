from core import full_inference_program, download_music
import sys, os
import gradio as gr
import regex as re
from assets.i18n.i18n import I18nAuto
import torch
import shutil
import unicodedata
import gradio as gr
from assets.i18n.i18n import I18nAuto


i18n = I18nAuto()


now_dir = os.getcwd()
sys.path.append(now_dir)


model_root = os.path.join(now_dir, "logs")
audio_root = os.path.join(now_dir, "audio_files", "original_files")


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
        file.endswith((".pth", ".onnx"))
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


vocals_model_names = [
    "Mel-Roformer by KimberleyJSN",
    "BS-Roformer by ViperX",
    "MDX23C",
]


karaoke_models_names = [
    "Mel-Roformer Karaoke by aufr33 and viperx",
    "UVR-BVE",
]


denoise_models_names = [
    "Mel-Roformer Denoise Normal by aufr33",
    "Mel-Roformer Denoise Aggressive by aufr33",
    "UVR Denoise",
]


dereverb_models_names = [
    "MDX23C DeReverb by aufr33 and jarredou",
    "UVR-Deecho-Dereverb",
    "MDX Reverb HQ by FoxJoy",
    "BS-Roformer Dereverb by anvuew",
]


deeecho_models_names = ["UVR-Deecho-Normal", "UVR-Deecho-Aggressive"]


def get_indexes():

    indexes_list = [
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(model_root_relative)
        for filename in filenames
        if filename.endswith(".index") and "trained" not in filename
    ]

    return indexes_list if indexes_list else ""


def match_index(model_file_value):
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


def output_path_fn(input_audio_path):
    original_name_without_extension = os.path.basename(input_audio_path).rsplit(".", 1)[
        0
    ]
    new_name = original_name_without_extension + "_output.wav"
    output_path = os.path.join(os.path.dirname(input_audio_path), new_name)

    return output_path


def get_number_of_gpus():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()

        return "-".join(map(str, range(num_gpus)))

    else:

        return "-"


def max_vram_gpu(gpu):

    if torch.cuda.is_available():
        gpu_properties = torch.cuda.get_device_properties(gpu)
        total_memory_gb = round(gpu_properties.total_memory / 1024 / 1024 / 1024)

        return total_memory_gb / 2

    else:

        return "0"


def format_title(title):

    formatted_title = (
        unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("utf-8")
    )

    formatted_title = re.sub(r"[\u2500-\u257F]+", "", formatted_title)
    formatted_title = re.sub(r"[^\w\s.-]", "", formatted_title)
    formatted_title = re.sub(r"\s+", "_", formatted_title)

    return formatted_title


def save_to_wav(upload_audio):

    file_path = upload_audio
    formated_name = format_title(os.path.basename(file_path))
    target_path = os.path.join(audio_root_relative, formated_name)

    if os.path.exists(target_path):
        os.remove(target_path)

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    shutil.copy(file_path, target_path)
    
    return target_path, output_path_fn(target_path)


def delete_outputs():
    gr.Info(f"Outputs cleared!")
    for root, _, files in os.walk(audio_root_relative, topdown=False):
        for name in files:
            if name.endswith(tuple(sup_audioext)) and name.__contains__("_output"):
                os.remove(os.path.join(root, name))


def change_choices():
    names = [
        os.path.join(root, file)
        for root, _, files in os.walk(model_root_relative, topdown=False)
        for file in files
        if (
            file.endswith((".pth", ".onnx"))
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
    )






















def download_music_tab():
    with gr.Row():
        link = gr.Textbox(
            label=i18n("Music URL"),
            lines=1,
        )
    output = gr.Textbox(
        label=i18n("Output Information"),
        info=i18n("The output information will be displayed here."),
    )

    download = gr.Button(i18n("Download"))
    download.click(
        download_music,
        inputs=[link],
        outputs=[output],
    )


def full_inference_tab():
    default_weight = names[0] if names else None

    with gr.Row():
        with gr.Row():
            model_file = gr.Dropdown(
                label=i18n("Voice Model"),
                info=i18n("Select the voice model to use for the conversion."),
                choices=sorted(names, key=lambda path: os.path.getsize(path)),
                interactive=True,
                value=default_weight,
                allow_custom_value=True,
            )
            index_file = gr.Dropdown(
                label=i18n("Index File"),
                info=i18n("Select the index file to use for the conversion."),
                choices=get_indexes(),
                value=match_index(default_weight) if default_weight else "",
                interactive=True,
                allow_custom_value=True,
            )
        with gr.Column():
            with gr.Row():
                unload_button = gr.Button(i18n("Unload Voice"))
                refresh_button = gr.Button(i18n("Refresh"))

            unload_button.click(
                fn=lambda: (
                    {"value": "", "__type__": "update"},
                    {"value": "", "__type__": "update"},
                ),
                inputs=[],
                outputs=[model_file, index_file],
            )

            model_file.select(
                fn=lambda model_file_value: match_index(model_file_value),
                inputs=[model_file],
                outputs=[index_file],
            )

    with gr.Tab(i18n("Single")):
        with gr.Column():
            upload_audio = gr.Audio(
                label=i18n("Upload Audio"),
                type="filepath",
                editable=False,
                sources="upload",
            )

            with gr.Row():
                audio = gr.Dropdown(
                    label=i18n("Select Audio"),
                    info=i18n("Select the audio to convert."),
                    choices=sorted(audio_paths),
                    value=audio_paths[0] if audio_paths else "",
                    interactive=True,
                    allow_custom_value=True,
                )

        with gr.Accordion(i18n("Advanced Settings"), open=False):
            with gr.Accordion(i18n("RVC Settings"), open=False):
                output_path = gr.Textbox(
                    label=i18n("Output Path"),
                    placeholder=i18n("Enter output path"),
                    info=i18n(
                        "The path where the output audio will be saved, by default in audio_files/rvc/output.wav"
                    ),
                    value=os.path.join(now_dir, "audio_files", "rvc"),
                    interactive=False,
                    visible=False,
                )

                infer_backing_vocals = gr.Checkbox(
                    label=i18n("Infer Backing Vocals"),
                    info=i18n("Infer the bakcing vocals too."),
                    visible=True,
                    value=False,
                    interactive=True,
                )

                with gr.Row():
                    infer_backing_vocals_model = gr.Dropdown(
                        label=i18n("Backing Vocals Model"),
                        info=i18n(
                            "Select the backing vocals model to use for the conversion."
                        ),
                        choices=sorted(names, key=lambda path: os.path.getsize(path)),
                        interactive=True,
                        value=default_weight,
                        visible=False,
                        allow_custom_value=False,
                    )

                    infer_backing_vocals_index = gr.Dropdown(
                        label=i18n("Backing Vocals Index File"),
                        info=i18n(
                            "Select the backing vocals index file to use for the conversion."
                        ),
                        choices=get_indexes(),
                        value=match_index(default_weight) if default_weight else "",
                        interactive=True,
                        visible=False,
                        allow_custom_value=True,
                    )

                    with gr.Column():

                        refresh_button_infer_backing_vocals = gr.Button(
                            i18n("Refresh"),
                            visible=False,
                        )

                        unload_button_infer_backing_vocals = gr.Button(
                            i18n("Unload Voice"),
                            visible=False,
                        )

                        unload_button_infer_backing_vocals.click(
                            fn=lambda: (
                                {"value": "", "__type__": "update"},
                                {"value": "", "__type__": "update"},
                            ),
                            inputs=[],
                            outputs=[
                                infer_backing_vocals_model,
                                infer_backing_vocals_index,
                            ],
                        )

                        infer_backing_vocals_model.select(
                            fn=lambda model_file_value: match_index(model_file_value),
                            inputs=[infer_backing_vocals_model],
                            outputs=[infer_backing_vocals_index],
                        )

                with gr.Accordion(
                    i18n("RVC Settings for Backing vocals"), open=False, visible=False
                ) as back_rvc_settings:

                    export_format_rvc_back = gr.Radio(
                        label=i18n("Export Format"),
                        info=i18n("Select the format to export the audio."),
                        choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
                        value="MP3",
                        interactive=True,
                        visible=False,
                    )

                    split_audio_back = gr.Checkbox(
                        label=i18n("Split Audio"),
                        info=i18n(
                            "Split the audio into chunks for inference to obtain better results in some cases."
                        ),
                        visible=True,
                        value=False,
                        interactive=True,
                    )

                    pitch_extract_back = gr.Radio(
                        label=i18n("Pitch Extractor"),
                        info=i18n("Pitch extract Algorith."),
                        choices=["rmvpe", "crepe", "crepe-tiny", "fcpe"],
                        value="rmvpe",
                        interactive=True,
                    )

                    hop_length_back = gr.Slider(
                        label=i18n("Hop Length"),
                        info=i18n("Hop length for pitch extraction."),
                        minimum=1,
                        maximum=512,
                        step=1,
                        value=64,
                        visible=False,
                    )

                    embedder_model_back = gr.Radio(
                        label=i18n("Embedder Model"),
                        info=i18n("Model used for learning speaker embedding."),
                        choices=[
                            "contentvec",
                            "chinese-hubert-base",
                            "japanese-hubert-base",
                            "korean-hubert-base",
                        ],
                        value="contentvec",
                        interactive=True,
                    )

                    autotune_back = gr.Checkbox(
                        label=i18n("Autotune"),
                        info=i18n(
                            "Apply a soft autotune to your inferences, recommended for singing conversions."
                        ),
                        visible=True,
                        value=False,
                        interactive=True,
                    )

                    pitch_back = gr.Slider(
                        label=i18n("Pitch"),
                        info=i18n("Adjust the pitch of the audio."),
                        minimum=-12,
                        maximum=12,
                        step=1,
                        value=0,
                        interactive=True,
                    )

                    filter_radius_back = gr.Slider(
                        minimum=0,
                        maximum=7,
                        label=i18n("Filter Radius"),
                        info=i18n(
                            "If the number is greater than or equal to three, employing median filtering on the collected tone results has the potential to decrease respiration."
                        ),
                        value=3,
                        step=1,
                        interactive=True,
                    )

                    index_rate_back = gr.Slider(
                        minimum=0,
                        maximum=1,
                        label=i18n("Search Feature Ratio"),
                        info=i18n(
                            "Influence exerted by the index file; a higher value corresponds to greater influence. However, opting for lower values can help mitigate artifacts present in the audio."
                        ),
                        value=0.75,
                        interactive=True,
                    )

                    rms_mix_rate_back = gr.Slider(
                        minimum=0,
                        maximum=1,
                        label=i18n("Volume Envelope"),
                        info=i18n(
                            "Substitute or blend with the volume envelope of the output. The closer the ratio is to 1, the more the output envelope is employed."
                        ),
                        value=0.25,
                        interactive=True,
                    )

                    protect_back = gr.Slider(
                        minimum=0,
                        maximum=0.5,
                        label=i18n("Protect Voiceless Consonants"),
                        info=i18n(
                            "Safeguard distinct consonants and breathing sounds to prevent electro-acoustic tearing and other artifacts. Pulling the parameter to its maximum value of 0.5 offers comprehensive protection. However, reducing this value might decrease the extent of protection while potentially mitigating the indexing effect."
                        ),
                        value=0.33,
                        interactive=True,
                    )

                clear_outputs_infer = gr.Button(
                    i18n("Clear Outputs (Deletes all audios in assets/audios)")
                )

                export_format_rvc = gr.Radio(
                    label=i18n("Export Format"),
                    info=i18n("Select the format to export the audio."),
                    choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
                    value="FLAC",
                    interactive=True,
                    visible=False,
                )

                split_audio = gr.Checkbox(
                    label=i18n("Split Audio"),
                    info=i18n(
                        "Split the audio into chunks for inference to obtain better results in some cases."
                    ),
                    visible=True,
                    value=False,
                    interactive=True,
                )

                pitch_extract = gr.Radio(
                    label=i18n("Pitch Extractor"),
                    info=i18n("Pitch extract Algorith."),
                    choices=["rmvpe", "crepe", "crepe-tiny", "fcpe"],
                    value="rmvpe",
                    interactive=True,
                )

                hop_length = gr.Slider(
                    label=i18n("Hop Length"),
                    info=i18n("Hop length for pitch extraction."),
                    minimum=1,
                    maximum=512,
                    step=1,
                    value=64,
                    visible=False,
                )

                embedder_model = gr.Radio(
                    label=i18n("Embedder Model"),
                    info=i18n("Model used for learning speaker embedding."),
                    choices=[
                        "contentvec",
                        "chinese-hubert-base",
                        "japanese-hubert-base",
                        "korean-hubert-base",
                    ],
                    value="contentvec",
                    interactive=True,
                )

                autotune = gr.Checkbox(
                    label=i18n("Autotune"),
                    info=i18n(
                        "Apply a soft autotune to your inferences, recommended for singing conversions."
                    ),
                    visible=True,
                    value=False,
                    interactive=True,
                )

                pitch = gr.Slider(
                    label=i18n("Pitch"),
                    info=i18n("Adjust the pitch of the audio."),
                    minimum=-12,
                    maximum=12,
                    step=1,
                    value=0,
                    interactive=True,
                )

                filter_radius = gr.Slider(
                    minimum=0,
                    maximum=7,
                    label=i18n("Filter Radius"),
                    info=i18n(
                        "If the number is greater than or equal to three, employing median filtering on the collected tone results has the potential to decrease respiration."
                    ),
                    value=3,
                    step=1,
                    interactive=True,
                )

                index_rate = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Search Feature Ratio"),
                    info=i18n(
                        "Influence exerted by the index file; a higher value corresponds to greater influence. However, opting for lower values can help mitigate artifacts present in the audio."
                    ),
                    value=0.75,
                    interactive=True,
                )

                rms_mix_rate = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Volume Envelope"),
                    info=i18n(
                        "Substitute or blend with the volume envelope of the output. The closer the ratio is to 1, the more the output envelope is employed."
                    ),
                    value=0.25,
                    interactive=True,
                )

                protect = gr.Slider(
                    minimum=0,
                    maximum=0.5,
                    label=i18n("Protect Voiceless Consonants"),
                    info=i18n(
                        "Safeguard distinct consonants and breathing sounds to prevent electro-acoustic tearing and other artifacts. Pulling the parameter to its maximum value of 0.5 offers comprehensive protection. However, reducing this value might decrease the extent of protection while potentially mitigating the indexing effect."
                    ),
                    value=0.33,
                    interactive=True,
                )

            with gr.Accordion(i18n("Audio Separation Settings"), open=False):

                use_tta = gr.Checkbox(
                    label=i18n("Use TTA"),
                    info=i18n("Use Test Time Augmentation."),
                    visible=True,
                    value=False,
                    interactive=True,
                )

                batch_size = gr.Slider(
                    minimum=1,
                    maximum=24,
                    step=1,
                    label=i18n("Batch Size"),
                    info=i18n("Set the batch size for the separation."),
                    value=1,
                    interactive=True,
                )

                vocal_model = gr.Dropdown(
                    label=i18n("Vocals Model"),
                    info=i18n("Select the vocals model to use for the separation."),
                    choices=sorted(vocals_model_names),
                    interactive=True,
                    value="Mel-Roformer by KimberleyJSN",
                    allow_custom_value=False,
                )

                karaoke_model = gr.Dropdown(
                    label=i18n("Karaoke Model"),
                    info=i18n("Select the karaoke model to use for the separation."),
                    choices=sorted(karaoke_models_names),
                    interactive=True,
                    value="Mel-Roformer Karaoke by aufr33 and viperx",
                    allow_custom_value=False,
                )

                dereverb_model = gr.Dropdown(
                    label=i18n("Dereverb Model"),
                    info=i18n("Select the dereverb model to use for the separation."),
                    choices=sorted(dereverb_models_names),
                    interactive=True,
                    value="UVR-Deecho-Dereverb",
                    allow_custom_value=False,
                )

                deecho = gr.Checkbox(
                    label=i18n("Deeecho"),
                    info=i18n("Apply deeecho to the audio."),
                    visible=True,
                    value=True,
                    interactive=True,
                )

                deeecho_model = gr.Dropdown(
                    label=i18n("Deeecho Model"),
                    info=i18n("Select the deeecho model to use for the separation."),
                    choices=sorted(deeecho_models_names),
                    interactive=True,
                    value="UVR-Deecho-Normal",
                    allow_custom_value=False,
                )

                denoise = gr.Checkbox(
                    label=i18n("Denoise"),
                    info=i18n("Apply denoise to the audio."),
                    visible=True,
                    value=False,
                    interactive=True,
                )

                denoise_model = gr.Dropdown(
                    label=i18n("Denoise Model"),
                    info=i18n("Select the denoise model to use for the separation."),
                    choices=sorted(denoise_models_names),
                    interactive=True,
                    value="Mel-Roformer Denoise Normal by aufr33",
                    allow_custom_value=False,
                    visible=False,
                )

            with gr.Accordion(i18n("Audio post-process Settings"), open=False):

                change_inst_pitch = gr.Slider(
                    label=i18n("Change Instrumental Pitch"),
                    info=i18n("Change the pitch of the instrumental."),
                    minimum=-12,
                    maximum=12,
                    step=1,
                    value=0,
                    interactive=True,
                )

                delete_audios = gr.Checkbox(
                    label=i18n("Delete Audios"),
                    info=i18n("Delete the audios after the conversion."),
                    visible=True,
                    value=False,
                    interactive=True,
                )

                reverb = gr.Checkbox(
                    label=i18n("Reverb"),
                    info=i18n("Apply reverb to the audio."),
                    visible=True,
                    value=False,
                    interactive=True,
                )

                reverb_room_size = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Reverb Room Size"),
                    info=i18n("Set the room size of the reverb."),
                    value=0.5,
                    interactive=True,
                    visible=False,
                )

                reverb_damping = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Reverb Damping"),
                    info=i18n("Set the damping of the reverb."),
                    value=0.5,
                    interactive=True,
                    visible=False,
                )

                reverb_wet_gain = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Reverb Wet Gain"),
                    info=i18n("Set the wet gain of the reverb."),
                    value=0.33,
                    interactive=True,
                    visible=False,
                )

                reverb_dry_gain = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Reverb Dry Gain"),
                    info=i18n("Set the dry gain of the reverb."),
                    value=0.4,
                    interactive=True,
                    visible=False,
                )

                reverb_width = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Reverb Width"),
                    info=i18n("Set the width of the reverb."),
                    value=1.0,
                    interactive=True,
                    visible=False,
                )

                vocals_volume = gr.Slider(
                    label=i18n("Vocals Volume"),
                    info=i18n("Adjust the volume of the vocals."),
                    minimum=-10,
                    maximum=0,
                    step=1,
                    value=-3,
                    interactive=True,
                )

                instrumentals_volume = gr.Slider(
                    label=i18n("Instrumentals Volume"),
                    info=i18n("Adjust the volume of the Instrumentals."),
                    minimum=-10,
                    maximum=0,
                    step=1,
                    value=-3,
                    interactive=True,
                )

                backing_vocals_volume = gr.Slider(
                    label=i18n("Backing Vocals Volume"),
                    info=i18n("Adjust the volume of the backing vocals."),
                    minimum=-10,
                    maximum=0,
                    step=1,
                    value=-3,
                    interactive=True,
                )

                export_format_final = gr.Radio(
                    label=i18n("Export Format"),
                    info=i18n("Select the format to export the audio."),
                    choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
                    value="FLAC",
                    interactive=True,
                )

            with gr.Accordion(i18n("Device Settings"), open=False):

                devices = gr.Textbox(
                    label=i18n("Device"),
                    info=i18n(
                        "Select the device to use for the conversion. 0 to ∞ separated by - and for CPU leave only an -"
                    ),
                    value=get_number_of_gpus(),
                    interactive=True,
                )

        with gr.Row():
            convert_button = gr.Button(i18n("Convert"))

        with gr.Row():
            vc_output1 = gr.Textbox(
                label=i18n("Output Information"),
                info=i18n("The output information will be displayed here."),
            )
            vc_output2 = gr.Audio(label=i18n("Export Audio"))

    with gr.Tab(i18n("Download Music")):

        download_music_tab()

    

    def update_dropdown_visibility(checkbox):

        return gr.update(visible=checkbox)

    def update_reverb_sliders_visibility(reverb_checked):

        return {
            reverb_room_size: gr.update(visible=reverb_checked),
            reverb_damping: gr.update(visible=reverb_checked),
            reverb_wet_gain: gr.update(visible=reverb_checked),
            reverb_dry_gain: gr.update(visible=reverb_checked),
            reverb_width: gr.update(visible=reverb_checked),
        }

    def update_visibility_infer_backing(infer_backing_vocals):

        visible = infer_backing_vocals

        return (
            {"visible": visible, "__type__": "update"},
            {"visible": visible, "__type__": "update"},
            {"visible": visible, "__type__": "update"},
            {"visible": visible, "__type__": "update"},
            {"visible": visible, "__type__": "update"},
        )

    def update_hop_length_visibility(pitch_extract_value):

        return gr.update(visible=pitch_extract_value in ["crepe", "crepe-tiny"])


    
    refresh_button.click(
        fn=change_choices,
        inputs=[],
        outputs=[model_file, index_file, audio],
    )

    refresh_button_infer_backing_vocals.click(
        fn=change_choices,
        inputs=[],
        outputs=[infer_backing_vocals_model, infer_backing_vocals_index],
    )

    upload_audio.upload(
        fn=save_to_wav,
        inputs=[upload_audio],
        outputs=[audio, output_path],
    )

    clear_outputs_infer.click(
        fn=delete_outputs,
        inputs=[],
        outputs=[],
    )

    convert_button.click(
        full_inference_program,
        inputs=[
            model_file,
            index_file,
            audio,
            output_path,
            export_format_rvc,
            split_audio,
            autotune,
            vocal_model,
            karaoke_model,
            dereverb_model,
            deecho,
            deeecho_model,
            denoise,
            denoise_model,
            reverb,
            vocals_volume,
            instrumentals_volume,
            backing_vocals_volume,
            export_format_final,
            devices,
            pitch,
            filter_radius,
            index_rate,
            rms_mix_rate,
            protect,
            pitch_extract,
            hop_length,
            reverb_room_size,
            reverb_damping,
            reverb_wet_gain,
            reverb_dry_gain,
            reverb_width,
            embedder_model,
            delete_audios,
            use_tta,
            batch_size,
            infer_backing_vocals,
            infer_backing_vocals_model,
            infer_backing_vocals_index,
            change_inst_pitch,
            pitch_back,
            filter_radius_back,
            index_rate_back,
            rms_mix_rate_back,
            protect_back,
            pitch_extract_back,
            hop_length_back,
            export_format_rvc_back,
            split_audio_back,
            autotune_back,
            embedder_model_back,
        ],
        outputs=[vc_output1, vc_output2],
    )

    deecho.change(
        fn=update_dropdown_visibility,
        inputs=deecho,
        outputs=deeecho_model,
    )

    denoise.change(
        fn=update_dropdown_visibility,
        inputs=denoise,
        outputs=denoise_model,
    )

    reverb.change(
        fn=update_reverb_sliders_visibility,
        inputs=reverb,
        outputs=[
            reverb_room_size,
            reverb_damping,
            reverb_wet_gain,
            reverb_dry_gain,
            reverb_width,
        ],
    )

    pitch_extract.change(
        fn=update_hop_length_visibility,
        inputs=pitch_extract,
        outputs=hop_length,
    )

    infer_backing_vocals.change(
        fn=update_visibility_infer_backing,
        inputs=[infer_backing_vocals],
        outputs=[
            infer_backing_vocals_model,
            infer_backing_vocals_index,
            refresh_button_infer_backing_vocals,
            unload_button_infer_backing_vocals,
            back_rvc_settings,
        ],
    )

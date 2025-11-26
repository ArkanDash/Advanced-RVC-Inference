import torch
import warnings
import sys, os
warnings.filterwarnings("ignore", category=UserWarning)
import shutil
import gradio as gr
import regex as re
import unicodedata
from pathlib import Path

now_dir = os.getcwd()
sys.path.append(now_dir)

from advanced_rvc_inference.core import full_inference_program
from advanced_rvc_inference.lib.i18n import I18nAuto

i18n = I18nAuto()

# Define path variables using Path objects for better cross-platform compatibility
# Use relative path for model_root as requested
model_root = Path("./assets/weights")
audio_root = Path(now_dir) / "assets" / "audios"
audio_root_opt = Path(now_dir) / "assets" / "audios" / "output"

# Convert to strings for compatibility with existing functions
model_root_str = str(model_root)
audio_root_str = str(audio_root)
audio_root_opt_str = str(audio_root_opt)

sup_audioext = {
    "wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", 
    "alac", "wma", "aiff", "webm", "ac3",
}

# Helper function to safely get file lists
def get_file_list(root_dir, extensions, exclude_patterns=None):
    """
    Get a list of files in a directory with specified extensions.
    
    Args:
        root_dir: Root directory to search
        extensions: Tuple/list of file extensions to include
        exclude_patterns: Optional list of patterns to exclude
        
    Returns:
        List of file paths
    """
    root_dir = Path(root_dir)
    file_list = []
    
    for root, _, files in os.walk(root_dir, topdown=False):
        for file in files:
            if file.endswith(extensions):
                if exclude_patterns and any(pattern in file for pattern in exclude_patterns):
                    continue
                file_list.append(str(Path(root) / file))
    
    return file_list

# Get model files using the relative path
names = get_file_list(
    model_root_str, 
    (".pth", ".onnx"), 
    exclude_patterns=["G_", "D_"]
)

# Get index files
indexes_list = get_file_list(
    model_root_str, 
    (".index",), 
    exclude_patterns=["trained"]
)

# Get audio files
audio_paths = get_file_list(
    audio_root_str, 
    tuple(sup_audioext), 
    exclude_patterns=["_output"]
)

# Model name lists
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
    """Get all index files"""
    return get_file_list(
        model_root_str, 
        (".index",), 
        exclude_patterns=["trained"]
    )

def match_index(model_file_value):
    """Find matching index file for a model"""
    if not model_file_value:
        return ""
        
    model_folder = os.path.dirname(model_file_value)
    model_name = os.path.basename(model_file_value)
    index_files = get_indexes()
    
    # Try to extract base model name
    pattern = r"^(.*?)_"
    match = re.match(pattern, model_name)
    
    # Look for matching index files
    for index_file in index_files:
        index_dir = os.path.dirname(index_file)
        index_name = os.path.basename(index_file)
        
        # Same directory
        if index_dir == model_folder:
            return index_file
        # Matching base name
        elif match and match.group(1) in index_name:
            return index_file
        # Exact model name in index
        elif model_name in index_name:
            return index_file
    
    return ""

def output_path_fn(input_audio_path, custom_output_path=None):
    """
    Resolve output path
    
    Args:
        input_audio_path: Path to input audio file
        custom_output_path: Optional custom output path
        
    Returns:
        Resolved output path
    """
    if custom_output_path:
        return custom_output_path
    
    # Generate default output path
    input_path = Path(input_audio_path)
    output_name = f"{input_path.stem}_output.wav"
    return str(audio_root_opt / output_name)

def enhanced_full_inference_wrapper(
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
):
    """
    Enhanced wrapper for full_inference_program with improved path handling
    """
    # Use path manager to resolve output path
    resolved_output_path = output_path_fn(audio, output_path)
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(resolved_output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Call the original function with the resolved path
        return full_inference_program(
            model_path=model_file,
            index_path=index_file,
            input_audio_path=audio,
            output_path=resolved_output_path,
            export_format_rvc=export_format_rvc,
            split_audio=split_audio,
            autotune=autotune,
            vocal_model=vocal_model,
            karaoke_model=karaoke_model,
            dereverb_model=dereverb_model,
            deecho=deecho,
            deecho_model=deeecho_model,
            denoise=denoise,
            denoise_model=denoise_model,
            reverb=reverb,
            vocals_volume=vocals_volume,
            instrumentals_volume=instrumentals_volume,
            backing_vocals_volume=backing_vocals_volume,
            export_format_final=export_format_final,
            devices=devices,
            pitch=pitch,
            filter_radius=filter_radius,
            index_rate=index_rate,
            rms_mix_rate=rms_mix_rate,
            protect=protect,
            pitch_extract=pitch_extract,
            hop_length=hop_length,  # Fixed typo from hop_lenght
            reverb_room_size=reverb_room_size,
            reverb_damping=reverb_damping,
            reverb_wet_gain=reverb_wet_gain,
            reverb_dry_gain=reverb_dry_gain,
            reverb_width=reverb_width,
            embedder_model=embedder_model,
            delete_audios=delete_audios,
            use_tta=use_tta,
            batch_size=batch_size,
            infer_backing_vocals=infer_backing_vocals,
            infer_backing_vocals_model=infer_backing_vocals_model,
            infer_backing_vocals_index=infer_backing_vocals_index,
            change_inst_pitch=change_inst_pitch,
            pitch_back=pitch_back,
            filter_radius_back=filter_radius_back,
            index_rate_back=index_rate_back,
            rms_mix_rate_back=rms_mix_rate_back,
            protect_back=protect_back,
            pitch_extract_back=pitch_extract_back,
            hop_length_back=hop_length_back,
            export_format_rvc_back=export_format_rvc_back,
            split_audio_back=split_audio_back,
            autotune_back=autotune_back,
            embedder_model_back=embedder_model_back,
        )
    except Exception as e:
        # Log error and return error message
        error_msg = f"Error during inference: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return error_msg, None

def get_number_of_gpus():
    """Get the number of available GPUs"""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        return "-".join(map(str, range(num_gpus)))
    else:
        return "-"

def max_vram_gpu(gpu):
    """Get the maximum VRAM for a GPU"""
    if torch.cuda.is_available():
        gpu_properties = torch.cuda.get_device_properties(gpu)
        total_memory_gb = round(gpu_properties.total_memory / 1024 / 1024 / 1024)
        return total_memory_gb / 2
    else:
        return "0"

def format_title(title):
    """Format a title to be filename-safe"""
    formatted_title = (
        unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("utf-8")
    )
    formatted_title = re.sub(r"[\u2500-\u257F]+", "", formatted_title)
    formatted_title = re.sub(r"[^\w\s.-]", "", formatted_title)
    formatted_title = re.sub(r"\s+", "_", formatted_title)
    return formatted_title

def save_to_wav(upload_audio):
    """Save uploaded audio to the audio directory"""
    if not upload_audio:
        return "", ""
        
    file_path = upload_audio
    formated_name = format_title(os.path.basename(file_path))
    target_path = audio_root / formated_name

    # Remove existing file if it exists
    if target_path.exists():
        os.remove(target_path)

    # Ensure directory exists
    os.makedirs(target_path.parent, exist_ok=True)
    
    # Copy file
    shutil.copy(file_path, target_path)
    
    # Return path and default output path
    return str(target_path), output_path_fn(str(target_path))

def delete_outputs():
    """Delete all output audio files"""
    gr.Info(f"Outputs cleared!")
    for root, _, files in os.walk(audio_root, topdown=False):
        for name in files:
            if name.endswith(tuple(sup_audioext)) and "_output" in name:
                os.remove(os.path.join(root, name))

def change_choices():
    """Refresh dropdown choices"""
    # Get updated file lists
    new_names = get_file_list(
        model_root_str, 
        (".pth", ".onnx"), 
        exclude_patterns=["G_", "D_"]
    )
    
    new_indexes = get_file_list(
        model_root_str, 
        (".index",), 
        exclude_patterns=["trained"]
    )
    
    new_audio_paths = get_file_list(
        audio_root_str, 
        tuple(sup_audioext), 
        exclude_patterns=["_output"]
    )
    
    return (
        {"choices": sorted(new_names), "__type__": "update"},
        {"choices": sorted(new_indexes), "__type__": "update"},
        {"choices": sorted(new_audio_paths), "__type__": "update"},
    )

def full_inference_tab():
    """Create the full inference tab UI"""
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
            refresh_button = gr.Button(i18n("Refresh"))
            unload_button = gr.Button(i18n("Unload Voice"))

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
                    value=audio_root_opt_str,
                    interactive=True,
                    visible=True,
                )
                
                infer_backing_vocals = gr.Checkbox(
                    label=i18n("Infer Backing Vocals"),
                    info=i18n("Infer the backing vocals too."),
                    visible=True,
                    value=False,
                    interactive=True,
                )
                
                # Backing vocals controls
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
                
                # Backing vocals RVC settings
                with gr.Accordion(
                    i18n("RVC Settings for Backing vocals"), open=False, visible=False
                ) as back_rvc_settings:
                    export_format_rvc_back = gr.Radio(
                        label=i18n("Export Format"),
                        info=i18n("Select the format to export the audio."),
                        choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
                        value="FLAC",
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
                    
                    pitch_extract_back = gr.Dropdown(
                        label=i18n("Pitch Extractor"),
                        info=i18n("Advanced pitch extraction algorithm (KRVC Kernel optimized)."),
                        choices=[
                            "rmvpe",
                            "mangio-crepe",
                            "mangio-crepe-tiny",
                            "crepe",
                            "crepe-tiny",
                            "mangio-dbs",
                            "fcpe",
                            "mangio-dt",
                            "pm",
                            "harvest",
                            "dio",
                            "pyin",
                            "pyworld-harvest",
                            "pyworld-dio",
                            "parselmouth",
                            "swipe",
                            "rapt",
                            "shs",
                            "mangio-swipe",
                            "mangio-rapt",
                            "mangio-shs",
                            "crepe-full",
                            "crepe-tiny-1024",
                            "crepe-tiny-2048",
                            "crepe-small",
                            "crepe-small-1024",
                            "crepe-small-2048",
                            "crepe-medium",
                            "crepe-medium-1024",
                            "crepe-medium-2048",
                            "crepe-large",
                            "crepe-large-1024",
                            "crepe-large-2048",
                            "mangio-crepe-full",
                            "mangio-crepe-tiny-1024",
                            "mangio-crepe-tiny-2048",
                            "mangio-crepe-small",
                            "mangio-crepe-small-1024",
                            "mangio-crepe-small-2048",
                            "mangio-crepe-medium",
                            "mangio-crepe-medium-1024",
                            "mangio-crepe-medium-2048",
                            "mangio-crepe-large",
                            "mangio-crepe-large-1024",
                            "mangio-crepe-large-2048",
                            "fcpe-legacy",
                            "fcpe-previous",
                            "fcpe-nvidia",
                            "rmvpe-clipping",
                            "rmvpe-medfilt",
                            "rmvpe-clipping-medfilt",
                            "harvest-clipping",
                            "harvest-medfilt",
                            "harvest-clipping-medfilt",
                            "dio-clipping",
                            "dio-medfilt",
                            "dio-clipping-medfilt",
                            "pyin-clipping",
                            "pyin-medfilt",
                            "pyin-clipping-medfilt",
                            "yin",
                            "pyyin",
                            "pyworld-yin",
                            "pyworld-reaper",
                            "pysptk-yin",
                            "reaper",
                            "pichtr",
                            "sigproc",
                            "snac",
                            "world Harvest",
                            "world Dio",
                            "pyworld-Harvest",
                            "pyworld-Dio",
                            "torch-dio",
                            "torch-harvest",
                            "torch-yin",
                            "torch-pitchshift",
                            "torch-pitchtracking",
                            "autotuned-harvest",
                            "autotuned-crepe",
                            "autotuned-fcpe",
                            "autotuned-rmvpe",
                            "mixed-harvest-crepe",
                            "mixed-crepe-fcpe",
                            "mixed-fcpe-rmvpe",
                            "hybrid[harvest+crepe]",
                            "hybrid[rmvpe+harvest]",
                            "hybrid[rmvpe+crepe]",
                            "hybrid[rmvpe+fcpe]",
                            "hybrid[harvest+fcpe]",
                            "hybrid[crepe+fcpe]",
                            "hybrid[rmvpe+harvest+crepe]",
                            "hybrid[rmvpe+harvest+fcpe]",
                            "hybrid[rmvpe+crepe+fcpe]",
                            "hybrid[mixed-all]"
                        ],
                        value="rmvpe",
                        interactive=True,
                        multiselect=False
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
                    
                    embedder_model_back = gr.Dropdown(
                        label=i18n("Embedder Model"),
                        info=i18n("Model used for learning speaker embedding (KRVC Kernel optimized)."),
                        choices=[
                            "contentvec",
                            "chinese-hubert-base",
                            "japanese-hubert-base",
                            "korean-hubert-base",
                            "vietnamese-hubert-base",
                            "spanish-hubert-base",
                            "french-hubert-base",
                            "german-hubert-base",
                            "english-hubert-base",
                            "portuguese-hubert-base",
                            "arabic-hubert-base",
                            "russian-hubert-base",
                            "italian-hubert-base",
                            "dutch-hubert-base",
                            "mandarin-hubert-base",
                            "cantonese-hubert-base",
                            "thai-hubert-base",
                            "korean-kss",
                            "korean-ksponspeech",
                            "japanese-jvs",
                            "japanese-m_ailabs",
                            "whisper-english",
                            "whisper-large-v2",
                            "whisper-large-v3",
                            "whisper-medium",
                            "whisper-small",
                            "whisper-tiny",
                            "whisper-large-v1",
                            "whisper-large-v3-turbo",
                            "hubert-base-lt",
                            "contentvec-mel",
                            "contentvec-ctc",
                            "dono-ctc",
                            "japanese-hubert-audio",
                            "ksin-melo-tts",
                            "mless-melo-tts",
                            "polish-hubert-base",
                            "spanish-wav2vec2",
                            "vocos-encodec",
                            "chinese-wav2vec2",
                            "nicht-ai-voice",
                            "multilingual-v2",
                            "multilingual-v1",
                            "speecht5",
                            "encodec_24khz",
                            "encodec_48khz",
                            "vits-universal",
                            "vits-japanese",
                            "vits-korean",
                            "vits-chinese",
                            "vits-thai",
                            "vits-vietnamese",
                            "vits-arabic",
                            "vits-russian",
                            "vits-french",
                            "vits-spanish",
                            "vits-german",
                            "vits-italian",
                            "vits-portuguese",
                            "vits-mandarin",
                            "vits-cantonese",
                            "vits-dutch",
                            "vits-polish",
                            "fairseq-v1",
                            "fairseq-v2",
                            "fairseq-w2v2",
                            "fairseq-hubert",
                            "onnx-contentvec",
                            "onnx-japanese-hubert",
                            "onnx-chinese-hubert",
                            "onnx-korean-hubert",
                            "onnx-multilingual-hubert"
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
                
                pitch_extract = gr.Dropdown(
                    label=i18n("Pitch Extractor"),
                    info=i18n("Advanced pitch extraction algorithm (KRVC Kernel optimized)."),
                    choices=[
                        "rmvpe",
                        "mangio-crepe",
                        "mangio-crepe-tiny",
                        "crepe",
                        "crepe-tiny",
                        "mangio-dbs",
                        "fcpe",
                        "mangio-dt",
                        "pm",
                        "harvest",
                        "dio",
                        "pyin",
                        "pyworld-harvest",
                        "pyworld-dio",
                        "parselmouth",
                        "swipe",
                        "rapt",
                        "shs",
                        "mangio-swipe",
                        "mangio-rapt",
                        "mangio-shs",
                        "crepe-full",
                        "crepe-tiny-1024",
                        "crepe-tiny-2048",
                        "crepe-small",
                        "crepe-small-1024",
                        "crepe-small-2048",
                        "crepe-medium",
                        "crepe-medium-1024",
                        "crepe-medium-2048",
                        "crepe-large",
                        "crepe-large-1024",
                        "crepe-large-2048",
                        "mangio-crepe-full",
                        "mangio-crepe-tiny-1024",
                        "mangio-crepe-tiny-2048",
                        "mangio-crepe-small",
                        "mangio-crepe-small-1024",
                        "mangio-crepe-small-2048",
                        "mangio-crepe-medium",
                        "mangio-crepe-medium-1024",
                        "mangio-crepe-medium-2048",
                        "mangio-crepe-large",
                        "mangio-crepe-large-1024",
                        "mangio-crepe-large-2048",
                        "fcpe-legacy",
                        "fcpe-previous",
                        "fcpe-nvidia",
                        "rmvpe-clipping",
                        "rmvpe-medfilt",
                        "rmvpe-clipping-medfilt",
                        "harvest-clipping",
                        "harvest-medfilt",
                        "harvest-clipping-medfilt",
                        "dio-clipping",
                        "dio-medfilt",
                        "dio-clipping-medfilt",
                        "pyin-clipping",
                        "pyin-medfilt",
                        "pyin-clipping-medfilt",
                        "yin",
                        "pyyin",
                        "pyworld-yin",
                        "pyworld-reaper",
                        "pysptk-yin",
                        "reaper",
                        "pichtr",
                        "sigproc",
                        "snac",
                        "world Harvest",
                        "world Dio",
                        "pyworld-Harvest",
                        "pyworld-Dio",
                        "torch-dio",
                        "torch-harvest",
                        "torch-yin",
                        "torch-pitchshift",
                        "torch-pitchtracking",
                        "autotuned-harvest",
                        "autotuned-crepe",
                        "autotuned-fcpe",
                        "autotuned-rmvpe",
                        "mixed-harvest-crepe",
                        "mixed-crepe-fcpe",
                        "mixed-fcpe-rmvpe",
                        "hybrid[harvest+crepe]",
                        "hybrid[rmvpe+harvest]",
                        "hybrid[rmvpe+crepe]",
                        "hybrid[rmvpe+fcpe]",
                        "hybrid[harvest+fcpe]",
                        "hybrid[crepe+fcpe]",
                        "hybrid[rmvpe+harvest+crepe]",
                        "hybrid[rmvpe+harvest+fcpe]",
                        "hybrid[rmvpe+crepe+fcpe]",
                        "hybrid[mixed-all]"
                    ],
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
                
                embedder_model = gr.Dropdown(
                    label=i18n("Embedder Model"),
                    info=i18n("Model used for learning speaker embedding (KRVC Kernel optimized)."),
                    choices=[
                        "contentvec",
                        "chinese-hubert-base",
                        "japanese-hubert-base",
                        "korean-hubert-base",
                        "vietnamese-hubert-base",
                        "spanish-hubert-base",
                        "french-hubert-base",
                        "german-hubert-base",
                        "english-hubert-base",
                        "portuguese-hubert-base",
                        "arabic-hubert-base",
                        "russian-hubert-base",
                        "italian-hubert-base",
                        "dutch-hubert-base",
                        "mandarin-hubert-base",
                        "cantonese-hubert-base",
                        "thai-hubert-base",
                        "korean-kss",
                        "korean-ksponspeech",
                        "japanese-jvs",
                        "japanese-m_ailabs",
                        "whisper-english",
                        "whisper-large-v2",
                        "whisper-large-v3",
                        "whisper-medium",
                        "whisper-small",
                        "whisper-tiny",
                        "whisper-large-v1",
                        "whisper-large-v3-turbo",
                        "hubert-base-lt",
                        "contentvec-mel",
                        "contentvec-ctc",
                        "dono-ctc",
                        "japanese-hubert-audio",
                        "ksin-melo-tts",
                        "mless-melo-tts",
                        "polish-hubert-base",
                        "spanish-wav2vec2",
                        "vocos-encodec",
                        "chinese-wav2vec2",
                        "nicht-ai-voice",
                        "multilingual-v2",
                        "multilingual-v1",
                        "speecht5",
                        "encodec_24khz",
                        "encodec_48khz",
                        "vits-universal",
                        "vits-japanese",
                        "vits-korean",
                        "vits-chinese",
                        "vits-thai",
                        "vits-vietnamese",
                        "vits-arabic",
                        "vits-russian",
                        "vits-french",
                        "vits-spanish",
                        "vits-german",
                        "vits-italian",
                        "vits-portuguese",
                        "vits-mandarin",
                        "vits-cantonese",
                        "vits-dutch",
                        "vits-polish",
                        "fairseq-v1",
                        "fairseq-v2",
                        "fairseq-w2v2",
                        "fairseq-hubert",
                        "onnx-contentvec",
                        "onnx-japanese-hubert",
                        "onnx-chinese-hubert",
                        "onnx-korean-hubert",
                        "onnx-multilingual-hubert"
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
                    value=True,
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

    # Event handlers
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
        enhanced_full_inference_wrapper,
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

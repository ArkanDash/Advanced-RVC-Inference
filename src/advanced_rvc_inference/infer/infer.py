import asyncio
import gc
import os

import edge_tts
import gradio as gr
import torch
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from fairseq.data.dictionary import Dictionary
from pydub import AudioSegment
from scipy.io import wavfile

from rvc.infer.config import Config
from rvc.infer.pipeline import VC
from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.lib.my_utils import load_audio

# Define paths to folders and files (constants)
RVC_MODELS_DIR = os.path.join(os.getcwd(), "models", "RVC_models")
OUTPUT_DIR = os.path.join(os.getcwd(), "output", "RVC_output")
HUBERT_BASE_PATH = os.path.join(os.getcwd(), "rvc", "models", "embedders", "hubert_base.pt")

# Create directories if they don't exist
os.makedirs(RVC_MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize configuration
config = Config()


# Displays task execution progress
def display_progress(percent, message, is_print, progress=gr.Progress()):
    if is_print:
        print(message)
    progress(percent, desc=message)


# Loads the RVC model and index by model name
def load_rvc_model(rvc_model):
    # Construct the path to the model directory
    model_dir = os.path.join(RVC_MODELS_DIR, rvc_model)
    # Get the list of files in the model directory
    model_files = os.listdir(model_dir)

    # Find the model file with .pth extension
    rvc_model_path = next((os.path.join(model_dir, f) for f in model_files if f.endswith(".pth")), None)
    # Find the index file with .index extension
    rvc_index_path = next((os.path.join(model_dir, f) for f in model_files if f.endswith(".index")), None)

    # Check if the model file exists
    if not rvc_model_path:
        raise ValueError(
            f"\033[91mERROR!\033[0m Model {rvc_model} not found. You may have made a mistake in the model name or provided an incorrect link during installation."
        )

    return rvc_model_path, rvc_index_path


# Loads the Hubert model
def load_hubert(model_path):
    torch.serialization.add_safe_globals([Dictionary])
    model, _, _ = load_model_ensemble_and_task([model_path], suffix="")
    hubert = model[0].to(config.device).float()
    hubert.eval()
    return hubert


# Gets the voice converter
def get_vc(model_path):
    # Load the model state from the file
    cpt = torch.load(model_path, map_location="cpu", weights_only=True)

    # Check the model format validity
    if "config" not in cpt or "weight" not in cpt:
        raise ValueError(f"Invalid format for {model_path}. Use a voice model trained on RVC v2.")

    # Extract model parameters
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]

    use_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    vocoder = cpt.get("vocoder", "HiFi-GAN")
    input_dim = 768 if version == "v2" else 256

    # Initialize the synthesizer
    net_g = Synthesizer(*cpt["config"], use_f0=use_f0, text_enc_hidden_dim=input_dim, vocoder=vocoder)

    # Remove unnecessary layer
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g = net_g.to(config.device).float()
    net_g.eval()

    # Initialize the voice converter object
    vc = VC(tgt_sr, config)
    return cpt, version, net_g, tgt_sr, vc, use_f0


# Converts the file to the user-selected format
def convert_audio(input_audio, output_audio, output_format):
    # Load the audio file
    audio = AudioSegment.from_file(input_audio)
    # Save the audio file in the selected format
    audio.export(output_audio, format=output_format)


# Synthesizes text to speech using edge_tts
async def text_to_speech(voice, text, rate, volume, pitch, output_path):
    if not -100 <= rate <= 100:
        raise ValueError("Rate must be in the range of -100% to +100%")
    if not -100 <= volume <= 100:
        raise ValueError("Volume must be in the range of -100% to +100%")
    if not -100 <= pitch <= 100:
        raise ValueError("Pitch must be in the range of -100Hz to +100Hz")

    rate = f"+{rate}%" if rate >= 0 else f"{rate}%"
    volume = f"+{volume}%" if volume >= 0 else f"{volume}%"
    pitch = f"+{pitch}Hz" if pitch >= 0 else f"{pitch}Hz"

    communicate = edge_tts.Communicate(voice=voice, text=text, rate=rate, volume=volume, pitch=pitch)
    await communicate.save(output_path)


# Performs inference using RVC
def rvc_infer(
    rvc_model=None,
    input_path=None,
    f0_method="rmvpe",
    f0_min=50,
    f0_max=1100,
    hop_length=128,
    rvc_pitch=0,
    protect=0.5,
    index_rate=0,
    volume_envelope=1,
    autopitch=False,
    autopitch_threshold=155.0,
    autotune=False,
    autotune_strength=1.0,
    output_format="wav",
):
    if not rvc_model:
        raise ValueError("Select a voice model for conversion.")
    if not os.path.exists(input_path):
        raise ValueError(f"File '{input_path}' not found. Ensure it has been uploaded or check the correctness of the path.")

    display_progress(0, "\n[⚙️] Starting the generation pipeline...", True)

    # Load the Hubert model
    display_progress(0.1, "Loading Hubert model...", False)
    hubert_model = load_hubert(HUBERT_BASE_PATH)
    # Load the RVC model and index
    display_progress(0.2, "Loading RVC model and index...", False)
    model_path, index_path = load_rvc_model(rvc_model)
    # Get the voice converter
    display_progress(0.3, "Obtaining voice converter...", False)
    cpt, version, net_g, tgt_sr, vc, use_f0 = get_vc(model_path)

    # Construct the output file name
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    if len(base_name) > 50:
        gr.Warning("File name exceeds 50 characters and will be shortened for convenience.")
        base_name = "Made_in_PolGen"  # Change the file name if the original is longer than 50 characters
    output_path = os.path.join(OUTPUT_DIR, f"{base_name}_({rvc_model}).{output_format}")

    # Load the audio file
    display_progress(0.4, "Loading audio file...", False)
    audio = load_audio(input_path, 16000)

    display_progress(0.5, f"[🌌] Converting audio — {base_name}...", True)
    audio_opt = vc.pipeline(
        hubert_model,
        net_g,
        0,
        audio,
        0 if autopitch else rvc_pitch,
        f0_min,
        f0_max,
        f0_method,
        index_path,
        index_rate,
        use_f0,
        volume_envelope,
        version,
        protect,
        hop_length,
        autopitch,
        autopitch_threshold,
        autotune,
        autotune_strength,
    )
    # Save the file and convert it to the selected format
    display_progress(0.8, "[💫] Saving result...", True)
    wavfile.write(output_path, tgt_sr, audio_opt)
    convert_audio(output_path, output_path, output_format)

    # Free memory
    display_progress(0.9, "Freeing memory...", False)
    del hubert_model, cpt, net_g, vc
    gc.collect()
    torch.cuda.empty_cache()

    display_progress(1.0, f"[✅] Conversion completed — {output_path}", True)
    return gr.Audio(output_path, label=os.path.basename(output_path))


def rvc_edgetts_infer(
    # RVC
    rvc_model=None,
    f0_method="rmvpe",
    f0_min=50,
    f0_max=1100,
    hop_length=128,
    rvc_pitch=0,
    protect=0.5,
    index_rate=0,
    volume_envelope=1,
    autopitch=False,
    autopitch_threshold=155.0,
    autotune=False,
    autotune_strength=1.0,
    output_format="wav",
    # EdgeTTS
    tts_voice=None,
    tts_text=None,
    tts_rate=0,
    tts_volume=0,
    tts_pitch=0,
):
    if not tts_text:
        raise ValueError("Enter the required text in the input field.")
    if not tts_voice:
        raise ValueError("Select a language and voice for speech synthesis.")

    display_progress(1.0, "[🎙️] Synthesizing speech...", False)
    input_path = os.path.join(OUTPUT_DIR, "TTS_Voice.wav")
    asyncio.run(text_to_speech(tts_voice, tts_text, tts_rate, tts_volume, tts_pitch, input_path))

    output_path = rvc_infer(
        rvc_model=rvc_model,
        input_path=input_path,
        f0_method=f0_method,
        f0_min=f0_min,
        f0_max=f0_max,
        hop_length=hop_length,
        rvc_pitch=rvc_pitch,
        protect=protect,
        index_rate=index_rate,
        volume_envelope=volume_envelope,
        autopitch=autopitch,
        autopitch_threshold=autopitch_threshold,
        autotune=autotune,
        autotune_strength=autotune_strength,
        output_format=output_format,
    )

    return input_path, output_path

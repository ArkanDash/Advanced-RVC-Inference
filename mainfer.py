import argparse
import gc, os
import hashlib
import json
import shlex
import subprocess
from contextlib import suppress
from urllib.parse import urlparse, parse_qs
import gradio as gr
import librosa
import numpy as np
import soundfile as sf
from pedalboard import Pedalboard, Reverb, Compressor, HighpassFilter
from pedalboard.io import AudioFile
from pydub import AudioSegment
import yt_dlp
from rvc import Config, load_hubert, get_vc, rvc_infer
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

rvc_models_dir = os.path.join(BASE_DIR, 'weights')
output_dir = os.path.join(BASE_DIR, 'song_output')



def raise_exception(error_msg):
    raise Exception(error_msg)

def display_progress(message):
    print(message)

def get_rvc_model(voice_model):
    rvc_model_filename, rvc_index_filename = None, None
    model_dir = os.path.join(rvc_models_dir, voice_model)
    for file in os.listdir(model_dir):
        ext = os.path.splitext(file)[1]
        if ext == '.pth':
            rvc_model_filename = file
        if ext == '.index':
            rvc_index_filename = file

    if rvc_model_filename is None:
        error_msg = f'No model file exists in {model_dir}.'
        raise_exception(error_msg)

    return os.path.join(model_dir, rvc_model_filename), os.path.join(model_dir, rvc_index_filename) if rvc_index_filename else ''



def yt_download(link):
    ydl_opts = {
        'format': 'bestaudio',
        'outtmpl': '%(title)s',
        'nocheckcertificate': True,
        'ignoreerrors': True,
        'no_warnings': True,
        'quiet': True,
        'extractaudio': True,
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(link, download=True)
        download_path = ydl.prepare_filename(result, outtmpl='%(title)s.mp3')

    return download_path





def voice_change(
    voice_model, 
    vocals_path, 
    output_path, 
    pitch_change, 
    f0_method, 
    index_rate, 
    filter_radius, 
    rms_mix_rate, 
    protect, 
    crepe_hop_length,
):
    """
    Function to perform voice change using the specified RVC model.

    Args:
        voice_model (str): The name or path of the voice model to use.
        vocals_path (str): Path to the input vocals audio file.
        output_path (str): Path to save the output audio file.
        pitch_change (float): Pitch shift value.
        f0_method (str): Fundamental frequency estimation method.
        index_rate (float): Index rate for the model.
        filter_radius (int): Smoothing filter radius.
        rms_mix_rate (float): RMS mix rate for volume adjustments.
        protect (float): Protection parameter to avoid artifacts.
        crepe_hop_length (int): Hop length for the Crepe algorithm.
        is_webui (bool): Flag indicating if the function is used in a web interface.

    Returns:
        Any: Output of the RVC inference process.
    """
    # Load the RVC model paths
    rvc_model_path, rvc_index_path = get_rvc_model(voice_model)

    # Initialize device and configuration
    device = 'cuda:0'
    config = Config(device, True)

    # Load the HuBERT model
    hubert_model = load_hubert(
        device, config.is_half, os.path.join(rvc_models_dir, 'hubert_base.pt')
    )

    # Load the voice conversion model
    cpt, version, net_g, tgt_sr, vc = get_vc(
        device, config.is_half, config, rvc_model_path
    )

    try:
        # Perform voice conversion
        output = rvc_infer(
            rvc_index_path, index_rate, vocals_path, output_path, pitch_change,
            f0_method, cpt, version, net_g, filter_radius, tgt_sr, rms_mix_rate,
            protect, crepe_hop_length, vc, hubert_model
        )
    finally:
        # Ensure proper memory cleanup
        del hubert_model, cpt
        gc.collect()

    return output


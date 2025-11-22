import os
import sys
import time
import torch
import yt_dlp
import shutil
import librosa
import argparse
import warnings

import numpy as np
import soundfile as sf

from urllib.parse import urlparse
from distutils.util import strtobool

sys.path.append(os.getcwd())

from main.app.variables import config, logger, translations
from main.inference.separate_music import _separate, vr_models

dataset_temp = "dataset_temp"

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--create_dataset", action='store_true')
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--output_dirs", type=str, default="./dataset")
    parser.add_argument("--skip_seconds", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--skip_start_audios", type=str, default="0")
    parser.add_argument("--skip_end_audios", type=str, default="0")
    parser.add_argument("--separate", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--model_name", type=str, default="MDXNET_Main")
    parser.add_argument("--reverb_model", type=str, default="MDX-Reverb")
    parser.add_argument("--denoise_model", type=str, default="Normal")
    parser.add_argument("--sample_rate", type=int, default=48000)
    parser.add_argument("--shifts", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--aggression", type=int, default=5)
    parser.add_argument("--hop_length", type=int, default=1024)
    parser.add_argument("--window_size", type=int, default=512)
    parser.add_argument("--segments_size", type=int, default=256)
    parser.add_argument("--post_process_threshold", type=float, default=0.2)
    parser.add_argument("--enable_tta", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--enable_denoise", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--high_end_process", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--enable_post_process", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--separate_reverb", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--clean_dataset", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--clean_strength", type=float, default=0.7)

    return parser.parse_args()

def main():
    args = parse_arguments()
    input_data, output_dirs, skip_seconds, skip_start_audios, skip_end_audios, separate, model_name, reverb_model, denoise_model, sample_rate, shifts, batch_size, overlap, aggression, hop_length, window_size, segments_size, post_process_threshold, enable_tta, enable_denoise, high_end_process, enable_post_process, separate_reverb, clean_dataset, clean_strength = args.input_data, args.output_dirs, args.skip_seconds, args.skip_start_audios, args.skip_end_audios, args.separate, args.model_name, args.reverb_model, args.denoise_model, args.sample_rate, args.shifts, args.batch_size, args.overlap, args.aggression, args.hop_length, args.window_size, args.segments_size, args.post_process_threshold, args.enable_tta, args.enable_denoise, args.high_end_process, args.enable_post_process, args.separate_reverb, args.clean_dataset, args.clean_strength

    create_dataset(
        input_data,
        output_dirs,
        skip_seconds,
        skip_start_audios,
        skip_end_audios,
        separate,
        model_name, 
        reverb_model, 
        denoise_model,
        sample_rate,
        shifts, 
        batch_size, 
        overlap, 
        aggression,
        hop_length, 
        window_size,
        segments_size, 
        post_process_threshold,
        enable_tta,
        enable_denoise,
        high_end_process,
        enable_post_process,
        separate_reverb,
        clean_dataset,
        clean_strength
    )

def create_dataset(
    input_data,
    output_dirs,
    skip_seconds,
    skip_start_audios,
    skip_end_audios,
    separate,
    model_name, 
    reverb_model="MDX-Reverb", 
    denoise_model="Normal",
    sample_rate=48000,
    shifts=2, 
    batch_size=1, 
    overlap=0.25, 
    aggression=5,
    hop_length=1024, 
    window_size=512,
    segments_size=256, 
    post_process_threshold=0.2,
    enable_tta=False,
    enable_denoise=False,
    high_end_process=False,
    enable_post_process=False,
    separate_reverb=False,
    clean_dataset=False,
    clean_strength=0.7
):
    log_data = {
        translations['audio_path']: input_data, 
        translations['output_path']: output_dirs, 
        translations['skip']: skip_seconds,
        translations['separator_tab']: separate,
        translations['modelname']: model_name, 
        translations['dereveb_audio']: separate_reverb,
        translations['sr']: sample_rate, 
        translations['shift']: shifts, 
        translations['batch_size']: batch_size, 
        translations['overlap']: overlap, 
        translations['aggression']: aggression,
        translations['hop_length']: hop_length,
        translations['window_size']: window_size,
        translations['segments_size']: segments_size, 
        translations['post_process_threshold']: post_process_threshold,
        translations['enable_tta']: enable_tta,
        translations['denoise_mdx']: enable_denoise, 
        translations['high_end_process']: high_end_process,
        translations['enable_post_process']: enable_post_process,
        translations['clear_dataset']: clean_dataset
    }

    if clean_dataset: log_data[translations['clean_strength']] = clean_strength
    if separate_reverb: log_data[translations['dereveb_model']] = reverb_model
    if enable_denoise and model_name in list(vr_models.keys()): log_data["Denoise Model"] = denoise_model
    if skip_seconds:
        log_data[translations['skip_start']] = skip_start_audios
        log_data[translations['skip_end']] = skip_end_audios

    for key, value in log_data.items():
        logger.debug(f"{key}: {value}")

    start_time = time.time()
    inputs_data = input_data.replace(", ", ",").split(",")

    pid_path = os.path.join("assets", "create_dataset_pid.txt")
    with open(pid_path, "w") as pid_file:
        pid_file.write(str(os.getpid()))

    try:
        if os.path.exists(dataset_temp): shutil.rmtree(dataset_temp, ignore_errors=True)
        else: os.makedirs(dataset_temp, exist_ok=True)

        audio_path = [
            downloader(
                url, 
                f"audio_{str(inputs_data.index(url))}"
            ) if is_url(url) else url
            for url in inputs_data
        ]
        
        if skip_seconds:
            skip_start_audios, skip_end_audios = skip_start_audios.replace(", ", ",").split(","), skip_end_audios.replace(", ", ",").split(",")

            if len(skip_start_audios) < len(audio_path) or len(skip_end_audios) < len(audio_path): 
                logger.warning(translations["skip<audio"])
                sys.exit(1)
            elif len(skip_start_audios) > len(audio_path) or len(skip_end_audios) > len(audio_path): 
                logger.warning(translations["skip>audio"])
                sys.exit(1)
            else:
                audio_path = [
                    skip_duration(
                        audio,
                        skip_start_audio,
                        skip_end_audio
                    )
                    for audio, skip_start_audio, skip_end_audio in zip(
                        audio_path, 
                        skip_start_audios, 
                        skip_end_audios
                    )
                ]
                    
        if separate:
            audio_path = [
                separate_main(
                    audio, 
                    audio_path.index(audio),
                    model_name, 
                    sample_rate,
                    reverb_model, 
                    denoise_model,
                    shifts, 
                    batch_size, 
                    overlap, 
                    aggression,
                    hop_length, 
                    window_size,
                    segments_size, 
                    post_process_threshold,
                    enable_tta,
                    enable_denoise,
                    high_end_process,
                    enable_post_process,
                    separate_reverb
                )
                for audio in audio_path
            ]

        if clean_dataset: 
            from main.tools.noisereduce import TorchGate
            tg = TorchGate(sr, prop_decrease=clean_strength).to(config.device)
        
        for audio in audio_path:
            data, sr = read_file(audio)

            if len(data.shape) > 1: data = librosa.to_mono(data.T)
            if sr != sample_rate: data = librosa.resample(data, orig_sr=sr, target_sr=sample_rate, res_type="soxr_vhq")
            if clean_dataset: data = tg(torch.from_numpy(data).unsqueeze(0).to(config.device).float()).squeeze(0).cpu().detach().numpy()

            sf.write(audio, data, sr)
            output_path = os.path.join(output_dirs, os.path.basename(audio))

            if os.path.exists(output_path): os.remove(output_path)
            shutil.move(audio, output_path)

        if os.path.exists(dataset_temp): shutil.rmtree(dataset_temp, ignore_errors=True)
    except Exception as e:
        logger.error(f"{translations['create_dataset_error']}: {e}")
        import traceback
        logger.error(traceback.format_exc())

    elapsed_time = time.time() - start_time
    if os.path.exists(pid_path): os.remove(pid_path)

    logger.info(translations["create_dataset_success"].format(elapsed_time=f"{elapsed_time:.2f}"))
    return output_dirs

def separate_main(
    input_path, 
    index,
    model_name, 
    sample_rate,
    reverb_model="MDX-Reverb", 
    denoise_model="Normal",
    shifts=2, 
    batch_size=1, 
    overlap=0.25, 
    aggression=5,
    hop_length=1024, 
    window_size=512,
    segments_size=256, 
    post_process_threshold=0.2,
    enable_tta=False,
    enable_denoise=False,
    high_end_process=False,
    enable_post_process=False,
    separate_reverb=False
):
    original_vocals, _, _, _ = _separate(
        input_path,
        dataset_temp,
        model_name, 
        reverb_model=reverb_model,
        denoise_model=denoise_model,
        sample_rate=sample_rate,
        shifts=shifts, 
        batch_size=batch_size, 
        overlap=overlap, 
        aggression=aggression,
        hop_length=hop_length, 
        window_size=window_size,
        segments_size=segments_size, 
        post_process_threshold=post_process_threshold,
        enable_tta=enable_tta,
        enable_denoise=enable_denoise, 
        high_end_process=high_end_process,
        enable_post_process=enable_post_process,
        separate_reverb=separate_reverb
    )

    vocals = os.path.join(dataset_temp, f"dataset_{index}.wav")
    os.rename(original_vocals, vocals)

    return vocals

def is_url(path):
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def downloader(
    url, 
    name
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        ydl_opts = {
            "format": "bestaudio/best", 
            "outtmpl": os.path.join(dataset_temp, f"{name}"), 
            "postprocessors": [{
                "key": "FFmpegExtractAudio", 
                "preferredcodec": "wav", 
                "preferredquality": "192"
            }], 
            "no_warnings": True, 
            "noplaylist": True, 
            "noplaylist": True, 
            "verbose": False
        }

        logger.info(f"{translations['starting_download']}: {url}...")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url)  
            logger.info(f"{translations['download_success']}: {url}")

    return os.path.join(dataset_temp, f"{name}" + ".wav")

def read_file(file):
    try:
        data, sr = sf.read(file, dtype=np.float32)
    except:
        data, sr = librosa.load(file, sr=None)

    return data, sr

def skip_duration(
    audio,
    skip_start_audio,
    skip_end_audio
):
    skip_start(audio, int(skip_start_audio))
    skip_end(audio, int(skip_end_audio))

    return audio

def skip_start(
    input_file, 
    seconds
):
    data, sr = read_file(input_file)
    total_duration = len(data) / sr
    
    if seconds <= 0: 
        logger.warning(translations["=<0"])
    elif seconds >= total_duration: 
        logger.warning(translations["skip_warning"].format(seconds=seconds, total_duration=f"{total_duration:.2f}"))
    else: 
        logger.info(f"{translations['skip_start']}: {input_file}...")
        sf.write(input_file, data[int(seconds * sr):], sr)

        logger.info(translations["skip_start_audio"].format(input_file=input_file))

def skip_end(
    input_file, 
    seconds
):
    data, sr = read_file(input_file)
    total_duration = len(data) / sr

    if seconds <= 0: 
        logger.warning(translations["=<0"])
    elif seconds > total_duration: 
        logger.warning(translations["skip_warning"].format(seconds=seconds, total_duration=f"{total_duration:.2f}"))
    else: 
        logger.info(f"{translations['skip_end']}: {input_file}...")
        sf.write(input_file, data[:-int(seconds * sr)], sr)

        logger.info(translations["skip_end_audio"].format(input_file=input_file))

if __name__ == "__main__": main()
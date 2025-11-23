import os
import sys
import time
import torch
import shutil
import warnings
import argparse

import numpy as np

from tqdm import tqdm
from distutils.util import strtobool

sys.path.append(os.getcwd())

from main.app.variables import config, logger, translations, configs
from main.library.utils import load_audio, load_embedders_model, extract_features

warnings.filterwarnings("ignore")

F0_MIN, F0_MAX, HOP_SIZE, SAMPLE_RATE, FRAME_LENGTH = 50, 1100, 160, 16000, 2048

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--create_reference", action='store_true')
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--reference_name", type=str, default="reference")
    parser.add_argument("--pitch_guidance", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--use_energy", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--version", type=str, default="v2")
    parser.add_argument("--embedder_model", type=str, default="hubert_base")
    parser.add_argument("--embedders_mode", type=str, default="fairseq")
    parser.add_argument("--f0_method", type=str, default="rmvpe")
    parser.add_argument("--f0_onnx", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--f0_up_key", type=int, default=0)
    parser.add_argument("--filter_radius", type=int, default=3)
    parser.add_argument("--f0_autotune", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--f0_autotune_strength", type=float, default=1)
    parser.add_argument("--proposal_pitch", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--proposal_pitch_threshold", type=float, default=255.0)
    parser.add_argument("--alpha", type=float, default=0.5)

    return parser.parse_args()

def main():
    args = parse_arguments()
    audio_path, reference_name, pitch_guidance, use_energy, version, embedder_model, embedders_mode, f0_method, f0_onnx, f0_up_key, filter_radius, f0_autotune, f0_autotune_strength, proposal_pitch, proposal_pitch_threshold, alpha = args.audio_path, args.reference_name, args.pitch_guidance, args.use_energy, args.version, args.embedder_model, args.embedders_mode, args.f0_method, args.f0_onnx, args.f0_up_key, args.filter_radius, args.f0_autotune, args.f0_autotune_strength, args.proposal_pitch, args.proposal_pitch_threshold, args.alpha

    create_reference(
        audio_path, 
        reference_name, 
        pitch_guidance, 
        use_energy, 
        version, 
        embedder_model, 
        embedders_mode, 
        f0_method, 
        f0_onnx, 
        f0_up_key, 
        filter_radius, 
        f0_autotune, 
        f0_autotune_strength, 
        proposal_pitch, 
        proposal_pitch_threshold,
        alpha
    )

def create_reference(
    audio_path, 
    reference_name,
    pitch_guidance = True,
    use_energy = False,
    version = "v2",
    embedder_model = "hubert_base", 
    embedders_mode = "fairseq", 
    f0_method = "rmvpe",
    f0_onnx = False,
    f0_up_key = 0,
    filter_radius = 3,
    f0_autotune = False,
    f0_autotune_strength = 1,
    proposal_pitch = False,
    proposal_pitch_threshold = 255.0,
    alpha = 0.5
):
    device = config.device
    is_half = config.is_half

    if not audio_path:
        logger.warning(translations["not_found_audio"])
        sys.exit(1)

    output_reference = os.path.join(configs["reference_path"], f"{reference_name}_{version}_{embedder_model}_{pitch_guidance}_{use_energy}")
    if os.path.exists(output_reference): shutil.rmtree(reference_name, ignore_errors=True)

    os.makedirs(output_reference)
    logger.info(translations["start_create_reference"])
    start_time = time.time()

    with tqdm(total=5, desc=translations["create_reference"], ncols=100, unit="a") as pbar:
        audio = load_audio(audio_path, sample_rate=SAMPLE_RATE)
        pbar.update(1)

        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1: audio /= audio_max

        trimmed_len = (len(audio) // 320) * 320
        audio = audio[:trimmed_len]

        audio_pad = torch.nn.functional.pad(
            torch.from_numpy(audio).to(
                torch.float16 if is_half else torch.float32
            ).to(device).unsqueeze(0), 
            (40, 40), 
            mode="reflect"
        )
        pbar.update(1)

        embedder = load_embedders_model(embedder_model, embedders_mode)
        if isinstance(embedder, torch.nn.Module): embedder = embedder.to(torch.float16 if is_half else torch.float32).eval().to(device)

        with torch.no_grad():
            feats = extract_features(embedder, audio_pad.view(1, -1), version, device=device)

        np.save(os.path.join(output_reference, "feats.npy"), feats.squeeze(0).float().cpu().numpy(), allow_pickle=False)
        pbar.update(1)

        if pitch_guidance:
            from main.library.predictors.Generator import Generator

            generator = Generator(
                sample_rate=SAMPLE_RATE, 
                hop_length=HOP_SIZE, 
                f0_min=F0_MIN, 
                f0_max=F0_MAX, 
                alpha=alpha, 
                is_half=is_half, 
                device=device, 
                f0_onnx_mode=f0_onnx, 
                del_onnx_model=True
            )

            pitch, pitchf = generator.calculator(
                x_pad=config.x_pad, 
                f0_method=f0_method, 
                x=audio, 
                f0_up_key=f0_up_key, 
                p_len=audio.shape[0] // 160 + 1, 
                filter_radius=filter_radius, 
                f0_autotune=f0_autotune, 
                f0_autotune_strength=f0_autotune_strength, 
                manual_f0=None, 
                proposal_pitch=proposal_pitch, 
                proposal_pitch_threshold=proposal_pitch_threshold
            )

            np.save(os.path.join(output_reference, "pitch_coarse.npy"), pitch, allow_pickle=False)
            np.save(os.path.join(output_reference, "pitch_fine.npy"), pitchf, allow_pickle=False)

        pbar.update(1)

        if use_energy:
            from main.inference.extracting.rms import RMSEnergyExtractor
            rms = RMSEnergyExtractor(frame_length=FRAME_LENGTH, hop_length=HOP_SIZE, center=True, pad_mode="reflect").to(device).eval()

            with torch.no_grad():
                energy = rms(audio_pad)

            np.save(os.path.join(output_reference, "energy.npy"), energy.float().cpu().numpy(), allow_pickle=False)

        pbar.update(1)

    logger.info(translations["create_reference_success"].format(elapsed_time=f"{(time.time() - start_time):.2f}"))

if __name__ == "__main__": main()
import os
import sys
import time
import argparse

from distutils.util import strtobool

sys.path.append(os.getcwd())

from main.library.utils import pydub_load
from main.library.uvr5_lib.separator import Separator
from main.app.variables import config, logger, translations, vr_models, demucs_models, mdx_models, karaoke_models, reverb_models, denoise_models

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--separate_music", action='store_true')
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dirs", type=str, default="./audios")
    parser.add_argument("--export_format", type=str, default="wav")
    parser.add_argument("--model_name", type=str, default="MDXNET_Main")
    parser.add_argument("--karaoke_model", type=str, default="MDX-Version-1")
    parser.add_argument("--reverb_model", type=str, default="MDX-Reverb")
    parser.add_argument("--denoise_model", type=str, default="Normal")
    parser.add_argument("--sample_rate", type=int, default=44100)
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
    parser.add_argument("--separate_backing", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--separate_reverb", type=lambda x: bool(strtobool(x)), default=False)

    return parser.parse_args()

def main():
    args = parse_arguments()
    input_path, output_dirs, export_format, model_name, karaoke_model, reverb_model, denoise_model, sample_rate, shifts, batch_size, overlap, aggression, hop_length, window_size, segments_size, post_process_threshold, enable_tta, enable_denoise, high_end_process, enable_post_process, separate_backing, separate_reverb = args.input_path, args.output_dirs, args.export_format, args.model_name, args.karaoke_model, args.reverb_model, args.denoise_model, args.sample_rate, args.shifts, args.batch_size, args.overlap, args.aggression, args.hop_length, args.window_size, args.segments_size, args.post_process_threshold, args.enable_tta, args.enable_denoise, args.high_end_process, args.enable_post_process, args.separate_backing, args.separate_reverb

    separate(
        input_path,
        output_dirs,
        export_format, 
        model_name, 
        karaoke_model,
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
        separate_backing,
        separate_reverb
    )

def separate(
    input_path,
    output_dirs,
    export_format="wav", 
    model_name="MDXNET_Main", 
    karaoke_model="MDX-Version-1",
    reverb_model="MDX-Reverb",
    denoise_model="Normal",
    sample_rate=44100,
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
    separate_backing=False,
    separate_reverb=False
):
    start_time = time.time()
    pid_path = os.path.join("assets", "separate_pid.txt")

    with open(pid_path, "w") as pid_file:
        pid_file.write(str(os.getpid()))

    try:
        input_path = input_path.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        output_dirs = os.path.dirname(output_dirs) or output_dirs

        log_data = {
            translations['audio_path']: input_path, 
            translations['output_path']: output_dirs, 
            translations['export_format']: export_format, 
            translations['shift']: shifts, 
            translations['segments_size']: segments_size, 
            translations['overlap']: overlap, 
            translations['modelname']: model_name, 
            translations['denoise_mdx']: enable_denoise, 
            translations['hop_length']: hop_length, 
            translations['batch_size']: batch_size, 
            translations['sr']: sample_rate,
            translations['separator_backing']: separate_backing,
            translations['dereveb_audio']: separate_reverb,
            translations['aggression']: aggression,
            translations['window_size']: window_size,
            translations['post_process_threshold']: post_process_threshold,
            translations['enable_tta']: enable_tta,
            translations['high_end_process']: high_end_process,
            translations['enable_post_process']: enable_post_process
        }

        if separate_backing: log_data[translations['backing_model_ver']] = karaoke_model
        if separate_reverb: log_data[translations['dereveb_model']] = reverb_model
        if enable_denoise and model_name in list(vr_models.keys()): log_data["Denoise Model"] = denoise_model

        for key, value in log_data.items():
            logger.debug(f"{key}: {value}")

        output_files = []
        files = [os.path.join(input_path, f) for f in os.listdir(input_path)] if os.path.isdir(input_path) else [input_path]

        for file in files:
            if os.path.isfile(file):
                output_files.append(_separate(
                    input_path,
                    output_dirs,
                    model_name, 
                    karaoke_model,
                    reverb_model,
                    denoise_model,
                    export_format, 
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
                    separate_backing,
                    separate_reverb
                ))
    except Exception as e:
        logger.error(f"{translations['separator_error']}: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    if os.path.exists(pid_path): os.remove(pid_path)
    elapsed_time = time.time() - start_time

    logger.info(translations["separator_success"].format(elapsed_time=f"{elapsed_time:.2f}"))
    return output_files

def _separate(
    input_path,
    output_dirs,
    model_name, 
    karaoke_model="MDX-Version-1",
    reverb_model="MDX-Reverb",
    denoise_model="Normal",
    export_format="wav", 
    sample_rate=44100,
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
    separate_backing=False,
    separate_reverb=False
):
    main_vocals, backing_vocals = None, None

    filename, _ = os.path.splitext(os.path.basename(input_path))
    output_dirs = os.path.join(output_dirs, filename)

    os.makedirs(output_dirs, exist_ok=True)
    clean_file(output_dirs, export_format)

    if model_name in list(demucs_models.keys()):
        original_vocals, instruments = demucs_main(
            input_path,
            output_dirs,
            model_name,
            export_format,
            segments_size,
            overlap,
            shifts,
            sample_rate
        )
    elif model_name in list(vr_models.keys()):
        original_vocals, instruments = vr_main(
            input_path,
            output_dirs,
            vr_models.get(model_name, model_name),
            export_format,
            batch_size,
            window_size, 
            aggression, 
            enable_denoise,
            denoise_model,
            enable_tta, 
            enable_post_process,
            post_process_threshold, 
            high_end_process,
            sample_rate,
        )
    else:
        original_vocals, instruments = mdx_main(
            input_path,
            output_dirs,
            mdx_models.get(model_name, model_name),
            export_format,
            segments_size,
            overlap,
            enable_denoise,
            hop_length,
            batch_size,
            sample_rate,
        )
    
    if separate_backing:
        if karaoke_model.startswith("MDX"):
            main_vocals, backing_vocals = mdx_main(
                original_vocals,
                output_dirs,
                karaoke_models.get(karaoke_model, karaoke_model),
                export_format,
                segments_size,
                overlap,
                enable_denoise,
                hop_length,
                batch_size,
                sample_rate,
                mode="karaoke"
            )
        else:
            main_vocals, backing_vocals = vr_main(
                original_vocals,
                output_dirs,
                karaoke_models.get(karaoke_model, karaoke_model),
                export_format,
                batch_size,
                window_size, 
                aggression, 
                enable_denoise,
                denoise_model,
                enable_tta, 
                enable_post_process,
                post_process_threshold, 
                high_end_process,
                sample_rate,
                mode="karaoke"
            )

    if separate_reverb:
        dereverb = [original_vocals]
        if separate_backing: dereverb.append(main_vocals)

        for audio in dereverb:
            if karaoke_model.startswith("MDX"):
                _, no_reverb_vocals = mdx_main(
                    audio,
                    output_dirs,
                    reverb_models.get(reverb_model, reverb_model),
                    export_format,
                    segments_size,
                    overlap,
                    enable_denoise,
                    hop_length,
                    batch_size,
                    sample_rate,
                    mode="reverb"
                )
            else:
                _, no_reverb_vocals = vr_main(
                    audio,
                    output_dirs,
                    reverb_models.get(reverb_model, reverb_model),
                    export_format,
                    batch_size,
                    window_size, 
                    aggression, 
                    enable_denoise,
                    denoise_model,
                    enable_tta, 
                    enable_post_process,
                    post_process_threshold, 
                    high_end_process,
                    sample_rate,
                    mode="reverb"
                )
            
            if "Original_Vocals" in os.path.basename(no_reverb_vocals): original_vocals = no_reverb_vocals
            else: main_vocals = no_reverb_vocals
    
    return original_vocals, instruments, main_vocals, backing_vocals

def vr_main(
    input_path,
    output_dirs,
    model_name,
    export_format="wav",
    batch_size=1, 
    window_size=512, 
    aggression=5, 
    enable_denoise=False,
    denoise_model="Normal",
    enable_tta=False, 
    enable_post_process=False, 
    post_process_threshold=0.2, 
    high_end_process=False,
    sample_rate=44100,
    mode="original"
):
    exists_file(input_path, output_dirs)

    logger.info(f"{translations['separator_process_2']}...")

    output_list = separate_main(
        audio_file=input_path, 
        model_filename=model_name, 
        export_format=export_format, 
        output_dir=output_dirs, 
        batch_size=batch_size,
        window_size=window_size,
        aggression=aggression,
        enable_tta=enable_tta,
        enable_post_process=enable_post_process,
        post_process_threshold=post_process_threshold,
        high_end_process=high_end_process,
        sample_rate=sample_rate
    )

    if enable_denoise:
        denoise_list = []
        for audio in output_list:
            audio_path = os.path.join(output_dirs, audio)

            denoise_file = separate_main(
                audio_file=audio_path, 
                model_filename=denoise_models.get(denoise_model, denoise_model), 
                export_format=export_format, 
                output_dir=output_dirs, 
                batch_size=batch_size,
                window_size=window_size,
                aggression=aggression,
                enable_tta=enable_tta,
                enable_post_process=enable_post_process,
                post_process_threshold=post_process_threshold,
                high_end_process=high_end_process,
                sample_rate=sample_rate
            )

            if os.path.exists(audio_path): os.remove(audio_path)

            for file in denoise_file:
                file_path = os.path.join(output_dirs, file)

                if "_(Noise)_" in file and os.path.exists(file_path): os.remove(file_path)
                elif "_(No Noise)_" in file: 
                    filename = "".join([file.split("_(No Noise)_")[0], ".", export_format])
                    os.rename(file_path, os.path.join(output_dirs, filename))

                    denoise_list.append(filename)

    logger.info(translations["separator_success_2"])
    return process_file(denoise_list if enable_denoise else output_list, output_dirs, export_format, mode)

def demucs_main(
    input_path,
    output_dirs,
    model_name,
    export_format="wav",
    segments_size=256,
    overlap=0.25,
    shifts=2,
    sample_rate=44100
):
    exists_file(input_path, output_dirs)
    
    logger.info(f"{translations['separator_process_2']}...")

    output_list = separate_main(
        audio_file=input_path, 
        output_dir=output_dirs, 
        model_filename=demucs_models.get(model_name, model_name), 
        export_format=export_format, 
        segment_size=(segments_size / 2), 
        overlap=overlap, 
        shifts=shifts, 
        sample_rate=sample_rate
    )

    logger.info(translations["separator_success_2"])
    return process_file(output_list, output_dirs, export_format, mode="4stem")

def mdx_main(
    input_path,
    output_dirs,
    model_name,
    export_format="wav",
    segments_size=256,
    overlap=0.25,
    enable_denoise=False,
    hop_length=1024,
    batch_size=1,
    sample_rate=44100,
    mode="original"
):
    exists_file(input_path, output_dirs)

    logger.info(f"{translations['separator_process_2']}...")

    output_list = separate_main(
        audio_file=input_path, 
        model_filename=model_name, 
        export_format=export_format, 
        output_dir=output_dirs, 
        segment_size=segments_size, 
        overlap=overlap, 
        batch_size=batch_size, 
        hop_length=hop_length, 
        enable_denoise=enable_denoise, 
        sample_rate=sample_rate
    )

    logger.info(translations["separator_success_2"])
    return process_file(output_list, output_dirs, export_format, mode)

def process_file(input_list, output_dirs, export_format="wav", mode="original"):
    demucs_inst = []

    reverb_audio, no_reverb_audio = None, None
    main_audio, backing_audio = os.path.join(output_dirs, f"Main_Vocals.{export_format}"), os.path.join(output_dirs, f"Backing_Vocals.{export_format}")
    original_audio, instruments_audio = os.path.join(output_dirs, f"Original_Vocals.{export_format}"), os.path.join(output_dirs, f"Instruments.{export_format}")

    for file in input_list:
        file_path = os.path.join(output_dirs, file)
        if not os.path.exists(file_path): logger.warning(translations["not_found"].format(name=file_path))

        if mode == "original":
            if "_(Instrumental)_" in file: os.rename(file_path, instruments_audio)
            elif "_(Vocals)_" in file: os.rename(file_path, original_audio)
        elif mode == "4stem":
            if "_(Vocals)_" in file: os.rename(file_path, original_audio)
            elif "_(Drums)_" in file or "_(Bass)_" in file or "_(Other)_" in file: demucs_inst.append(file_path)
        elif mode == "reverb":
            filename = file.split("_(")[0]

            reverb_audio = os.path.join(output_dirs, "".join([filename, "_Reverb.", export_format]))
            no_reverb_audio = os.path.join(output_dirs, "".join([filename, "_No_Reverb.", export_format]))

            if "_(Reverb)_" in file or "_(Echo)_" in file: os.rename(file_path, reverb_audio)
            elif "_(No Reverb)_" in file or "_(No Echo)_" in file: os.rename(file_path, no_reverb_audio)
        elif mode == "karaoke":
            if "_(Instrumental)_" in file: os.rename(file_path, backing_audio)
            elif "_(Vocals)_" in file: os.rename(file_path, main_audio)

    if mode == "reverb": return reverb_audio, no_reverb_audio
    if mode == "karaoke": return main_audio, backing_audio 

    if mode == "4stem":
        demucs_audio = pydub_load(demucs_inst[0])
        for file in demucs_inst[1:]:
            demucs_audio = demucs_audio.overlay(pydub_load(file))

        demucs_audio.export(instruments_audio, format=export_format)

        for f in demucs_inst:
            if os.path.exists(f): os.remove(f)

    return original_audio, instruments_audio

def exists_file(input_path, output_dirs):
    if not os.path.exists(input_path): 
        logger.warning(translations["input_not_valid"])
        sys.exit(1)
    
    if not os.path.exists(output_dirs): 
        logger.warning(translations["output_not_valid"])
        sys.exit(1)

def clean_file(output_dirs, export_format):
    for f in [
        "Original_Vocals.", 
        "Original_Vocals_Reverb.",
        "Original_Vocals_No_Reverb.", 
        "Main_Vocals.",
        "Main_Vocals_Reverb.", 
        "Main_Vocals_No_Reverb.",
        "Instruments.",
        "Backing_Vocals."
    ]:
        file_path = os.path.join(output_dirs, f + export_format)
        if os.path.exists(file_path): os.remove(file_path)

def separate_main(
    audio_file=None, 
    model_filename="UVR-MDX-NET_Main_340.onnx", 
    export_format="wav", 
    output_dir=".", 
    segment_size=256, 
    overlap=0.25, 
    batch_size=1, 
    hop_length=1024, 
    enable_denoise=False, 
    shifts=2, 
    window_size=512,
    aggression=5,
    enable_tta=False,
    enable_post_process=False,
    post_process_threshold=0.2,
    high_end_process=False,
    sample_rate=44100
):
    try:
        separator = Separator(
            logger=logger, 
            output_dir=output_dir, 
            output_format=export_format, 
            output_bitrate=None, 
            normalization_threshold=0.9, 
            sample_rate=sample_rate, 
            mdx_params={
                "hop_length": hop_length, 
                "segment_size": segment_size, 
                "overlap": overlap, 
                "batch_size": batch_size, 
                "enable_denoise": enable_denoise
            }, 
            demucs_params={
                "segment_size": segment_size, 
                "shifts": shifts, 
                "overlap": overlap, 
                "segments_enabled": config.configs.get("demucs_segments_enable", True)
            },
            vr_params={
                "batch_size": batch_size, 
                "window_size": window_size, 
                "aggression": aggression, 
                "enable_tta": enable_tta, 
                "enable_post_process": enable_post_process, 
                "post_process_threshold": post_process_threshold, 
                "high_end_process": high_end_process
            }
        )
        separator.load_model(model_filename=model_filename)

        return separator.separate(audio_file)
    except:
        logger.debug(translations["default_setting"])
        separator = Separator(
            logger=logger, 
            output_dir=output_dir, 
            output_format=export_format, 
            output_bitrate=None, 
            normalization_threshold=0.9, 
            sample_rate=44100, 
            mdx_params={
                "hop_length": 1024, 
                "segment_size": 256, 
                "overlap": 0.25, 
                "batch_size": 1, 
                "enable_denoise": enable_denoise
            }, 
            demucs_params={
                "segment_size": 128, 
                "shifts": 2, 
                "overlap": 0.25, 
                "segments_enabled": config.configs.get("demucs_segments_enable", True)
            },
            vr_params={
                "batch_size": 1, 
                "window_size": 512, 
                "aggression": 5, 
                "enable_tta": False, 
                "enable_post_process": False, 
                "post_process_threshold": 0.2, 
                "high_end_process": False
            }
        )
        separator.load_model(model_filename=model_filename)

        return separator.separate(audio_file)
    
if __name__ == "__main__": main()
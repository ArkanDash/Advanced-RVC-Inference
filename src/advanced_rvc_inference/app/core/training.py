import os
import sys
import time
import shutil
import codecs
import threading
import subprocess

sys.path.append(os.getcwd())

from main.tools import huggingface
from main.app.core.ui import gr_info, gr_warning
from main.app.variables import python, translations, configs

def if_done(done, p):
    while 1:
        if p.poll() is None: time.sleep(0.5)
        else: break

    done[0] = True

def log_read(done, name):
    log_file = os.path.join(configs["logs_path"], "app.log")

    f = open(log_file, "w", encoding="utf-8")
    f.close()

    while 1:
        with open(log_file, "r", encoding="utf-8") as f:
            yield "".join(line for line in f.readlines() if "DEBUG" not in line and name in line and line.strip() != "")

        time.sleep(1)
        if done[0]: break

    with open(log_file, "r", encoding="utf-8") as f:
        log = "".join(line for line in f.readlines() if "DEBUG" not in line and line.strip() != "")

    yield log

def create_dataset(
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
):
    gr_info(translations["start"].format(start=translations["create"]))

    p = subprocess.Popen(f'{python} {configs["create_dataset_path"]} --input_data "{input_data}" --output_dirs "{output_dirs}" --skip_seconds {skip_seconds} --skip_start_audios "{skip_start_audios}" --skip_end_audios "{skip_end_audios}" --separate {separate} --model_name "{model_name}" --reverb_model "{reverb_model}" --denoise_model "{denoise_model}" --sample_rate {sample_rate} --shifts {shifts} --batch_size {batch_size} --overlap {overlap} --aggression {aggression} --hop_length {hop_length} --window_size {window_size} --segments_size {segments_size} --post_process_threshold {post_process_threshold} --enable_tta {enable_tta} --enable_denoise {enable_denoise} --high_end_process {high_end_process} --enable_post_process {enable_post_process} --separate_reverb {separate_reverb} --clean_dataset {clean_dataset} --clean_strength {clean_strength}', shell=True)
    done = [False]

    threading.Thread(target=if_done, args=(done, p)).start()

    for log in log_read(done, "create_dataset"):
        yield log

def create_reference(audio_path, reference_name, pitch_guidance, use_energy, version, embedder_model, embedders_mode, f0_method, f0_onnx, f0_up_key, filter_radius, f0_autotune, f0_autotune_strength, proposal_pitch, proposal_pitch_threshold, alpha=0.5):
    gr_info(translations["start"].format(start=translations["create_reference"]))

    p = subprocess.Popen(f'{python} {configs["create_reference_path"]} --audio_path "{audio_path}" --reference_name "{reference_name}" --pitch_guidance {pitch_guidance} --use_energy {use_energy} --version {version} --embedder_model {embedder_model} --embedders_mode {embedders_mode} --f0_method {f0_method} --f0_onnx {f0_onnx} --f0_up_key {f0_up_key} --filter_radius {filter_radius} --f0_autotune {f0_autotune} --f0_autotune_strength {f0_autotune_strength} --proposal_pitch {proposal_pitch} --proposal_pitch_threshold {proposal_pitch_threshold} --alpha {alpha}', shell=True)
    done = [False]

    threading.Thread(target=if_done, args=(done, p)).start()

    for log in log_read(done, "create_reference"):
        yield log

def preprocess(model_name, sample_rate, cpu_core, cut_preprocess, process_effects, dataset, clean_dataset, clean_strength, chunk_len=3.0, overlap_len=0.3, normalization_mode="none"):
    sr = int(float(sample_rate.rstrip("k")) * 1000)

    if not model_name: return gr_warning(translations["provide_name"])
    if not os.path.exists(dataset) or not any(f.lower().endswith(("wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3")) for f in os.listdir(dataset) if os.path.isfile(os.path.join(dataset, f))): return gr_warning(translations["not_found_data"])
    
    model_dir = os.path.join(configs["logs_path"], model_name)
    if os.path.exists(model_dir): shutil.rmtree(model_dir, ignore_errors=True)

    p = subprocess.Popen(f'{python} {configs["preprocess_path"]} --model_name "{model_name}" --dataset_path "{dataset}" --sample_rate {sr} --cpu_cores {cpu_core} --cut_preprocess {cut_preprocess} --process_effects {process_effects} --clean_dataset {clean_dataset} --clean_strength {clean_strength} --chunk_len {chunk_len} --overlap_len {overlap_len} --normalization_mode {normalization_mode}', shell=True)
    done = [False]

    threading.Thread(target=if_done, args=(done, p)).start()
    os.makedirs(model_dir, exist_ok=True)

    for log in log_read(done, "preprocess"):
        yield log

def extract(model_name, version, method, pitch_guidance, hop_length, cpu_cores, gpu, sample_rate, embedders, custom_embedders, onnx_f0_mode, embedders_mode, f0_autotune, f0_autotune_strength, hybrid_method, rms_extract, alpha=0.5):
    f0method, embedder_model = (method if method != "hybrid" else hybrid_method), (embedders if embedders != "custom" else custom_embedders)
    sr = int(float(sample_rate.rstrip("k")) * 1000)

    if not model_name: return gr_warning(translations["provide_name"])
    model_dir = os.path.join(configs["logs_path"], model_name)

    try:
        if not any(os.path.isfile(os.path.join(model_dir, "sliced_audios", f)) for f in os.listdir(os.path.join(model_dir, "sliced_audios"))) or not any(os.path.isfile(os.path.join(model_dir, "sliced_audios_16k", f)) for f in os.listdir(os.path.join(model_dir, "sliced_audios_16k"))): return gr_warning(translations["not_found_data_preprocess"])
    except:
        return gr_warning(translations["not_found_data_preprocess"])
    
    p = subprocess.Popen(f'{python} {configs["extract_path"]} --model_name "{model_name}" --rvc_version {version} --f0_method {f0method} --pitch_guidance {pitch_guidance} --hop_length {hop_length} --cpu_cores {cpu_cores} --gpu {gpu} --sample_rate {sr} --embedder_model {embedder_model} --f0_onnx {onnx_f0_mode} --embedders_mode {embedders_mode} --f0_autotune {f0_autotune} --f0_autotune_strength {f0_autotune_strength} --rms_extract {rms_extract} --alpha {alpha}', shell=True)
    done = [False]

    threading.Thread(target=if_done, args=(done, p)).start()
    os.makedirs(model_dir, exist_ok=True)

    for log in log_read(done, "extract"):
        yield log

def create_index(model_name, rvc_version, index_algorithm):
    if not model_name: return gr_warning(translations["provide_name"])
    model_dir = os.path.join(configs["logs_path"], model_name)
    
    try:
        if not any(os.path.isfile(os.path.join(model_dir, f"{rvc_version}_extracted", f)) for f in os.listdir(os.path.join(model_dir, f"{rvc_version}_extracted"))): return gr_warning(translations["not_found_data_extract"])
    except:
        return gr_warning(translations["not_found_data_extract"])
    
    p = subprocess.Popen(f'{python} {configs["create_index_path"]} --model_name "{model_name}" --rvc_version {rvc_version} --index_algorithm {index_algorithm}', shell=True)
    done = [False]

    threading.Thread(target=if_done, args=(done, p)).start()
    os.makedirs(model_dir, exist_ok=True)

    for log in log_read(done, "create_index"):
        yield log

def training(model_name, rvc_version, save_every_epoch, save_only_latest, save_every_weights, total_epoch, sample_rate, batch_size, gpu, pitch_guidance, not_pretrain, custom_pretrained, pretrain_g, pretrain_d, detector, threshold, clean_up, cache, model_author, vocoder, checkpointing, deterministic, benchmark, optimizer, energy_use, custom_reference=False, reference_name="", multiscale_mel_loss=False):
    sr = int(float(sample_rate.rstrip("k")) * 1000)
    if not model_name: return gr_warning(translations["provide_name"])

    model_dir = os.path.join(configs["logs_path"], model_name)
    if os.path.exists(os.path.join(model_dir, "train_pid.txt")): os.remove(os.path.join(model_dir, "train_pid.txt"))
    
    try:
        if not any(os.path.isfile(os.path.join(model_dir, f"{rvc_version}_extracted", f)) for f in os.listdir(os.path.join(model_dir, f"{rvc_version}_extracted"))): return gr_warning(translations["not_found_data_extract"])
    except:
        return gr_warning(translations["not_found_data_extract"])
    
    if not not_pretrain:
        if not custom_pretrained: 
            pretrain_dir = configs["pretrained_v2_path"] if rvc_version == 'v2' else configs["pretrained_v1_path"]
            download_version = codecs.decode(f"uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/cergenvarq_", "rot13") + f"{rvc_version}/"

            pretrained_selector = {
                True: {
                    32000: ("f0G32k.pth", "f0D32k.pth"), 
                    40000: ("f0G40k.pth", "f0D40k.pth"), 
                    48000: ("f0G48k.pth", "f0D48k.pth")
                }, 
                False: {
                    32000: ("G32k.pth", "D32k.pth"), 
                    40000: ("G40k.pth", "D40k.pth"), 
                    48000: ("G48k.pth", "D48k.pth")
                }
            }

            pg2, pd2 = "", ""
            pg, pd = pretrained_selector[pitch_guidance][sr]

            if energy_use: pg2, pd2 = pg2 + "ENERGY_", pd2 + "ENERGY_"
            if vocoder != 'Default': pg2, pd2 = pg2 + vocoder + "_", pd2 + vocoder + "_"

            pg2, pd2 = pg2 + pg, pd2 + pd
            pretrained_G, pretrained_D = (
                os.path.join(
                    pretrain_dir,
                    pg2
                ), 
                os.path.join(
                    pretrain_dir,
                    pd2
                )
            )

            try:
                if not os.path.exists(pretrained_G):
                    gr_info(translations["download_pretrained"].format(dg="G", rvc_version=rvc_version))
                    huggingface.HF_download_file(
                        "".join(
                            [
                                download_version, 
                                pg2
                            ]
                        ),
                        os.path.join(
                            pretrain_dir,
                            pg2
                        )
                    )
                        
                if not os.path.exists(pretrained_D):
                    gr_info(translations["download_pretrained"].format(dg="D", rvc_version=rvc_version))
                    huggingface.HF_download_file(
                        "".join(
                            [
                                download_version, 
                                pd2
                            ]
                        ), 
                        os.path.join(
                            pretrain_dir,
                            pd2
                        )
                    )
            except:
                gr_warning(translations["not_use_pretrain_error_download"])
                pretrained_G = pretrained_D = None
        else:
            if not pretrain_g: return gr_warning(translations["provide_pretrained"].format(dg="G"))
            if not pretrain_d: return gr_warning(translations["provide_pretrained"].format(dg="D"))
            
            pg2, pd2 = pretrain_g, pretrain_d
            pretrained_G, pretrained_D = (
                (os.path.join(configs["pretrained_custom_path"], pg2) if not os.path.exists(pg2) else pg2), 
                (os.path.join(configs["pretrained_custom_path"], pd2) if not os.path.exists(pd2) else pd2)
            )

            if not os.path.exists(pretrained_G): return gr_warning(translations["not_found_pretrain"].format(dg="G"))
            if not os.path.exists(pretrained_D): return gr_warning(translations["not_found_pretrain"].format(dg="D"))
    else: 
        pretrained_G = pretrained_D = None
        gr_warning(translations["not_use_pretrain"])

    if custom_reference:
        reference_path = os.path.join(configs["reference_path"], reference_name)

        if not os.path.exists(reference_path):
            gr_warning(translations["not_found_reference"])

            custom_reference = False
            reference_path = None
    else: reference_path = None

    gr_info(translations["start"].format(start=translations["training"]))

    p = subprocess.Popen(f'{python} {configs["train_path"]} --model_name "{model_name}" --rvc_version {rvc_version} --save_every_epoch {save_every_epoch} --save_only_latest {save_only_latest} --save_every_weights {save_every_weights} --total_epoch {total_epoch} --batch_size {batch_size} --gpu {gpu} --pitch_guidance {pitch_guidance} --overtraining_detector {detector} --overtraining_threshold {threshold} --cleanup {clean_up} --cache_data_in_gpu {cache} --g_pretrained_path "{pretrained_G}" --d_pretrained_path "{pretrained_D}" --model_author "{model_author}" --vocoder "{vocoder}" --checkpointing {checkpointing} --deterministic {deterministic} --benchmark {benchmark} --optimizer {optimizer} --energy_use {energy_use} --use_custom_reference {custom_reference} --reference_path {reference_path} --multiscale_mel_loss {multiscale_mel_loss}', shell=True)
    done = [False]

    with open(os.path.join(model_dir, "train_pid.txt"), "w") as pid_file:
        pid_file.write(str(p.pid))

    threading.Thread(target=if_done, args=(done, p)).start()

    for log in log_read(done, "train"):
        lines = log.splitlines()
        if len(lines) > 50: log = "\n".join(lines[-50:])
        yield log
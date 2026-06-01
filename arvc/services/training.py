import os
import sys
import time
import shutil
import codecs
import threading
import subprocess

sys.path.append(os.getcwd())

from arvc.utils import huggingface
from arvc.utils.feedback import gr_info, gr_warning
from arvc.utils.variables import python, translations, configs, logger
from arvc.engine.models.optimizers import get_optimizer_choices
from arvc.engine.models.generators import get_vocoder_choices

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

def preprocess(model_name, sample_rate, cpu_core, cut_preprocess, process_effects, dataset, clean_dataset, clean_strength, chunk_len=3.0, overlap_len=0.3, normalization_mode="post"):
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

def training(model_name, rvc_version, save_every_epoch, save_only_latest, save_every_weights, total_epoch, sample_rate, batch_size, gpu, pitch_guidance, not_pretrain, custom_pretrained, pretrain_g, pretrain_d, detector, threshold, clean_up, cache, model_author, vocoder, checkpointing, deterministic, benchmark, optimizer, energy_use, custom_reference=False, reference_name="", multiscale_mel_loss=False, cosine_lr=False, newpytorch=True):
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
            download_version = configs.get(
                f"pretrained_{rvc_version}_url",
                f"https://huggingface.co/buckets/R-Kentaren/Ultimate-RVC-Models/resolve/pretrained_{rvc_version}/"
            )

            pretrained_selector = {
                True: {  # pitch_guidance (f0 models)
                    24000: ("f0G24k.pth", "f0D24k.pth"),
                    32000: ("f0G32k.pth", "f0D32k.pth"),
                    40000: ("f0G40k.pth", "f0D40k.pth"),
                    44100: ("f0G40k.pth", "f0D40k.pth"),  # reuse 40k pretrained
                    48000: ("f0G48k.pth", "f0D48k.pth"),
                },
                False: {  # no pitch guidance
                    24000: ("G24k.pth", "D24k.pth"),
                    32000: ("G32k.pth", "D32k.pth"),
                    40000: ("G40k.pth", "D40k.pth"),
                    44100: ("G40k.pth", "D40k.pth"),  # reuse 40k pretrained
                    48000: ("G48k.pth", "D48k.pth"),
                }
            }

            pg, pd = pretrained_selector[pitch_guidance][sr]

            # NOTE: The default pretrained models in the bucket do NOT have
            # vocoder or energy prefixes.  Only attempt vocoder/energy-prefixed
            # filenames if those specific files already exist locally (e.g. the
            # user manually placed them).  Otherwise fall back to the standard
            # filenames (f0G40k.pth, f0D40k.pth, etc.) which are guaranteed
            # to exist in the R-Kentaren/Ultimate-RVC-Models bucket for 32k/40k/48k.
            pg_prefixed, pd_prefixed = pg, pd

            if energy_use:
                _pg_e = "ENERGY_" + pg
                _pd_e = "ENERGY_" + pd
                if os.path.exists(os.path.join(pretrain_dir, _pg_e)):
                    pg_prefixed = _pg_e
                if os.path.exists(os.path.join(pretrain_dir, _pd_e)):
                    pd_prefixed = _pd_e

            if vocoder not in ('Default', 'HiFi-GAN'):
                _pg_v = vocoder + "_" + pg_prefixed
                _pd_v = vocoder + "_" + pd_prefixed
                if os.path.exists(os.path.join(pretrain_dir, _pg_v)):
                    pg_prefixed = _pg_v
                if os.path.exists(os.path.join(pretrain_dir, _pd_v)):
                    pd_prefixed = _pd_v

            pretrained_G = os.path.join(pretrain_dir, pg_prefixed)
            pretrained_D = os.path.join(pretrain_dir, pd_prefixed)

            # Primary and fallback pretrained URLs
            primary_url = download_version
            # Fallback: R-Kentaren/Ultimate-RVC-Models HuggingFace Storage Bucket
            _default_fallback = (
                f"https://huggingface.co/buckets/R-Kentaren/Ultimate-RVC-Models/resolve/pretrained_{rvc_version}/"
            )
            fallback_url = configs.get(
                f"pretrained_{rvc_version}_fallback_url",
                _default_fallback
            )

            # 24k pretrained models do not exist in any known repo.
            # Fallback to 32k pretrained for 24k training.
            sr_for_pretrained = sr
            if sr == 24000:
                logger.warning("24k pretrained models are not available; falling back to 32k pretrained.")
                sr_for_pretrained = 32000
                # Re-derive filenames for the fallback sample rate (without vocoder prefix)
                pg_prefixed, pd_prefixed = pretrained_selector[pitch_guidance][sr_for_pretrained]
                pretrained_G = os.path.join(pretrain_dir, pg_prefixed)
                pretrained_D = os.path.join(pretrain_dir, pd_prefixed)

            def _download_pretrained(file_url, file_path, url_sources):
                """Try downloading from multiple URL sources, return True on success."""
                for src_url in url_sources:
                    try:
                        full_url = src_url + os.path.basename(file_path)
                        logger.info(f"Trying download: {full_url}")
                        huggingface.HF_download_file(full_url, file_path)
                        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                            return True
                    except Exception as e:
                        logger.warning(f"Download failed from {src_url}: {e}")
                        continue
                return False

            try:
                url_sources = [primary_url, fallback_url]

                if not os.path.exists(pretrained_G):
                    gr_info(translations["download_pretrained"].format(dg="G", rvc_version=rvc_version))
                    if not _download_pretrained(download_version + pg_prefixed, pretrained_G, url_sources):
                        logger.error(f"Failed to download pretrained G ({pg_prefixed}) from all sources")
                        pretrained_G = None

                if not os.path.exists(pretrained_D):
                    gr_info(translations["download_pretrained"].format(dg="D", rvc_version=rvc_version))
                    if not _download_pretrained(download_version + pd_prefixed, pretrained_D, url_sources):
                        logger.error(f"Failed to download pretrained D ({pd_prefixed}) from all sources")
                        pretrained_D = None

                # Verify files actually exist after download
                if pretrained_G and not os.path.exists(pretrained_G):
                    logger.error(f"Pretrained G file missing after download: {pretrained_G}")
                    pretrained_G = None
                if pretrained_D and not os.path.exists(pretrained_D):
                    logger.error(f"Pretrained D file missing after download: {pretrained_D}")
                    pretrained_D = None

                if pretrained_G is None and pretrained_D is None:
                    gr_warning(translations["not_use_pretrain_error_download"])
            except Exception as e:
                logger.error(f"Pretrained model download error: {e}")
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

    p = subprocess.Popen(f'{python} {configs["train_path"]} --model_name "{model_name}" --rvc_version {rvc_version} --save_every_epoch {save_every_epoch} --save_only_latest {save_only_latest} --save_every_weights {save_every_weights} --total_epoch {total_epoch} --batch_size {batch_size} --gpu {gpu} --pitch_guidance {pitch_guidance} --overtraining_detector {detector} --overtraining_threshold {threshold} --cleanup {clean_up} --cache_data_in_gpu {cache} --g_pretrained_path "{pretrained_G}" --d_pretrained_path "{pretrained_D}" --model_author "{model_author}" --vocoder "{vocoder}" --checkpointing {checkpointing} --deterministic {deterministic} --benchmark {benchmark} --optimizer {optimizer} --energy_use {energy_use} --use_custom_reference {custom_reference} --reference_path {reference_path} --multiscale_mel_loss {multiscale_mel_loss} --use_cosine_annealing_lr {cosine_lr} --newpytorch {newpytorch}', shell=True)
    done = [False]

    with open(os.path.join(model_dir, "train_pid.txt"), "w") as pid_file:
        pid_file.write(str(p.pid))

    threading.Thread(target=if_done, args=(done, p)).start()

    for log in log_read(done, "train"):
        lines = log.splitlines()
        if len(lines) > 50: log = "\n".join(lines[-50:])
        yield log

def one_click_train(
    model_name, rvc_version, sample_rate, dataset_path,
    pitch_guidance, f0_method, total_epoch, batch_size, save_every,
    gpu, vocoder, optimizer, model_author=""
):
    """One-click training pipeline: preprocess → extract → train → create index.

    This is a generator that yields progress log strings so the Gradio UI
    can display real-time progress.

    Args:
        model_name: Name of the model to train.
        rvc_version: RVC version ("v1" or "v2").
        sample_rate: Sample rate string (e.g. "40k").
        dataset_path: Path to the dataset folder.
        pitch_guidance: Whether to use pitch guidance.
        f0_method: F0 extraction method (e.g. "rmvpe").
        total_epoch: Total training epochs.
        batch_size: Training batch size.
        save_every: Save checkpoint every N epochs.
        gpu: GPU identifier string (e.g. "0" or "-").
        vocoder: Vocoder name (e.g. "Default", "BigVGAN", "MRF-HiFi-GAN", "RefineGAN").
        optimizer: Optimizer name (e.g. "AdamW").
        model_author: Optional model author name.

    Yields:
        Progress log strings.
    """
    all_logs = ""

    # ── Step 1: Preprocessing ──
    gr_info(translations["start"].format(start="Preprocessing"))
    step1_log = ""
    for log in preprocess(
        model_name=model_name,
        sample_rate=sample_rate,
        cpu_core=os.cpu_count() or 4,
        cut_preprocess="Automatic",
        process_effects=True,
        dataset=dataset_path,
        clean_dataset=False,
        clean_strength=0.7,
        chunk_len=3.0,
        overlap_len=0.3,
        normalization_mode="post",
    ):
        step1_log = log
        all_logs += f"=== Step 1: Preprocessing ===\n{log}\n\n"
        yield all_logs

    if not step1_log:
        gr_warning("Preprocessing did not produce any output.")
        all_logs += "Preprocessing failed or produced no output.\n"
        yield all_logs
        return

    # ── Step 2: Feature Extraction ──
    gr_info(translations["start"].format(start="Feature Extraction"))
    step2_log = ""
    for log in extract(
        model_name=model_name,
        version=rvc_version,
        method=f0_method,
        pitch_guidance=pitch_guidance,
        hop_length=128,
        cpu_cores=os.cpu_count() or 4,
        gpu=gpu,
        sample_rate=sample_rate,
        embedders="hubert_base",
        custom_embedders="hubert_base",
        onnx_f0_mode=False,
        embedders_mode="fairseq",
        f0_autotune=False,
        f0_autotune_strength=1.0,
        hybrid_method="hybrid[pm+crepe-tiny]",
        rms_extract=False,
        alpha=0.5,
    ):
        step2_log = log
        all_logs += f"=== Step 2: Feature Extraction ===\n{log}\n\n"
        yield all_logs

    if not step2_log:
        gr_warning("Feature extraction did not produce any output.")
        all_logs += "Feature extraction failed or produced no output.\n"
        yield all_logs
        return

    # ── Step 3: Training ──
    gr_info(translations["start"].format(start="Training"))
    step3_log = ""
    for log in training(
        model_name=model_name,
        rvc_version=rvc_version,
        save_every_epoch=save_every,
        save_only_latest=True,
        save_every_weights=True,
        total_epoch=total_epoch,
        sample_rate=sample_rate,
        batch_size=batch_size,
        gpu=gpu,
        pitch_guidance=pitch_guidance,
        not_pretrain=False,
        custom_pretrained=False,
        pretrain_g="",
        pretrain_d="",
        detector=False,
        threshold=50,
        clean_up=False,
        cache=True,
        model_author=model_author,
        vocoder=vocoder,
        checkpointing=False,
        deterministic=False,
        benchmark=False,
        optimizer=optimizer,
        energy_use=False,
        custom_reference=False,
        reference_name="",
        multiscale_mel_loss=False,
        cosine_lr=True,
    ):
        step3_log = log
        all_logs += f"=== Step 3: Training ===\n{log}\n\n"
        yield all_logs

    if not step3_log:
        gr_warning("Training did not produce any output.")
        all_logs += "Training failed or produced no output.\n"
        yield all_logs
        return

    # ── Step 4: Index Creation ──
    gr_info(translations["start"].format(start="Index Creation"))
    step4_log = ""
    for log in create_index(
        model_name=model_name,
        rvc_version=rvc_version,
        index_algorithm="Auto",
    ):
        step4_log = log
        all_logs += f"=== Step 4: Index Creation ===\n{log}\n\n"
        yield all_logs

    if not step4_log:
        gr_warning("Index creation did not produce any output.")
        all_logs += "Index creation failed or produced no output.\n"
        yield all_logs
        return

    gr_info("One-click training pipeline completed successfully!")
    all_logs += "=== One-Click Training Complete ===\n"
    yield all_logs
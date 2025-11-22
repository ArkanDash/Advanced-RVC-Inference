import os
import re
import gc
import sys
import shutil
import datetime
import subprocess

import numpy as np

sys.path.append(os.getcwd())

from main.app.variables import logger, config, configs, translations, python
from main.app.core.ui import gr_info, gr_warning, gr_error, process_output, replace_export_format

def convert(pitch, filter_radius, index_rate, rms_mix_rate, protect, hop_length, f0_method, input_path, output_path, pth_path, index_path, f0_autotune, clean_audio, clean_strength, export_format, embedder_model, resample_sr, split_audio, f0_autotune_strength, checkpointing, f0_onnx, embedders_mode, formant_shifting, formant_qfrency, formant_timbre, f0_file, proposal_pitch, proposal_pitch_threshold, audio_processing=False, alpha=0.5):    
    subprocess.run([
        python, 
        configs["convert_path"], 
        "--pitch", str(pitch), 
        "--filter_radius", str(filter_radius), 
        "--index_rate", str(index_rate), 
        "--rms_mix_rate", str(rms_mix_rate), 
        "--protect", str(protect), 
        "--hop_length", str(hop_length), 
        "--f0_method", f0_method, 
        "--input_path", input_path, 
        "--output_path", output_path, 
        "--pth_path", pth_path, 
        "--index_path", index_path, 
        "--f0_autotune", str(f0_autotune), 
        "--clean_audio", str(clean_audio), 
        "--clean_strength", str(clean_strength), 
        "--export_format", export_format, 
        "--embedder_model", embedder_model, 
        "--resample_sr", str(resample_sr), 
        "--split_audio", str(split_audio), 
        "--f0_autotune_strength", str(f0_autotune_strength), 
        "--checkpointing", str(checkpointing), 
        "--f0_onnx", str(f0_onnx), 
        "--embedders_mode", embedders_mode, 
        "--formant_shifting", str(formant_shifting), 
        "--formant_qfrency", str(formant_qfrency), 
        "--formant_timbre", str(formant_timbre), 
        "--f0_file", f0_file, 
        "--proposal_pitch", str(proposal_pitch), 
        "--proposal_pitch_threshold", str(proposal_pitch_threshold),
        "--audio_processing", str(audio_processing),
        "--alpha", str(alpha)
    ])

def convert_audio(clean, autotune, use_audio, use_original, convert_backing, not_merge_backing, merge_instrument, pitch, clean_strength, model, index, index_rate, input, output, format, method, hybrid_method, hop_length, embedders, custom_embedders, resample_sr, filter_radius, rms_mix_rate, protect, split_audio, f0_autotune_strength, input_audio_name, checkpointing, onnx_f0_mode, formant_shifting, formant_qfrency, formant_timbre, f0_file, embedders_mode, proposal_pitch, proposal_pitch_threshold, audio_processing=False, alpha=0.5):
    model_path = os.path.join(configs["weights_path"], model) if not os.path.exists(model) else model

    return_none = [None]*6
    return_none[5] = {"visible": True, "__type__": "update"}

    if not use_audio:
        if merge_instrument or not_merge_backing or convert_backing or use_original:
            gr_warning(translations["turn_on_use_audio"])
            return return_none

    if use_original:
        if convert_backing:
            gr_warning(translations["turn_off_convert_backup"])
            return return_none
        elif not_merge_backing:
            gr_warning(translations["turn_off_merge_backup"])
            return return_none

    if not model or not os.path.exists(model_path) or os.path.isdir(model_path) or not model.endswith((".pth", ".onnx")):
        gr_warning(translations["provide_file"].format(filename=translations["model"]))
        return return_none

    f0method, embedder_model = (method if method != "hybrid" else hybrid_method), (embedders if embedders != "custom" else custom_embedders)

    if use_audio:
        output_audio = os.path.join(configs["audios_path"], input_audio_name)

        from main.library.utils import pydub_load
        
        def get_audio_file(label):
            matching_files = [f for f in os.listdir(output_audio) if label in f]

            if not matching_files: return translations["notfound"]   
            return os.path.join(output_audio, matching_files[0])

        output_path = os.path.join(output_audio, f"Convert_Vocals.{format}")
        output_backing = os.path.join(output_audio, f"Convert_Backing.{format}")
        output_merge_backup = os.path.join(output_audio, f"Vocals+Backing.{format}")
        output_merge_instrument = os.path.join(output_audio, f"Vocals+Instruments.{format}")

        if os.path.exists(output_audio): os.makedirs(output_audio, exist_ok=True)
        output_path = process_output(output_path)

        if use_original:
            original_vocal = get_audio_file('Original_Vocals_No_Reverb.')

            if original_vocal == translations["notfound"]: original_vocal = get_audio_file('Original_Vocals.')

            if original_vocal == translations["notfound"]: 
                gr_warning(translations["not_found_original_vocal"])
                return return_none
            
            input_path = original_vocal
        else:
            main_vocal = get_audio_file('Main_Vocals_No_Reverb.')
            backing_vocal = get_audio_file('Backing_Vocals.')

            if main_vocal == translations["notfound"]: main_vocal = get_audio_file('Main_Vocals.')
            if main_vocal == translations["notfound"]: 
                gr_warning(translations["not_found_main_vocal"])
                return return_none
            
            if not not_merge_backing and backing_vocal == translations["notfound"]: 
                gr_warning(translations["not_found_backing_vocal"])
                return return_none
            
            input_path = main_vocal
            backing_path = backing_vocal

        gr_info(translations["convert_vocal"])

        convert(pitch, filter_radius, index_rate, rms_mix_rate, protect, hop_length, f0method, input_path, output_path, model_path, index, autotune, clean, clean_strength, format, embedder_model, resample_sr, split_audio, f0_autotune_strength, checkpointing, onnx_f0_mode, embedders_mode, formant_shifting, formant_qfrency, formant_timbre, f0_file, proposal_pitch, proposal_pitch_threshold, audio_processing, alpha)

        gr_info(translations["convert_success"])

        if convert_backing:
            output_backing = process_output(output_backing)

            gr_info(translations["convert_backup"])

            convert(pitch, filter_radius, index_rate, rms_mix_rate, protect, hop_length, f0method, backing_path, output_backing, model_path, index, autotune, clean, clean_strength, format, embedder_model, resample_sr, split_audio, f0_autotune_strength, checkpointing, onnx_f0_mode, embedders_mode, formant_shifting, formant_qfrency, formant_timbre, f0_file, proposal_pitch, proposal_pitch_threshold, audio_processing, alpha)

            gr_info(translations["convert_backup_success"])

        try:
            if not not_merge_backing and not use_original:
                backing_source = output_backing if convert_backing else backing_vocal

                output_merge_backup = process_output(output_merge_backup)

                gr_info(translations["merge_backup"])

                pydub_load(output_path, volume=-4).overlay(pydub_load(backing_source, volume=-6)).export(output_merge_backup, format=format)

                gr_info(translations["merge_success"])

            if merge_instrument:    
                vocals = output_merge_backup if not not_merge_backing and not use_original else output_path

                output_merge_instrument = process_output(output_merge_instrument)

                gr_info(translations["merge_instruments_process"])

                instruments = get_audio_file('Instruments.')
                
                if instruments == translations["notfound"]: 
                    gr_warning(translations["not_found_instruments"])
                    output_merge_instrument = None
                else: pydub_load(instruments, volume=-7).overlay(pydub_load(vocals, volume=-4 if use_original else None)).export(output_merge_instrument, format=format)
                
                gr_info(translations["merge_success"])
        except:
            return return_none

        return [(None if use_original else output_path), output_backing, (None if not_merge_backing and use_original else output_merge_backup), (output_path if use_original else None), (output_merge_instrument if merge_instrument else None), {"visible": True, "__type__": "update"}]
    else:
        if not input or not os.path.exists(input): 
            gr_warning(translations["input_not_valid"])
            return return_none
        
        if not output:
            gr_warning(translations["output_not_valid"])
            return return_none
        
        output = replace_export_format(output, format)

        if os.path.isdir(input):
            gr_info(translations["is_folder"])

            if not [f for f in os.listdir(input) if f.lower().endswith(("wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"))]:
                gr_warning(translations["not_found_in_folder"])
                return return_none
            
            gr_info(translations["batch_convert"])

            output_dir = os.path.dirname(output) or output
            convert(pitch, filter_radius, index_rate, rms_mix_rate, protect, hop_length, f0method, input, output_dir, model_path, index, autotune, clean, clean_strength, format, embedder_model, resample_sr, split_audio, f0_autotune_strength, checkpointing, onnx_f0_mode, embedders_mode, formant_shifting, formant_qfrency, formant_timbre, f0_file, proposal_pitch, proposal_pitch_threshold, audio_processing, alpha)

            gr_info(translations["batch_convert_success"])

            return return_none
        else:
            output_dir = os.path.dirname(output) or output

            if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
            output = process_output(output)

            gr_info(translations["convert_vocal"])

            convert(pitch, filter_radius, index_rate, rms_mix_rate, protect, hop_length, f0method, input, output, model_path, index, autotune, clean, clean_strength, format, embedder_model, resample_sr, split_audio, f0_autotune_strength, checkpointing, onnx_f0_mode, embedders_mode, formant_shifting, formant_qfrency, formant_timbre, f0_file, proposal_pitch, proposal_pitch_threshold, audio_processing, alpha)

            gr_info(translations["convert_success"])

            return_none[0] = output
            return return_none

def convert_selection(clean, autotune, use_audio, use_original, convert_backing, not_merge_backing, merge_instrument, pitch, clean_strength, model, index, index_rate, input, output, format, method, hybrid_method, hop_length, embedders, custom_embedders, resample_sr, filter_radius, rms_mix_rate, protect, split_audio, f0_autotune_strength, checkpointing, onnx_f0_mode, formant_shifting, formant_qfrency, formant_timbre, f0_file, embedders_mode, proposal_pitch, proposal_pitch_threshold, audio_processing=False, alpha=0.5):
    if use_audio:
        gr_info(translations["search_separate"])
        choice = [f for f in os.listdir(configs["audios_path"]) if os.path.isdir(os.path.join(configs["audios_path"], f))] if config.debug_mode else [f for f in os.listdir(configs["audios_path"]) if os.path.isdir(os.path.join(configs["audios_path"], f)) and any(file.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3")) for file in os.listdir(os.path.join(configs["audios_path"], f)))]

        gr_info(translations["found_choice"].format(choice=len(choice)))

        if len(choice) == 0: 
            gr_warning(translations["separator==0"])

            return [{"choices": [], "value": "", "interactive": False, "visible": False, "__type__": "update"}, None, None, None, None, None, {"visible": True, "__type__": "update"}, {"visible": False, "__type__": "update"}]
        elif len(choice) == 1:
            convert_output = convert_audio(clean, autotune, use_audio, use_original, convert_backing, not_merge_backing, merge_instrument, pitch, clean_strength, model, index, index_rate, None, None, format, method, hybrid_method, hop_length, embedders, custom_embedders, resample_sr, filter_radius, rms_mix_rate, protect, split_audio, f0_autotune_strength, choice[0], checkpointing, onnx_f0_mode, formant_shifting, formant_qfrency, formant_timbre, f0_file, embedders_mode, proposal_pitch, proposal_pitch_threshold, audio_processing, alpha)

            return [{"choices": [], "value": "", "interactive": False, "visible": False, "__type__": "update"}, convert_output[0], convert_output[1], convert_output[2], convert_output[3], convert_output[4], {"visible": True, "__type__": "update"}, {"visible": False, "__type__": "update"}]
        else: return [{"choices": choice, "value": choice[0], "interactive": True, "visible": True, "__type__": "update"}, None, None, None, None, None, {"visible": False, "__type__": "update"}, {"visible": True, "__type__": "update"}]
    else:
        main_convert = convert_audio(clean, autotune, use_audio, use_original, convert_backing, not_merge_backing, merge_instrument, pitch, clean_strength, model, index, index_rate, input, output, format, method, hybrid_method, hop_length, embedders, custom_embedders, resample_sr, filter_radius, rms_mix_rate, protect, split_audio, f0_autotune_strength, None, checkpointing, onnx_f0_mode, formant_shifting, formant_qfrency, formant_timbre, f0_file, embedders_mode, proposal_pitch, proposal_pitch_threshold, audio_processing, alpha)

        return [{"choices": [], "value": "", "interactive": False, "visible": False, "__type__": "update"}, main_convert[0], None, None, None, None, {"visible": True, "__type__": "update"}, {"visible": False, "__type__": "update"}]

def whisper_process(model_size, input_audio, configs, device, out_queue, word_timestamps=True):
    from main.library.speaker_diarization.whisper import load_model

    try:
        segments = load_model(model_size, device=device).transcribe(input_audio, fp16=configs.get("fp16", False), word_timestamps=word_timestamps)
        out_queue.put(segments["segments"])
    except Exception as e:
        out_queue.put(e)
    finally:
        del segments
        gc.collect()

def convert_with_whisper(num_spk, model_size, cleaner, clean_strength, autotune, f0_autotune_strength, checkpointing, model_1, model_2, model_index_1, model_index_2, pitch_1, pitch_2, index_strength_1, index_strength_2, export_format, input_audio, output_audio, onnx_f0_mode, method, hybrid_method, hop_length, embed_mode, embedders, custom_embedders, resample_sr, filter_radius, rms_mix_rate, protect, formant_shifting, formant_qfrency_1, formant_timbre_1, formant_qfrency_2, formant_timbre_2, proposal_pitch, proposal_pitch_threshold, audio_processing=False, alpha=0.5):
    import librosa
    import multiprocessing as mp

    from pydub import AudioSegment
    from sklearn.cluster import AgglomerativeClustering

    from main.library.utils import clear_gpu_cache
    from main.library.speaker_diarization.audio import Audio
    from main.library.speaker_diarization.segment import Segment
    from main.library.utils import check_spk_diarization, pydub_load
    from main.library.speaker_diarization.embedding import SpeechBrainPretrainedSpeakerEmbedding
    
    check_spk_diarization(model_size)
    model_pth_1, model_pth_2 = os.path.join(configs["weights_path"], model_1) if not os.path.exists(model_1) else model_1, os.path.join(configs["weights_path"], model_2) if not os.path.exists(model_2) else model_2

    if (not model_1 or not os.path.exists(model_pth_1) or os.path.isdir(model_pth_1) or not model_pth_1.endswith((".pth", ".onnx"))) and (not model_2 or not os.path.exists(model_pth_2) or os.path.isdir(model_pth_2) or not model_pth_2.endswith((".pth", ".onnx"))):
        gr_warning(translations["provide_file"].format(filename=translations["model"]))
        return None
    
    if not model_1: model_pth_1 = model_pth_2
    if not model_2: model_pth_2 = model_pth_1

    if not input_audio or not os.path.exists(input_audio) or os.path.isdir(input_audio): 
        gr_warning(translations["input_not_valid"])
        return None
        
    if not output_audio:
        gr_warning(translations["output_not_valid"])
        return None
    
    output_audio = process_output(output_audio)
    gr_info(translations["start_whisper"])
    
    try:
        try:
            mp.set_start_method("spawn")
        except:
            pass

        whisper_queue = mp.Queue()
        whisperprocess = mp.Process(target=whisper_process, args=(model_size, input_audio, configs, config.device, whisper_queue, True))
        whisperprocess.start()

        segments = whisper_queue.get()
        audio = Audio()

        embedding_model = SpeechBrainPretrainedSpeakerEmbedding(embedding=os.path.join(configs["speaker_diarization_path"], "models", "speechbrain"), device=config.device)
        y, sr = librosa.load(input_audio, sr=None)  
        duration = len(y) / sr
            
        def segment_embedding(segment):
            waveform, _ = audio.crop(input_audio, Segment(segment["start"], min(duration, segment["end"])))
            return embedding_model(waveform.mean(dim=0, keepdim=True)[None] if waveform.shape[0] == 2 else waveform[None])  
        
        def time(secs):
            return datetime.timedelta(seconds=round(secs))
        
        def merge_audio(files_list, time_stamps, original_file_path, output_path, format):
            def extract_number(filename):
                match = re.search(r'_(\d+)', filename)
                return int(match.group(1)) if match else 0

            total_duration = len(pydub_load(original_file_path))
            combined = AudioSegment.empty() 
            current_position = 0 

            for file, (start_i, end_i) in zip(sorted(files_list, key=extract_number), time_stamps):
                if start_i > current_position: combined += AudioSegment.silent(duration=start_i - current_position)  
                
                combined += pydub_load(file)  
                current_position = end_i

            if current_position < total_duration: combined += AudioSegment.silent(duration=total_duration - current_position)
            combined.export(output_path, format=format)

            return output_path

        embeddings = np.zeros(shape=(len(segments), 192))
        for i, segment in enumerate(segments):
            embeddings[i] = segment_embedding(segment)

        labels = AgglomerativeClustering(num_spk).fit(np.nan_to_num(embeddings)).labels_
        for i in range(len(segments)):
            segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

        merged_segments, current_text = [], []
        current_speaker, current_start = None, None

        for i, segment in enumerate(segments):
            speaker = segment["speaker"]
            start_time = segment["start"]
            text = segment["text"][1:]  

            if speaker == current_speaker:
                current_text.append(text)
                end_time = segment["end"]
            else:
                if current_speaker is not None: merged_segments.append({"speaker": current_speaker, "start": current_start, "end": end_time, "text": " ".join(current_text)})
                
                current_speaker = speaker
                current_start = start_time
                current_text = [text]
                end_time = segment["end"]

        if current_speaker is not None: merged_segments.append({"speaker": current_speaker, "start": current_start, "end": end_time, "text": " ".join(current_text)})

        gr_info(translations["whisper_done"])

        x = ""
        for segment in merged_segments:
            x += f"\n{segment['speaker']} {str(time(segment['start']))} - {str(time(segment['end']))}\n"
            x += segment["text"] + "\n"

        logger.info(x)

        del audio, embedding_model, segments, labels
        clear_gpu_cache()
        gc.collect()

        gr_info(translations["process_audio"])

        audio = pydub_load(input_audio)
        output_folder = "audios_temp"

        if os.path.exists(output_folder): shutil.rmtree(output_folder, ignore_errors=True)
        for f in [output_folder, os.path.join(output_folder, "1"), os.path.join(output_folder, "2")]:
            os.makedirs(f, exist_ok=True)

        time_stamps, processed_segments = [], []
        for i, segment in enumerate(merged_segments):
            start_ms = int(segment["start"] * 1000) 
            end_ms = int(segment["end"] * 1000)

            index = i + 1

            segment_filename = os.path.join(output_folder, "1" if i % 2 == 1 else "2", f"segment_{index}.wav")
            audio[start_ms:end_ms].export(segment_filename, format="wav")

            processed_segments.append(os.path.join(output_folder, "1" if i % 2 == 1 else "2", f"segment_{index}_output.wav"))
            time_stamps.append((start_ms, end_ms))

        f0method, embedder_model = (method if method != "hybrid" else hybrid_method), (embedders if embedders != "custom" else custom_embedders)

        gr_info(translations["process_done_start_convert"])

        convert(pitch_1, filter_radius, index_strength_1, rms_mix_rate, protect, hop_length, f0method, os.path.join(output_folder, "1"), output_folder, model_pth_1, model_index_1, autotune, cleaner, clean_strength, "wav", embedder_model, resample_sr, False, f0_autotune_strength, checkpointing, onnx_f0_mode, embed_mode, formant_shifting, formant_qfrency_1, formant_timbre_1, "", proposal_pitch, proposal_pitch_threshold, audio_processing, alpha)
        convert(pitch_2, filter_radius, index_strength_2, rms_mix_rate, protect, hop_length, f0method, os.path.join(output_folder, "2"), output_folder, model_pth_2, model_index_2, autotune, cleaner, clean_strength, "wav", embedder_model, resample_sr, False, f0_autotune_strength, checkpointing, onnx_f0_mode, embed_mode, formant_shifting, formant_qfrency_2, formant_timbre_2, "", proposal_pitch, proposal_pitch_threshold, audio_processing, alpha)

        gr_info(translations["convert_success"])
        return merge_audio(processed_segments, time_stamps, input_audio, replace_export_format(output_audio, export_format), export_format)
    except Exception as e:
        gr_error(translations["error_occurred"].format(e=e))
        import traceback
        logger.debug(traceback.format_exc())
        return None
    finally:
        if os.path.exists("audios_temp"): shutil.rmtree("audios_temp", ignore_errors=True)

def convert_tts(clean, autotune, pitch, clean_strength, model, index, index_rate, input, output, format, method, hybrid_method, hop_length, embedders, custom_embedders, resample_sr, filter_radius, rms_mix_rate, protect, split_audio, f0_autotune_strength, checkpointing, onnx_f0_mode, formant_shifting, formant_qfrency, formant_timbre, f0_file, embedders_mode, proposal_pitch, proposal_pitch_threshold, audio_processing=False, alpha=0.5):
    model_path = os.path.join(configs["weights_path"], model) if not os.path.exists(model) else model

    if not model_path or not os.path.exists(model_path) or os.path.isdir(model_path) or not model.endswith((".pth", ".onnx")):
        gr_warning(translations["provide_file"].format(filename=translations["model"]))
        return None

    if not input or not os.path.exists(input): 
        gr_warning(translations["input_not_valid"])
        return None
    
    if os.path.isdir(input): 
        input_audio = [f for f in os.listdir(input) if "tts" in f and f.lower().endswith(("wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"))]
        
        if not input_audio:
            gr_warning(translations["not_found_in_folder"])
            return None
        
        input = os.path.join(input, input_audio[0])
    
    if not output:
        gr_warning(translations["output_not_valid"])
        return None
    
    output = replace_export_format(output, format)
    if os.path.isdir(output): output = os.path.join(output, f"tts.{format}")

    output_dir = os.path.dirname(output)
    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    
    output = process_output(output)

    f0method = method if method != "hybrid" else hybrid_method
    embedder_model = embedders if embedders != "custom" else custom_embedders

    gr_info(translations["convert_vocal"])

    convert(pitch, filter_radius, index_rate, rms_mix_rate, protect, hop_length, f0method, input, output, model_path, index, autotune, clean, clean_strength, format, embedder_model, resample_sr, split_audio, f0_autotune_strength, checkpointing, onnx_f0_mode, embedders_mode, formant_shifting, formant_qfrency, formant_timbre, f0_file, proposal_pitch, proposal_pitch_threshold, audio_processing, alpha)

    gr_info(translations["convert_success"])
    return output
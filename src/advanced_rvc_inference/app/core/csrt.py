import os
import sys

sys.path.append(os.getcwd())

from main.app.core.inference import whisper_process
from main.library.utils import check_spk_diarization
from main.app.core.ui import gr_info, gr_warning, process_output
from main.app.variables import config, translations, configs, logger

def create_srt(model_size, input_audio, output_file, word_timestamps):
    import multiprocessing as mp

    if not input_audio or not os.path.exists(input_audio) or os.path.isdir(input_audio): 
        gr_warning(translations["input_not_valid"])
        return [None]*2
    
    if not output_file.endswith(".srt"): output_file += ".srt"
        
    if not output_file:
        gr_warning(translations["output_not_valid"])
        return [None]*2
    
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)

    info = ""
    output_file = process_output(output_file)

    check_spk_diarization(model_size, speechbrain=False)
    gr_info(translations["csrt"])

    try:
        mp.set_start_method("spawn")
    except:
        pass

    whisper_queue = mp.Queue()
    whisperprocess = mp.Process(target=whisper_process, args=(model_size, input_audio, configs, config.device, whisper_queue, word_timestamps))
    whisperprocess.start()

    segments = whisper_queue.get()

    with open(output_file, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments):
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()
            
            index = f"{i+1}\n"
            timestamp = f"{format_timestamp(start)} --> {format_timestamp(end)}\n"
            text1 = f"{text}\n\n"

            f.write(index)
            f.write(timestamp)
            f.write(text1)

            info = info + index + timestamp + text1
        logger.info(info)
    
    gr_info(translations["success"])

    return [{"value": output_file, "visible": True, "__type__": "update"}, info]

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)

    seconds = int(seconds % 60)
    miliseconds = int((seconds - int(seconds)) * 1000)

    return f"{hours:02}:{minutes:02}:{seconds:02},{miliseconds:03}"
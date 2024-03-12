import torch, os, traceback, sys, warnings, shutil, numpy as np
import gradio as gr
import librosa
import asyncio
import rarfile
import edge_tts
import yt_dlp
import ffmpeg
import gdown
import subprocess
import wave
import soundfile as sf
from scipy.io import wavfile
from datetime import datetime
from urllib.parse import urlparse
from mega import Mega

os.system("python models.py")

now_dir = os.getcwd()
tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.environ["TEMP"] = tmp
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from fairseq import checkpoint_utils
from vc_infer_pipeline import VC
from config import Config
config = Config()

tts_voice_list = asyncio.get_event_loop().run_until_complete(edge_tts.list_voices())
voices = [f"{v['ShortName']}-{v['Gender']}" for v in tts_voice_list]

hubert_model = None

f0method_mode = ["pm", "harvest", "crepe"]
f0method_info = "PM is fast, Harvest is good but extremely slow, and Crepe effect is good but requires GPU (Default: PM)"

if os.path.isfile("rmvpe.pt"):
    f0method_mode.insert(2, "rmvpe")
    f0method_info = "PM is fast, Harvest is good but extremely slow, Rvmpe is alternative to harvest (might be better), and Crepe effect is good but requires GPU (Default: PM)"

def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()

load_hubert()

weight_root = "weights"
index_root = "weights/index"
weights_model = []
weights_index = []
for _, _, model_files in os.walk(weight_root):
    for file in model_files:
        if file.endswith(".pth"):
            weights_model.append(file)
for _, _, index_files in os.walk(index_root):
    for file in index_files:
        if file.endswith('.index') and "trained" not in file:
            weights_index.append(os.path.join(index_root, file))

def check_models():
    weights_model = []
    weights_index = []
    for _, _, model_files in os.walk(weight_root):
        for file in model_files:
            if file.endswith(".pth"):
                weights_model.append(file)
    for _, _, index_files in os.walk(index_root):
        for file in index_files:
            if file.endswith('.index') and "trained" not in file:
                weights_index.append(os.path.join(index_root, file))
    return (
        gr.Dropdown.update(choices=sorted(weights_model), value=weights_model[0]),
        gr.Dropdown.update(choices=sorted(weights_index))
    )

def clean():
    return (
        gr.Dropdown.update(value=""),
        gr.Slider.update(visible=False)
    )

def vc_single(
    sid,
    vc_audio_mode,
    input_audio_path,
    input_upload_audio,
    vocal_audio,
    tts_text,
    tts_voice,
    f0_up_key,
    f0_file,
    f0_method,
    file_index,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect
):  # spk_item, input_audio0, vc_transform0,f0_file,f0method0
    global tgt_sr, net_g, vc, hubert_model, version, cpt
    try:
        logs = []
        print(f"Converting...")
        logs.append(f"Converting...")
        yield "\n".join(logs), None
        if vc_audio_mode == "Input path" or "Youtube" and input_audio_path != "":
            audio, sr = librosa.load(input_audio_path, sr=16000, mono=True)
        elif vc_audio_mode == "Upload audio":
            selected_audio = input_upload_audio
            if vocal_audio:
                selected_audio = vocal_audio
            elif input_upload_audio:
                selected_audio = input_upload_audio
            sampling_rate, audio = selected_audio
            duration = audio.shape[0] / sampling_rate
            audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio.transpose(1, 0))
            if sampling_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
        elif vc_audio_mode == "TTS Audio":
            if tts_text is None or tts_voice is None:
                return "You need to enter text and select a voice", None
            asyncio.run(edge_tts.Communicate(tts_text, "-".join(tts_voice.split('-')[:-1])).save("tts.mp3"))
            audio, sr = librosa.load("tts.mp3", sr=16000, mono=True)
            input_audio_path = "tts.mp3"
        f0_up_key = int(f0_up_key)
        times = [0, 0, 0]
        if hubert_model == None:
            load_hubert()
        if_f0 = cpt.get("f0", 1)
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            sid,
            audio,
            input_audio_path,
            times,
            f0_up_key,
            f0_method,
            file_index,
            # file_big_npy,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            f0_file=f0_file
        )
        if resample_sr >= 16000 and tgt_sr != resample_sr:
            tgt_sr = resample_sr
        index_info = (
            "Using index:%s." % file_index
            if os.path.exists(file_index)
            else "Index not used."
        )
        print("Success.\n %s\nTime:\n npy:%ss, f0:%ss, infer:%ss" % (
            index_info,
            times[0],
            times[1],
            times[2],
        ))
        info = f"{index_info}\n[{datetime.now().strftime('%Y-%m-%d %H:%M')}]: npy: {times[0]}, f0: {times[1]}s, infer: {times[2]}s"
        logs.append(info)
        yield "\n".join(logs), (tgt_sr, audio_opt)
    except:
        info = traceback.format_exc()
        print(info)
        logs.append(info)
        yield "\n".join(logs), None

def get_vc(sid, to_return_protect0):
    global n_spk, tgt_sr, net_g, vc, cpt, version, weights_index
    if sid == "" or sid == []:
        global hubert_model
        if hubert_model is not None:  # 考虑到轮询, 需要加个判断看是否 sid 是由有模型切换到无模型的
            print("clean_empty_cache")
            del net_g, n_spk, vc, hubert_model, tgt_sr  # ,cpt
            hubert_model = net_g = n_spk = vc = hubert_model = tgt_sr = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            ###楼下不这么折腾清理不干净
            if_f0 = cpt.get("f0", 1)
            version = cpt.get("version", "v1")
            if version == "v1":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs256NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
            elif version == "v2":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs768NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
            del net_g, cpt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cpt = None
        return (
            gr.Slider.update(maximum=2333, visible=False),
            gr.Slider.update(visible=True),
            gr.Dropdown.update(choices=sorted(weights_index), value=""),
            gr.Markdown.update(value="# <center> No model selected")
        )
    print(f"Loading {sid} model...")
    selected_model = sid[:-4]
    cpt = torch.load(os.path.join(weight_root, sid), map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    if if_f0 == 0:
        to_return_protect0 = {
            "visible": False,
            "value": 0.5,
            "__type__": "update",
        }
    else:
        to_return_protect0 = {
            "visible": True,
            "value": to_return_protect0,
            "__type__": "update",
        }
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]
    weights_index = []
    for _, _, index_files in os.walk(index_root):
        for file in index_files:
            if file.endswith('.index') and "trained" not in file:
                weights_index.append(os.path.join(index_root, file))
    if weights_index == []:
        selected_index = gr.Dropdown.update(value="")
    else:
        selected_index = gr.Dropdown.update(value=weights_index[0])
    for index, model_index in enumerate(weights_index):
        if selected_model in model_index:
            selected_index = gr.Dropdown.update(value=weights_index[index])
            break
    return (
        gr.Slider.update(maximum=n_spk, visible=True),
        to_return_protect0,
        selected_index,
        gr.Markdown.update(
            f'## <center> {selected_model}\n'+
            f'### <center> RVC {version} Model'
        )
    )

def find_audio_files(folder_path, extensions):
    audio_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                audio_files.append(file)
    return audio_files

def vc_multi(
    spk_item,
    vc_input,
    vc_output,
    vc_transform0,
    f0method0,
    file_index,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
):
    global tgt_sr, net_g, vc, hubert_model, version, cpt
    logs = []
    logs.append("Converting...")
    yield "\n".join(logs)
    print()
    try:
        if os.path.exists(vc_input):
            folder_path = vc_input
            extensions = [".mp3", ".wav", ".flac", ".ogg"]
            audio_files = find_audio_files(folder_path, extensions)
            for index, file in enumerate(audio_files, start=1):
                audio, sr = librosa.load(os.path.join(folder_path, file), sr=16000, mono=True)
                input_audio_path = folder_path, file
                f0_up_key = int(vc_transform0)
                times = [0, 0, 0]
                if hubert_model == None:
                    load_hubert()
                if_f0 = cpt.get("f0", 1)
                audio_opt = vc.pipeline(
                    hubert_model,
                    net_g,
                    spk_item,
                    audio,
                    input_audio_path,
                    times,
                    f0_up_key,
                    f0method0,
                    file_index,
                    index_rate,
                    if_f0,
                    filter_radius,
                    tgt_sr,
                    resample_sr,
                    rms_mix_rate,
                    version,
                    protect,
                    f0_file=None
                )
                if resample_sr >= 16000 and tgt_sr != resample_sr:
                    tgt_sr = resample_sr
                output_path = f"{os.path.join(vc_output, file)}"
                os.makedirs(os.path.join(vc_output), exist_ok=True)
                sf.write(
                    output_path,
                    audio_opt,
                    tgt_sr,
                )
                info = f"{index} / {len(audio_files)} | {file}"
                print(info)
                logs.append(info)
                yield "\n".join(logs)
        else:
            logs.append("Folder not found or path doesn't exist.")
            yield "\n".join(logs)
    except:
        info = traceback.format_exc()
        print(info)
        logs.append(info)
        yield "\n".join(logs)

def download_audio(url, audio_provider):
    logs = []
    os.makedirs("dl_audio", exist_ok=True)
    if url == "":
        logs.append("URL required!")
        yield None, "\n".join(logs)
        return None, "\n".join(logs)
    if audio_provider == "Youtube":
        logs.append("Downloading the audio...")
        yield None, "\n".join(logs)
        ydl_opts = {
            'noplaylist': True,
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            "outtmpl": 'result/dl_audio/audio',
        }
        audio_path = "result/dl_audio/audio.wav"
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        logs.append("Download Complete.")
        yield audio_path, "\n".join(logs)

def cut_vocal_and_inst_yt(split_model):
    logs = []
    logs.append("Starting the audio splitting process...")
    yield "\n".join(logs), None, None, None
    command = f"demucs --two-stems=vocals -n {split_model} result/dl_audio/audio.wav -o output"
    result = subprocess.Popen(command.split(), stdout=subprocess.PIPE, text=True)
    for line in result.stdout:
        logs.append(line)
        yield "\n".join(logs), None, None, None
    print(result.stdout)
    vocal = f"output/{split_model}/audio/vocals.wav"
    inst = f"output/{split_model}/audio/no_vocals.wav"
    logs.append("Audio splitting complete.")
    yield "\n".join(logs), vocal, inst, vocal

def cut_vocal_and_inst(split_model, audio_data):
    logs = []
    vocal_path = "output/result/audio.wav"
    os.makedirs("output/result", exist_ok=True)
    wavfile.write(vocal_path, audio_data[0], audio_data[1])
    logs.append("Starting the audio splitting process...")
    yield "\n".join(logs), None, None
    command = f"demucs --two-stems=vocals -n {split_model} {vocal_path} -o output"
    result = subprocess.Popen(command.split(), stdout=subprocess.PIPE, text=True)
    for line in result.stdout:
        logs.append(line)
        yield "\n".join(logs), None, None
    print(result.stdout)
    vocal = f"output/{split_model}/audio/vocals.wav"
    inst = f"output/{split_model}/audio/no_vocals.wav"
    logs.append("Audio splitting complete.")
    yield "\n".join(logs), vocal, inst
    
def combine_vocal_and_inst(audio_data, vocal_volume, inst_volume, split_model):
    os.makedirs("output/result", exist_ok=True)
    vocal_path = "output/result/output.wav"
    output_path = "output/result/combine.mp3"
    inst_path = f"output/{split_model}/audio/no_vocals.wav"
    wavfile.write(vocal_path, audio_data[0], audio_data[1])
    command =  f'ffmpeg -y -i {inst_path} -i {vocal_path} -filter_complex [0:a]volume={inst_volume}[i];[1:a]volume={vocal_volume}[v];[i][v]amix=inputs=2:duration=longest[a] -map [a] -b:a 320k -c:a libmp3lame {output_path}'
    result = subprocess.run(command.split(), stdout=subprocess.PIPE)
    print(result.stdout.decode())
    return output_path

def download_and_extract_models(urls):
    logs = []
    os.makedirs("zips", exist_ok=True)
    os.makedirs(os.path.join("zips", "extract"), exist_ok=True)
    os.makedirs(os.path.join(weight_root), exist_ok=True)
    os.makedirs(os.path.join(index_root), exist_ok=True)
    for link in urls.splitlines():
        url = link.strip()
        if not url:
            raise gr.Error("URL Required!")
            return "No URLs provided."
        model_zip = urlparse(url).path.split('/')[-2] + '.zip'
        model_zip_path = os.path.join('zips', model_zip)
        logs.append(f"Downloading...")
        yield "\n".join(logs)
        if "drive.google.com" in url:
            gdown.download(url, os.path.join("zips", "extract"), quiet=False)
        elif "mega.nz" in url:
            m = Mega()
            m.download_url(url, 'zips')
        else:
            os.system(f"wget {url} -O {model_zip_path}")
        logs.append(f"Extracting...")
        yield "\n".join(logs)
        for filename in os.listdir("zips"):
            archived_file = os.path.join("zips", filename)
            if filename.endswith(".zip"):
                shutil.unpack_archive(archived_file, os.path.join("zips", "extract"), 'zip')
            elif filename.endswith(".rar"):
                with rarfile.RarFile(archived_file, 'r') as rar:
                    rar.extractall(os.path.join("zips", "extract"))
        for _, dirs, files in os.walk(os.path.join("zips", "extract")):
            logs.append(f"Searching Model and Index...")
            yield "\n".join(logs)
            model = False
            index = False
            if files:
                for file in files:
                    if file.endswith(".pth"):
                        basename = file[:-4]
                        shutil.move(os.path.join("zips", "extract", file), os.path.join(weight_root, file))
                        model = True
                    if file.endswith('.index') and "trained" not in file:
                        shutil.move(os.path.join("zips", "extract", file), os.path.join(index_root, file))
                        index = True
            else:
                logs.append("No model in main folder.")
                yield "\n".join(logs)
                logs.append("Searching in subfolders...")
                yield "\n".join(logs)
                for sub_dir in dirs:
                    for _, _, sub_files in os.walk(os.path.join("zips", "extract", sub_dir)):
                        for file in sub_files:
                            if file.endswith(".pth"):
                                basename = file[:-4]
                                shutil.move(os.path.join("zips", "extract", sub_dir, file), os.path.join(weight_root, file))
                                model = True
                            if file.endswith('.index') and "trained" not in file:
                                shutil.move(os.path.join("zips", "extract", sub_dir, file), os.path.join(index_root, file))
                                index = True  
                        shutil.rmtree(os.path.join("zips", "extract", sub_dir))
            if index is False:
                logs.append("Model only file, no Index file detected.")
                yield "\n".join(logs)
        logs.append("Download Completed!")
        yield "\n".join(logs)
    logs.append("Successfully download all models! Refresh your model list to load the model")
    yield "\n".join(logs)

def use_microphone(microphone):
    if microphone == True:
        return gr.Audio.update(source="microphone")
    else:
        return gr.Audio.update(source="upload")

def change_audio_mode(vc_audio_mode):
    if vc_audio_mode == "Input path":
        return (
            # Input & Upload
            gr.Textbox.update(visible=True),
            gr.Checkbox.update(visible=False),
            gr.Audio.update(visible=False),
            # Youtube
            gr.Dropdown.update(visible=False),
            gr.Textbox.update(visible=False),
            gr.Textbox.update(visible=False),
            gr.Button.update(visible=False),
            # Splitter
            gr.Dropdown.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Button.update(visible=True),
            gr.Button.update(visible=False),
            gr.Audio.update(visible=False),
            gr.Audio.update(visible=True),
            gr.Audio.update(visible=True),
            gr.Slider.update(visible=True),
            gr.Slider.update(visible=True),
            gr.Audio.update(visible=True),
            gr.Button.update(visible=True),
            # TTS
            gr.Textbox.update(visible=False),
            gr.Dropdown.update(visible=False)
        )
    elif vc_audio_mode == "Upload audio":
        return (
            # Input & Upload
            gr.Textbox.update(visible=False),
            gr.Checkbox.update(visible=True),
            gr.Audio.update(visible=True),
            # Youtube
            gr.Dropdown.update(visible=False),
            gr.Textbox.update(visible=False),
            gr.Textbox.update(visible=False),
            gr.Button.update(visible=False),
            # Splitter
            gr.Dropdown.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Button.update(visible=False),
            gr.Button.update(visible=True),
            gr.Audio.update(visible=False),
            gr.Audio.update(visible=True),
            gr.Audio.update(visible=True),
            gr.Slider.update(visible=True),
            gr.Slider.update(visible=True),
            gr.Audio.update(visible=True),
            gr.Button.update(visible=True),
            # TTS
            gr.Textbox.update(visible=False),
            gr.Dropdown.update(visible=False)
        )
    elif vc_audio_mode == "Youtube":
        return (
            # Input & Upload
            gr.Textbox.update(visible=False),
            gr.Checkbox.update(visible=False),
            gr.Audio.update(visible=False),
            # Youtube
            gr.Dropdown.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Button.update(visible=True),
            # Splitter
            gr.Dropdown.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Button.update(visible=True),
            gr.Button.update(visible=False),
            gr.Audio.update(visible=True),
            gr.Audio.update(visible=True),
            gr.Audio.update(visible=True),
            gr.Slider.update(visible=True),
            gr.Slider.update(visible=True),
            gr.Audio.update(visible=True),
            gr.Button.update(visible=True),
            # TTS
            gr.Textbox.update(visible=False),
            gr.Dropdown.update(visible=False)
        )
    elif vc_audio_mode == "TTS Audio":
        return (
            # Input & Upload
            gr.Textbox.update(visible=False),
            gr.Checkbox.update(visible=False),
            gr.Audio.update(visible=False),
            # Youtube
            gr.Dropdown.update(visible=False),
            gr.Textbox.update(visible=False),
            gr.Textbox.update(visible=False),
            gr.Button.update(visible=False),
            # Splitter
            gr.Dropdown.update(visible=False),
            gr.Textbox.update(visible=False),
            gr.Button.update(visible=False),
            gr.Button.update(visible=False),
            gr.Audio.update(visible=False),
            gr.Audio.update(visible=False),
            gr.Audio.update(visible=False),
            gr.Slider.update(visible=False),
            gr.Slider.update(visible=False),
            gr.Audio.update(visible=False),
            gr.Button.update(visible=False),
            # TTS
            gr.Textbox.update(visible=True),
            gr.Dropdown.update(visible=True)
        )
        
with gr.Blocks() as app:
    gr.Markdown(
        "# <center> Advanced RVC Inference\n"
    )
    with gr.Row():
        sid = gr.Dropdown(
            label="Weight",
            choices=sorted(weights_model),
        )
        file_index = gr.Dropdown(
            label="List of index file",
            choices=sorted(weights_index),
            interactive=True,
        )
        spk_item = gr.Slider(
            minimum=0,
            maximum=2333,
            step=1,
            label="Speaker ID",
            value=0,
            visible=False,
            interactive=True,
        )
        refresh_model = gr.Button("Refresh model list", variant="primary")
        clean_button = gr.Button("Clear Model from memory", variant="primary")
        refresh_model.click(
            fn=check_models, inputs=[], outputs=[sid, file_index]
        )
        clean_button.click(fn=clean, inputs=[], outputs=[sid, spk_item])
    with gr.TabItem("Inference"):
        selected_model = gr.Markdown(value="# <center> No model selected")
        with gr.Row():
            with gr.Column():
                vc_audio_mode = gr.Dropdown(label="Input voice", choices=["Input path", "Upload audio", "Youtube", "TTS Audio"], allow_custom_value=False, value="Upload audio")
                # Input
                vc_input = gr.Textbox(label="Input audio path", visible=False)
                # Upload
                vc_microphone_mode = gr.Checkbox(label="Use Microphone", value=False, visible=True, interactive=True)
                vc_upload = gr.Audio(label="Upload audio file", source="upload", visible=True, interactive=True)
                # Youtube
                vc_download_audio = gr.Dropdown(label="Provider", choices=["Youtube"], allow_custom_value=False, visible=False, value="Youtube", info="Select provider (Default: Youtube)")
                vc_link = gr.Textbox(label="Youtube URL", visible=False, info="Example: https://www.youtube.com/watch?v=Nc0sB1Bmf-A", placeholder="https://www.youtube.com/watch?v=...")
                vc_log_yt = gr.Textbox(label="Output Information", visible=False, interactive=False)
                vc_download_button = gr.Button("Download Audio", variant="primary", visible=False)
                vc_audio_preview = gr.Audio(label="Downloaded Audio Preview", visible=False)
                # TTS
                tts_text = gr.Textbox(label="TTS text", info="Text to speech input", visible=False)
                tts_voice = gr.Dropdown(label="Edge-tts speaker", choices=voices, visible=False, allow_custom_value=False, value="en-US-AnaNeural-Female")
                # Splitter
                vc_split_model = gr.Dropdown(label="Splitter Model", choices=["hdemucs_mmi", "htdemucs", "htdemucs_ft", "mdx", "mdx_q", "mdx_extra_q"], allow_custom_value=False, visible=True, value="htdemucs", info="Select the splitter model (Default: htdemucs)")
                vc_split_log = gr.Textbox(label="Output Information", visible=True, interactive=False)
                vc_split_yt = gr.Button("Split Audio", variant="primary", visible=False)
                vc_split = gr.Button("Split Audio", variant="primary", visible=True)
                vc_vocal_preview = gr.Audio(label="Vocal Preview", interactive=False, visible=True)
                vc_inst_preview = gr.Audio(label="Instrumental Preview", interactive=False, visible=True)
            with gr.Column():
                vc_transform0 = gr.Number(
                    label="Transpose", 
                    info='Type "12" to change from male to female convertion or Type "-12" to change female to male convertion.',
                    value=0
                )
                f0method0 = gr.Radio(
                    label="Pitch extraction algorithm",
                    info=f0method_info,
                    choices=f0method_mode,
                    value="pm",
                    interactive=True,
                )
                index_rate0 = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Retrieval feature ratio",
                    value=0.7,
                    interactive=True,
                )
                filter_radius0 = gr.Slider(
                    minimum=0,
                    maximum=7,
                    label="Apply Median Filtering",
                    info="The value represents the filter radius and can reduce breathiness.",
                    value=3,
                    step=1,
                    interactive=True,
                )
                resample_sr0 = gr.Slider(
                    minimum=0,
                    maximum=48000,
                    label="Resample the output audio",
                    info="Resample the output audio in post-processing to the final sample rate. Set to 0 for no resampling",
                    value=0,
                    step=1,
                    interactive=True,
                )
                rms_mix_rate0 = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Volume Envelope",
                    info="Use the volume envelope of the input to replace or mix with the volume envelope of the output. The closer the ratio is to 1, the more the output envelope is used",
                    value=1,
                    interactive=True,
                )
                protect0 = gr.Slider(
                    minimum=0,
                    maximum=0.5,
                    label="Voice Protection",
                    info="Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy",
                    value=0.5,
                    step=0.01,
                    interactive=True,
                )
                f0_file0 = gr.File(
                    label="F0 curve file (Optional)",
                    info="One pitch per line, Replace the default F0 and pitch modulation"
                )
            with gr.Column():
                vc_log = gr.Textbox(label="Output Information", interactive=False)
                vc_output = gr.Audio(label="Output Audio", interactive=False)
                vc_convert = gr.Button("Convert", variant="primary")
                vc_vocal_volume = gr.Slider(
                    minimum=0,
                    maximum=10,
                    label="Vocal volume",
                    value=1,
                    interactive=True,
                    step=1,
                    info="Adjust vocal volume (Default: 1}",
                    visible=True
                )
                vc_inst_volume = gr.Slider(
                    minimum=0,
                    maximum=10,
                    label="Instrument volume",
                    value=1,
                    interactive=True,
                    step=1,
                    info="Adjust instrument volume (Default: 1}",
                    visible=True
                )
                vc_combined_output = gr.Audio(label="Output Combined Audio", visible=True)
                vc_combine =  gr.Button("Combine",variant="primary", visible=True)
        vc_convert.click(
            vc_single,
            [
                spk_item,
                vc_audio_mode,
                vc_input,
                vc_upload,
                vc_vocal_preview,
                tts_text,
                tts_voice,
                vc_transform0,
                f0_file0,
                f0method0,
                file_index,
                index_rate0,
                filter_radius0,
                resample_sr0,
                rms_mix_rate0,
                protect0,
            ],
            [vc_log, vc_output],
        )
        vc_download_button.click(
            fn=download_audio, 
            inputs=[vc_link, vc_download_audio], 
            outputs=[vc_audio_preview, vc_log_yt]
        )
        vc_split_yt.click(
            fn=cut_vocal_and_inst_yt, 
            inputs=[vc_split_model], 
            outputs=[vc_split_log, vc_vocal_preview, vc_inst_preview, vc_input]
        )
        vc_split.click(
            fn=cut_vocal_and_inst, 
            inputs=[vc_split_model, vc_upload],
            outputs=[vc_split_log, vc_vocal_preview, vc_inst_preview]
        )
        vc_combine.click(
            fn=combine_vocal_and_inst,
            inputs=[vc_output, vc_vocal_volume, vc_inst_volume, vc_split_model],
            outputs=[vc_combined_output]
        )
        vc_microphone_mode.change(
            fn=use_microphone,
            inputs=vc_microphone_mode,
            outputs=vc_upload
        )
        vc_audio_mode.change(
            fn=change_audio_mode,
            inputs=[vc_audio_mode],
            outputs=[
                # Input & Upload
                vc_input,
                vc_microphone_mode,
                vc_upload,
                # Youtube
                vc_download_audio,
                vc_link,
                vc_log_yt,
                vc_download_button,
                # Splitter
                vc_split_model,
                vc_split_log,
                vc_split_yt,
                vc_split,
                vc_audio_preview,
                vc_vocal_preview,
                vc_inst_preview,
                vc_vocal_volume,
                vc_inst_volume,
                vc_combined_output,
                vc_combine,
                # TTS
                tts_text,
                tts_voice
            ]
        )
        sid.change(fn=get_vc, inputs=[sid, protect0], outputs=[spk_item, protect0, file_index, selected_model])
    with gr.TabItem("Batch Inference"):
        with gr.Row():
            with gr.Column():
                vc_input_bat = gr.Textbox(label="Input audio path (folder)", visible=True)
                vc_output_bat = gr.Textbox(label="Output audio path (folder)", value="result/batch", visible=True)
            with gr.Column():
                vc_transform0_bat = gr.Number(
                    label="Transpose", 
                    info='Type "12" to change from male to female convertion or Type "-12" to change female to male convertion.',
                    value=0
                )
                f0method0_bat = gr.Radio(
                    label="Pitch extraction algorithm",
                    info=f0method_info,
                    choices=f0method_mode,
                    value="pm",
                    interactive=True,
                )
                index_rate0_bat = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Retrieval feature ratio",
                    value=0.7,
                    interactive=True,
                )
                filter_radius0_bat = gr.Slider(
                    minimum=0,
                    maximum=7,
                    label="Apply Median Filtering",
                    info="The value represents the filter radius and can reduce breathiness.",
                    value=3,
                    step=1,
                    interactive=True,
                )
                resample_sr0_bat = gr.Slider(
                    minimum=0,
                    maximum=48000,
                    label="Resample the output audio",
                    info="Resample the output audio in post-processing to the final sample rate. Set to 0 for no resampling",
                    value=0,
                    step=1,
                    interactive=True,
                )
                rms_mix_rate0_bat = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Volume Envelope",
                    info="Use the volume envelope of the input to replace or mix with the volume envelope of the output. The closer the ratio is to 1, the more the output envelope is used",
                    value=1,
                    interactive=True,
                )
                protect0_bat = gr.Slider(
                    minimum=0,
                    maximum=0.5,
                    label="Voice Protection",
                    info="Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy",
                    value=0.5,
                    step=0.01,
                    interactive=True,
                )
            with gr.Column():
                vc_log_bat = gr.Textbox(label="Output Information", interactive=False)
                vc_convert_bat = gr.Button("Convert", variant="primary")
        vc_convert_bat.click(
            vc_multi,
            [
                spk_item,
                vc_input_bat,
                vc_output_bat,
                vc_transform0_bat,
                f0method0_bat,
                file_index,
                index_rate0_bat,
                filter_radius0_bat,
                resample_sr0_bat,
                rms_mix_rate0_bat,
                protect0_bat,
            ],
            [vc_log_bat],
        )
    with gr.TabItem("Model Downloader"):
        gr.Markdown(
            "# <center> Model Downloader (Beta)\n"+
            "#### <center> To download multi link you have to put your link to the textbox and every link separated by space\n"+
            "#### <center> Support Direct Link, Mega, Google Drive, etc"
        )
        with gr.Column():
            md_text = gr.Textbox(label="URL")
        with gr.Row():
            md_download = gr.Button(label="Convert", variant="primary")
            md_download_logs = gr.Textbox(label="Output information", interactive=False)
            md_download.click(
                fn=download_and_extract_models,
                inputs=[md_text],
                outputs=[md_download_logs]
            )
    with gr.TabItem("Settings"):
        gr.Markdown(
            "# <center> Settings\n"+
            "#### <center> Work in progress"
        )
    app.queue(concurrency_count=1, max_size=50, api_open=config.api).launch(share=config.colab)

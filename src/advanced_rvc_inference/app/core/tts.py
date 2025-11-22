import os
import sys
import pysrt
import codecs
import librosa
import asyncio
import requests
import tempfile

sys.path.append(os.getcwd())

from main.app.variables import translations
from main.app.core.ui import gr_info, gr_warning, gr_error

def synthesize_tts(prompt, voice, speed, output, pitch, google):
    if not google: 
        from edge_tts import Communicate
        asyncio.run(Communicate(text=prompt, voice=voice, rate=f"+{speed}%" if speed >= 0 else f"{speed}%", pitch=f"+{pitch}Hz" if pitch >= 0 else f"{pitch}Hz").save(output))
    else: 
        response = requests.get(codecs.decode("uggcf://genafyngr.tbbtyr.pbz/genafyngr_ggf", "rot13"), params={"ie": "UTF-8", "q": prompt, "tl": voice, "ttsspeed": speed, "client": "tw-ob"}, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"})

        if response.status_code == 200:
            with open(output, "wb") as f:
                f.write(response.content)

            if pitch != 0 or speed != 0:
                y, sr = librosa.load(output, sr=None)

                if pitch != 0: y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch)
                if speed != 0: y = librosa.effects.time_stretch(y, rate=speed)

                import soundfile as sf
                sf.write(file=output, data=y, samplerate=sr, format=os.path.splitext(os.path.basename(output))[-1].lower().replace('.', ''))
        else: gr_error(f"{response.status_code}, {response.text}")

def srt_tts(srt_file, out_file, voice, rate = 0, sr = 24000, google = False):
    import numpy as np
    import soundfile as sf

    def time_stretch(y, sr, target_duration):
        rate = (len(y) / sr) / target_duration
        if rate != 1.0: y = librosa.effects.time_stretch(y=y.astype(np.float32), rate=rate)

        n_target = int(round(target_duration * sr))
        return np.pad(y, (0, n_target - len(y))) if len(y) < n_target else y[:n_target]

    def pysrttime_to_seconds(t):
        return (t.hours * 60 + t.minutes) * 60 + t.seconds + t.milliseconds / 1000

    subs = pysrt.open(srt_file)
    if not subs: raise ValueError(translations["srt"])

    final_audio = np.zeros(int(round(pysrttime_to_seconds(subs[-1].end) * sr)), dtype=np.float32)

    with tempfile.TemporaryDirectory() as tempdir:
        for idx, seg in enumerate(subs):
            wav_path = os.path.join(tempdir, f"seg_{idx}.wav")
            synthesize_tts(" ".join(seg.text.splitlines()), voice, 0, wav_path, rate, google)

            audio, file_sr = sf.read(wav_path, dtype=np.float32)
            if file_sr != sr: audio = np.interp(np.linspace(0, len(audio) - 1, int(len(audio) * sr / file_sr)), np.arange(len(audio)), audio)
            adjusted = time_stretch(audio, sr, pysrttime_to_seconds(seg.duration))

            start_sample = int(round(pysrttime_to_seconds(seg.start) * sr))
            end_sample = start_sample + adjusted.shape[0]

            if end_sample > final_audio.shape[0]:
                adjusted = adjusted[: final_audio.shape[0] - start_sample]
                end_sample = final_audio.shape[0]

            final_audio[start_sample:end_sample] += adjusted

    sf.write(out_file, final_audio, sr)

def TTS(prompt, voice, speed, output, pitch, google, srt_input):
    if not srt_input: srt_input = ""

    if not prompt and not srt_input.endswith(".srt"):
        gr_warning(translations["enter_the_text"])
        return None
    
    if not voice:
        gr_warning(translations["choose_voice"])
        return None
    
    if not output: 
        gr_warning(translations["output_not_valid"])
        return None
    
    if os.path.isdir(output): output = os.path.join(output, f"tts.wav")
    gr_info(translations["convert"].format(name=translations["text"]))

    output_dir = os.path.dirname(output) or output
    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)

    if srt_input.endswith(".srt"): srt_tts(srt_input, output, voice, 0, 24000, google)
    else: synthesize_tts(prompt, voice, speed, output, pitch, google)

    gr_info(translations["success"])
    return output
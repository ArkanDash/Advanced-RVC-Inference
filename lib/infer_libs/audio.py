import numpy as np
import av
import ffmpeg
import os
import traceback
import sys
import subprocess

platform_stft_mapping = {
    'linux': os.path.join(os.getcwd(), 'stftpitchshift'),
    'darwin': os.path.join(os.getcwd(), 'stftpitchshift'),
    'win32': os.path.join(os.getcwd(), 'stftpitchshift.exe'),
}

stft = platform_stft_mapping.get(sys.platform)

def wav2(i, o, format):
    inp = av.open(i, 'rb')
    if format == "m4a": format = "mp4"
    out = av.open(o, 'wb', format=format)
    if format == "ogg": format = "libvorbis"
    if format == "mp4": format = "aac"

    ostream = out.add_stream(format)

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame): out.mux(p)

    for p in ostream.encode(None): out.mux(p)

    out.close()
    inp.close()

def load_audio(file, sr, DoFormant=False, Quefrency=1.0, Timbre=1.0):
    formanted = False
    file = file.strip(' \n"')
    if not os.path.exists(file):
        raise RuntimeError(
            "Wrong audio path, that does not exist."
        )

    try:
        if DoFormant:
            print("Starting formant shift. Please wait as this process takes a while.")
            formanted_file = f"{os.path.splitext(os.path.basename(file))[0]}_formanted{os.path.splitext(os.path.basename(file))[1]}"
            command = (
                f'{stft} -i "{file}" -q "{Quefrency}" '
                f'-t "{Timbre}" -o "{formanted_file}"'
            )
            subprocess.run(command, shell=True)
            file = formanted_file
            print(f"Formanted {file}\n")

        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = (
            file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # Prevent small white copy path head and tail with spaces and " and return
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )

        return np.frombuffer(out, np.float32).flatten()

    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

def check_audio_duration(file):
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")

        probe = ffmpeg.probe(file)

        duration = float(probe['streams'][0]['duration'])

        if duration < 0.76:
            print(
                f"Audio file, {file.split('/')[-1]}, under ~0.76s detected - file is too short. Target at least 1-2s for best results."
            )
            return False

        return True
    except Exception as e:
        raise RuntimeError(f"Failed to check audio duration: {e}")
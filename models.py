

import subprocess



command1 = [
    "aria2c",
    "--console-log-level=error",
    "-c",
    "-x", "16",
    "-s", "16",
    "-k", "1M",
    "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt",
    "-d", "/content/Advanced-RVC-Inference",
    "-o", "hubert_base.pt"
]


command2 = [
    "aria2c",
    "--console-log-level=error",
    "-c",
    "-x", "16",
    "-s", "16",
    "-k", "1M",
    "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
    "-d", "/content/Advanced-RVC-Inference",
    "-o", "rmvpe.pt"
]


subprocess.run(command1)


subprocess.run(command2)

print("done")

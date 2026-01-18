import os 

os.system("aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt -d . -o hubert_base.pt")
os.system("aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt -d . -o rmvpe.pt")
os.system("aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/sail-rvc/yoimiya-jp/resolve/main/model.pth -d ./weights -o yoimiya.pth")
os.system("aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/sail-rvc/yoimiya-jp/resolve/main/model.index -d ./weights/index -o yoimiya.index")

os.system("python infer.py")
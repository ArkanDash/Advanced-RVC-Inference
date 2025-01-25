import subprocess
import os

def download_file(url, output_name, destination):
    command = [
        "aria2c",
        "--console-log-level=error",
        "-c",
        "-x", "16",
        "-s", "16",
        "-k", "1M",
        url,
        "-d", destination,
        "-o", output_name
    ]
    subprocess.run(command)

if __name__ == "__main__":
    current_directory = os.getcwd()
    
    # List of files to download
    files_to_download = [
        {
            "url": "https://huggingface.co/theNeofr/rvc-base/resolve/main/hubert_base.pt",
            "output_name": "hubert_base.pt"
        },
        {
            "url": "https://huggingface.co/theNeofr/rvc-base/resolve/main/rmvpe.pt",
            "output_name": "rmvpe.pt"
        },
        {
            "url": "https://huggingface.co/theNeofr/rvc-base/resolve/main/fcpe.pt",
            "output_name": "fcpe.pt"
        }
    ]

    # Download each file
    for file in files_to_download:
        download_file(file["url"], file["output_name"], current_directory)

    print("Download completed.")

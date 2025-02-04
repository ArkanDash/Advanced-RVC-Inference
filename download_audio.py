import os
import argparse
import yt_dlp


class MyLogger(object):
    def debug(self, msg):
        print("[DEBUG]", msg)

    def warning(self, msg):
        print("[WARNING]", msg)

    def error(self, msg):
        print("[ERROR]", msg)


def progress_hook(info):
    status = info.get("status")
    if status == "downloading":
        downloaded = info.get("downloaded_bytes", 0)
        total = info.get("total_bytes", info.get("total_bytes_estimate", 0))
        if total:
            percent = downloaded / total * 100
            print(f"[DEBUG] Downloading: {percent:.2f}%")
    elif status == "finished":
        print("[DEBUG] Download finished, now converting to WAV...")


def download_youtube_audio(url, output_path):
    os.makedirs(output_path, exist_ok=True)

    outtmpl = os.path.join(output_path, "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "logger": MyLogger(),
        "progress_hooks": [progress_hook],
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "verbose": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


# Command-line interface for local usage.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a YouTube video's audio as WAV using yt-dlp with debugging output."
    )
    parser.add_argument("url", help="The URL of the YouTube video to download.")
    parser.add_argument(
        "--output",
        default="downloads",
        help="Custom output directory (default: 'downloads').",
    )
    args = parser.parse_args()
    download_youtube_audio(args.url, args.output)


# gyatt dyum made by NeoDev

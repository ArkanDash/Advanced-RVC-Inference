# Download music fallback
import os
import sys
import subprocess

def download_music(url, output_dir):
    """Download music from URL using yt-dlp"""
    try:
        cmd = ['yt-dlp', '-x', '--audio-format', 'wav', '-o', os.path.join(output_dir, '%(title)s.%(ext)s'), url]
        subprocess.run(cmd, check=True)
        return True
    except:
        return False

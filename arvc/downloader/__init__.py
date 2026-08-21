"""
File-host downloader utilities for Advanced RVC Inference.

This subpackage groups together all downloaders for the various file-hosting
services that the project supports for fetching pretrained models, embedders,
and other assets:

- ``gdown``       — Google Drive (large files, folder downloads)
- ``huggingface`` — Hugging Face Hub (pretrained models, datasets)
- ``mediafire``   — MediaFire shared URLs
- ``meganz``      — MEGA.nz (encrypted file shares)
- ``pixeldrain``  — Pixeldrain file hosting

All downloaders expose a uniform ``download_url(url, dest_path, *args, **kwargs)``
interface where possible, plus service-specific helpers (e.g. ``HF_download_file``
for Hugging Face).
"""

from . import gdown, huggingface, mediafire, meganz, pixeldrain

__all__ = [
    "gdown",
    "huggingface",
    "mediafire",
    "meganz",
    "pixeldrain",
]

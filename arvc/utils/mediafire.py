import os
import sys
import requests

from bs4 import BeautifulSoup

# SECURITY PATCH: hard limits for the MediaFire downloader — same as the
# other downloader modules. Prevents disk-fill DoS and path traversal via
# attacker-controlled filenames in MediaFire's HTML.
MAX_DOWNLOAD_BYTES = 8 * 1024 * 1024 * 1024  # 8 GB
ALLOWED_EXTENSIONS = (
    ".pth", ".pt", ".onnx", ".index", ".zip", ".tar", ".tar.gz",
    ".tgz", ".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4",
    ".json", ".npy", ".npz", ".bin", ".txt", ".ckpt",
)


def _sanitize_filename(name: str) -> str:
    """Force attacker-controlled MediaFire filename to a single basename component."""
    if not name:
        return "mediafire_download.bin"
    name = os.path.basename(name)
    name = name.replace(os.path.sep, "_").replace("/", "_").replace("\\", "_")
    name = name.lstrip(".-")
    lower = name.lower()
    if not any(lower.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        name = name + ".bin"
    return name


def Mediafire_Download(url, output=None, filename=None):
    if not filename:
        filename = url.split('/')[-2]
    # SECURITY PATCH: sanitize the filename BEFORE joining — MediaFire URLs
    # contain attacker-controlled path segments.
    filename = _sanitize_filename(filename)
    if not output:
        output = os.path.dirname(os.path.realpath(__file__))
    output_file = os.path.join(output, filename)

    sess = requests.session()
    sess.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6)"})

    try:
        # SECURITY PATCH: was no timeout on either sess.get(url) or the
        # streaming download get. Add 300s timeouts so a hung MediaFire
        # server can't hang the worker thread indefinitely.
        html_resp = sess.get(url, timeout=300)
        download_href = BeautifulSoup(html_resp.content, "html.parser").find(id="downloadButton").get("href")

        with requests.get(download_href, stream=True, timeout=300) as r:
            r.raise_for_status()

            # SECURITY PATCH: enforce size cap up front
            content_length = int(r.headers.get('content-length', 0) or 0)
            if content_length and content_length > MAX_DOWNLOAD_BYTES:
                raise ValueError(
                    f"MediaFire download size {content_length} bytes exceeds the "
                    f"{MAX_DOWNLOAD_BYTES}-byte safety limit. Aborting."
                )

            with open(output_file, "wb") as f:
                download_progress = 0

                for chunk in r.iter_content(chunk_size=1024):
                    if not chunk:
                        continue
                    download_progress += len(chunk)
                    if download_progress > MAX_DOWNLOAD_BYTES:
                        f.close()
                        try:
                            os.remove(output_file)
                        except OSError:
                            pass
                        raise ValueError(
                            f"Streamed MediaFire download exceeded the "
                            f"{MAX_DOWNLOAD_BYTES}-byte safety limit. Aborting."
                        )
                    f.write(chunk)

                    if content_length:
                        sys.stdout.write(
                            f"\r[{filename}]: {int(100 * download_progress / content_length)}% "
                            f"({round(download_progress / 1024 / 1024, 2)}mb/"
                            f"{round(content_length / 1024 / 1024, 2)}mb)"
                        )
                        sys.stdout.flush()

        sys.stdout.write("\n")
        return output_file
    except Exception as e:
        raise RuntimeError(e)

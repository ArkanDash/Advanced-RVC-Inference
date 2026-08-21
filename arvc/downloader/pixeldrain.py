import os
import requests

# SECURITY PATCH: hard limits for the PixelDrain downloader.
# - MAX_DOWNLOAD_BYTES: rejects files larger than 8 GB (RVC models / datasets
#   are never this large; an attacker could otherwise fill the disk).
# - ALLOWED_EXTENSIONS: only saves files with model-friendly extensions to
#   prevent disguised executable payloads (.exe, .sh, .bat) being written
#   next to the user's model library.
MAX_DOWNLOAD_BYTES = 8 * 1024 * 1024 * 1024  # 8 GB
ALLOWED_EXTENSIONS = (
    ".pth", ".pt", ".onnx", ".index", ".zip", ".tar", ".tar.gz",
    ".tgz", ".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4",
    ".json", ".npy", ".npz", ".bin", ".txt", ".ckpt",
)


def _sanitize_filename(name: str) -> str:
    """Force attacker-controlled Content-Disposition filename to a single basename component.

    SECURITY PATCH: was verbatim use of the Content-Disposition header value
    joined to output_dir — a malicious PixelDrain link could set the
    filename to `../../../../etc/cron.d/evil` and we'd write through the
    traversal.
    """
    if not name:
        return "pixeldrain_download.bin"
    name = os.path.basename(name)
    name = name.replace(os.path.sep, "_").replace("/", "_").replace("\\", "_")
    name = name.lstrip(".-")
    lower = name.lower()
    if not any(lower.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        name = name + ".bin"
    return name


def pixeldrain(url, output_dir):
    try:
        # SECURITY PATCH: was no timeout + loaded entire file into memory
        # via `response.content`. Now stream to disk with a 5-min timeout
        # and enforce a hard size cap.
        response = requests.get(
            f"https://pixeldrain.com/api/file/{url.split('pixeldrain.com/u/')[1]}",
            stream=True,
            timeout=300,
        )

        if response.status_code == 200:
            # SECURITY PATCH: was `response.headers.get("Content-Disposition")`
            # — `AttributeError` if the header is missing. Default to a UUID
            # name if absent. Then sanitize the filename before joining.
            cd = response.headers.get("Content-Disposition") or ""
            if "filename=" in cd:
                raw_name = cd.split("filename=")[-1].strip('";')
            else:
                # No Content-Disposition — derive from the URL slug
                raw_name = url.split("pixeldrain.com/u/")[-1].split("?")[0] or "pixeldrain_download"
            safe_name = _sanitize_filename(raw_name)
            file_path = os.path.join(output_dir, safe_name)

            # SECURITY PATCH: enforce size cap on Content-Length up front
            content_length = int(response.headers.get("Content-Length", 0) or 0)
            if content_length and content_length > MAX_DOWNLOAD_BYTES:
                raise ValueError(
                    f"PixelDrain download size {content_length} bytes exceeds the "
                    f"{MAX_DOWNLOAD_BYTES}-byte safety limit. Aborting."
                )

            bytes_written = 0
            with open(file_path, "wb") as newfile:
                for chunk in response.iter_content(chunk_size=512 * 1024):
                    if not chunk:
                        continue
                    bytes_written += len(chunk)
                    if bytes_written > MAX_DOWNLOAD_BYTES:
                        newfile.close()
                        try:
                            os.remove(file_path)
                        except OSError:
                            pass
                        raise ValueError(
                            f"Streamed PixelDrain download exceeded the "
                            f"{MAX_DOWNLOAD_BYTES}-byte safety limit. Aborting."
                        )
                    newfile.write(chunk)
            return file_path
        else:
            return None
    except Exception as e:
        raise RuntimeError(e)

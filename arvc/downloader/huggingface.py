import os
import tqdm
import requests

try:
    import wget
except ImportError:
    # SECURITY PATCH: was bare `except:` — would swallow KeyboardInterrupt.
    wget = None

# Maximum number of retries for failed downloads
MAX_RETRIES = 3
# Timeout in seconds for each request
REQUEST_TIMEOUT = 300
# SECURITY PATCH: hard size cap and extension whitelist — same as the other
# downloader modules. Prevents disk-fill DoS via a 100 GB file hosted on
# HuggingFace, and prevents disguised .exe payloads being saved next to
# the user's model library.
MAX_DOWNLOAD_BYTES = 8 * 1024 * 1024 * 1024  # 8 GB
ALLOWED_EXTENSIONS = (
    ".pth", ".pt", ".onnx", ".index", ".zip", ".tar", ".tar.gz",
    ".tgz", ".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4",
    ".json", ".npy", ".npz", ".bin", ".txt", ".ckpt",
)


def _sanitize_filename(name: str) -> str:
    """Force attacker-controlled HF filename to a single basename component.

    Used for the URL basename path. HF URLs are typically trusted, but a
    user can be tricked into pasting a malicious HF URL with a path
    segment like `../../../../etc/cron.d/evil` — this guard prevents that.
    """
    if not name:
        return "hf_download.bin"
    name = os.path.basename(name)
    name = name.replace(os.path.sep, "_").replace("/", "_").replace("\\", "_")
    name = name.lstrip(".-")
    lower = name.lower()
    if not any(lower.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        name = name + ".bin"
    return name

def HF_download_file(url, output_path=None):
    """Download a file from HuggingFace (regular repo or Storage Bucket).

    Handles:
    - Regular repo URLs: https://huggingface.co/{user}/{repo}/resolve/main/{path}
    - Storage Bucket URLs: https://huggingface.co/buckets/{user}/{bucket}/resolve/{path}
    - 302 redirects (both URL types redirect to CDN/XetHub signed URLs)
    - Automatic retry on transient failures

    Args:
        url: The HuggingFace file URL to download.
        output_path: Local path to save the file. If None, uses the URL basename.
                     If a directory, saves the file inside that directory.

    Returns:
        The output path of the downloaded file.

    Raises:
        ValueError: If the download fails after all retries.
    """
    # Normalize URL: replace /blob/ and /tree/ with /resolve/, strip query params
    url = url.replace("/blob/", "/resolve/").replace("/tree/", "/resolve/").replace("?download=true", "").strip()

    # Determine output path
    url_filename = os.path.basename(url)
    # SECURITY PATCH: sanitize the URL-derived filename before joining
    url_filename = _sanitize_filename(url_filename)
    if output_path is None:
        output_path = url_filename
    elif os.path.isdir(output_path):
        output_path = os.path.join(output_path, url_filename)

    # Ensure parent directory exists
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    # Try wget first if available
    if wget is not None:
        try:
            wget.download(url, out=output_path)
            return output_path
        except Exception:
            # Fall through to requests-based download
            pass

    # requests-based download with retry logic
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # allow_redirects=True is the default for requests.get(),
            # so 302 redirects from /resolve/ URLs are followed automatically.
            response = requests.get(
                url,
                stream=True,
                timeout=REQUEST_TIMEOUT,
                allow_redirects=True,
                headers={"User-Agent": "Advanced-RVC-Inference/1.0"},
            )

            if response.status_code == 200:
                content_length = int(response.headers.get("content-length", 0))
                # SECURITY PATCH: enforce size cap up front to prevent disk-fill DoS
                if content_length and content_length > MAX_DOWNLOAD_BYTES:
                    raise ValueError(
                        f"Download size {content_length} bytes exceeds the "
                        f"{MAX_DOWNLOAD_BYTES}-byte safety limit. Aborting."
                    )
                progress_bar = tqdm.tqdm(
                    total=content_length if content_length > 0 else None,
                    desc=url_filename,
                    ncols=100,
                    unit="byte",
                    leave=False,
                )

                bytes_written = 0
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=10 * 1024 * 1024):
                        if chunk:  # filter out keep-alive new chunks
                            # SECURITY PATCH: enforce streaming size cap
                            bytes_written += len(chunk)
                            if bytes_written > MAX_DOWNLOAD_BYTES:
                                progress_bar.close()
                                f.close()
                                try:
                                    os.remove(output_path)
                                except OSError:
                                    pass
                                raise ValueError(
                                    f"Streamed download exceeded the "
                                    f"{MAX_DOWNLOAD_BYTES}-byte safety limit. Aborting."
                                )
                            progress_bar.update(len(chunk))
                            f.write(chunk)

                progress_bar.close()

                # Verify the file was actually written (not empty)
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    return output_path
                else:
                    raise ValueError(f"Downloaded file is empty: {output_path}")

            elif response.status_code in (301, 302, 303, 307, 308):
                # Should not happen with allow_redirects=True, but handle just in case
                redirect_url = response.headers.get("Location", "")
                if redirect_url:
                    url = redirect_url
                    continue
                else:
                    raise ValueError(f"Redirect ({response.status_code}) without Location header")
            else:
                raise ValueError(
                    f"HTTP {response.status_code} for {url}\n"
                    f"Response: {response.text[:500] if response.text else 'empty'}"
                )

        except requests.exceptions.ConnectionError as e:
            last_error = e
            if attempt < MAX_RETRIES:
                import time
                time.sleep(2 * attempt)  # Exponential backoff
        except requests.exceptions.Timeout as e:
            last_error = e
            if attempt < MAX_RETRIES:
                import time
                time.sleep(2 * attempt)
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < MAX_RETRIES:
                import time
                time.sleep(2 * attempt)

    # All retries failed
    raise ValueError(
        f"Failed to download {url} after {MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )

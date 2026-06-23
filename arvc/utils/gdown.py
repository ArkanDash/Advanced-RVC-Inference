import os
import re
import sys
import json
import tqdm
import codecs
import tempfile
import requests

from urllib.parse import urlparse, parse_qs, unquote


from arvc.utils.variables import translations

# SECURITY PATCH: hard limits for the Google Drive downloader.
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
    """Strip path separators and return a basename-only string.

    SECURITY PATCH: was `os.path.basename(url)` — Google Drive's
    `Content-Disposition` header is fully attacker-controlled, so a malicious
    shared file named `../../../../etc/cron.d/evil` would be joined into a
    system path. We force the result to a single filename component.
    """
    if not name:
        return "gdown_download"
    # Replace any path separator characters with underscores
    name = name.replace(os.path.sep, "_").replace("/", "_").replace("\\", "_")
    # Strip leading dots / dashes to prevent hidden-file or arg-injection quirks
    name = name.lstrip(".-")
    # Verify extension
    lower = name.lower()
    if not any(lower.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        # Append .bin to unknown extensions instead of rejecting — caller
        # may rename later, but we never write a .exe / .sh / .bat to disk.
        name = name + ".bin"
    return name

def parse_url(url):
    parsed = urlparse(url)
    is_download_link = parsed.path.endswith("/uc")
    if not parsed.hostname in ("drive.google.com", "docs.google.com"): return None, is_download_link
    file_id = parse_qs(parsed.query).get("id", [None])[0]

    if file_id is None:
        for pattern in (r"^/file/d/(.*?)/(edit|view)$", r"^/file/u/[0-9]+/d/(.*?)/(edit|view)$", r"^/document/d/(.*?)/(edit|htmlview|view)$", r"^/document/u/[0-9]+/d/(.*?)/(edit|htmlview|view)$", r"^/presentation/d/(.*?)/(edit|htmlview|view)$", r"^/presentation/u/[0-9]+/d/(.*?)/(edit|htmlview|view)$", r"^/spreadsheets/d/(.*?)/(edit|htmlview|view)$", r"^/spreadsheets/u/[0-9]+/d/(.*?)/(edit|htmlview|view)$"):
            match = re.match(pattern, parsed.path)
            if match:
                file_id = match.group(1)
                break
    return file_id, is_download_link

def get_url_from_gdrive_confirmation(contents):
    for pattern in (r'href="(\/uc\?export=download[^"]+)', r'href="/open\?id=([^"]+)"', r'"downloadUrl":"([^"]+)'):
        match = re.search(pattern, contents)
        if match:
            url = match.group(1)
            if pattern == r'href="/open\?id=([^"]+)"': url = (codecs.decode("uggcf://qevir.hfrepbagrag.tbbtyr.pbz/qbjaybnq?vq=", "rot13") + url + "&confirm=t&uuid=" + re.search(r'<input\s+type="hidden"\s+name="uuid"\s+value="([^"]+)"', contents).group(1))
            elif pattern == r'"downloadUrl":"([^"]+)': url = url.replace("\\u003d", "=").replace("\\u0026", "&")
            else: url = codecs.decode("uggcf://qbpf.tbbtyr.pbz", "rot13") + url.replace("&", "&")
            return url

    match = re.search(r'<p class="uc-error-subcaption">(.*)</p>', contents)
    if match: raise Exception(match.group(1))
    raise Exception(translations["gdown_error"])

def _get_session(use_cookies, return_cookies_file=False):
    sess = requests.session()
    sess.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6)"})
    cookies_file = os.path.join(os.path.expanduser("~"), ".cache/gdown/cookies.json")

    if os.path.exists(cookies_file) and use_cookies:
        with open(cookies_file) as f:
            for k, v in json.load(f):
                sess.cookies[k] = v
    return (sess, cookies_file) if return_cookies_file else sess

def gdown_download(url=None, output=None):
    file_id = None

    if url is None: raise ValueError(translations["gdown_value_error"])

    if "/file/d/" in url: file_id = url.split("/d/")[1].split("/")[0]
    elif "open?id=" in url: file_id = url.split("open?id=")[1].split("/")[0]
    elif "/download?id=" in url: file_id = url.split("/download?id=")[1].split("&")[0]

    if file_id:
        url = f"{codecs.decode('uggcf://qevir.tbbtyr.pbz/hp?vq=', 'rot13')}{file_id}"
        url_origin = url

        sess, cookies_file = _get_session(use_cookies=True, return_cookies_file=True)
        gdrive_file_id, is_gdrive_download_link = parse_url(url)

        if gdrive_file_id:
            url = f"{codecs.decode('uggcf://qevir.tbbtyr.pbz/hp?vq=', 'rot13')}{gdrive_file_id}"
            url_origin = url
            is_gdrive_download_link = True

        while 1:
            res = sess.get(url, stream=True, verify=True)
            if url == url_origin and res.status_code == 500:
                url = f"{codecs.decode('uggcf://qevir.tbbtyr.pbz/bcra?vq=', 'rot13')}{gdrive_file_id}"
                continue

            os.makedirs(os.path.dirname(cookies_file), exist_ok=True)
            with open(cookies_file, "w") as f:
                json.dump([(k, v) for k, v in sess.cookies.items() if not k.startswith("download_warning_")], f, indent=2)

            if "Content-Disposition" in res.headers: break
            if not (gdrive_file_id and is_gdrive_download_link): break

            try:
                url = get_url_from_gdrive_confirmation(res.text)
            except Exception as e:
                raise Exception(e)

        if gdrive_file_id and is_gdrive_download_link:
            content_disposition = unquote(res.headers["Content-Disposition"])
            match = re.search(r"filename\*=UTF-8''(.*)", content_disposition) or re.search(r'filename=["\']?(.*?)["\']?$', content_disposition)
            filename_from_url = (match.group(1).replace(os.path.sep, "_") if match else os.path.basename(url))
        else: filename_from_url = os.path.basename(url)

        # SECURITY PATCH: sanitize attacker-controlled filename + add extension whitelist
        filename_from_url = _sanitize_filename(filename_from_url)
        output = os.path.join(output or ".", filename_from_url)

        # SECURITY PATCH: was `tempfile.mktemp(...)` — TOCTOU symlink race.
        # Replace with `mkstemp` which atomically creates the file with a
        # file descriptor we own, then close it and reopen in append mode.
        fd, tmp_file = tempfile.mkstemp(
            suffix=tempfile.template,
            prefix=os.path.basename(output),
            dir=os.path.dirname(output),
        )
        os.close(fd)
        f = open(tmp_file, "ab")

        if tmp_file is not None and f.tell() != 0:
            res = sess.get(
                url,
                headers={"Range": f"bytes={f.tell()}-"},
                stream=True,
                verify=True,
                timeout=300,  # SECURITY PATCH: was no timeout → hung server hangs worker
            )

        try:
            total_expected = int(res.headers.get("Content-Length", 0))
            # SECURITY PATCH: enforce hard size cap to prevent disk-fill DoS
            if total_expected and total_expected > MAX_DOWNLOAD_BYTES:
                raise ValueError(
                    f"Download size {total_expected} bytes exceeds the "
                    f"{MAX_DOWNLOAD_BYTES}-byte safety limit. Aborting."
                )
            with tqdm.tqdm(desc=os.path.basename(output), total=total_expected, ncols=100, unit="byte") as pbar:
                bytes_written = 0
                for chunk in res.iter_content(chunk_size=512 * 1024):
                    f.write(chunk)
                    bytes_written += len(chunk)
                    if bytes_written > MAX_DOWNLOAD_BYTES:
                        raise ValueError(
                            f"Streamed download exceeded the {MAX_DOWNLOAD_BYTES}-byte "
                            f"safety limit. Aborting mid-stream."
                        )
                    pbar.update(len(chunk))

                pbar.close()
        finally:
            f.close()
            try:
                os.rename(tmp_file, output)
            except OSError:
                # SECURITY PATCH: was silent `pass` — at least log the failure
                # so the caller knows `output` is missing.
                try:
                    os.remove(tmp_file)
                except OSError:
                    pass
                raise RuntimeError(
                    f"Failed to move downloaded file from '{tmp_file}' to '{output}'. "
                    f"Partial download was discarded."
                )
            sess.close()

        return output
    return None
import os
import re
import sys
import json
import tqdm
import codecs
import tempfile
import requests

from urllib.parse import urlparse, parse_qs, unquote

sys.path.append(os.getcwd())

from main.app.variables import translations

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
            filename_from_url = (re.search(r"filename\*=UTF-8''(.*)", content_disposition) or re.search(r'filename=["\']?(.*?)["\']?$', content_disposition)).group(1).replace(os.path.sep, "_")
        else: filename_from_url = os.path.basename(url)

        output = os.path.join(output or ".", filename_from_url)
        tmp_file = tempfile.mktemp(suffix=tempfile.template, prefix=os.path.basename(output), dir=os.path.dirname(output))
        f = open(tmp_file, "ab")

        if tmp_file is not None and f.tell() != 0: res = sess.get(url, headers={"Range": f"bytes={f.tell()}-"}, stream=True, verify=True)

        try:
            with tqdm.tqdm(desc=os.path.basename(output), total=int(res.headers.get("Content-Length", 0)), ncols=100, unit="byte") as pbar:
                for chunk in res.iter_content(chunk_size=512 * 1024):
                    f.write(chunk)
                    pbar.update(len(chunk))

                pbar.close()
                if tmp_file: f.close()
        finally:
            os.rename(tmp_file, output)
            sess.close()

        return output
    return None
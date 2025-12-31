import os
import sys
import requests

from bs4 import BeautifulSoup

def Mediafire_Download(url, output=None, filename=None):
    if not filename: filename = url.split('/')[-2]
    if not output: output = os.path.dirname(os.path.realpath(__file__))
    output_file = os.path.join(output, filename)

    sess = requests.session()
    sess.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6)"})

    try:
        with requests.get(BeautifulSoup(sess.get(url).content, "html.parser").find(id="downloadButton").get("href"), stream=True) as r:
            r.raise_for_status()

            with open(output_file, "wb") as f:
                total_length = int(r.headers.get('content-length'))
                download_progress = 0

                for chunk in r.iter_content(chunk_size=1024):
                    download_progress += len(chunk)
                    f.write(chunk)

                    sys.stdout.write(f"\r[{filename}]: {int(100 * download_progress / total_length)}% ({round(download_progress / 1024 / 1024, 2)}mb/{round(total_length / 1024 / 1024, 2)}mb)")
                    sys.stdout.flush()

        sys.stdout.write("\n")
        return output_file
    except Exception as e:
        raise RuntimeError(e)
import os
import requests

def pixeldrain(url, output_dir):
    try:
        response = requests.get(f"https://pixeldrain.com/api/file/{url.split('pixeldrain.com/u/')[1]}")

        if response.status_code == 200:
            file_path = os.path.join(output_dir, (response.headers.get("Content-Disposition").split("filename=")[-1].strip('";')))

            with open(file_path, "wb") as newfile:
                newfile.write(response.content)
            return file_path
        else: return None
    except Exception as e:
        raise RuntimeError(e)
import os
import sys
import json
import codecs
import requests

sys.path.append(os.getcwd())

from main.app.core.ui import gr_info, gr_warning
from main.app.variables import translations, configs

def stop_pid(pid_file, model_name=None, train=False):
    try:
        pid_file_path = os.path.join("assets", f"{pid_file}.txt") if model_name is None else os.path.join(configs["logs_path"], model_name, f"{pid_file}.txt")

        if not os.path.exists(pid_file_path): return gr_warning(translations["not_found_pid"])
        else:
            with open(pid_file_path, "r") as pid_file:
                pids = [int(pid) for pid in pid_file.readlines()]

            for pid in pids:
                os.kill(pid, 9)

            if os.path.exists(pid_file_path): os.remove(pid_file_path)

        pid_file_path = os.path.join(configs["logs_path"], model_name, "config.json")

        if train and os.path.exists(pid_file_path):
            with open(pid_file_path, "r") as pid_file:
                pid_data = json.load(pid_file)
                pids = pid_data.get("process_pids", [])

            with open(pid_file_path, "w") as pid_file:
                pid_data.pop("process_pids", None)

                json.dump(pid_data, pid_file, indent=4)

            for pid in pids:
                os.kill(pid, 9)

            gr_info(translations["end_pid"])
    except:
        pass

def google_translate(text, source='auto', target='vi'):
    if text == "": return gr_warning(translations["prompt_warning"])

    try:
        import textwrap

        def translate_chunk(chunk):
            response = requests.get(codecs.decode("uggcf://genafyngr.tbbtyrncvf.pbz/genafyngr_n/fvatyr", "rot13"), params={'client': 'gtx', 'sl': source, 'tl': target, 'dt': 't', 'q': chunk})
            return ''.join([i[0] for i in response.json()[0]]) if response.status_code == 200 else chunk

        translated_text = ''
        for chunk in textwrap.wrap(text, 5000, break_long_words=False, break_on_hyphens=False):
            translated_text += translate_chunk(chunk)

        return translated_text
    except:
        return text
import os
import re
import sys
import json
import codecs
import shutil
import yt_dlp
import warnings
import requests

from bs4 import BeautifulSoup

sys.path.append(os.getcwd())

from advanced_rvc_inference.utils import huggingface, gdown, meganz, mediafire, pixeldrain
from advanced_rvc_inference.core.process import move_files_from_directory, fetch_pretrained_data, extract_name_model
from advanced_rvc_inference.core.ui import gr_info, gr_warning, gr_error, process_output, replace_url, replace_modelname

from advanced_rvc_inference.utils.variables import logger, translations, model_options, configs

def download_url(url):
    if not url: 
        gr_warning(translations["provide_url"])
        return [None]*3

    if not os.path.exists(configs["audios_path"]): os.makedirs(configs["audios_path"], exist_ok=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        ydl_opts = {
            "format": "bestaudio/best", 
            "cookies": "advanced_rvc_inference/assets/config.txt",
            "postprocessors": [{
                "key": "FFmpegExtractAudio", 
                "preferredcodec": "wav", 
                "preferredquality": "192"
            }], 
            "quiet": True, 
            "no_warnings": True, 
            "noplaylist": True, 
            "verbose": False
        }

        gr_info(translations["start"].format(start=translations["download_music"]))

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            audio_output = os.path.join(configs["audios_path"], re.sub(r'\s+', '-', re.sub(r'[^\w\s\u4e00-\u9fff\uac00-\ud7af\u0400-\u04FF\u1100-\u11FF]', '', ydl.extract_info(url, download=False).get('title', 'video')).strip()))
            if os.path.exists(audio_output): shutil.rmtree(audio_output, ignore_errors=True)

            ydl_opts['outtmpl'] = audio_output
            
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: 
            audio_output = process_output(audio_output + ".wav")
            
            ydl.download([url])

        gr_info(translations["success"])
        return [audio_output, audio_output, translations["success"]]

def move_file(file, download_dir, model, use_orig_weight_name=False):
    weights_dir = configs["weights_path"]
    logs_dir = configs["logs_path"]

    if not os.path.exists(weights_dir): os.makedirs(weights_dir, exist_ok=True)
    if not os.path.exists(logs_dir): os.makedirs(logs_dir, exist_ok=True)

    if file.endswith(".zip"): shutil.unpack_archive(file, download_dir)
    move_files_from_directory(download_dir, weights_dir, logs_dir, model, use_orig_weight_name)

def download_model(url=None, model=None):
    """Download an RVC voice model from a URL.

    Supports HuggingFace, Google Drive, MediaFire, PixelDrain, and Mega links.
    When no model name is provided, the original weight filename is preserved
    (use_orig_weight_name=True behavior from Vietnamese-RVC).

    Args:
        url: Download URL for the model file.
        model: Optional model name. If None, the original filename is used.

    Returns:
        Status message string.
    """
    if not url: return gr_warning(translations["provide_url"])

    url = replace_url(url)
    download_dir = "download_model"
    use_orig_weight_name = False

    os.makedirs(download_dir, exist_ok=True)
    
    try:
        gr_info(translations["start"].format(start=translations["download"]))

        if "huggingface.co" in url: file = huggingface.HF_download_file(url, download_dir)
        elif "google.com" in url: file = gdown.gdown_download(url, download_dir)
        elif "mediafire.com" in url: file = mediafire.Mediafire_Download(url, download_dir)
        elif "pixeldrain.com" in url: file = pixeldrain.pixeldrain(url, download_dir)
        elif "mega.nz" in url: file = meganz.mega_download_url(url, download_dir)
        else:
            gr_warning(translations["not_support_url"])
            return translations["not_support_url"]
        
        if not model: 
            use_orig_weight_name = True
            modelname = os.path.basename(file)

            model = (
                extract_name_model(modelname) 
                if modelname.endswith(".index") else 
                os.path.splitext(modelname)[0]
            )

            if model is None: model = os.path.splitext(modelname)[0]

        model = replace_modelname(model)

        move_file(file, download_dir, model, use_orig_weight_name)
        gr_info(translations["success"])

        return translations["success"]
    except Exception as e:
        gr_error(message=translations["error_occurred"].format(e=e))
        return translations["error_occurred"].format(e=e)
    finally:
        shutil.rmtree(download_dir, ignore_errors=True)
        
def download_pretrained_model(choices, model, sample_rate):
    pretraineds_custom_path = configs["pretrained_custom_path"]

    if choices == translations["list_model"]:
        paths = fetch_pretrained_data()[model][sample_rate]

        if not os.path.exists(pretraineds_custom_path): os.makedirs(pretraineds_custom_path, exist_ok=True)
        url = codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/cergenvarq_phfgbz/", "rot13") + paths

        gr_info(translations["download_pretrain"])
        file = huggingface.HF_download_file(replace_url(url), os.path.join(pretraineds_custom_path, paths))

        if file.endswith(".zip"): 
            shutil.unpack_archive(file, pretraineds_custom_path)
            os.remove(file)

        gr_info(translations["success"])
        return translations["success"]
    elif choices == translations["download_url"]:
        pretrain_is_zip = model.endswith(".zip") or model.endswith(".zip?download=true") or sample_rate.endswith(".zip") or sample_rate.endswith(".zip?download=true")
        urls = []

        if not model and not pretrain_is_zip: 
            gr_warning(translations["provide_pretrain"].format(dg="D"))
            return [None]*2

        if not sample_rate and not pretrain_is_zip: 
            gr_warning(translations["provide_pretrain"].format(dg="G"))
            return [None]*2

        gr_info(translations["download_pretrain"])

        if model: urls.append(model)
        if sample_rate: urls.append(sample_rate)

        for url in urls:
            url = replace_url(url)
        
            if "huggingface.co" in url: file = huggingface.HF_download_file(url, pretraineds_custom_path)
            elif "google.com" in url: file = gdown.gdown_download(url, pretraineds_custom_path)
            elif "mediafire.com" in url: file = mediafire.Mediafire_Download(url, pretraineds_custom_path)
            elif "pixeldrain.com" in url: file = pixeldrain.pixeldrain(url, pretraineds_custom_path)
            elif "mega.nz" in url: file = meganz.mega_download_url(url, pretraineds_custom_path)
            else:
                gr_warning(translations["not_support_url"])
                return translations["not_support_url"], translations["not_support_url"]
            
            if file.endswith(".zip"):
                shutil.unpack_archive(file, pretraineds_custom_path)
                if os.path.exists(file): os.remove(file)

        gr_info(translations["success"])
        return translations["success"], translations["success"]

SEARCH_API_URL = codecs.decode("uggcf://ibvpr-zbqryf.pbz/srgpu_qngn.cuc", "rot13")
MAX_SEARCH_PAGES = 10
SEARCH_REQUEST_TIMEOUT = 15


def fetch_models_data(search):
    """Fetch model data from voice-models.com with paginated search.

    Uses the pagination HTML to detect the last page (by checking for
    the 'next disabled' class) instead of relying on empty table data.
    """
    all_table_data = []
    page = 1
    is_last_page = False

    while page <= MAX_SEARCH_PAGES and not is_last_page:
        try:
            response = requests.post(
                url=SEARCH_API_URL,
                data={"page": page, "search": search},
                timeout=SEARCH_REQUEST_TIMEOUT,
            )

            if response.status_code != 200:
                logger.debug(f"{translations['code_error']} {response.status_code}")
                break

            try:
                data = response.json()
            except (json.JSONDecodeError, ValueError):
                logger.debug(translations["json_error"])
                break

            table_data = data.get("table", "")
            pagination_html = data.get("pagination", "")

            if not table_data.strip():
                break

            all_table_data.append(table_data)

            # Check pagination for last page indicator
            # The API returns "next disabled" class when there are no more pages
            if pagination_html and "next disabled" in pagination_html:
                is_last_page = True
            else:
                page += 1

        except requests.Timeout:
            logger.debug(f"Search request timed out on page {page}")
            break
        except requests.RequestException as e:
            logger.debug(translations["requests_error"].format(e=e))
            break

    return all_table_data


def _extract_model_url(row):
    """Extract the download URL from a search result row.

    Priority:
    1. data-clipboard-text attribute (direct HuggingFace URL)
    2. <a> tag href with easyaivoice.com/run?url= wrapper
    Returns the direct URL or None if no HuggingFace URL is found.
    """
    # Method 1: Direct URL from clipboard button (most reliable)
    copy_btn = row.find("button", attrs={"data-clipboard-text": True})
    if copy_btn and copy_btn.get("data-clipboard-text"):
        url = copy_btn["data-clipboard-text"]
        if "huggingface" in url:
            return url

    # Method 2: Fall back to <a> tag with redirect wrapper
    for a_tag in row.find_all("a", href=True):
        href = a_tag["href"]
        if "huggingface" in href:
            # Strip redirect wrapper if present
            return href.replace("https://easyaivoice.com/run?url=", "")

    return None


def _extract_model_name(row):
    """Extract the model display name from a search result row."""
    name_tag = row.find("a", class_="fs-5")
    if name_tag:
        return name_tag.get_text(strip=True)
    return None


def _extract_model_size(row):
    """Extract the model file size badge from a search result row."""
    badge = row.find("span", class_="badge")
    if badge:
        size_text = badge.get_text(strip=True)
        return size_text
    return None

def search_models(name):
    if not name:
        gr_warning(translations["provide_name"])
        return [None] * 2

    gr_info(translations["start"].format(start=translations["search"]))

    tables = fetch_models_data(name)

    if len(tables) == 0:
        gr_info(translations["not_found"].format(name=name))
        return [None] * 2

    model_options.clear()

    for table in tables:
        for row in BeautifulSoup(table, "html.parser").select("tr"):
            model_name = _extract_model_name(row)
            model_url = _extract_model_url(row)
            model_size = _extract_model_size(row)

            if not model_name or not model_url:
                continue

            clean_name = replace_modelname(model_name)

            # Append size info to display name for better UX
            if model_size:
                display_name = f"{clean_name} ({model_size})"
            else:
                display_name = clean_name

            model_options[display_name] = model_url

    gr_info(translations["found"].format(results=len(model_options)))
    return [
        {"value": "", "choices": model_options, "interactive": True, "visible": True, "__type__": "update"},
        {"value": translations["downloads"], "visible": True, "__type__": "update"},
    ]

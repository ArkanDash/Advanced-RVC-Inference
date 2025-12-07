import os
import sys
import json
import shutil
import requests
import tempfile
import numpy as np
import math
import gradio as gr

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


now_dir = os.getcwd()
sys.path.insert(0, now_dir)

from advanced_rvc_inference.core import run_download_script
from advanced_rvc_inference.rvc.lib.utils import format_title

from advanced_rvc_inference.assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

# Vietnamese-RVC inspired F0 methods and download functionality
F0_METHODS = {
    "rmvpe": "RMVPE (Recommended)",
    "fcpe": "FCPE (Fast Cepstral Pitch Estimator)",
    "fcpe_legacy": "FCPE Legacy",
    "crepe": "CREPE (Deep Learning)",
    "penn": "Penn",
    "djcm": "DJCM",
    "pesto": "Pesto",
    "swift": "Swift",
    "hybrid[rmvpe+fcpe]": "Hybrid RMVPE+FCPE",
    "hybrid[rmvpe+crepe]": "Hybrid RMVPE+CREPE"
}

EMBEDDER_MODES = {
    "fairseq": "Fairseq (Default)",
    "onnx": "ONNX",
    "transformers": "Transformers",
    "whisper": "Whisper"
}

gradio_temp_dir = os.path.join(tempfile.gettempdir(), "gradio")

if os.path.exists(gradio_temp_dir):
    shutil.rmtree(gradio_temp_dir)


def get_modelname(f0_method: str, f0_onnx: bool = False) -> str:
    """Determine the appropriate model name based on F0 extraction method."""
    if f0_onnx:
        suffix = '.onnx'
    else:
        suffix = '.pt' if 'crepe' in f0_method else '.pth'
    
    if f0_method == 'rmvpe':
        return f'rmvpe{suffix}'
    elif f0_method == 'fcpe':
        return f'fcpe{suffix}'
    elif f0_method == 'fcpe_legacy':
        return f'fcpe_legacy{suffix}'
    elif f0_method == 'ddsp_200k':
        return f'ddsp_200k{suffix}'
    elif f0_method == 'crepe':
        return f'crepe{suffix}'
    elif f0_method == 'penn':
        return f'penn{suffix}'
    elif f0_method == 'djcm':
        return f'djcm{suffix}'
    elif f0_method == 'pesto':
        return f'pesto{suffix}'
    elif f0_method == 'swift':
        return f'swift{suffix}'
    elif f0_method.startswith('hybrid'):
        # For hybrid methods, we use the first method in the hybrid
        method_part = f0_method.split('[')[1].split('+')[0]
        return get_modelname(method_part, f0_onnx)
    else:
        return f'{f0_method}{suffix}'


def autotune_f0(note_dict, f0: np.ndarray, f0_autotune_strength: float) -> np.ndarray:
    """Autotune the F0 array by shifting each frequency towards the nearest note."""
    autotuned_f0 = np.copy(f0)
    for i, freq in enumerate(f0):
        if freq > 0:  # Only process voiced frames
            closest_note = min(note_dict.keys(), key=lambda x: abs(x - freq))
            shift_amount = (closest_note - freq) * f0_autotune_strength
            autotuned_f0[i] = freq + shift_amount
    return autotuned_f0


def extract_median_f0(f0: np.ndarray) -> float:
    """Calculate the median F0 value from an F0 array."""
    # Replace zeros with NaN and interpolate
    f0_no_zeros = np.where(f0 == 0, np.nan, f0)
    # Interpolate NaN values
    indices = np.arange(len(f0_no_zeros))
    f0_interpolated = np.interp(indices, indices[~np.isnan(f0_no_zeros)], f0_no_zeros[~np.isnan(f0_no_zeros)])
    return float(np.median(f0_interpolated))


def proposal_f0_up_key(f0: np.ndarray, target_f0: float = 155.0, limit: int = 12) -> int:
    """Propose an F0 up-key based on the difference between median F0 and target F0."""
    try:
        median_f0 = extract_median_f0(f0)
        if median_f0 <= 0:
            return 0
        semitone_diff = round(12 * math.log2(target_f0 / median_f0))
        return max(-limit, min(limit, semitone_diff))
    except (ValueError, IndexError):
        return 0


def save_drop_model(dropbox):
    if "pth" not in dropbox and "index" not in dropbox:
        raise gr.Error(
            message="The file you dropped is not a valid model file. Please try again."
        )

    file_name = format_title(os.path.basename(dropbox))
    model_name = file_name

    if ".pth" in model_name:
        model_name = model_name.split(".pth")[0]
    elif ".index" in model_name:
        replacements = ["nprobe_1_", "_v1", "_v2", "added_"]
        for rep in replacements:
            model_name = model_name.replace(rep, "")
        model_name = model_name.split(".index")[0]

    model_path = os.path.join("advanced_rvc_inference", "logs", model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if os.path.exists(os.path.join(model_path, file_name)):
        os.remove(os.path.join(model_path, file_name))
    shutil.move(dropbox, os.path.join(model_path, file_name))
    print(f"{file_name} saved in {model_path}")
    gr.Info(f"{file_name} saved in {model_path}")

    return None


# F0 Models and Embedders URLs (Vietnamese-RVC inspired)
F0_MODELS_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"
EMBEDDERS_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"

json_url = "https://huggingface.co/IAHispano/Applio/raw/main/pretrains.json"


def check_assets(f0_method: str, hubert: str = "hubert_base", f0_onnx: bool = False, embedders_mode: str = "fairseq"):
    """Check for and download necessary predictor and embedder models."""
    success = True
    
    try:
        # Download F0 predictor if needed
        if f0_method and f0_method != "none":
            predictor_name = get_modelname(f0_method, f0_onnx)
            success &= download_predictor(predictor_name)
        
        # Download embedder
        success &= download_embedder(embedders_mode, hubert)
        
        return success
    except Exception as e:
        print(f"Error checking assets: {e}")
        return False


def download_predictor(predictor: str) -> bool:
    """Download a specified predictor model."""
    predictors_path = os.path.join("advanced_rvc_inference", "rvc", "models", "predictors")
    os.makedirs(predictors_path, exist_ok=True)
    
    predictor_path = os.path.join(predictors_path, predictor)
    
    if not os.path.exists(predictor_path):
        try:
            url = f"{F0_MODELS_URL}/{predictor}"
            download_file_from_url(url, predictor_path)
            return os.path.exists(predictor_path)
        except Exception as e:
            print(f"Failed to download predictor {predictor}: {e}")
            return False
    
    return True


def download_embedder(embedders_mode: str, hubert: str) -> bool:
    """Download an embedder model based on mode and hubert model name."""
    embedders_path = os.path.join("advanced_rvc_inference", "rvc", "models", "embedders")
    os.makedirs(embedders_path, exist_ok=True)
    
    if embedders_mode == "fairseq":
        embedder_path = os.path.join(embedders_path, f"{hubert}.pth")
    elif embedders_mode == "transformers":
        embedder_path = os.path.join(embedders_path, hubert)
    elif embedders_mode == "whisper":
        embedder_path = os.path.join(embedders_path, f"{hubert}.pt")
    else:
        embedder_path = os.path.join(embedders_path, f"{hubert}.onnx")
    
    if not os.path.exists(embedder_path):
        try:
            if embedders_mode == "fairseq":
                url = f"{EMBEDDERS_URL}/hubert_base.pt"
            elif embedders_mode == "transformers":
                url = f"https://huggingface.co/{hubert}/resolve/main/pytorch_model.bin"
                embedder_dir = os.path.join(embedders_path, hubert)
                os.makedirs(embedder_dir, exist_ok=True)
                download_file_from_url(url, os.path.join(embedder_dir, "pytorch_model.bin"))
                return True
            elif embedders_mode == "whisper":
                url = f"{EMBEDDERS_URL}/whisper/{hubert}.pt"
            else:
                url = f"{EMBEDDERS_URL}/onnx/{hubert}.onnx"
            
            download_file_from_url(url, embedder_path)
            return os.path.exists(embedder_path)
        except Exception as e:
            print(f"Failed to download embedder {hubert}: {e}")
            return False
    
    return True


def fetch_pretrained_data():
    pretraineds_custom_path = os.path.join("advanced_rvc_inference", "rvc", "models", "pretraineds", "custom")
    os.makedirs(pretraineds_custom_path, exist_ok=True)
    try:
        with open(
            os.path.join(pretraineds_custom_path, json_url.split("/")[-1]), "r"
        ) as f:
            data = json.load(f)
    except:
        try:
            response = requests.get(json_url)
            response.raise_for_status()
            data = response.json()
            with open(
                os.path.join(pretraineds_custom_path, json_url.split("/")[-1]),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    data,
                    f,
                    indent=2,
                    separators=(",", ": "),
                    ensure_ascii=False,
                )
        except:
            data = {
                "Titan": {
                    "32k": {"D": "null", "G": "null"},
                },
            }
    return data


def get_pretrained_list():
    data = fetch_pretrained_data()
    return list(data.keys())


def get_pretrained_sample_rates(model):
    data = fetch_pretrained_data()
    return list(data[model].keys())


def get_file_size(url):
    response = requests.head(url)
    return int(response.headers.get("content-length", 0))


def download_file_from_url(url, destination_path, progress_bar=None):
    """Download a file from URL with optional progress tracking."""
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    block_size = 1024
    with open(destination_path, "wb") as file:
        for data in response.iter_content(block_size):
            file.write(data)
            if progress_bar:
                progress_bar.update(len(data))


def download_file(url, destination_path, progress_bar):
    """Legacy download function for backward compatibility."""
    download_file_from_url(url, destination_path, progress_bar)


def download_pretrained_model(model, sample_rate, url_g="", url_d=""):
    save_path = os.path.join("advanced_rvc_inference", "rvc", "models", "pretraineds", "custom")
    os.makedirs(save_path, exist_ok=True)
    tasks = []

    if url_g or url_d:
        tasks = [
            (u, os.path.join(save_path, os.path.basename(u)))
            for u in [url_g, url_d]
            if u
        ]
        if not tasks:
            return gr.Warning(i18n("Please provide at least one URL."))
    else:
        data = fetch_pretrained_data()
        paths = data[model][sample_rate]
        tasks = [
            (
                f"https://huggingface.co/{p}",
                os.path.join(save_path, os.path.basename(p)),
            )
            for p in [paths["D"], paths["G"]]
        ]

    gr.Info(i18n("Downloading pretrained model..."))

    with tqdm(
        total=sum(get_file_size(u) for u, _ in tasks),
        unit="iB",
        unit_scale=True,
        desc="Downloading files",
    ) as pbar:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(download_file, url, dst, pbar) for url, dst in tasks
            ]
            for f in futures:
                f.result()

    gr.Info(i18n("Pretrained model downloaded successfully!"))
    print("Pretrained model downloaded successfully!")


def update_sample_rate_dropdown(model):
    return gr.Dropdown.update(
        choices=get_pretrained_sample_rates(model),
        value=get_pretrained_sample_rates(model)[0]
    )


def download_handler(is_custom, model, sample_rate, url_g, url_d):
    if is_custom:
        download_pretrained_model(
            None,
            None,
            url_g.replace("?download=true", ""),
            url_d.replace("?download=true", ""),
        )
    else:
        download_pretrained_model(model, sample_rate, "", "")


def download_f0_embedder_handler(f0_method, hubert_model, embedder_mode, f0_onnx):
    """Download F0 predictor and embedder models."""
    gr.Info(i18n(f"Downloading F0 predictor ({f0_method}) and embedder ({hubert_model})..."))
    
    try:
        success = check_assets(f0_method, hubert_model, f0_onnx, embedder_mode)
        
        if success:
            gr.Info(i18n("F0 predictor and embedder models downloaded successfully!"))
            return i18n("Download completed successfully!")
        else:
            gr.Warning(i18n("Some downloads failed. Please check the console for details."))
            return i18n("Download completed with some errors.")
    except Exception as e:
        gr.Error(i18n(f"Download failed: {str(e)}"))
        return i18n(f"Download failed: {str(e)}")


def download_tab():
    with gr.Column():
        gr.Markdown(value=i18n("## Download Model"))
        model_link = gr.Textbox(
            label=i18n("Model Link"),
            placeholder=i18n("Introduce the model link"),
            interactive=True,
        )
        model_download_output_info = gr.Textbox(
            label=i18n("Output Information"),
            info=i18n("The output information will be displayed here."),
            value="",
            max_lines=8,
            interactive=False,
        )
        model_download_button = gr.Button(i18n("Download Model"))
        model_download_button.click(
            fn=run_download_script,
            inputs=[model_link],
            outputs=[model_download_output_info],
        )
        gr.Markdown(value=i18n("## Drop files"))
        dropbox = gr.File(
            label=i18n(
                "Drag your .pth file and .index file into this space. Drag one and then the other."
            ),
            type="filepath",
        )

        dropbox.upload(
            fn=save_drop_model,
            inputs=[dropbox],
            outputs=[dropbox],
        )
        
        # New F0 Models and Embedders Download Section
        gr.Markdown(value=i18n("## Download F0 Models and Embedders"))
        
        with gr.Group():
            f0_method = gr.Dropdown(
                label=i18n("F0 Method"),
                info=i18n("Select the F0 extraction method to download."),
                choices=list(F0_METHODS.keys()),
                value="rmvpe",
                interactive=True,
            )
            
            embedder_mode = gr.Dropdown(
                label=i18n("Embedder Mode"),
                info=i18n("Select the embedder mode."),
                choices=list(EMBEDDER_MODES.keys()),
                value="fairseq",
                interactive=True,
            )
            
            hubert_model = gr.Dropdown(
                label=i18n("Hubert/Embedder Model"),
                info=i18n("Select the Hubert or embedder model to download."),
                choices=["hubert_base", "whisper-base", "whisper-small", "whisper-medium"],
                value="hubert_base",
                interactive=True,
            )
            
            f0_onnx = gr.Checkbox(
                label=i18n("Use ONNX Models"),
                info=i18n("Use ONNX versions of F0 models if available."),
                value=False,
                interactive=True,
            )
            
            download_f0_embedder_button = gr.Button(i18n("Download F0 & Embedder"))
            download_output = gr.Textbox(
                label=i18n("Download Status"),
                info=i18n("The download status will be displayed here."),
                value="",
                max_lines=5,
                interactive=False,
            )
            
            download_f0_embedder_button.click(
                fn=download_f0_embedder_handler,
                inputs=[f0_method, hubert_model, embedder_mode, f0_onnx],
                outputs=[download_output],
            )

        gr.Markdown(value=i18n("## Download Pretrained Models"))

        with gr.Group():
            with gr.Group(visible=True) as default:
                pretrained_model = gr.Dropdown(
                    label=i18n("Pretrained"),
                    info=i18n("Select the pretrained model you want to download."),
                    choices=get_pretrained_list(),
                    value="Titan",
                    interactive=True,
                )
                pretrained_sample_rate = gr.Dropdown(
                    label=i18n("Sampling Rate"),
                    info=i18n("And select the sampling rate."),
                    choices=get_pretrained_sample_rates(pretrained_model.value),
                    value="40k",
                    interactive=True,
                    allow_custom_value=True,
                )

            with gr.Group(visible=False) as custom:
                url_g = gr.Textbox(label=i18n("Pretrained G"), interactive=True)
                url_d = gr.Textbox(label=i18n("Pretrained D"), interactive=True)

            use_custom = gr.Checkbox(
                label=i18n("Custom Pretrained"),
                value=False,
                interactive=True,
            )

        download_pretrained = gr.Button(i18n("Download"))

        pretrained_model.change(
            update_sample_rate_dropdown,
            inputs=[pretrained_model],
            outputs=[pretrained_sample_rate],
        )

        use_custom.change(
            fn=lambda x: (
                gr.Group.update(visible=not x),
                gr.Group.update(visible=x),
            ),
            inputs=[use_custom],
            outputs=[default, custom],
        )

        download_pretrained.click(
            fn=download_handler,
            inputs=[use_custom, pretrained_model, pretrained_sample_rate, url_g, url_d],
            outputs=[],
        )

import os
import shutil
import gc
import torch
from multiprocessing import cpu_count
from lib.modules import VC
from lib.split_audio import split_silence_nonsilent, adjust_audio_lengths, combine_silence_nonsilent

class Configs:
    def __init__(self, device, is_half):
        self.device = device
        self.is_half = is_half
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            #if (
#                    ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
#                    or "P40" in self.gpu_name.upper()
#                    or "1060" in self.gpu_name
#                    or "1070" in self.gpu_name
#                    or "1080" in self.gpu_name
#            ):
#                print("16 series/10 series P40 forced single precision")
#                self.is_half = False
#                for config_file in ["32k.json", "40k.json", "48k.json"]:
#                    with open(BASE_DIR / "src" / "configs" / config_file, "r") as f:
#                        strr = f.read().replace("true", "false")
#                    with open(BASE_DIR / "src" / "configs" / config_file, "w") as f:
#                        f.write(strr)
#                with open(BASE_DIR / "src" / "trainset_preprocess_pipeline_print.py", "r") as f:
#                    strr = f.read().replace("3.7", "3.0")
#                with open(BASE_DIR / "src" / "trainset_preprocess_pipeline_print.py", "w") as f:
#                    f.write(strr)
#            else:
#                self.gpu_name = None
#            self.gpu_mem = int(
#                torch.cuda.get_device_properties(i_device).total_memory
#                / 1024
#                / 1024
#                / 1024
#                + 0.4
#            )
#            if self.gpu_mem <= 4:
#                with open(BASE_DIR / "src" / "trainset_preprocess_pipeline_print.py", "r") as f:
#                    strr = f.read().replace("3.7", "3.0")
#                with open(BASE_DIR / "src" / "trainset_preprocess_pipeline_print.py", "w") as f:
#                    f.write(strr)
        elif torch.backends.mps.is_available():
            print("No supported N-card found, use MPS for inference")
            self.device = "mps"
        else:
            print("No supported N-card found, use CPU for inference")
            self.device = "cpu"

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # 6G memory config
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G memory config
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem != None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max

def get_model(voice_model):
    model_dir = os.path.join(os.getcwd(), "models", voice_model)
    model_filename, index_filename = None, None
    for file in os.listdir(model_dir):
        ext = os.path.splitext(file)[1]
        if ext == '.pth':
            model_filename = file
        if ext == '.index':
            index_filename = file

    if model_filename is None:
        print(f'No model file exists in {models_dir}.')
        return None, None

    return os.path.join(model_dir, model_filename), os.path.join(model_dir, index_filename) if index_filename else ''

def infer_audio(
    model_name,
    audio_path,
    f0_change=0,
    f0_method="rmvpe+",
    min_pitch="50",
    max_pitch="1100",
    crepe_hop_length=128,
    index_rate=0.75,
    filter_radius=3,
    rms_mix_rate=0.25,
    protect=0.33,
    split_infer=False,
    min_silence=500,
    silence_threshold=-50,
    seek_step=1,
    keep_silence=100,
    do_formant=False,
    quefrency=0,
    timbre=1,
    f0_autotune=False,
    audio_format="wav",
    resample_sr=0,
    hubert_model_path="assets/hubert/hubert_base.pt",
    rmvpe_model_path="assets/rmvpe/rmvpe.pt",
    fcpe_model_path="assets/fcpe/fcpe.pt"
    ):
    os.environ["rmvpe_model_path"] = rmvpe_model_path
    os.environ["fcpe_model_path"] = fcpe_model_path
    configs = Configs('cuda:0', True)
    vc = VC(configs)
    pth_path, index_path = get_model(model_name)
    vc_data = vc.get_vc(pth_path, protect, 0.5)
    
    if split_infer:
        inferred_files = []
        temp_dir = os.path.join(os.getcwd(), "seperate", "temp")
        os.makedirs(temp_dir, exist_ok=True)
        print("Splitting audio to silence and nonsilent segments.")
        silence_files, nonsilent_files = split_silence_nonsilent(audio_path, min_silence, silence_threshold, seek_step, keep_silence)
        print(f"Total silence segments: {len(silence_files)}.\nTotal nonsilent segments: {len(nonsilent_files)}.")
        for i, nonsilent_file in enumerate(nonsilent_files):
            print(f"Inferring nonsilent audio {i+1}")
            inference_info, audio_data, output_path = vc.vc_single(
            0,
            nonsilent_file,
            f0_change,
            f0_method,
            index_path,
            index_path,
            index_rate,
            filter_radius,
            resample_sr,
            rms_mix_rate,
            protect,
            audio_format,
            crepe_hop_length,
            do_formant,
            quefrency,
            timbre,
            min_pitch,
            max_pitch,
            f0_autotune,
            hubert_model_path
            )
            if inference_info[0] == "Success.":
                print("Inference ran successfully.")
                print(inference_info[1])
                print("Times:\nnpy: %.2fs f0: %.2fs infer: %.2fs\nTotal time: %.2fs" % (*inference_info[2],))
            else:
                print(f"An error occurred while processing.\n{inference_info[0]}")
                return None
            inferred_files.append(output_path)
        print("Adjusting inferred audio lengths.")
        adjusted_inferred_files = adjust_audio_lengths(nonsilent_files, inferred_files)
        print("Combining silence and inferred audios.")
        output_count = 1
        while True:
            output_path = os.path.join(os.getcwd(), "output", f"{os.path.splitext(os.path.basename(audio_path))[0]}{model_name}{f0_method.capitalize()}_{output_count}.{audio_format}")
            if not os.path.exists(output_path):
                break
            output_count += 1
        output_path = combine_silence_nonsilent(silence_files, adjusted_inferred_files, keep_silence, output_path)
        [shutil.move(inferred_file, temp_dir) for inferred_file in inferred_files]
        shutil.rmtree(temp_dir)
    else:
        inference_info, audio_data, output_path = vc.vc_single(
            0,
            audio_path,
            f0_change,
            f0_method,
            index_path,
            index_path,
            index_rate,
            filter_radius,
            resample_sr,
            rms_mix_rate,
            protect,
            audio_format,
            crepe_hop_length,
            do_formant,
            quefrency,
            timbre,
            min_pitch,
            max_pitch,
            f0_autotune,
            hubert_model_path
        )
        if inference_info[0] == "Success.":
            print("Inference ran successfully.")
            print(inference_info[1])
            print("Times:\nnpy: %.2fs f0: %.2fs infer: %.2fs\nTotal time: %.2fs" % (*inference_info[2],))
        else:
            print(f"An error occurred while processing.\n{inference_info[0]}")
            del configs, vc
            gc.collect()
            return inference_info[0]
    
    del configs, vc
    gc.collect()
    return output_path
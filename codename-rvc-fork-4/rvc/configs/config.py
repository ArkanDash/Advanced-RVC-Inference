import torch
import os
import json

arch_config_paths = {
    "hifi_mrf_refine": [
        os.path.join("hifi_mrf_refine", "48000.json"),
        os.path.join("hifi_mrf_refine", "40000.json"),
        os.path.join("hifi_mrf_refine", "32000.json"),
    ],
    "ringformer_v1": [
        os.path.join("ringformer_v1", "48000.json"),
        os.path.join("ringformer_v1", "40000.json"),
        os.path.join("ringformer_v1", "32000.json"),
        os.path.join("ringformer_v1", "24000.json"),
    ],
    "ringformer_v2": [
        os.path.join("ringformer_v2", "48000.json"),
        os.path.join("ringformer_v2", "40000.json"),
        os.path.join("ringformer_v2", "32000.json"),
        os.path.join("ringformer_v2", "24000.json"),
    ],
    "pcph_gan": [
        os.path.join("pcph_gan", "48000.json"),
        os.path.join("pcph_gan", "40000.json"),
        os.path.join("pcph_gan", "32000.json"),
    ],
}

def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class Config:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        initial_precision = self.get_precision()

        if self.device == "cpu":
            self.is_half = False
            print("[CONFIG] Running on CPU, forcing fp32 precision.")
        else:
            self.is_half = initial_precision == "fp16"  or initial_precision == "bf16"
            print(f"[CONFIG] Running on CUDA, training-only precision loaded from config: {initial_precision}")
        self.gpu_name = (
            torch.cuda.get_device_name(int(self.device.split(":")[-1]))
            if self.device.startswith("cuda")
            else None
        )

        self.json_config = self.load_config_json("hifi_mrf_refine")
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def load_config_json(self, vocoder_arch="hifi_mrf_refine"):
        configs = {}
        for config_file in arch_config_paths.get(vocoder_arch, arch_config_paths["hifi_mrf_refine"]):
            config_path = os.path.join("rvc", "configs", config_file)
            with open(config_path, "r") as f:
                configs[config_file] = json.load(f)
        return configs


    def set_precision(self, precision):
        if precision not in ["fp32", "fp16", "bf16"]:
            raise ValueError("Invalid precision type. Must be 'fp32', 'fp16' or 'bf16'.")

        fp16_run_value = precision == "fp16"
        bf16_run_value = precision == "bf16"

        self.is_half =  fp16_run_value or bf16_run_value

        for config_path in arch_config_paths["hifi_mrf_refine"]:
            full_config_path = os.path.join("rvc", "configs", config_path)
            try:
                with open(full_config_path, "r") as f:
                    config = json.load(f)
                config["train"]["fp16_run"] = fp16_run_value
                config["train"]["bf16_run"] = bf16_run_value
                with open(full_config_path, "w") as f:
                    json.dump(config, f, indent=4)
            except FileNotFoundError:
                print(f"File not found: {full_config_path}")

        for config_path in arch_config_paths["ringformer_v1"]:
            full_config_path = os.path.join("rvc", "configs", config_path)
            try:
                with open(full_config_path, "r") as f:
                    config = json.load(f)
                config["train"]["fp16_run"] = fp16_run_value
                config["train"]["bf16_run"] = bf16_run_value
                with open(full_config_path, "w") as f:
                    json.dump(config, f, indent=4)
            except FileNotFoundError:
                print(f"File not found: {full_config_path}")

        for config_path in arch_config_paths["ringformer_v2"]:
            full_config_path = os.path.join("rvc", "configs", config_path)
            try:
                with open(full_config_path, "r") as f:
                    config = json.load(f)
                config["train"]["fp16_run"] = fp16_run_value
                config["train"]["bf16_run"] = bf16_run_value
                with open(full_config_path, "w") as f:
                    json.dump(config, f, indent=4)
            except FileNotFoundError:
                print(f"File not found: {full_config_path}")

        for config_path in arch_config_paths["ringformer_v3"]:
            full_config_path = os.path.join("rvc", "configs", config_path)
            try:
                with open(full_config_path, "r") as f:
                    config = json.load(f)
                config["train"]["fp16_run"] = fp16_run_value
                config["train"]["bf16_run"] = bf16_run_value
                with open(full_config_path, "w") as f:
                    json.dump(config, f, indent=4)
            except FileNotFoundError:
                print(f"File not found: {full_config_path}")

        for config_path in arch_config_paths["pcph_gan"]:
            full_config_path = os.path.join("rvc", "configs", config_path)
            try:
                with open(full_config_path, "r") as f:
                    config = json.load(f)
                config["train"]["fp16_run"] = fp16_run_value
                config["train"]["bf16_run"] = bf16_run_value
                with open(full_config_path, "w") as f:
                    json.dump(config, f, indent=4)
            except FileNotFoundError:
                print(f"File not found: {full_config_path}")

        return f"Precision set to: {precision}."


    def get_precision(self):
        if not arch_config_paths:
            raise FileNotFoundError("No configuration paths provided.")

        full_config_path = os.path.join("rvc", "configs", arch_config_paths["hifi_mrf_refine"][0])
        try:
            with open(full_config_path, "r") as f:
                config = json.load(f)

            fp16_run_value = config["train"].get("fp16_run", False)
            bf16_run_value = config["train"].get("bf16_run", False)

            if fp16_run_value:
                precision = "fp16"
            elif bf16_run_value:
                precision = "bf16"
            else:
                precision = "fp32"
            return precision
        except FileNotFoundError:
            print(f"File not found: {full_config_path}")
            return None


    def check_precision(self):
        if not arch_config_paths:
            raise FileNotFoundError("No configuration paths provided.")

        full_config_path = os.path.join("rvc", "configs", arch_config_paths["hifi_mrf_refine"][0])
        try:
            with open(full_config_path, "r") as f:
                config = json.load(f)

            fp16_run_value = config["train"].get("fp16_run", False)
            bf16_run_value = config["train"].get("bf16_run", False)

            if fp16_run_value:
                precision = "fp16"
            elif bf16_run_value:
                precision = "bf16"
            else:
                precision = "fp32"

            runtime_precision = "fp32"
            if self.is_half and fp16_run_value:
                runtime_precision = "fp16"
            elif self.is_half and bf16_run_value:
                runtime_precision = "bf16"

            result = (
                f"Config File Precision: {precision}\n"
                f"Runtime Precision: {runtime_precision}\n"
                f"'is_half' Flag: {self.is_half}"
            )
            return result
        except FileNotFoundError:
            print(f"File not found: {full_config_path}")
            return "Configuration file not found."

    def device_config(self):
        if self.device.startswith("cuda"):
            self.set_cuda_config()
        else:
            self.device = "cpu"
            self.is_half = False
            self.set_precision("fp32")

        # Configuration for 6GB GPU memory
        x_pad, x_query, x_center, x_max = (
            (3, 10, 60, 65) if self.is_half else (1, 6, 38, 41)
        )
        if self.gpu_mem is not None and self.gpu_mem <= 4:
            # Configuration for 5GB GPU memory
            x_pad, x_query, x_center, x_max = (1, 5, 30, 32)

        return x_pad, x_query, x_center, x_max


    def set_cuda_config(self):
        i_device = int(self.device.split(":")[-1])
        self.gpu_name = torch.cuda.get_device_name(i_device)

        # GPUs that must be forced to fp32 ( They either don't support fp16 or the performance is tragic and outweights the pros.
        fp32_gpus = ["16", "P40", "P10", "1050", "1060", "1070", "1080"]

        if any(gpu_str.lower() in self.gpu_name.lower() for gpu_str in fp32_gpus):
            if self.is_half:
                print(f"[CONFIG WARNING] Your GPU ({self.gpu_name}) does NOT support fp16 and bf16 precision.")
                print(f"[CONFIG WARNING] Your GPU ({self.gpu_name}) does NOT support RingFormer architecture.")
                print("[CONFIG] Forcing precision to fp32.")
            self.is_half = False
            self.set_precision("fp32")

        self.gpu_mem = torch.cuda.get_device_properties(i_device).total_memory // (1024 ** 3)

def max_vram_gpu(gpu):
    if torch.cuda.is_available():
        gpu_properties = torch.cuda.get_device_properties(gpu)
        total_memory_gb = round(gpu_properties.total_memory / 1024 / 1024 / 1024)
        return total_memory_gb
    else:
        return "1"

def get_gpu_info():
    ngpu = torch.cuda.device_count()
    gpu_infos = []
    if torch.cuda.is_available() or ngpu != 0:
        for i in range(ngpu):
            gpu_name = torch.cuda.get_device_name(i)
            mem = int(
                torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024
                + 0.4
            )
            gpu_infos.append(f"{i}: {gpu_name} ({mem} GB)")
    if len(gpu_infos) > 0:
        gpu_info = "\n".join(gpu_infos)
    else:
        gpu_info = "Unfortunately, there is no compatible GPU available to support your training."
    return gpu_info


def get_number_of_gpus():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        return "-".join(map(str, range(num_gpus)))
    else:
        return "-"


def microarchitecture_capability_checker():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


def check_if_fp16():
    for arch_name, config_files in arch_config_paths.items():
        for config_file in config_files:
            full_config_path = os.path.join("rvc", "configs", config_file)
            try:
                with open(full_config_path, "r") as f:
                    config = json.load(f)
                if config.get("train", {}).get("fp16_run", False):
                    return True
            except FileNotFoundError:
                continue
    return False

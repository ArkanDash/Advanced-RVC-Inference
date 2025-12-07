import torch
import os
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)

class Config:
    def __init__(self):
        self.device = "cuda:0"
        self.is_half = True
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                "40 Series" in self.gpu_name
                or "RTX 40" in self.gpu_name
                or "40_" in self.gpu_name
                or "40-" in self.gpu_name
                or "Ampere" in self.gpu_name
            ):
                print("Half precision disabled for 40 Series/RTX 40 GPUs")
                self.is_half = False
            else:
                self.gpu_mem = torch.cuda.get_device_properties(i_device).total_memory
            return 1, 8, 64, 128
        elif torch.backends.mps.is_available():
            print("Using Mac Metal Performance Shaders (MPS) backend")
            return 1, 8, 64, 128
        else:
            print("Using CPU backend")
            return 1, 8, 64, 128

def get_gpu_info():
    try:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_info_list = []
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
                gpu_info_list.append(f"GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")
            return " | ".join(gpu_info_list)
        else:
            return "No GPU available"
    except:
        return "GPU info unavailable"

def get_number_of_gpus():
    try:
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        else:
            return 0
    except:
        return 0

def max_vram_gpu(gpu_id=0):
    try:
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            return torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
        return 0
    except:
        return 0
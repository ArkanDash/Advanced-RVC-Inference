from multiprocessing import cpu_count

import torch


# Device and parameter configuration
class Config:
    def __init__(self):
        # Determine the device to use
        self.device = self.get_device()
        # Get the number of CPU cores
        self.n_cpu = cpu_count()
        # Initialize GPU name and memory
        self.gpu_name = None
        self.gpu_mem = None
        # Configure device-specific parameters
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    # Determine the device to use
    def get_device(self):
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # Configure device-specific parameters
    def device_config(self):
        if torch.cuda.is_available():
            print("Device in use - CUDA")
            self._configure_gpu()
        elif torch.backends.mps.is_available():
            print("Device in use - MPS")
            self.device = "mps"
        else:
            print("Device in use - CPU")
            self.device = "cpu"

        # Set padding, query, center, and max values
        x_pad, x_query, x_center, x_max = (1, 6, 38, 41)
        # Adjust parameters if GPU memory is low
        if self.gpu_mem is not None and self.gpu_mem <= 4:
            x_pad, x_query, x_center, x_max = (1, 5, 30, 32)

        return x_pad, x_query, x_center, x_max

    # Configure GPU-specific settings
    def _configure_gpu(self):
        # Get GPU name
        self.gpu_name = torch.cuda.get_device_name(self.device)
        # Calculate GPU memory in GB
        self.gpu_mem = int(torch.cuda.get_device_properties(self.device).total_memory / 1024 / 1024 / 1024 + 0.4)

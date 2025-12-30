import numpy as np
import torch
import sys
import os

def print_file_shape(file_path):
    try:
        ext = os.path.splitext(file_path)[1]
        
        if ext == '.npy':
            # Directly load the .npy file with numpy (this is typically fast)
            data = np.load(file_path)
            print(f"Shape of '{file_path}' (NumPy): {data.shape}")

        elif ext == '.pt':
            # Only load as torch tensor if it's a .pt file
            data = torch.load(file_path, map_location='cpu')
            if hasattr(data, 'shape'):
                print(f"Shape of '{file_path}' (PyTorch Tensor): {data.shape}")
            else:
                print(f"Loaded '{file_path}' (PyTorch), but it's not a Tensor: type={type(data)}")

        else:
            print(f"Unsupported file type for '{file_path}'. Only .npy and .pt are supported.")

    except Exception as e:
        print(f"Failed to load '{file_path}': {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_shape.py <path_to_npy_or_pt_file>")
    else:
        file_path = sys.argv[1]
        print_file_shape(file_path)

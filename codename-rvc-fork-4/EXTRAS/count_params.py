import torch
import os

def count_params_in_checkpoint(pth_path):
    checkpoint = torch.load(pth_path, map_location="cpu")
    print(f"\nLoaded checkpoint: {pth_path}")
    
    if not isinstance(checkpoint, dict):
        print("Not a dict checkpoint; aborting.")
        return

    found_any = False
    for key, value in checkpoint.items():
        if isinstance(value, dict):
            param_count = sum(v.numel() for v in value.values() if isinstance(v, torch.Tensor))
            print(f"Key: '{key}' → Parameters: {param_count:,}")
            found_any = True

    if not found_any:
        # fallback: assume it's a plain state_dict
        param_count = sum(v.numel() for v in checkpoint.values() if isinstance(v, torch.Tensor))
        print(f"Top-level state_dict → Parameters: {param_count:,}")

if __name__ == "__main__":
    path = input("Enter the path to the .pth checkpoint file: ").strip()
    
    if not os.path.isfile(path):
        print(f"File not found: {path}")
    else:
        count_params_in_checkpoint(path)

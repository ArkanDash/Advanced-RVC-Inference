import os
import torch
from collections import OrderedDict

def extract(ckpt):
    a = ckpt["model"]
    opt = OrderedDict()
    opt["weight"] = {}
    for key in a.keys():
        if "enc_q" in key:
            continue
        opt["weight"][key] = a[key]
    print(f"[DEBUG] extract() returning keys: {list(opt['weight'].keys())}")
    return opt

def model_blender(name, path1, path2, ratio):
    try:
        message = f"Model {path1} and {path2} are merged with alpha {ratio}."
        print(f"[DEBUG] Starting model_blender with: {message}")
        
        # Load checkpoints
        ckpt1 = torch.load(path1, map_location="cpu", weights_only=True)
        ckpt2 = torch.load(path2, map_location="cpu", weights_only=True)
        print(f"[DEBUG] Loaded ckpt1 keys: {list(ckpt1.keys())}")
        print(f"[DEBUG] Loaded ckpt2 keys: {list(ckpt2.keys())}")

        # Check sample rate compatibility
        if ckpt1["sr"] != ckpt2["sr"]:
            err_msg = "The sample rates of the two models are not the same."
            print(f"[DEBUG] {err_msg}")
            # Ensure consistent tuple return: error message and None
            return err_msg, None

        # Retrieve configuration values
        cfg = ckpt1["config"]
        cfg_f0 = ckpt1["f0"]
        cfg_version = ckpt1["version"]
        cfg_sr = ckpt1["sr"]
        vocoder = ckpt1.get("vocoder", "HiFi-GAN")
        print(f"[DEBUG] Config: {cfg}, sr: {cfg_sr}, version: {cfg_version}")

        # Extract models if needed
        if "model" in ckpt1:
            print("[DEBUG] Extracting model from ckpt1")
            ckpt1 = extract(ckpt1)
        else:
            ckpt1 = ckpt1["weight"]
            print("[DEBUG] Using ckpt1['weight'] directly")
        if "model" in ckpt2:
            print("[DEBUG] Extracting model from ckpt2")
            ckpt2 = extract(ckpt2)
        else:
            ckpt2 = ckpt2["weight"]
            print("[DEBUG] Using ckpt2['weight'] directly")

        print(f"[DEBUG] ckpt1 model keys: {list(ckpt1.keys())}")
        print(f"[DEBUG] ckpt2 model keys: {list(ckpt2.keys())}")

        # Check model architecture compatibility
        if sorted(list(ckpt1.keys())) != sorted(list(ckpt2.keys())):
            err_msg = "Fail to merge the models. The model architectures are not the same."
            print(f"[DEBUG] {err_msg}")
            return err_msg, None

        # Blend model weights
        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt1.keys():
            if key == "emb_g.weight" and ckpt1[key].shape != ckpt2[key].shape:
                min_shape0 = min(ckpt1[key].shape[0], ckpt2[key].shape[0])
                print(f"[DEBUG] Blending key '{key}' with different shapes, using min shape: {min_shape0}")
                opt["weight"][key] = (
                    ratio * (ckpt1[key][:min_shape0].float())
                    + (1 - ratio) * (ckpt2[key][:min_shape0].float())
                ).half()
            else:
                opt["weight"][key] = (
                    ratio * (ckpt1[key].float())
                    + (1 - ratio) * (ckpt2[key].float())
                ).half()
            print(f"[DEBUG] Blended key '{key}': shape {opt['weight'][key].shape}")

        # Append additional configuration data
        opt["config"] = cfg
        opt["sr"] = cfg_sr
        opt["f0"] = cfg_f0
        opt["version"] = cfg_version
        opt["info"] = message
        opt["vocoder"] = vocoder

        output_path = os.path.join("logs", f"{name}.pth")
        torch.save(opt, output_path)
        print(f"[DEBUG] Model blending successful. Saved to: {output_path}")
        
        ret_tuple = (message, output_path)
        print(f"[DEBUG] Returning tuple: {ret_tuple}")
        return ret_tuple
    except Exception as error:
        print(f"[DEBUG] Exception in model_blender: {error}")
        ret_tuple = (str(error), None)
        print(f"[DEBUG] Returning error tuple: {ret_tuple}")
        return ret_tuple
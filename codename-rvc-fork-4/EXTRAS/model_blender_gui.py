import os
import torch
from collections import OrderedDict
import tkinter as tk
from tkinter import filedialog, messagebox
import gc

def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()

def normalize_sr(sr):
    """
    Normalize the sample rate value.
    If sr is a string ending with 'k' (e.g. "48k"), it converts it to an integer (e.g. 48000).
    Otherwise, it returns the original value.
    """
    if isinstance(sr, str) and sr.lower().endswith('k'):
        try:
            value = float(sr[:-1]) * 1000
            return int(value)
        except Exception as e:
            print(f"[DEBUG] Failed to normalize sr '{sr}': {e}")
            return sr
    return sr

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

def model_blender(name, path1, path2, ratio, out_dir):
    try:
        message = f"Model {path1} and {path2} are merged with alpha {ratio}."
        print(f"[DEBUG] Starting model_blender with: {message}")
        
        # Load checkpoints
        ckpt1 = torch.load(path1, map_location="cpu", weights_only=True)
        ckpt2 = torch.load(path2, map_location="cpu", weights_only=True)
        print(f"[DEBUG] Loaded ckpt1 keys: {list(ckpt1.keys())}")
        print(f"[DEBUG] Loaded ckpt2 keys: {list(ckpt2.keys())}")

        # Normalize and check sample rate compatibility
        sr1 = normalize_sr(ckpt1["sr"])
        sr2 = normalize_sr(ckpt2["sr"])
        print(f"[DEBUG] Normalized sample rates: Model A = {sr1}, Model B = {sr2}")
        if sr1 != sr2:
            err_msg = "The sample rates of the two models are not the same."
            print(f"[DEBUG] {err_msg}")
            return err_msg, None
        else:
            # Update sample rate keys to the normalized value
            ckpt1["sr"] = sr1
            ckpt2["sr"] = sr1

        # Retrieve configuration values
        cfg = ckpt1["config"]
        cfg_f0 = ckpt1["f0"]
        cfg_version = ckpt1["version"]
        cfg_sr = sr1  # normalized sample rate
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

        # Save to the output path specified by the user
        output_path = os.path.join(out_dir, f"{name}.pth")
        os.makedirs(out_dir, exist_ok=True)
        torch.save(opt, output_path)
        print(f"[DEBUG] Model blending successful. Saved to: {output_path}")
        
        ret_tuple = (message, output_path)
        print(f"[DEBUG] Returning tuple: {ret_tuple}")
        
        # Unload models and clear memory
        del ckpt1, ckpt2, opt
        clear_memory()

        return ret_tuple
    except Exception as error:
        print(f"[DEBUG] Exception in model_blender: {error}")
        ret_tuple = (str(error), None)
        print(f"[DEBUG] Returning error tuple: {ret_tuple}")
        clear_memory()
        return ret_tuple

class ModelBlenderGUI:
    def __init__(self, master):
        self.master = master
        master.title("Model Blender GUI")

        # Model A file
        self.label1 = tk.Label(master, text="Path to Model A:")
        self.label1.grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.entry_model_a = tk.Entry(master, width=50)
        self.entry_model_a.grid(row=0, column=1, padx=5, pady=5)
        self.button_browse_a = tk.Button(master, text="Browse", command=self.browse_model_a)
        self.button_browse_a.grid(row=0, column=2, padx=5, pady=5)

        # Model B file
        self.label2 = tk.Label(master, text="Path to Model B:")
        self.label2.grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.entry_model_b = tk.Entry(master, width=50)
        self.entry_model_b.grid(row=1, column=1, padx=5, pady=5)
        self.button_browse_b = tk.Button(master, text="Browse", command=self.browse_model_b)
        self.button_browse_b.grid(row=1, column=2, padx=5, pady=5)

        # Output file name
        self.label3 = tk.Label(master, text="Merged Model Name (without extension):")
        self.label3.grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.entry_output_name = tk.Entry(master, width=50)
        self.entry_output_name.grid(row=2, column=1, padx=5, pady=5)

        # Output directory
        self.label4 = tk.Label(master, text="Output Folder:")
        self.label4.grid(row=3, column=0, padx=5, pady=5, sticky="e")
        self.entry_output_dir = tk.Entry(master, width=50)
        self.entry_output_dir.grid(row=3, column=1, padx=5, pady=5)
        self.button_browse_out = tk.Button(master, text="Browse", command=self.browse_output_dir)
        self.button_browse_out.grid(row=3, column=2, padx=5, pady=5)

        # Merge ratio slider
        self.label5 = tk.Label(master, text="Merge ratio (Model A weight):")
        self.label5.grid(row=4, column=0, padx=5, pady=5, sticky="e")
        self.scale_ratio = tk.Scale(master, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL)
        self.scale_ratio.set(0.5)
        self.scale_ratio.grid(row=4, column=1, padx=5, pady=5, sticky="w")

        # Merge button
        self.button_merge = tk.Button(master, text="Merge Models", command=self.merge_models)
        self.button_merge.grid(row=5, column=1, padx=5, pady=15)

    def browse_model_a(self):
        file_path = filedialog.askopenfilename(
            title="Select Model A",
            filetypes=[("PyTorch Files", "*.pth *.pt"), ("All Files", "*.*")]
            )
        if file_path:
            self.entry_model_a.delete(0, tk.END)
            self.entry_model_a.insert(0, file_path)

    def browse_model_b(self):
        file_path = filedialog.askopenfilename(
            title="Select Model B", 
            filetypes=[("PyTorch Files", "*.pth *.pt"), ("All Files", "*.*")]
        )
        if file_path:
            self.entry_model_b.delete(0, tk.END)
            self.entry_model_b.insert(0, file_path)

    def browse_output_dir(self):
        folder_path = filedialog.askdirectory(title="Select Output Folder")
        if folder_path:
            self.entry_output_dir.delete(0, tk.END)
            self.entry_output_dir.insert(0, folder_path)

    def merge_models(self):
        model_a_path = self.entry_model_a.get()
        model_b_path = self.entry_model_b.get()
        output_name = self.entry_output_name.get().strip()
        out_dir = self.entry_output_dir.get().strip()
        ratio = self.scale_ratio.get()

        if not model_a_path or not model_b_path or not output_name or not out_dir:
            messagebox.showerror("Error", "Please specify both model paths, an output folder, and a merged model name.")
            return

        # Run the blender function with user-specified output directory
        msg, out_path = model_blender(output_name, model_a_path, model_b_path, ratio, out_dir)
        if out_path:
            messagebox.showinfo("Success", f"{msg}\nMerged model saved to: {out_path}")
        else:
            messagebox.showerror("Error", msg)

if __name__ == "__main__":
    root = tk.Tk()
    gui = ModelBlenderGUI(root)
    root.mainloop()

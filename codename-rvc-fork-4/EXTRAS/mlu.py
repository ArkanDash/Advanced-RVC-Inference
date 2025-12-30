import os
import torch
import torch.nn.functional as F
import copy
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

def cosine_similarity(tensor_a, tensor_b):
    if torch.equal(tensor_a, tensor_b):
        return 1.0
    a = tensor_a.view(-1).double()
    b = tensor_b.view(-1).double()

    dot = torch.dot(a, b)
    norm_a = a.norm(p=2)
    norm_b = b.norm(p=2)

    eps = 1e-14
    cos_sim = dot / (norm_a * norm_b + eps)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    return round(cos_sim.item(), 8)

def show_similarity_popup(parent, merged_weights, weights_b):
    layers_to_check = [
        "enc_p",
        "flow",
        "dec.conv_pre",
        "dec.ups",
        "dec.noise_convs",
        "dec.resblocks",
        "dec.conv_post",
    ]

    similarities = []
    for layer_prefix in layers_to_check:
        keys_a = [k for k in merged_weights.keys() if k.startswith(layer_prefix)]
        sim_sum = 0.0
        count = 0
        for key in keys_a:
            if key in weights_b:
                a = merged_weights[key]
                b = weights_b[key]
                if a.shape == b.shape:
                    sim = cosine_similarity(a, b)
                    sim_sum += sim
                    count += 1
        avg_sim = (sim_sum / count) if count > 0 else None
        similarities.append((layer_prefix, avg_sim))

    msg = "Similarity between Model B layers and Merged layers (cosine similarity):\n\n"
    for layer, sim in similarities:
        if sim is not None:
            msg += f"{layer}: {sim:.4f}\n"
        else:
            msg += f"{layer}: No matching parameters found\n"

    messagebox.showinfo("Layer Similarity Report", msg, parent=parent)


LAYER_INFO = {
    "enc_p": "Encoder P - Encodes input features a latent representation.",
    "flow": "Flow - Normalizing flow to map between prior and posterior latent representations",
    "dec.conv_pre": "Pre-processing conv layer - Maps the input features.",
    "dec.ups": "Upsampling layers - Expand temporal res. and shape the coarse structure of the waveform.",
    "dec.noise_convs": "Noise conv layers - Inject signal variation/noise.",
    "dec.resblocks": "Residual blocks - Shape both harmonic and temporal structures.",
    "dec.conv_post": "Post-processing conv layer - transform the last hidden features into waveform output.",
}


COLOR_MAP = {
    "primary": "#cdc8ff",
    "secondary": "#221c3d",
    "text_light": "#eeeeee",
    "text_dark": "#666666",
    "text_disabled": "#888888",
    "button_active_bg": "#cdc8ff",
    "button_fg": "#eeeeee",
    "slider_trough": "#8c7dcb",
}

PRIMARY = COLOR_MAP["primary"]
SECONDARY = COLOR_MAP["secondary"]
TEXT_COLOR = COLOR_MAP["text_light"]

def set_style(style, widget_class, *, background=None, foreground=None, font=None, padding=None, map_bg=None, map_fg=None):
    if background or foreground or font or padding is not None:
        style.configure(widget_class,
                        background=background if background else "",
                        foreground=foreground if foreground else "",
                        font=font if font else "",
                        padding=padding if padding else "")
    if map_bg or map_fg:
        if map_bg:
            style.map(widget_class, background=map_bg)
        if map_fg:
            style.map(widget_class, foreground=map_fg)


def blend_tensors(tensor_a, tensor_b, ratio_b):
    return tensor_a * (1 - ratio_b) + tensor_b * ratio_b

def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    weights = ckpt.get('weight', ckpt.get('model', None))
    return weights, ckpt

def merge_checkpoints(weights_a, weights_b, transfer_config):
    merged = {k: v.clone() for k, v in weights_b.items()}

    for prefix, cfg in transfer_config.items():
        mode = cfg['mode']
        ratio = cfg.get('ratio', 0.0)
        for key in merged.keys():
            if key.startswith(prefix) and key in weights_a:
                if mode == 'full':
                    merged[key] = weights_a[key]
                elif mode == 'merge':
                    a = weights_a[key]
                    b = merged[key]
                    if a.shape == b.shape:
                        merged[key] = blend_tensors(b, a, ratio)
    return merged

class LayerRow:
    def __init__(self, parent, layer_name, description, row):
        self.layer_name = layer_name
        self.mode_var = tk.StringVar(value="none")

        self.ratio_var = tk.DoubleVar(value=0.5)

        ttk.Label(parent, text=layer_name, foreground=PRIMARY).grid(row=row, column=0, sticky="w", pady=(10, 0))
        ttk.Label(parent, text=description, foreground=TEXT_COLOR, wraplength=500).grid(row=row+1, column=0, sticky="w")

        self.rb_full = ttk.Radiobutton(parent, text="Full Transfer A → B", variable=self.mode_var, value="full", command=self._toggle_slider)
        self.rb_merge = ttk.Radiobutton(parent, text="Merge A & B", variable=self.mode_var, value="merge", command=self._toggle_slider)
        self.rb_none = ttk.Radiobutton(parent, text="Do Nothing", variable=self.mode_var, value="none", command=self._toggle_slider)

        self.rb_full.grid(row=row, column=1, sticky="w", padx=5)
        self.rb_merge.grid(row=row, column=2, sticky="w", padx=5)
        self.rb_none.grid(row=row, column=3, sticky="w", padx=5)

        self.slider = ttk.Scale(parent, from_=0, to=1, variable=self.ratio_var, orient="horizontal", length=150, style="Horizontal.TScale")
        self.slider.grid(row=row+1, column=1, columnspan=2, sticky="w", padx=5)
        self.label_slider = ttk.Label(parent, text="A:B Blend Ratio")
        self.label_slider.grid(row=row+2, column=1, sticky="w", padx=5)

        self.ratio_entry = ttk.Entry(parent, textvariable=self.ratio_var, width=5)
        self.ratio_entry.grid(row=row+2, column=2, sticky="w", padx=5)
        self.ratio_entry.grid_remove()

        self.slider.configure(command=self._on_slider_change)
        self.ratio_var.trace_add("write", self._on_entry_change)


        self._toggle_slider()

    def _on_slider_change(self, value):
        self.ratio_var.set(round(float(value), 3))

    def _on_entry_change(self, *args):
        try:
            value = float(self.ratio_var.get())
            if 0.0 <= value <= 1.0:
                self.slider.set(value)
        except ValueError:
            pass

    def _toggle_slider(self):
        mode = self.mode_var.get()
        if mode == "merge":
            self.slider.state(["!disabled"])
            self.label_slider.configure(foreground=TEXT_COLOR)
            self.ratio_entry.grid()
        else:
            self.slider.state(["disabled"])
            self.label_slider.configure(foreground="#666666")
            self.ratio_entry.grid_remove()

class ModelMergerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Model Layer Utility 🍇")
        self.update_idletasks()
        self.geometry("850x920")
        self.minsize(850, 920)
        self.configure(bg=SECONDARY)

        self.weights_a = None
        self.ckpt_a = None
        self.weights_b = None
        self.ckpt_b = None

        self.layer_rows = {}

        self._create_widgets()

    def _create_widgets(self):
        style = ttk.Style(self)
        style.theme_use('clam')

        set_style(style, "TFrame",
                  background=COLOR_MAP["secondary"])

        set_style(style, "TLabel",
                  background=COLOR_MAP["secondary"],
                  foreground=COLOR_MAP["text_light"],
                  font=("Segoe UI", 10))

        set_style(style, "TButton",
                  background=COLOR_MAP["secondary"],
                  foreground=COLOR_MAP["button_fg"],
                  font=("Segoe UI", 10),
                  padding=6,
                  map_bg=[("active", COLOR_MAP["button_active_bg"]), ("pressed", COLOR_MAP["button_active_bg"])],
                  map_fg=[("disabled", COLOR_MAP["text_disabled"])])

        set_style(style, "TRadiobutton",
                  background=COLOR_MAP["secondary"],
                  foreground=COLOR_MAP["primary"],
                  font=("Segoe UI", 9),
                  map_fg=[("disabled", COLOR_MAP["text_dark"]), ("selected", COLOR_MAP["primary"])])

        style.configure("Horizontal.TScale",
                        background=COLOR_MAP["secondary"],
                        troughcolor=COLOR_MAP["slider_trough"],
                        bordercolor=COLOR_MAP["secondary"],
                        lightcolor=COLOR_MAP["slider_trough"],
                        darkcolor=COLOR_MAP["slider_trough"])

        frame = ttk.Frame(self, padding=15, style="TFrame")
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Model A (Source - donor of layers):").grid(row=0, column=0, sticky="w")
        self.entry_a = ttk.Entry(frame, width=80)
        self.entry_a.grid(row=1, column=0, sticky="w")
        ttk.Button(frame, text="Browse", command=self.browse_a).grid(row=1, column=1, sticky="w", padx=5)

        ttk.Label(frame, text="Model B (Target - gets layers from A):").grid(row=2, column=0, sticky="w", pady=(15, 0))
        self.entry_b = ttk.Entry(frame, width=80)
        self.entry_b.grid(row=3, column=0, sticky="w")
        ttk.Button(frame, text="Browse", command=self.browse_b).grid(row=3, column=1, sticky="w", padx=5)
        ttk.Button(frame, text="Swap A ↔ B", command=self.swap_models).grid(row=3, column=2, sticky="w", padx=5)


        ttk.Label(frame, text="Layer Transfer Settings:", font=("Segoe UI", 12, "bold")).grid(row=4, column=0, sticky="w", pady=(20, 5))

        row_start = 5
        for i, (layer, desc) in enumerate(LAYER_INFO.items()):
            lr = LayerRow(frame, layer, desc, row_start + i * 3)
            self.layer_rows[layer] = lr

        ttk.Label(frame, text="Output Folder:").grid(row=row_start + len(LAYER_INFO)*3, column=0, sticky="w", pady=(20, 0))
        self.output_folder_entry = ttk.Entry(frame, width=60)
        self.output_folder_entry.grid(row=row_start + len(LAYER_INFO)*3 + 1, column=0, sticky="w")
        ttk.Button(frame, text="Browse", command=self.browse_output).grid(row=row_start + len(LAYER_INFO)*3 + 1, column=1, sticky="w", padx=5)

        ttk.Label(frame, text="Output Filename:").grid(row=row_start + len(LAYER_INFO)*3 + 2, column=0, sticky="w", pady=(10, 0))
        self.output_filename_entry = ttk.Entry(frame, width=60)
        self.output_filename_entry.insert(0, "merged_model.pth")
        self.output_filename_entry.grid(row=row_start + len(LAYER_INFO)*3 + 3, column=0, sticky="w")

        ttk.Button(frame, text="Merge and Save", command=self.merge_and_save).grid(row=row_start + len(LAYER_INFO)*3 + 4, column=0, pady=(20, 0), sticky="w")
        self.status_label = ttk.Label(frame, text="")
        self.status_label.grid(row=row_start + len(LAYER_INFO)*3 + 5, column=0, sticky="w", pady=(10, 0))

    def swap_models(self):
        path_a = self.entry_a.get()
        path_b = self.entry_b.get()
        self.entry_a.delete(0, "end")
        self.entry_b.delete(0, "end")
        self.entry_a.insert(0, path_b)
        self.entry_b.insert(0, path_a)


    def browse_a(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch Checkpoint", "*.pth *.pt")])
        if path:
            self.entry_a.delete(0, "end")
            self.entry_a.insert(0, path)

    def browse_b(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch Checkpoint", "*.pth *.pt")])
        if path:
            self.entry_b.delete(0, "end")
            self.entry_b.insert(0, path)

    def browse_output(self):
        path = filedialog.askdirectory()
        if path:
            self.output_folder_entry.delete(0, "end")
            self.output_folder_entry.insert(0, path)

    def merge_and_save(self):
        path_a = self.entry_a.get()
        path_b = self.entry_b.get()
        output_folder = self.output_folder_entry.get()
        output_filename = self.output_filename_entry.get()

        if not os.path.isfile(path_a):
            self.status_label.configure(text="Model A path invalid!", foreground="red")
            return
        if not os.path.isfile(path_b):
            self.status_label.configure(text="Model B path invalid!", foreground="red")
            return
        if not os.path.isdir(output_folder):
            self.status_label.configure(text="Output folder invalid!", foreground="red")
            return
        if not output_filename.endswith(".pth"):
            self.status_label.configure(text="Filename must end with .pth", foreground="red")
            return

        self.status_label.configure(text="Loading checkpoints...")
        self.update()

        try:
            weights_a, ckpt_a = load_checkpoint(path_a)
            weights_b, ckpt_b = load_checkpoint(path_b)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load checkpoints: {e}")
            self.status_label.configure(text="Error loading checkpoints.", foreground="red")
            return

        self.status_label.configure(text="Merging weights...")
        self.update()

        transfer_cfg = {}
        for layer, lr in self.layer_rows.items():
            mode = lr.mode_var.get()
            if mode == "none":
                continue
            ratio = lr.ratio_var.get() if mode == "merge" else 0.0
            transfer_cfg[layer] = {"mode": mode, "ratio": ratio}

        merged_weights = merge_checkpoints(weights_a, weights_b, transfer_cfg)
        ckpt_b_copy = copy.deepcopy(ckpt_b)
        ckpt_b_copy['weight'] = merged_weights

        output_path = os.path.join(output_folder, output_filename)
        torch.save(ckpt_b_copy, output_path)

        self.status_label.configure(text=f"Saved merged checkpoint:\n{output_path}", foreground="green")

        show_similarity_popup(self, merged_weights, weights_b)


if __name__ == "__main__":
    app = ModelMergerApp()
    app.mainloop()
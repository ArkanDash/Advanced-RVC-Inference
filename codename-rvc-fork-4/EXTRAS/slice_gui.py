import os
import librosa
import soundfile as sf
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import scrolledtext
from tkinter import ttk
from tkinterdnd2 import TkinterDnD, DND_FILES
import threading
import tqdm


class AudioSlicerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Slicer GUI")
        self.root.geometry("640x520")
        self.root.configure(bg="#1b1721")

        self.input_path = None
        self.output_folder = None
        self.slice_length_ms = 3000

        self.setup_style()
        self.setup_gui()

    def setup_style(self):
        style = ttk.Style()
        style.theme_use('clam')

        style.configure("TFrame", background="#2c2634")
        style.configure("TLabel", background="#2c2634", foreground="white", font=("Segoe UI", 10))
        style.configure("TRadiobutton", background="#2c2634", foreground="white", font=("Segoe UI", 10))
        style.configure("TButton",
                        background="#423967", foreground="white",
                        font=("Segoe UI", 10), padding=6)
        style.map("TButton",
                  background=[("active", "#51457f")],
                  foreground=[("active", "white")])

        style.configure("TEntry", fieldbackground="#1b1721", foreground="white")
        style.configure("TText", background="#1b1721", foreground="white")

    def setup_gui(self):
        # Drop area
        self.drop_label = ttk.Label(self.root, text="Drop .wav file here or click to browse",
                                    relief="ridge", anchor="center", padding=10)
        self.drop_label.pack(pady=10, fill=tk.X, padx=10)
        self.drop_label.bind("<Button-1>", self.browse_file)

        # Enable drag-and-drop functionality
        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind('<<Drop>>', self.on_drop)

        # Sample rate, bit depth, and duration display
        self.info_label = ttk.Label(self.root, text="Waiting for file...", background="#1b1721", foreground="white")
        self.info_label.pack(pady=10, fill=tk.X, padx=10)

        # Slice selection
        self.slice_frame = ttk.Frame(self.root)
        self.slice_frame.pack(pady=5, padx=10, fill=tk.X)

        ttk.Label(self.slice_frame, text="Select slice length:").pack(side=tk.LEFT, padx=(0, 10))

        self.slice_var = tk.StringVar(value="3000")
        ttk.Radiobutton(self.slice_frame, text="3 sec", variable=self.slice_var, value="3000").pack(side=tk.LEFT)
        ttk.Radiobutton(self.slice_frame, text="3.7 sec", variable=self.slice_var, value="3700").pack(side=tk.LEFT)

        # Output folder
        self.out_frame = ttk.Frame(self.root)
        self.out_frame.pack(pady=10, padx=10, fill=tk.X)

        self.output_entry = ttk.Entry(self.out_frame, width=50)
        self.output_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        ttk.Button(self.out_frame, text="Browse Output Folder", command=self.browse_output).pack(side=tk.LEFT)

        # Slice button
        self.run_button = ttk.Button(self.root, text="Slice Audio", command=self.run)
        self.run_button.pack(pady=10)

        # Log display
        self.log_text = scrolledtext.ScrolledText(self.root, height=14, bg="#1b1721", fg="white", insertbackground="white")
        self.log_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def browse_file(self, event=None):
        path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if path:
            self.input_path = path
            self.drop_label.config(text=f"Loaded: {os.path.basename(path)}")
            self.update_info()

    def on_drop(self, event):
        file_path = event.data
        if file_path.endswith('.wav'):
            self.input_path = file_path
            self.drop_label.config(text=f"Loaded: {os.path.basename(file_path)}")
            self.update_info()

    def update_info(self):
        if not self.input_path:
            return

        y, sr = self.detect_sample_rate(self.input_path)
        bit_depth = self.probe_bit_depth(self.input_path)
        duration_ms = len(y) / sr * 1000  # Total duration in milliseconds

        # Convert duration to minutes and seconds
        duration_min_sec = self.format_duration(duration_ms / 1000)  # Convert ms to seconds for formatting
        self.info_label.config(text=f"Sample Rate: {sr} Hz | Bit Depth: {bit_depth} | Total Length: {duration_min_sec}")

    def format_duration(self, duration_sec):
        # Convert seconds to minutes and seconds (MM:SS)
        minutes = int(duration_sec // 60)
        seconds = int(duration_sec % 60)
        return f"{minutes:02}:{seconds:02}"

    def browse_output(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_folder = folder
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, folder)

    def run(self):
        if not self.input_path:
            messagebox.showerror("Error", "Please select a .wav file.")
            return

        self.output_folder = self.output_entry.get().strip()
        if not self.output_folder:
            messagebox.showerror("Error", "Please select an output folder.")
            return

        self.slice_length_ms = int(self.slice_var.get())
        self.log("Starting slicing process...")
        threading.Thread(target=self.slice_audio).start()

    def detect_sample_rate(self, input_path):
        y, sr = librosa.load(input_path, sr=None)
        return y, sr

    def probe_bit_depth(self, input_path):
        with sf.SoundFile(input_path) as file:
            return file.subtype

    def slice_audio_segment(self, y, sr, start_ms, end_ms, output_folder, index):
        start_sample = int(start_ms * sr / 1000)
        end_sample = int(end_ms * sr / 1000)
        slice_audio = y[start_sample:end_sample]
        slice_filename = os.path.join(output_folder, f"slice_{index}.wav")
        sf.write(slice_filename, slice_audio, sr)
        return slice_filename

    def slice_audio(self):
        y, sr = self.detect_sample_rate(self.input_path)
        bit_depth = self.probe_bit_depth(self.input_path)

        self.log(f"Sample rate: {sr} Hz")
        self.log(f"Bit depth: {bit_depth}")

        duration_ms = len(y) / sr * 1000
        num_slices = int(duration_ms // self.slice_length_ms)
        discarded_length = 0

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        tasks = [
            (y, sr, i * self.slice_length_ms, (i + 1) * self.slice_length_ms, self.output_folder, i)
            for i in range(num_slices)
        ]

        with ThreadPoolExecutor(max_workers=max(1, multiprocessing.cpu_count() - 2)) as executor:
            futures = [executor.submit(self.slice_audio_segment, *task) for task in tasks]

            for i, future in enumerate(tqdm.tqdm(futures, desc="Slicing", unit="slice")):
                future.result()
                self.log(f"Exported slice {i + 1}/{num_slices}")

        # Calculate the total length of all slices
        total_sliced_duration_ms = num_slices * self.slice_length_ms
        total_sliced_duration_sec = total_sliced_duration_ms / 1000  # In seconds

        # Convert the total sliced duration to min:sec format
        total_sliced_duration_min_sec = self.format_duration(total_sliced_duration_sec)

        remaining_ms = duration_ms - (num_slices * self.slice_length_ms)
        if remaining_ms < self.slice_length_ms:
            discarded_length += remaining_ms
            self.log(f"Discarded last segment of {remaining_ms / 1000:.2f} seconds (too short).")

        # Convert the original duration to min:sec format
        original_duration_min_sec = self.format_duration(duration_ms / 1000)

        self.log(f"Done. Total discarded: {discarded_length / 1000:.2f} seconds.")
        self.info_label.config(
            text=f"Total Duration of Slices: {total_sliced_duration_min_sec} | Original Duration: {original_duration_min_sec}"
        )


if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = AudioSlicerGUI(root)
    root.mainloop()

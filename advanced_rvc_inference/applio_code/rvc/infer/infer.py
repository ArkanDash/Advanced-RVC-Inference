import os
import sys
import time
import torch
import librosa
import logging
import traceback
import numpy as np
import soundfile as sf

from scipy.io import wavfile

now_dir = os.getcwd()
sys.path.append(now_dir)

from programs.applio_code.rvc.infer.pipeline import Pipeline as VC
from programs.applio_code.rvc.lib.utils import load_audio_infer, load_embedding
from programs.applio_code.rvc.lib.tools.split_audio import process_audio, merge_audio
from programs.applio_code.rvc.lib.algorithm.synthesizers import Synthesizer
from programs.applio_code.rvc.configs.config import Config

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("faiss.loader").setLevel(logging.WARNING)


class VoiceConverter:
    """
    A class for performing voice conversion using the Retrieval-Based Voice Conversion (RVC) method.
    """

    def __init__(self):
        """
        Initializes the VoiceConverter with default configuration, and sets up models and parameters.
        """
        self.config = Config()  # Load RVC configuration
        self.hubert_model = (
            None  # Initialize the Hubert model (for embedding extraction)
        )
        self.last_embedder_model = None  # Last used embedder model
        self.tgt_sr = None  # Target sampling rate for the output audio
        self.net_g = None  # Generator network for voice conversion
        self.vc = None  # Voice conversion pipeline instance
        self.cpt = None  # Checkpoint for loading model weights
        self.version = None  # Model version
        self.n_spk = None  # Number of speakers in the model
        self.use_f0 = None  # Whether the model uses F0

    def load_hubert(self, embedder_model: str, embedder_model_custom: str = None):
        """
        Loads the HuBERT model for speaker embedding extraction.
        """
        self.hubert_model = load_embedding(embedder_model, embedder_model_custom)
        self.hubert_model.to(self.config.device)
        self.hubert_model = (
            self.hubert_model.half()
            if self.config.is_half
            else self.hubert_model.float()
        )
        self.hubert_model.eval()

    @staticmethod
    def convert_audio_format(input_path, output_path, output_format):
        """
        Converts an audio file to a specified output format.
        """
        try:
            if output_format != "WAV":
                print(f"Converting audio to {output_format} format...")
                audio, sample_rate = librosa.load(input_path, sr=None)
                common_sample_rates = [
                    8000,
                    11025,
                    12000,
                    16000,
                    22050,
                    24000,
                    32000,
                    44100,
                    48000,
                ]
                target_sr = min(common_sample_rates, key=lambda x: abs(x - sample_rate))
                audio = librosa.resample(
                    audio, orig_sr=sample_rate, target_sr=target_sr
                )
                sf.write(output_path, audio, target_sr, format=output_format.lower())
            return output_path
        except Exception as error:
            print(f"An error occurred converting the audio format: {error}")

    def convert_audio(
        self,
        audio_input_path: str,
        audio_output_path: str,
        model_path: str,
        index_path: str,
        embedder_model: str,
        pitch: int,
        f0_file: str,
        f0_method: str,
        index_rate: float,
        volume_envelope: int,
        protect: float,
        hop_length: int,
        split_audio: bool,
        f0_autotune: bool,
        filter_radius: int,
        embedder_model_custom: str,
        export_format: str,
        resample_sr: int = 0,
        sid: int = 0,
    ):
        """
        Performs voice conversion on the input audio.
        """
        self.get_vc(model_path, sid)

        try:
            start_time = time.time()
            print(f"Converting audio '{audio_input_path}'...")
            audio = load_audio_infer(
                audio_input_path,
                16000,
            )
            audio_max = np.abs(audio).max() / 0.95

            if audio_max > 1:
                audio /= audio_max

            if not self.hubert_model or embedder_model != self.last_embedder_model:
                self.load_hubert(embedder_model, embedder_model_custom)
                self.last_embedder_model = embedder_model

            file_index = (
                index_path.strip()
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip()
                .replace("trained", "added")
            )

            if self.tgt_sr != resample_sr >= 16000:
                self.tgt_sr = resample_sr

            if split_audio:
                result, new_dir_path = process_audio(audio_input_path)
                if result == "Error":
                    return "Error with Split Audio", None

                dir_path = (
                    new_dir_path.strip().strip('"').strip("\n").strip('"').strip()
                )
                if dir_path:
                    paths = [
                        os.path.join(root, name)
                        for root, _, files in os.walk(dir_path, topdown=False)
                        for name in files
                        if name.endswith(".wav") and root == dir_path
                    ]
                try:
                    for path in paths:
                        self.convert_audio(
                            audio_input_path=path,
                            audio_output_path=path,
                            model_path=model_path,
                            index_path=index_path,
                            sid=sid,
                            pitch=pitch,
                            f0_file=None,
                            f0_method=f0_method,
                            index_rate=index_rate,
                            resample_sr=resample_sr,
                            volume_envelope=volume_envelope,
                            protect=protect,
                            hop_length=hop_length,
                            split_audio=False,
                            f0_autotune=f0_autotune,
                            filter_radius=filter_radius,
                            export_format=export_format,
                            embedder_model=embedder_model,
                            embedder_model_custom=embedder_model_custom,
                        )
                except Exception as error:
                    print(f"An error occurred processing the segmented audio: {error}")
                    print(traceback.format_exc())
                    return f"Error {error}"
                print("Finished processing segmented audio, now merging audio...")
                merge_timestamps_file = os.path.join(
                    os.path.dirname(new_dir_path),
                    f"{os.path.basename(audio_input_path).split('.')[0]}_timestamps.txt",
                )
                self.tgt_sr, audio_opt = merge_audio(merge_timestamps_file)
                os.remove(merge_timestamps_file)
                sf.write(audio_output_path, audio_opt, self.tgt_sr, format="WAV")
            else:
                audio_opt = self.vc.pipeline(
                    model=self.hubert_model,
                    net_g=self.net_g,
                    sid=sid,
                    audio=audio,
                    input_audio_path=audio_input_path,
                    pitch=pitch,
                    f0_method=f0_method,
                    file_index=file_index,
                    index_rate=index_rate,
                    pitch_guidance=self.use_f0,
                    filter_radius=filter_radius,
                    tgt_sr=self.tgt_sr,
                    resample_sr=resample_sr,
                    volume_envelope=volume_envelope,
                    version=self.version,
                    protect=protect,
                    hop_length=hop_length,
                    f0_autotune=f0_autotune,
                    f0_file=f0_file,
                )

            if audio_output_path:
                sf.write(audio_output_path, audio_opt, self.tgt_sr, format="WAV")
            output_path_format = audio_output_path.replace(
                ".wav", f".{export_format.lower()}"
            )
            audio_output_path = self.convert_audio_format(
                audio_output_path, output_path_format, export_format
            )

            elapsed_time = time.time() - start_time
            print(
                f"Conversion completed at '{audio_output_path}' in {elapsed_time:.2f} seconds."
            )

        except Exception as error:
            print(f"An error occurred during audio conversion: {error}")
            print(traceback.format_exc())

    def convert_audio_batch(
        self,
        audio_input_paths: str,
        audio_output_path: str,
        model_path: str,
        index_path: str,
        embedder_model: str,
        pitch: int,
        f0_file: str,
        f0_method: str,
        index_rate: float,
        volume_envelope: int,
        protect: float,
        hop_length: int,
        split_audio: bool,
        f0_autotune: bool,
        filter_radius: int,
        embedder_model_custom: str,
        export_format: str,
        resample_sr: int = 0,
        sid: int = 0,
        pid_file_path: str = None,
    ):
        """
        Performs voice conversion on a batch of input audio files.
        """
        pid = os.getpid()
        with open(pid_file_path, "w") as pid_file:
            pid_file.write(str(pid))
        try:
            if not self.hubert_model or embedder_model != self.last_embedder_model:
                self.load_hubert(embedder_model, embedder_model_custom)
                self.last_embedder_model = embedder_model
            self.get_vc(model_path, sid)
            file_index = (
                index_path.strip()
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip()
                .replace("trained", "added")
            )
            start_time = time.time()
            print(f"Converting audio batch '{audio_input_paths}'...")
            audio_files = [
                f
                for f in os.listdir(audio_input_paths)
                if f.endswith((".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus"))
            ]
            print(f"Detected {len(audio_files)} audio files for inference.")
            for i, audio_input_path in enumerate(audio_files):
                audio_output_paths = os.path.join(
                    audio_output_path,
                    f"{os.path.splitext(os.path.basename(audio_input_path))[0]}_output.{export_format.lower()}",
                )
                if os.path.exists(audio_output_paths):
                    continue
                print(f"Converting audio '{audio_input_path}'...")
                audio_input_path = os.path.join(audio_input_paths, audio_input_path)

                audio = load_audio_infer(
                    audio_input_path,
                    16000,
                )
                audio_max = np.abs(audio).max() / 0.95

                if audio_max > 1:
                    audio /= audio_max

                if self.tgt_sr != resample_sr >= 16000:
                    self.tgt_sr = resample_sr

                if split_audio:
                    result, new_dir_path = process_audio(audio_input_path)
                    if result == "Error":
                        return "Error with Split Audio", None

                    dir_path = (
                        new_dir_path.strip().strip('"').strip("\n").strip('"').strip()
                    )
                    if dir_path:
                        paths = [
                            os.path.join(root, name)
                            for root, _, files in os.walk(dir_path, topdown=False)
                            for name in files
                            if name.endswith(".wav") and root == dir_path
                        ]
                    try:
                        for path in paths:
                            self.convert_audio(
                                audio_input_path=path,
                                audio_output_path=path,
                                model_path=model_path,
                                index_path=index_path,
                                sid=sid,
                                pitch=pitch,
                                f0_file=None,
                                f0_method=f0_method,
                                index_rate=index_rate,
                                resample_sr=resample_sr,
                                volume_envelope=volume_envelope,
                                protect=protect,
                                hop_length=hop_length,
                                split_audio=False,
                                f0_autotune=f0_autotune,
                                filter_radius=filter_radius,
                                export_format=export_format,
                                embedder_model=embedder_model,
                                embedder_model_custom=embedder_model_custom,
                            )
                    except Exception as error:
                        print(
                            f"An error occurred processing the segmented audio: {error}"
                        )
                        print(traceback.format_exc())
                        return f"Error {error}"
                    print("Finished processing segmented audio, now merging audio...")
                    merge_timestamps_file = os.path.join(
                        os.path.dirname(new_dir_path),
                        f"{os.path.basename(audio_input_path).split('.')[0]}_timestamps.txt",
                    )
                    self.tgt_sr, audio_opt = merge_audio(merge_timestamps_file)
                    os.remove(merge_timestamps_file)
                else:
                    audio_opt = self.vc.pipeline(
                        model=self.hubert_model,
                        net_g=self.net_g,
                        sid=sid,
                        audio=audio,
                        input_audio_path=audio_input_path,
                        pitch=pitch,
                        f0_method=f0_method,
                        file_index=file_index,
                        index_rate=index_rate,
                        pitch_guidance=self.use_f0,
                        filter_radius=filter_radius,
                        tgt_sr=self.tgt_sr,
                        resample_sr=resample_sr,
                        volume_envelope=volume_envelope,
                        version=self.version,
                        protect=protect,
                        hop_length=hop_length,
                        f0_autotune=f0_autotune,
                        f0_file=f0_file,
                    )

                if audio_output_paths:
                    sf.write(audio_output_paths, audio_opt, self.tgt_sr, format="WAV")
                output_path_format = audio_output_paths.replace(
                    ".wav", f".{export_format.lower()}"
                )
                audio_output_paths = self.convert_audio_format(
                    audio_output_paths, output_path_format, export_format
                )
                print(f"Conversion completed at '{audio_output_paths}'.")
            elapsed_time = time.time() - start_time
            print(f"Batch conversion completed in {elapsed_time:.2f} seconds.")
            os.remove(pid_file_path)
        except Exception as error:
            print(f"An error occurred during audio conversion: {error}")
            print(traceback.format_exc())

    def get_vc(self, weight_root, sid):
        """
        Loads the voice conversion model and sets up the pipeline.
        """
        if sid == "" or sid == []:
            self.cleanup_model()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.load_model(weight_root)

        if self.cpt is not None:
            self.setup_network()
            self.setup_vc_instance()

    def cleanup_model(self):
        """
        Cleans up the model and releases resources.
        """
        if self.hubert_model is not None:
            del self.net_g, self.n_spk, self.vc, self.hubert_model, self.tgt_sr
            self.hubert_model = self.net_g = self.n_spk = self.vc = self.tgt_sr = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del self.net_g, self.cpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.cpt = None

    def load_model(self, weight_root):
        """
        Loads the model weights from the specified path.
        """
        self.cpt = (
            torch.load(weight_root, map_location="cpu")
            if os.path.isfile(weight_root)
            else None
        )

    def setup_network(self):
        """
        Sets up the network configuration based on the loaded checkpoint.
        """
        if self.cpt is not None:
            self.tgt_sr = self.cpt["config"][-1]
            self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]
            self.use_f0 = self.cpt.get("f0", 1)

            self.version = self.cpt.get("version", "v1")
            self.text_enc_hidden_dim = 768 if self.version == "v2" else 256
            self.net_g = Synthesizer(
                *self.cpt["config"],
                use_f0=self.use_f0,
                text_enc_hidden_dim=self.text_enc_hidden_dim,
                is_half=self.config.is_half,
            )
            del self.net_g.enc_q
            self.net_g.load_state_dict(self.cpt["weight"], strict=False)
            self.net_g.eval().to(self.config.device)
            self.net_g = (
                self.net_g.half() if self.config.is_half else self.net_g.float()
            )

    def setup_vc_instance(self):
        """
        Sets up the voice conversion pipeline instance based on the target sampling rate and configuration.
        """
        if self.cpt is not None:
            self.vc = VC(self.tgt_sr, self.config)
            self.n_spk = self.cpt["config"][-3]

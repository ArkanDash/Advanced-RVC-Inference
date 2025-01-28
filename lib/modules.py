import os, sys
import traceback
import logging
now_dir = os.getcwd()
sys.path.append(now_dir)
logger = logging.getLogger(__name__)
import numpy as np
import soundfile as sf
import torch
from io import BytesIO
from lib.infer_libs.audio import load_audio
from lib.infer_libs.audio import wav2
from lib.infer_libs.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from lib.pipeline import Pipeline
import time
import glob
from shutil import move
from fairseq import checkpoint_utils

sup_audioext = {
    "wav",
    "mp3",
    "flac",
    "ogg",
    "opus",
    "m4a",
    "mp4",
    "aac",
    "alac",
    "wma",
    "aiff",
    "webm",
    "ac3",
}

def note_to_hz(note_name):
    try:
        SEMITONES = {'C': -9, 'C#': -8, 'D': -7, 'D#': -6, 'E': -5, 'F': -4, 'F#': -3, 'G': -2, 'G#': -1, 'A': 0, 'A#': 1, 'B': 2}
        pitch_class, octave = note_name[:-1], int(note_name[-1])
        semitone = SEMITONES[pitch_class]
        note_number = 12 * (octave - 4) + semitone
        frequency = 440.0 * (2.0 ** (1.0/12)) ** note_number
        return frequency
    except:
        return None

def load_hubert(hubert_model_path, config):
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [hubert_model_path],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()

class VC:
    def __init__(self, config):
        self.n_spk = None
        self.tgt_sr = None
        self.net_g = None
        self.pipeline = None
        self.cpt = None
        self.version = None
        self.if_f0 = None
        self.version = None
        self.hubert_model = None

        self.config = config

    def get_vc(self, sid, *to_return_protect):
        logger.info("Get sid: " + sid)

        to_return_protect0 = {
            "visible": self.if_f0 != 0,
            "value": to_return_protect[0]
            if self.if_f0 != 0 and to_return_protect
            else 0.5,
            "__type__": "update",
        }
        to_return_protect1 = {
            "visible": self.if_f0 != 0,
            "value": to_return_protect[1]
            if self.if_f0 != 0 and to_return_protect
            else 0.33,
            "__type__": "update",
        }

        if sid == "" or sid == []:
            if self.hubert_model is not None:  # 考虑到轮询, 需要加个判断看是否 sid 是由有模型切换到无模型的
                logger.info("Clean model cache")
                del (
                    self.net_g,
                    self.n_spk,
                    self.vc,
                    self.hubert_model,
                    self.tgt_sr,
                )  # ,cpt
                self.hubert_model = (
                    self.net_g
                ) = self.n_spk = self.vc = self.hubert_model = self.tgt_sr = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                ###楼下不这么折腾清理不干净
                self.if_f0 = self.cpt.get("f0", 1)
                self.version = self.cpt.get("version", "v1")
                if self.version == "v1":
                    if self.if_f0 == 1:
                        self.net_g = SynthesizerTrnMs256NSFsid(
                            *self.cpt["config"], is_half=self.config.is_half
                        )
                    else:
                        self.net_g = SynthesizerTrnMs256NSFsid_nono(*self.cpt["config"])
                elif self.version == "v2":
                    if self.if_f0 == 1:
                        self.net_g = SynthesizerTrnMs768NSFsid(
                            *self.cpt["config"], is_half=self.config.is_half
                        )
                    else:
                        self.net_g = SynthesizerTrnMs768NSFsid_nono(*self.cpt["config"])
                del self.net_g, self.cpt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return (
                {"visible": False, "__type__": "update"},
                {
                    "visible": True,
                    "value": to_return_protect0,
                    "__type__": "update",
                },
                {
                    "visible": True,
                    "value": to_return_protect1,
                    "__type__": "update",
                },
                "",
                "",
            )
        #person = f'{os.getenv("weight_root")}/{sid}'
        person = f'{sid}'
        #logger.info(f"Loading: {person}")
        logger.info(f"Loading...")
        self.cpt = torch.load(person, map_location="cpu")
        self.tgt_sr = self.cpt["config"][-1]
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        self.if_f0 = self.cpt.get("f0", 1)
        self.version = self.cpt.get("version", "v1")

        synthesizer_class = {
            ("v1", 1): SynthesizerTrnMs256NSFsid,
            ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
            ("v2", 1): SynthesizerTrnMs768NSFsid,
            ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
        }

        self.net_g = synthesizer_class.get(
            (self.version, self.if_f0), SynthesizerTrnMs256NSFsid
        )(*self.cpt["config"], is_half=self.config.is_half)

        del self.net_g.enc_q

        self.net_g.load_state_dict(self.cpt["weight"], strict=False)
        self.net_g.eval().to(self.config.device)
        if self.config.is_half:
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()

        self.pipeline = Pipeline(self.tgt_sr, self.config)
        n_spk = self.cpt["config"][-3]
        #index = {"value": get_index_path_from_model(sid), "__type__": "update"}
        #logger.info("Select index: " + index["value"])

        return (
            (
                {"visible": False, "maximum": n_spk, "__type__": "update"},
                to_return_protect0,
                to_return_protect1
            )
            if to_return_protect
            else {"visible": False, "maximum": n_spk, "__type__": "update"}
        )

    def vc_single_dont_save(
        self,
        sid,
        input_audio_path1,
        f0_up_key,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
        crepe_hop_length,
        do_formant,
        quefrency,
        timbre,
        f0_min,
        f0_max,
        f0_autotune,
        hubert_model_path = "assets/hubert/hubert_base.pt"
    ):
        """
        Performs inference without saving
    
        Parameters:
        - sid (int)
        - input_audio_path1 (str)
        - f0_up_key (int)
        - f0_method (str)
        - file_index (str)
        - file_index2 (str)
        - index_rate (float)
        - filter_radius (int)
        - resample_sr (int)
        - rms_mix_rate (float)
        - protect (float)
        - crepe_hop_length (int)
        - do_formant (bool)
        - quefrency (float)
        - timbre (float)
        - f0_min (str)
        - f0_max (str)
        - f0_autotune (bool)
        - hubert_model_path (str)

        Returns:
        Tuple(Tuple(status, index_info, times), Tuple(sr, data)):
            - Tuple(status, index_info, times):
                - status (str): either "Success." or an error
                - index_info (str): index path if used
                - times (list): [npy_time, f0_time, infer_time, total_time]
            - Tuple(sr, data): Audio data results.
        """
        global total_time
        total_time = 0
        start_time = time.time()
        
        if not input_audio_path1:
            return "You need to upload an audio", None
        
        if not os.path.exists(input_audio_path1):
            return "Audio was not properly selected or doesn't exist", None
        
        f0_up_key = int(f0_up_key)
        if not f0_min.isdigit():
            f0_min = note_to_hz(f0_min)
            if f0_min:
                print(f"Converted Min pitch: freq - {f0_min}")
            else:
                f0_min = 50
                print("Invalid minimum pitch note. Defaulting to 50hz.")
        else:
            f0_min = float(f0_min)
        if not f0_max.isdigit():
            f0_max = note_to_hz(f0_max)
            if f0_max:
                print(f"Converted Max pitch: freq - {f0_max}")
            else:
                f0_max = 1100
                print("Invalid maximum pitch note. Defaulting to 1100hz.")
        else:
            f0_max = float(f0_max)
        
        try:
            print(f"Attempting to load {input_audio_path1}....")
            audio = load_audio(file=input_audio_path1,
                               sr=16000,
                               DoFormant=do_formant,
                               Quefrency=quefrency,
                               Timbre=timbre)
            
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max
            times = [0, 0, 0]

            if self.hubert_model is None:
                self.hubert_model = load_hubert(hubert_model_path, self.config)

            try:
                self.if_f0 = self.cpt.get("f0", 1)
            except NameError:
                message = "Model was not properly selected"
                print(message)
                return message, None
            
            if file_index and not file_index == "" and isinstance(file_index, str):
                file_index = file_index.strip(" ") \
                .strip('"') \
                .strip("\n") \
                .strip('"') \
                .strip(" ") \
                .replace("trained", "added")
            elif file_index2:
                file_index = file_index2
            else:
                file_index = ""  

            audio_opt = self.pipeline.pipeline(
                self.hubert_model,
                self.net_g,
                sid,
                audio,
                input_audio_path1,
                times,
                f0_up_key,
                f0_method,
                file_index,
                index_rate,
                self.if_f0,
                filter_radius,
                self.tgt_sr,
                resample_sr,
                rms_mix_rate,
                self.version,
                protect,
                crepe_hop_length,
                f0_autotune,
                f0_min=f0_min,
                f0_max=f0_max                 
            )

            if self.tgt_sr != resample_sr >= 16000:
                tgt_sr = resample_sr
            else:
                tgt_sr = self.tgt_sr
            index_info = (
                "Index: %s." % file_index
                if isinstance(file_index, str) and os.path.exists(file_index)
                else "Index not used."
            )
            end_time = time.time()
            total_time = end_time - start_time
            times.append(total_time)
            return (
                ("Success.", index_info, times),
                (tgt_sr, audio_opt),
            )
        except:
            info = traceback.format_exc()
            logger.warn(info)
            return (
                (info, None, [None, None, None, None]),
                (None, None)
            )

    def vc_single(
        self,
        sid,
        input_audio_path1,
        f0_up_key,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
        format1,
        crepe_hop_length,
        do_formant,
        quefrency,
        timbre,
        f0_min,
        f0_max,
        f0_autotune,
        hubert_model_path = "assets/hubert/hubert_base.pt"
    ):
        """
        Performs inference with saving
    
        Parameters:
        - sid (int)
        - input_audio_path1 (str)
        - f0_up_key (int)
        - f0_method (str)
        - file_index (str)
        - file_index2 (str)
        - index_rate (float)
        - filter_radius (int)
        - resample_sr (int)
        - rms_mix_rate (float)
        - protect (float)
        - format1 (str)
        - crepe_hop_length (int)
        - do_formant (bool)
        - quefrency (float)
        - timbre (float)
        - f0_min (str)
        - f0_max (str)
        - f0_autotune (bool)
        - hubert_model_path (str)

        Returns:
        Tuple(Tuple(status, index_info, times), Tuple(sr, data), output_path):
            - Tuple(status, index_info, times):
                - status (str): either "Success." or an error
                - index_info (str): index path if used
                - times (list): [npy_time, f0_time, infer_time, total_time]
            - Tuple(sr, data): Audio data results.
            - output_path (str): Audio results path
        """
        global total_time
        total_time = 0
        start_time = time.time()
        
        if not input_audio_path1:
            return "You need to upload an audio", None, None
        
        if not os.path.exists(input_audio_path1):
            return "Audio was not properly selected or doesn't exist", None, None

        f0_up_key = int(f0_up_key)
        if not f0_min.isdigit():
            f0_min = note_to_hz(f0_min)
            if f0_min:
                print(f"Converted Min pitch: freq - {f0_min}")
            else:
                f0_min = 50
                print("Invalid minimum pitch note. Defaulting to 50hz.")
        else:
            f0_min = float(f0_min)
        if not f0_max.isdigit():
            f0_max = note_to_hz(f0_max)
            if f0_max:
                print(f"Converted Max pitch: freq - {f0_max}")
            else:
                f0_max = 1100
                print("Invalid maximum pitch note. Defaulting to 1100hz.")
        else:
            f0_max = float(f0_max)

        try:
            print(f"Attempting to load {input_audio_path1}...")
            audio = load_audio(file=input_audio_path1,
                               sr=16000,
                               DoFormant=do_formant,
                               Quefrency=quefrency,
                               Timbre=timbre)
            
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max
            times = [0, 0, 0]

            if self.hubert_model is None:
                self.hubert_model = load_hubert(hubert_model_path, self.config)

            try:
                self.if_f0 = self.cpt.get("f0", 1)
            except NameError:
                message = "Model was not properly selected"
                print(message)
                return message, None
            if file_index and not file_index == "" and isinstance(file_index, str):
                file_index = file_index.strip(" ") \
                .strip('"') \
                .strip("\n") \
                .strip('"') \
                .strip(" ") \
                .replace("trained", "added")
            elif file_index2:
                file_index = file_index2
            else:
                file_index = ""

            audio_opt = self.pipeline.pipeline(
                self.hubert_model,
                self.net_g,
                sid,
                audio,
                input_audio_path1,
                times,
                f0_up_key,
                f0_method,
                file_index,
                index_rate,
                self.if_f0,
                filter_radius,
                self.tgt_sr,
                resample_sr,
                rms_mix_rate,
                self.version,
                protect,
                crepe_hop_length,
                f0_autotune,
                f0_min=f0_min,
                f0_max=f0_max                 
            )

            if self.tgt_sr != resample_sr >= 16000:
                tgt_sr = resample_sr
            else:
                tgt_sr = self.tgt_sr
            index_info = (
                "Index: %s." % file_index
                if isinstance(file_index, str) and os.path.exists(file_index)
                else "Index not used."
            )
            
            opt_root = os.path.join(os.getcwd(), "output")
            os.makedirs(opt_root, exist_ok=True)
            output_count = 1
            
            while True:
                opt_filename = f"{os.path.splitext(os.path.basename(input_audio_path1))[0]}{os.path.basename(os.path.dirname(file_index))}{f0_method.capitalize()}_{output_count}.{format1}"
                current_output_path = os.path.join(opt_root, opt_filename)
                if not os.path.exists(current_output_path):
                    break
                output_count += 1
            try:
                if format1 in ["wav", "flac"]:
                    sf.write(
                        current_output_path,
                        audio_opt,
                        self.tgt_sr,
                    )
                else:
                    with BytesIO() as wavf:
                        sf.write(
                            wavf,
                            audio_opt,
                            self.tgt_sr,
                            format="wav"
                        )
                        wavf.seek(0, 0)
                        with open(current_output_path, "wb") as outf:
                                wav2(wavf, outf, format1)
            except:
                info = traceback.format_exc()
            end_time = time.time()
            total_time = end_time - start_time
            times.append(total_time)
            return (
                ("Success.", index_info, times),
                (tgt_sr, audio_opt),
                current_output_path
            )
        except:
            info = traceback.format_exc()
            logger.warn(info)
            return (
                (info, None, [None, None, None, None]),
                (None, None),
                None
            )
import os
import sys
import json

import numpy as np

from fastapi import FastAPI, WebSocketDisconnect, WebSocket

sys.path.append(os.getcwd())

from advanced_rvc_inference.library.utils import clear_gpu_cache
from advanced_rvc_inference.utils.variables import configs, translations, logger
from advanced_rvc_inference.rvc.realtime.realtime import VoiceChanger, RVC_Realtime

app = FastAPI()
vc_instance = None

PIPELINE_SAMPLE_RATE = 16000
DEVICE_SAMPLE_RATE = 48000

@app.websocket("/ws-audio")
async def websocket_audio(ws: WebSocket):
    global vc_instance
    await ws.accept()

    logger.info(translations["ws_connected"])

    try:
        text = await ws.receive_text()
        params = json.loads(text)

        read_chunk_size = int(params["chunk_size"])
        block_frame = read_chunk_size * 128
        embedders = params["embedders"]

        model_pth = params["model_pth"]
        model_pth = os.path.join(configs["weights_path"], model_pth) if not os.path.exists(model_pth) else model_pth

        if not model_pth or not os.path.exists(model_pth) or os.path.isdir(model_pth) or not model_pth.endswith((".pth", ".onnx")):
            logger.warning(translations["provide_file"].format(filename=translations["model"]))
            await ws.send_text(json.dumps({"type": "warnings", "value": translations["provide_file"].format(filename=translations["model"])}))
            return
        
        logger.info(translations["start_realtime"])

        if vc_instance is None:
            vc_instance = VoiceChanger(
                read_chunk_size=read_chunk_size, 
                cross_fade_overlap_size=params["cross_fade_overlap_size"], 
                input_sample_rate=DEVICE_SAMPLE_RATE, 
                extra_convert_size=params["extra_convert_size"]
            )
            vc_instance.initialize(vc_model=RVC_Realtime(
                model_path=model_pth, 
                index_path=params["model_index"], 
                f0_method=params["f0_method"], 
                f0_onnx=params["f0_onnx"], 
                embedder_model=(embedders if embedders != "custom" else params["custom_embedders"]), 
                embedders_mode=params["embedders_mode"], 
                sample_rate=PIPELINE_SAMPLE_RATE, 
                hop_length=params["hop_length"], 
                silent_threshold=params["silent_threshold"], 
                input_sample_rate=DEVICE_SAMPLE_RATE, 
                output_sample_rate=DEVICE_SAMPLE_RATE, 
                vad_enabled=params["vad_enabled"], 
                vad_sensitivity=params["vad_sensitivity"], 
                vad_frame_ms=params["vad_frame_ms"], 
                clean_audio=params["clean_audio"], 
                clean_strength=params["clean_strength"]
            ))
        
        logger.info(translations["realtime_is_ready"])

        while 1:
            audio = await ws.receive_bytes()
            arr = np.frombuffer(audio, dtype=np.float32)

            if arr.size != block_frame:
                arr = np.pad(arr, (0, block_frame - arr.size)).astype(np.float32) if arr.size < block_frame else arr[:block_frame].astype(np.float32)

            audio_output, _, perf = vc_instance.on_request(
                arr * (params["input_audio_gain"] / 100.0), 
                f0_up_key=params["f0_up_key"], 
                index_rate=params["index_rate"], 
                protect=params["protect"], 
                filter_radius=params["filter_radius"], 
                rms_mix_rate=params["rms_mix_rate"], 
                f0_autotune=params["f0_autotune"], 
                f0_autotune_strength=params["f0_autotune_strength"], 
                proposal_pitch=params["proposal_pitch"], 
                proposal_pitch_threshold=params["proposal_pitch_threshold"]
            )

            await ws.send_text(json.dumps({"type": "latency", "value": perf[1]}))
            await ws.send_bytes(audio_output.tobytes())
    except WebSocketDisconnect:
        logger.info(translations["ws_disconnected"])
    except Exception as e:
        import traceback
        logger.debug(traceback.format_exc())
        logger.info(translations["error_occurred"].format(e=e))
    finally:
        if vc_instance is not None:
            del vc_instance
            vc_instance = None

        clear_gpu_cache()

        try:
            await ws.close()
        except:
            pass

        logger.info(translations["ws_closed"])
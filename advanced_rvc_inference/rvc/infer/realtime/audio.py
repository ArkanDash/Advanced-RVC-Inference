import os
import sys
import librosa
import traceback

import numpy as np
import sounddevice as sd

from queue import Queue

sys.path.append(os.getcwd())

from main.app.variables import logger, translations

class ServerAudioDevice:
    def __init__(self, index, name, host_api, max_input_channels, max_output_channels, default_samplerate):
        self.index = index 
        self.name = name
        self.host_api = host_api
        self.max_input_channels = max_input_channels
        self.max_output_channels = max_output_channels
        self.default_samplerate = default_samplerate

def check_the_device(device, type = "input"):
    stream_device = sd.InputStream if type == "input" else sd.OutputStream
    try:
        with stream_device(device=device["index"], dtype=np.float32, samplerate=device["default_samplerate"]):
            return True
    except Exception:
        return False

def list_audio_device():
    try:
        audio_device_list = sd.query_devices()
    except Exception as e:
        logger.error(translations["error_occurred"].format(e=e))
        audio_device_list = []
    except OSError as e:
        logger.debug(translations["error_occurred"].format(e=e))
        audio_device_list = []

    input_audio_device_list = [
        d for d in audio_device_list if d["max_input_channels"] > 0 and check_the_device(d, "input")
    ]
    output_audio_device_list = [
        d for d in audio_device_list if d["max_output_channels"] > 0 and check_the_device(d, "output")
    ]
    
    try:
        hostapis = sd.query_hostapis()
    except Exception as e:
        logger.error(translations["error_occurred"].format(e=e))
        hostapis = []
    except OSError as e:
        logger.debug(translations["error_occurred"].format(e=e))
        hostapis = []

    audio_input_device, audio_output_device = [], []

    for d in input_audio_device_list:
        input_audio_device = ServerAudioDevice(
            index=d["index"],
            name=d["name"],
            host_api=hostapis[d["hostapi"]]["name"],
            max_input_channels=d["max_input_channels"],
            max_output_channels=d["max_output_channels"],
            default_samplerate=d["default_samplerate"],
        )
        audio_input_device.append(input_audio_device)

    for d in output_audio_device_list:
        output_audio_device = ServerAudioDevice(
            index=d["index"],
            name=d["name"],
            host_api=hostapis[d["hostapi"]]["name"],
            max_input_channels=d["max_input_channels"],
            max_output_channels=d["max_output_channels"],
            default_samplerate=d["default_samplerate"],
        )
        audio_output_device.append(output_audio_device)

    return audio_input_device, audio_output_device

class Audio:
    def __init__(self, callbacks, f0_up_key = 0, index_rate = 0.5, protect = 0.5, filter_radius = 3, rms_mix_rate = 1, f0_autotune = False, f0_autotune_strength = 1, proposal_pitch = False, proposal_pitch_threshold = 255.0, input_audio_gain = 1.0, output_audio_gain = 1.0, monitor_audio_gain = 1.0, monitor = False):
        self.callbacks = callbacks
        self.mon_queue = Queue()
        self.performance = [0, 0, 0]
        self.input_stream = None
        self.output_stream = None
        self.monitor = None
        self.running = False
        self.input_audio_gain = input_audio_gain
        self.output_audio_gain = output_audio_gain
        self.monitor_audio_gain = monitor_audio_gain
        self.use_monitor = monitor
        self.f0_up_key = f0_up_key
        self.index_rate = index_rate
        self.protect = protect
        self.filter_radius = filter_radius
        self.rms_mix_rate = rms_mix_rate
        self.f0_autotune = f0_autotune
        self.f0_autotune_strength = f0_autotune_strength
        self.proposal_pitch = proposal_pitch
        self.proposal_pitch_threshold = proposal_pitch_threshold

    def get_input_audio_device(self, index):
        audioinput, _ = list_audio_device()
        serverAudioDevice = [x for x in audioinput if x.index == index]

        return serverAudioDevice[0] if len(serverAudioDevice) > 0 else None

    def get_output_audio_device(self, index):
        _, audiooutput = list_audio_device()
        serverAudioDevice = [x for x in audiooutput if x.index == index]

        return serverAudioDevice[0] if len(serverAudioDevice) > 0 else None
    
    def process_data(self, indata):
        indata = indata * self.input_audio_gain
        unpacked_data = librosa.to_mono(indata.T)

        return self.callbacks.change_voice(unpacked_data, self.f0_up_key, self.index_rate, self.protect, self.filter_radius, self.rms_mix_rate, self.f0_autotune, self.f0_autotune_strength, self.proposal_pitch, self.proposal_pitch_threshold)
    
    def process_data_with_time(self, indata):
        out_wav, _, perf, _ = self.process_data(indata)
        self.performance = perf

        self.callbacks.emit_to(self.performance)
        return out_wav
    
    def audio_stream_callback(self, indata, frames, times, status):
        try:
            out_wav = self.process_data_with_time(indata)
            self.mon_queue.put(out_wav)
        except Exception as e:
            logger.error(translations["error_occurred"].format(e=e))
            logger.debug(traceback.format_exc())

    def audio_queue(self, outdata, gain):
        try:
            mon_wav = self.mon_queue.get()

            while self.mon_queue.qsize() > 0:
                self.mon_queue.get()

            output_channels = outdata.shape[1]
            outdata[:] = (np.repeat(mon_wav, output_channels).reshape(-1, output_channels) * gain)
        except Exception as e:
            logger.error(translations["error_occurred"].format(e=e))
            logger.debug(traceback.format_exc())

    def run_audio_stream(self, block_frame, input_device_id, output_device_id, output_monitor_id, input_audio_sample_rate, output_monitor_sample_rate, input_max_channel, output_max_channel, output_monitor_max_channel, input_extra_setting, output_extra_setting, output_monitor_extra_setting):
        self.input_stream = sd.InputStream(
            callback=self.audio_stream_callback,
            latency="low",
            dtype=np.float32,
            device=input_device_id,
            blocksize=block_frame,
            samplerate=input_audio_sample_rate,
            channels=input_max_channel,
            extra_settings=input_extra_setting
        )
        self.output_stream = sd.OutputStream(
            callback=lambda outdata, frames, times, status: self.audio_queue(outdata, self.output_audio_gain),
            latency="low",
            dtype=np.float32,
            device=output_device_id,
            blocksize=block_frame,
            samplerate=input_audio_sample_rate,
            channels=output_max_channel,
            extra_settings=output_extra_setting
        )
        self.input_stream.start()
        self.output_stream.start()

        if self.use_monitor:
            self.monitor = sd.OutputStream(
                callback=lambda outdata, frames, times, status: self.audio_queue(outdata, self.monitor_audio_gain),
                latency="low",
                dtype=np.float32,
                device=output_monitor_id,
                blocksize=block_frame,
                samplerate=output_monitor_sample_rate,
                channels=output_monitor_max_channel,
                extra_settings=output_monitor_extra_setting
            )
            self.monitor.start()

    def stop(self):
        self.running = False

        if self.input_stream is not None:
            self.input_stream.close()
            self.input_stream = None

        if self.output_stream is not None:
            self.output_stream.close()
            self.output_stream = None

        if self.monitor is not None:
            self.monitor.close()
            self.monitor = None

    def start(self, input_device_id, output_device_id, output_monitor_id, exclusive_mode, asio_input_channel, asio_output_channel, asio_output_monitor_channel, read_chunk_size, input_audio_sample_rate, output_monitor_sample_rate):
        self.stop()

        input_audio_device, output_audio_device = self.get_input_audio_device(input_device_id), self.get_output_audio_device(output_device_id)
        input_channels, output_channels = input_audio_device.max_input_channels, output_audio_device.max_output_channels
    
        input_extra_setting, output_extra_setting, output_monitor_extra_setting, monitor_channels = None, None, None, None
        wasapi_exclusive_mode = bool(exclusive_mode)

        if input_audio_device and "WASAPI" in input_audio_device.host_api:
            input_extra_setting = sd.WasapiSettings(exclusive=wasapi_exclusive_mode, auto_convert=not wasapi_exclusive_mode)
        elif input_audio_device and "ASIO" in input_audio_device.host_api and asio_input_channel != -1:
            input_extra_setting = sd.AsioSettings(channel_selectors=[asio_input_channel])
            input_channels = 1

        if output_audio_device and "WASAPI" in output_audio_device.host_api:
            output_extra_setting = sd.WasapiSettings(exclusive=wasapi_exclusive_mode, auto_convert=not wasapi_exclusive_mode)
        elif input_audio_device and "ASIO" in input_audio_device.host_api and asio_output_channel != -1:
            output_extra_setting = sd.AsioSettings(channel_selectors=[asio_output_channel])
            output_channels = 1

        if self.use_monitor:
            output_monitor_device = self.get_output_audio_device(output_monitor_id)
            monitor_channels = output_monitor_device.max_output_channels

            if output_monitor_device and "WASAPI" in output_monitor_device.host_api:
                output_monitor_extra_setting = sd.WasapiSettings(exclusive=wasapi_exclusive_mode, auto_convert=not wasapi_exclusive_mode)
            elif output_monitor_device and "ASIO" in output_monitor_device.host_api and asio_output_monitor_channel != -1:
                output_monitor_extra_setting = sd.AsioSettings(channel_selectors=[asio_output_monitor_channel])
                monitor_channels = 1

        block_frame = int((read_chunk_size * 128 / 48000) * input_audio_sample_rate)

        try:
            self.run_audio_stream(block_frame, input_device_id, output_device_id, output_monitor_id, input_audio_sample_rate, output_monitor_sample_rate, input_channels, output_channels, monitor_channels, input_extra_setting, output_extra_setting, output_monitor_extra_setting)
            self.running = True
        except Exception as e:
            logger.error(translations["error_occurred"].format(e=e))
            logger.debug(traceback.format_exc())
from advanced_rvc_inference.utils.variables import logger, config, translations, theme, configs, language, allow_disk


js_code = """
() => {
    window._activeStream = null;
    window._audioCtx = null;
    window._workletNode = null;
    window._playbackNode = null;
    window._ws = null;

    function setStatus(msg, use_alert = true) {
        const realtimeStatus = document.querySelector("#realtime-status-info h2.output-class");
        if (use_alert) alert(msg);

        if (realtimeStatus) {
            realtimeStatus.innerText = msg;
            realtimeStatus.style.whiteSpace = "nowrap";
            realtimeStatus.style.textAlign = "center";
        }
    }

    async function addModuleFromString(ctx, codeStr) {
        const blob = new Blob([codeStr], {type: 'application/javascript'});
        const url = URL.createObjectURL(blob);

        await ctx.audioWorklet.addModule(url);
        URL.revokeObjectURL(url);
    };

    function createOutputRoute(audioCtx, playbackNode, sinkId, gainValue = 1.0) {
        const dest = audioCtx.createMediaStreamDestination();
        const gainNode = audioCtx.createGain();
        gainNode.gain.value = gainValue;

        playbackNode.connect(gainNode);
        gainNode.connect(dest);

        const el = document.createElement('audio');
        el.autoplay = true;
        el.srcObject = dest.stream;
        el.style.display = 'none';
        document.body.appendChild(el);

        if (el.setSinkId) el.setSinkId(sinkId).catch(err => console.error(err));
        return { dest, gainNode, el };
    }

    const inputWorkletSource = `
        class InputProcessor extends AudioWorkletProcessor {
            constructor() {
                super();
                this.buffer = new Float32Array(0);
                this.block_frame = 128;
                this.port.onmessage = (e) => {
                    if (e.data && e.data.block_frame) this.block_frame = e.data.block_frame;
                };
            }

            process(inputs) {
                const input = inputs[0];
                if (!input || !input[0]) return true;
                const frame = input[0];

                const newBuf = new Float32Array(this.buffer.length + frame.length);
                newBuf.set(this.buffer, 0);
                newBuf.set(frame, this.buffer.length);
                this.buffer = newBuf;

                while (this.buffer.length >= this.block_frame) {
                    const chunk = this.buffer.slice(0, this.block_frame);

                    this.port.postMessage({chunk}, [chunk.buffer]);
                    this.buffer = this.buffer.slice(this.block_frame);
                }

                return true;
            }
        }
        registerProcessor('input-processor', InputProcessor);
        `;

        const playbackWorkletSource = `
            class PlaybackProcessor extends AudioWorkletProcessor {
                constructor(options) {
                    super(options);
                    const bufferSize = options.processorOptions && options.processorOptions.bufferSize ? options.processorOptions.bufferSize: 98304;
                    this.buffer = new Float32Array(bufferSize); 
                    this.bufferCapacity = bufferSize; 
                    this.writePointer = 0;
                    this.readPointer = 0;
                    this.availableSamples = 0;
                    this.port.onmessage = (e) => {
                        if (e.data && e.data.chunk) {
                            const chunk = new Float32Array(e.data.chunk);
                            const chunkSize = chunk.length;

                            if (this.availableSamples + chunkSize > this.bufferCapacity) return;

                            for (let i = 0; i < chunkSize; i++) {
                                this.buffer[this.writePointer] = chunk[i];
                                this.writePointer = (this.writePointer + 1) % this.bufferCapacity;
                            }

                            this.availableSamples += chunkSize;
                        }
                    };
                }

                process(inputs, outputs) {
                    const output = outputs[0];
                    if (!output || !output[0]) return true;

                    const frame = output[0];
                    const frameSize = frame.length;

                    if (this.availableSamples >= frameSize) {
                        for (let i = 0; i < frameSize; i++) {
                            frame[i] = this.buffer[this.readPointer];
                            this.readPointer = (this.readPointer + 1) % this.bufferCapacity;
                        }
                        this.availableSamples -= frameSize;
                    } else {
                        frame.fill(0);
                    }

                    if (output.length > 1) output[1].set(output[0]);
                    return true;
                }
            }
            registerProcessor('playback-processor', PlaybackProcessor);
            `;

    window.getAudioDevices = async function() {
        if (!navigator.mediaDevices) {
            setStatus("__MEDIA_DEVICES__");
            return {"inputs": {}, "outputs": {}};
        }

        try {
            await navigator.mediaDevices.getUserMedia({ audio: true });
        } catch (err) {
            console.error(err);
            setStatus("__MIC_INACCESSIBLE__")

            return {"inputs": {}, "outputs": {}};
        }

        const devices = await navigator.mediaDevices.enumerateDevices();
        const inputs = {};
        const outputs = {};
        
        for (const device of devices) {
            if (device.kind === "audioinput") {
                inputs[device.label] = device.deviceId
            } else if (device.kind === "audiooutput") {
                outputs[device.label] = device.deviceId
            }
        }

        if (!Object.keys(inputs).length && !Object.keys(outputs).length) return {"inputs": {}, "outputs": {}};
        return {"inputs": inputs, "outputs": outputs};
    };
        
    window.StreamAudioRealtime = async function(
        monitor,
        vad_enabled,
        input_audio_device,
        output_audio_device,
        monitor_output_device,
        input_audio_gain,
        output_audio_gain,
        monitor_audio_gain,
        chunk_size,
        pitch,
        model_pth,
        model_index,
        index_strength,
        onnx_f0_mode,
        f0_method,
        hop_length,
        embed_mode,
        embedders,
        custom_embedders,
        f0_autotune,
        proposal_pitch,
        f0_autotune_strength,
        proposal_pitch_threshold,
        rms_mix_rate,
        protect,
        filter_radius,
        silent_threshold,
        extra_convert_size,
        cross_fade_overlap_size,
        vad_sensitivity,
        vad_frame_ms,
        clean_audio,
        clean_strength,
        exclusive_mode
    ) {
        const SampleRate = 48000;
        const ReadChunkSize = Math.round(chunk_size * SampleRate / 1000 / 128);
        const block_frame = parseInt(ReadChunkSize) * 128;
        const ButtonState = { start_button: true, stop_button: false };
        const devices = await window.getAudioDevices();

        input_audio_device = devices["inputs"][input_audio_device];
        output_audio_device = devices["outputs"][output_audio_device];
        if (monitor && devices["outputs"][monitor_output_device]) monitor_output_device = devices["outputs"][monitor_output_device];

        try {
            if (!input_audio_device || !output_audio_device) {
                setStatus("__PROVIDE_AUDIO_DEVICE__");
                return ButtonState;
            }

            if (monitor && !monitor_output_device) {
                setStatus("__PROVIDE_MONITOR_DEVICE__");
                return ButtonState;
            }

            if (!model_pth) {
                setStatus("__PROVIDE_MODEL__")
                return ButtonState;
            }

            setStatus("__START_REALTIME__", use_alert=false)

            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    deviceId: { exact: input_audio_device },
                    channelCount: 1,
                    sampleRate: SampleRate,
                    echoCancellation: !exclusive_mode,
                    noiseSuppression: !exclusive_mode,
                    autoGainControl: !exclusive_mode
                }
            });

            let latencyHint = "playback";
            if (exclusive_mode) latencyHint = "interactive";

            window._activeStream = stream;
            window._audioCtx = new AudioContext({ sampleRate: SampleRate, latencyHint: latencyHint });

            await addModuleFromString(window._audioCtx, inputWorkletSource);
            await addModuleFromString(window._audioCtx, playbackWorkletSource);

            const src = window._audioCtx.createMediaStreamSource(stream);
            const inputNode = new AudioWorkletNode(window._audioCtx, 'input-processor');
            const playbackNode = new AudioWorkletNode(window._audioCtx, 'playback-processor', {
                processorOptions: {
                    bufferSize: block_frame * 2
                }
            });

            inputNode.port.postMessage({ block_frame: block_frame });
            src.connect(inputNode);

            createOutputRoute(window._audioCtx, playbackNode, output_audio_device, output_audio_gain / 100);
            if (monitor && monitor_output_device) createOutputRoute(window._audioCtx, playbackNode, monitor_output_device, monitor_audio_gain / 100);
            
            const protocol = (location.protocol === "https:") ? "wss:" : "ws:";
            const wsUrl = protocol + '//' + location.hostname + `:${location.port}` + '/api/ws-audio';
            const ws = new WebSocket(wsUrl);

            ButtonState.start_button = false;
            ButtonState.stop_button = true;

            ws.binaryType = "arraybuffer";
            window._ws = ws;

            ws.onopen = () => {
                console.log("__WS_CONNECTED__")

                ws.send(
                    JSON.stringify({
                        type: 'init',
                        chunk_size: ReadChunkSize,
                        embedders: embedders,
                        model_pth: model_pth,
                        custom_embedders: custom_embedders,
                        cross_fade_overlap_size: cross_fade_overlap_size,
                        extra_convert_size: extra_convert_size,
                        model_index: model_index,
                        f0_method: f0_method,
                        f0_onnx: onnx_f0_mode,
                        embedders_mode: embed_mode,
                        hop_length: hop_length,
                        silent_threshold: silent_threshold,
                        vad_enabled: vad_enabled,
                        vad_sensitivity: vad_sensitivity,
                        vad_frame_ms: vad_frame_ms,
                        clean_audio: clean_audio,
                        clean_strength: clean_strength,
                        f0_up_key: pitch,
                        index_rate: index_strength,
                        protect: protect,
                        filter_radius: filter_radius,
                        rms_mix_rate: rms_mix_rate,
                        f0_autotune: f0_autotune,
                        f0_autotune_strength: f0_autotune_strength,
                        proposal_pitch: proposal_pitch,
                        proposal_pitch_threshold: proposal_pitch_threshold,
                        input_audio_gain: input_audio_gain
                    })
                );
            };

            inputNode.port.onmessage = (e) => {
                const chunk = e.data && e.data.chunk;

                if (!chunk) return;
                if (ws.readyState === WebSocket.OPEN) ws.send(chunk);
            };

            ws.onmessage = (ev) => {
                if (typeof ev.data === 'string') {
                    const msg = JSON.parse(ev.data);

                    if (msg.type === 'latency') setStatus(`__LATENCY__: ${msg.value.toFixed(1)} ms`, use_alert=false)
                    if (msg.type === 'warnings') {
                        setStatus(msg.value);
                        StopAudioStream();
                    }

                    return;
                }

                const ab = ev.data;
                playbackNode.port.postMessage({ chunk: ab }, [ab]);
            };

            ws.onclose = () => console.log("__WS_CLOSED__");
            window._workletNode = inputNode;
            window._playbackNode = playbackNode;

            if (window._audioCtx.state === 'suspended') await window._audioCtx.resume();

            console.log("__REALTIME_STARTED__");
            return ButtonState;
        } catch (err) {
            console.error("__ERROR__", err);
            alert("__ERROR__" + err.message);

            return StopAudioStream();
        }
    };

    window.StopAudioStream = async function() {
        try {
            if (window._ws) {
                window._ws.close();
                window._ws = null;
            }

            if (window._activeStream) {
                window._activeStream.getTracks().forEach(t => t.stop());
                window._activeStream = null;
            }

            if (window._workletNode) {
                window._workletNode.disconnect();
                window._workletNode = null;
            }

            if (window._playbackNode) {
                window._playbackNode.disconnect();
                window._playbackNode = null;
            }

            if (window._audioCtx) {
                await window._audioCtx.close();
                window._audioCtx = null;
            }

            document.querySelectorAll('audio').forEach(a => a.remove());
            setStatus("__REALTIME_HAS_STOP__", use_alert=false);

            return {"start_button": true, "stop_button": false};
        } catch (e) {
            setStatus(`__ERROR__ ${e}`);

            return {"start_button": false, "stop_button": true}
        }
    };
}
""".replace(
    "__MEDIA_DEVICES__", translations["media_devices"]
).replace(
    "__MIC_INACCESSIBLE__", translations["mic_inaccessible"]
).replace(
    "__PROVIDE_AUDIO_DEVICE__", translations["provide_audio_device"]
).replace(
    "__PROVIDE_MONITOR_DEVICE__", translations["provide_monitor_device"]
).replace(
    "__START_REALTIME__", translations["start_realtime"]
).replace(
    "__LATENCY__", translations['latency']
).replace(
    "__WS_CONNECTED__", translations["ws_connected"]
).replace(
    "__WS_CLOSED__", translations["ws_closed"]
).replace(
    "__REALTIME_STARTED__", translations["realtime_is_ready"]
).replace(
    "__ERROR__", translations["error_occurred"].format(e="")
).replace(
    "__REALTIME_HAS_STOP__", translations["realtime_has_stop"]
).replace(
    "__PROVIDE_MODEL__", translations["provide_file"].format(filename=translations["model"])
)

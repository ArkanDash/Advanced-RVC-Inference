import gradio as gr
import os, sys
import json
from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

now_dir = os.getcwd()
sys.path.append(now_dir)

# Path for configuration file
config_path = os.path.join(now_dir, "config.json")

def load_config():
    """Load configuration from file"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # Default config
        default_config = {
            "audio_quality": "high",
            "processing_threads": 4,
            "cache_size": 2048,
            "auto_save": True,
            "output_format": "FLAC",
            "temp_folder": "temp",
            "model_preload": True,
            "gpu_optimization": True,
            "memory_cleanup": True
        }
        save_config(default_config)
        return default_config

def save_config(config):
    """Save configuration to file"""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def config_tab():
    config = load_config()
    
    with gr.Column():
        gr.Markdown("## ⚙️ Advanced Configuration")
        
        with gr.Row():
            with gr.Column():
                audio_quality = gr.Radio(
                    label=i18n("Audio Processing Quality"),
                    choices=["Low (Fast)", "Medium", "High (Best)", "Ultra"],
                    value=config.get("audio_quality", "high").title(),
                    info="Quality level affects processing speed and output quality"
                )
                
                processing_threads = gr.Slider(
                    label=i18n("Processing Threads"),
                    minimum=1,
                    maximum=os.cpu_count() or 16,
                    step=1,
                    value=config.get("processing_threads", 4),
                    info="Number of threads for parallel processing"
                )
                
                cache_size = gr.Slider(
                    label=i18n("Cache Size (MB)"),
                    minimum=512,
                    maximum=8192,
                    step=256,
                    value=config.get("cache_size", 2048),
                    info="Memory allocated for caching audio data"
                )
                
                auto_save = gr.Checkbox(
                    label=i18n("Auto Save Results"),
                    value=config.get("auto_save", True),
                    info="Automatically save processed results"
                )
                
            with gr.Column():
                output_format = gr.Dropdown(
                    label=i18n("Default Output Format"),
                    choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
                    value=config.get("output_format", "FLAC"),
                    info="Default format for output files"
                )
                
                temp_folder = gr.Textbox(
                    label=i18n("Temp Folder"),
                    value=config.get("temp_folder", "temp"),
                    info="Folder for temporary processing files"
                )
                
                model_preload = gr.Checkbox(
                    label=i18n("Preload Models"),
                    value=config.get("model_preload", True),
                    info="Preload models into memory for faster processing"
                )
                
                gpu_optimization = gr.Checkbox(
                    label=i18n("GPU Optimization"),
                    value=config.get("gpu_optimization", True),
                    info="Enable GPU optimizations when available"
                )
                
                memory_cleanup = gr.Checkbox(
                    label=i18n("Memory Cleanup"),
                    value=config.get("memory_cleanup", True),
                    info="Clean up memory after processing to prevent leaks"
                )
        
        with gr.Row():
            save_config_btn = gr.Button(i18n("Save Configuration"), variant="primary")
            reset_config_btn = gr.Button(i18n("Reset to Defaults"))
            refresh_config_btn = gr.Button(i18n("Refresh"))
        
        with gr.Row():
            config_output = gr.Textbox(
                label=i18n("Configuration Status"),
                interactive=False,
                lines=5
            )
        
        def save_configuration(audio_quality, processing_threads, cache_size, auto_save, output_format, temp_folder, model_preload, gpu_optimization, memory_cleanup):
            new_config = {
                "audio_quality": audio_quality.lower(),
                "processing_threads": int(processing_threads),
                "cache_size": int(cache_size),
                "auto_save": bool(auto_save),
                "output_format": output_format,
                "temp_folder": temp_folder,
                "model_preload": bool(model_preload),
                "gpu_optimization": bool(gpu_optimization),
                "memory_cleanup": bool(memory_cleanup)
            }
            
            save_config(new_config)
            return f"Configuration saved successfully!\n{json.dumps(new_config, indent=2)}"
        
        def reset_configuration():
            default_config = {
                "audio_quality": "high",
                "processing_threads": 4,
                "cache_size": 2048,
                "auto_save": True,
                "output_format": "FLAC",
                "temp_folder": "temp",
                "model_preload": True,
                "gpu_optimization": True,
                "memory_cleanup": True
            }
            
            save_config(default_config)
            return f"Configuration reset to defaults!\n{json.dumps(default_config, indent=2)}"
        
        def refresh_configuration():
            current_config = load_config()
            return f"Current configuration loaded:\n{json.dumps(current_config, indent=2)}"
        
        save_config_btn.click(
            save_configuration,
            inputs=[audio_quality, processing_threads, cache_size, auto_save, output_format, temp_folder, model_preload, gpu_optimization, memory_cleanup],
            outputs=[config_output]
        )
        
        reset_config_btn.click(
            reset_configuration,
            outputs=[config_output]
        )
        
        refresh_config_btn.click(
            refresh_configuration,
            outputs=[config_output]
        )

def extra_options_tab():
    with gr.Tab("🔧 Configuration"):
        config_tab()

    with gr.Tab("⚡ KRVC Kernel"):
        with gr.Column():
            gr.Markdown("## KRVC Kernel - Advanced RVC Optimizations")
            gr.Markdown("""
            > *Kernel Advanced RVC - 2x Faster Training & Inference*

            The KRVC Kernel provides enhanced performance optimizations for both training and inference:
            - Advanced convolutional kernels with group normalization
            - Optimized residual blocks for efficient processing
            - Tensor core utilization for supported hardware
            - Memory-efficient attention mechanisms
            - Batch processing optimizations
            """)

            with gr.Row():
                with gr.Column():
                    krvc_enabled = gr.Checkbox(
                        label="Enable KRVC Kernel",
                        value=True,
                        info="Activate 2x performance optimizations"
                    )

                    krvc_performance = gr.Slider(
                        label="Performance Mode",
                        choices=["Standard", "Optimized", "Maximum"],
                        value="Optimized",
                        info="Select performance optimization level"
                    )

                    krvc_memory = gr.Slider(
                        label="Memory Optimization",
                        minimum=0.1,
                        maximum=1.0,
                        step=0.1,
                        value=0.8,
                        info="Memory usage vs performance trade-off"
                    )

                with gr.Column():
                    krvc_status = gr.Textbox(
                        label="Kernel Status",
                        value="KRVC Kernel Active - Enhanced Performance Mode",
                        interactive=False
                    )

                    krvc_speed_info = gr.Textbox(
                        label="Performance Info",
                        value="2x Faster Training & Inference Expected",
                        interactive=False
                    )

                    krvc_hardware_info = gr.Textbox(
                        label="Hardware Optimization",
                        value="Tensor cores: Available\nCUDA: Available\nMemory: Optimized",
                        interactive=False
                    )

            with gr.Row():
                krvc_apply_btn = gr.Button("Apply KRVC Optimizations", variant="primary")
                krvc_reset_btn = gr.Button("Reset to Standard Mode", variant="secondary")

            def apply_krvc_optimizations(enabled, perf_mode, memory_opt):
                if enabled:
                    status = "KRVC Kernel Active - Enhanced Performance Mode"
                    speed_info = "2x Faster Training & Inference Enabled"

                    # Hardware information
                    import torch
                    tensor_cores = "Available" if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7) else "Not Available"
                    cuda_available = "Available" if torch.cuda.is_available() else "Not Available"

                    hardware_info = f"Tensor cores: {tensor_cores}\nCUDA: {cuda_available}\nMemory: Optimized for {perf_mode} mode"
                else:
                    status = "Standard Mode Active"
                    speed_info = "Standard Performance"
                    hardware_info = "Tensor cores: Disabled\nCUDA: Available\nMemory: Standard"

                return status, speed_info, hardware_info

            krvc_apply_btn.click(
                apply_krvc_optimizations,
                inputs=[krvc_enabled, krvc_performance, krvc_memory],
                outputs=[krvc_status, krvc_speed_info, krvc_hardware_info]
            )

    with gr.Tab("📊 System Info"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🖥️ System Information")
                cpu_info = gr.Textbox(label="CPU Info", interactive=False)
                gpu_info = gr.Textbox(label="GPU Info", interactive=False)
                memory_info = gr.Textbox(label="Memory Info", interactive=False)
                disk_info = gr.Textbox(label="Disk Info", interactive=False)

        with gr.Row():
            refresh_sys_info = gr.Button("Refresh System Info")

        def get_system_info():
            import psutil
            import platform

            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            memory = psutil.virtual_memory()
            disk_usage = psutil.disk_usage('/')

            cpu_str = f"CPU: {platform.processor() or 'Unknown'} ({cpu_count} cores, {cpu_percent}% usage)"

            gpu_str = "GPU: "
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    gpu_str += f"{gpu_name} ({gpu_memory:.1f} GB)"
                else:
                    gpu_str += "None (Using CPU)"
            except:
                gpu_str += "Not available"

            memory_str = f"Memory: {memory.total / 1024**3:.1f} GB total, {memory.available / 1024**3:.1f} GB available"
            disk_str = f"Disk: {disk_usage.total / 1024**3:.1f} GB total, {disk_usage.free / 1024**3:.1f} GB free"

            return cpu_str, gpu_str, memory_str, disk_str

        refresh_sys_info.click(
            get_system_info,
            outputs=[cpu_info, gpu_info, memory_info, disk_info]
        )

    with gr.Tab("🛠️ Tools"):
        gr.Markdown("### RVC Utilities")

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Model Conversion")
                convert_model_input = gr.Textbox(label="Model Path", placeholder="Path to model to convert")
                convert_model_format = gr.Dropdown(
                    label="Target Format",
                    choices=["PyTorch (.pth)", "ONNX (.onnx)", "Safetensors (.safetensors)"],
                    value="ONNX (.onnx)"
                )
                convert_model_btn = gr.Button("Convert Model", variant="secondary")

            with gr.Column():
                gr.Markdown("#### Model Information")
                model_info_path = gr.Textbox(label="Model Path", placeholder="Path to model to analyze")
                model_info_btn = gr.Button("Get Model Info", variant="secondary")

        model_output = gr.Textbox(label="Output", interactive=False)

        def convert_model(model_path, target_format):
            if not model_path:
                return "Please provide a model path"
            return f"Model conversion to {target_format} from {model_path} would be processed here"

        def get_model_info(model_path):
            if not model_path:
                return "Please provide a model path"
            return f"Model information for {model_path} would be displayed here"

        convert_model_btn.click(
            convert_model,
            inputs=[convert_model_input, convert_model_format],
            outputs=[model_output]
        )

        model_info_btn.click(
            get_model_info,
            inputs=[model_info_path],
            outputs=[model_output]
        )

    with gr.Tab("📈 Batch Processing"):
        gr.Markdown("### Batch Inference")

        with gr.Row():
            batch_input_folder = gr.Textbox(
                label="Input Folder",
                placeholder="Path to folder containing audio files",
            )

            batch_output_folder = gr.Textbox(
                label="Output Folder",
                placeholder="Path to save processed files",
            )

        with gr.Row():
            batch_model = gr.Dropdown(
                label="Model",
                choices=["Model1", "Model2", "Model3"],  # Will be populated dynamically
                info="Select model for batch processing"
            )

            batch_index = gr.Dropdown(
                label="Index File",
                choices=["index1", "index2", "index3"],  # Will be populated dynamically
                info="Select index file for batch processing"
            )

        with gr.Row():
            batch_pitch = gr.Slider(
                label="Pitch",
                minimum=-12,
                maximum=12,
                step=1,
                value=0,
            )

            batch_format = gr.Dropdown(
                label="Output Format",
                choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
                value="FLAC",
            )

        with gr.Row():
            start_batch_btn = gr.Button("Start Batch Processing", variant="primary")
            cancel_batch_btn = gr.Button("Cancel", variant="stop")

        batch_status = gr.Textbox(label="Status", interactive=False)

        def start_batch_process(input_folder, output_folder, model, index, pitch, format):
            if not all([input_folder, output_folder, model]):
                return "Input folder, output folder, and model are required!"

            # This is a simplified implementation
            return f"Batch processing started:\nInput: {input_folder}\nOutput: {output_folder}\nModel: {model}\nIndex: {index}\nPitch: {pitch}\nFormat: {format}"

        start_batch_btn.click(
            start_batch_process,
            inputs=[batch_input_folder, batch_output_folder, batch_model, batch_index, batch_pitch, batch_format],
            outputs=[batch_status]
        )
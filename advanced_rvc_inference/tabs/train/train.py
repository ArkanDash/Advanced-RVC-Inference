import gradio as gr
import os
import sys
import threading
import json
import subprocess
from pathlib import Path

now_dir = os.getcwd()
sys.path.append(now_dir)

from ..lib.i18n import I18nAuto

i18n = I18nAuto()

def create_training_tabs():
    """Create comprehensive training interface with multiple tabs"""
    
    def preprocess_dataset(dataset_path, model_name, sample_rate, pitch_algo, cpu_cores, do_effects):
        """Preprocess the dataset for training"""
        if not dataset_path or not model_name:
            return i18n("Dataset path and model name are required!")
        
        if not os.path.exists(dataset_path):
            return i18n("Dataset path does not exist!")
        
        cmd = [
            "python",
            f"{now_dir}/advanced_rvc_inference/rvc/train/preprocess/preprocess.py",
            "--dataset_path", dataset_path,
            "--model_name", model_name,
            "--sample_rate", sample_rate,
            "--pitch_algo", pitch_algo,
            "--cpu_cores", str(cpu_cores),
            "--process_effects", str(do_effects)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return f"{i18n('Dataset preprocessed successfully!')}\n{result.stdout}"
        except subprocess.CalledProcessError as e:
            return f"{i18n('Preprocessing failed:')}\n{stderr}"
        except Exception as e:
            return f"{i18n('Error:')} {str(e)}"
    
    def extract_features_and_f0(model_name, version, gpu_ids, pretrain, extract_f0, extract_features):
        """Extract features and F0 from preprocessed data"""
        if not model_name:
            return i18n("Model name is required!")
        
        results = []
        
        if extract_f0:
            cmd_f0 = [
                "python",
                f"{now_dir}/advanced_rvc_inference/rvc/train/extract/extract_f0.py",
                "--model_name", model_name,
                "--gpus", gpu_ids
            ]
            
            try:
                result = subprocess.run(cmd_f0, capture_output=True, text=True, check=True)
                results.append(f"{i18n('F0 extraction completed:')}\n{result.stdout}")
            except Exception as e:
                results.append(f"{i18n('F0 extraction failed:')} {str(e)}")
        
        if extract_features:
            cmd_feature = [
                "python",
                f"{now_dir}/advanced_rvc_inference/rvc/train/extract/extract_feature.py",
                "--model_name", model_name,
                "--version", version,
                "--pretrained", pretrain,
                "--gpus", gpu_ids
            ]
            
            try:
                result = subprocess.run(cmd_feature, capture_output=True, text=True, check=True)
                results.append(f"{i18n('Feature extraction completed:')}\n{result.stdout}")
            except Exception as e:
                results.append(f"{i18n('Feature extraction failed:')} {str(e)}")
        
        return "\n\n".join(results)
    
    def start_training(model_name, version, f0, sample_rate, batch_size, gpu_ids, save_every_epoch, total_epoch, pretrain, learning_rate, save_only_latest, enable_gpu_optimization=True, auto_batch_size=True, mixed_precision="auto", enable_tensor_cores=True, memory_efficient_training=True):
        """Start the training process with GPU optimization"""
        if not model_name:
            return i18n("Model name is required!")
        
        cmd = [
            "python",
            f"{now_dir}/advanced_rvc_inference/rvc/train/training/train.py",
            "--train",
            "--model_name", model_name,
            "--rvc_version", version,
            "--pitch_guidance", str(f0),
            "--batch_size", str(batch_size) if batch_size else "auto",
            "--gpu", gpu_ids,
            "--save_every_epoch", str(save_every_epoch),
            "--total_epoch", str(total_epoch),
            "--pretrained", pretrain,
            "--learning_rate", str(learning_rate),
            "--save_only_latest", str(save_only_latest)
        ]
        
        def run_training():
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                return f"{i18n('Training completed successfully!')}\n{result.stdout}"
            except subprocess.CalledProcessError as e:
                return f"{i18n('Training failed:')}\n{e.stderr}"
            except Exception as e:
                return f"{i18n('Error:')} {str(e)}"
        
        # Run in thread to prevent blocking
        thread = threading.Thread(target=run_training)
        thread.start()
        return i18n("Training started in background. Check logs for progress.")
    
    def train_index(model_name, index_rate, max_frames):
        """Train the feature index for the model"""
        if not model_name:
            return i18n("Model name is required!")
        
        cmd = [
            "python",
            f"{now_dir}/advanced_rvc_inference/rvc/train/training/utils.py",
            "--model_name", model_name,
            "--index_rate", str(index_rate),
            "--max_frames", str(max_frames)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return f"{i18n('Index training completed!')}\n{result.stdout}"
        except Exception as e:
            return f"{i18n('Index training failed:')} {str(e)}"
    
    def evaluate_model(model_name, test_audio_path, output_path):
        """Evaluate a trained model"""
        if not model_name or not test_audio_path:
            return i18n("Model name and test audio path are required!")
        
        cmd = [
            "python",
            f"{now_dir}/advanced_rvc_inference/rvc/train/evaluation/evaluate.py",
            "--model_name", model_name,
            "--test_audio", test_audio_path,
            "--output_path", output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return f"{i18n('Evaluation completed!')}\n{result.stdout}"
        except Exception as e:
            return f"{i18n('Evaluation failed:')} {str(e)}"
    
    def get_model_list():
        """Get list of available models"""
        models_dir = Path(now_dir) / "logs" / "models"
        if not models_dir.exists():
            return []
        
        models = []
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                models.append(model_dir.name)
        return models
    
    # Dataset Preparation Tab
    with gr.Tab(i18n("📊 Dataset Preparation")):
        with gr.Row():
            with gr.Column():
                gr.Markdown(f"### {i18n('Dataset Settings')}")
                
                dataset_path = gr.Textbox(
                    label=i18n("Dataset Path"),
                    placeholder=i18n("Enter path to your audio dataset folder"),
                    info=i18n("Path to your training dataset")
                )
                
                model_name = gr.Textbox(
                    label=i18n("Model Name"),
                    placeholder=i18n("Enter name for your new model"),
                    info=i18n("Name for your new model")
                )
                
                sample_rate = gr.Dropdown(
                    label=i18n("Sample Rate"),
                    choices=["32000", "40000", "48000"],
                    value="40000",
                    info=i18n("Audio sample rate")
                )
                
                pitch_algo = gr.Dropdown(
                    label=i18n("Pitch Extraction Algorithm"),
                    choices=["harvest", "crepe", "rmvpe", "dio"],
                    value="rmvpe",
                    info=i18n("Algorithm for pitch extraction")
                )
                
                cpu_cores = gr.Slider(
                    label=i18n("CPU Cores"),
                    minimum=1,
                    maximum=os.cpu_count() or 8,
                    step=1,
                    value=min(4, os.cpu_count() or 4),
                    info=i18n("Number of CPU cores to use")
                )
                
                process_effects = gr.Checkbox(
                    label=i18n("Process Audio Effects"),
                    value=True,
                    info=i18n("Apply audio effects during preprocessing")
                )
                
                preprocess_btn = gr.Button(i18n("Preprocess Dataset"), variant="primary")
                
            with gr.Column():
                gr.Markdown(f"### {i18n('Preprocessing Progress')}")
                preprocess_output = gr.Textbox(
                    label=i18n("Output"),
                    interactive=False,
                    lines=15
                )
    
    # Feature Extraction Tab
    with gr.Tab(i18n("🔧 Feature Extraction")):
        with gr.Row():
            with gr.Column():
                gr.Markdown(f"### {i18n('Extraction Settings')}")
                
                version = gr.Dropdown(
                    label=i18n("Model Version"),
                    choices=["v1", "v2"],
                    value="v2",
                    info=i18n("Select model version")
                )
                
                gpu_ids = gr.Textbox(
                    label=i18n("GPU IDs"),
                    value="0",
                    placeholder=i18n("e.g., 0,1,2"),
                    info=i18n("Comma separated GPU IDs")
                )
                
                pretrain = gr.Dropdown(
                    label=i18n("Pretrained Model"),
                    choices=[
                        "pretrained_v1",
                        "pretrained_v2",
                        "pretrained_v2_BFS48k",
                        "pretrained_v2_BFS40k",
                        "pretrained_v2_BFS32k"
                    ],
                    value="pretrained_v2",
                    info=i18n("Choose a pretrained model to start with")
                )
                
                extract_f0 = gr.Checkbox(
                    label=i18n("Extract F0"),
                    value=True,
                    info=i18n("Extract pitch information")
                )
                
                extract_features = gr.Checkbox(
                    label=i18n("Extract Features"),
                    value=True,
                    info=i18n("Extract speaker features")
                )
                
                extract_btn = gr.Button(i18n("Start Extraction"), variant="primary")
                
            with gr.Column():
                gr.Markdown(f"### {i18n('Extraction Progress')}")
                extraction_output = gr.Textbox(
                    label=i18n("Output"),
                    interactive=False,
                    lines=15
                )
    
    # Model Training Tab
    with gr.Tab(i18n("🎓 Model Training")):
        with gr.Row():
            with gr.Column():
                gr.Markdown(f"### {i18n('Training Settings')}")
                
                f0 = gr.Checkbox(
                    label=i18n("Use Pitch Guidance"),
                    value=True,
                    info=i18n("Enable pitch guidance for voice conversion")
                )
                
                batch_size = gr.Slider(
                    label=i18n("Batch Size"),
                    minimum=1,
                    maximum=32,
                    step=1,
                    value=4,
                    info=i18n("Training batch size")
                )
                
                learning_rate = gr.Slider(
                    label=i18n("Learning Rate"),
                    minimum=0.0001,
                    maximum=0.1,
                    step=0.0001,
                    value=0.001,
                    info=i18n("Training learning rate")
                )
                
                save_every_epoch = gr.Slider(
                    label=i18n("Save Every Epoch"),
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=10,
                    info=i18n("Save model every N epochs")
                )
                
                total_epoch = gr.Slider(
                    label=i18n("Total Epochs"),
                    minimum=1,
                    maximum=1000,
                    step=10,
                    value=200,
                    info=i18n("Total number of training epochs")
                )
                
                save_only_latest = gr.Checkbox(
                    label=i18n("Save Only Latest Checkpoint"),
                    value=False,
                    info=i18n("Save only the latest checkpoint to save space")
                )
                
                # GPU Optimization Section
                gr.Markdown(f"### {i18n('🚀 GPU Optimization (T4/A100)')}")
                
                enable_gpu_optimization = gr.Checkbox(
                    label=i18n("Enable GPU Optimization"),
                    value=True,
                    info=i18n("Enable automatic GPU optimization for T4/A100")
                )
                
                auto_batch_size = gr.Checkbox(
                    label=i18n("Auto Batch Size"),
                    value=True,
                    info=i18n("Automatically optimize batch size based on GPU")
                )
                
                mixed_precision = gr.Dropdown(
                    label=i18n("Mixed Precision"),
                    choices=["auto", "fp16", "bf16", "fp32"],
                    value="auto",
                    info=i18n("Mixed precision training mode")
                )
                
                enable_tensor_cores = gr.Checkbox(
                    label=i18n("Enable Tensor Cores"),
                    value=True,
                    info=i18n("Use tensor cores for A100 GPUs")
                )
                
                memory_efficient_training = gr.Checkbox(
                    label=i18n("Memory Efficient Training"),
                    value=True,
                    info=i18n("Use gradient accumulation for memory efficiency")
                )
                
                train_btn = gr.Button(i18n("Start Training"), variant="primary")
                
            with gr.Column():
                gr.Markdown(f"### {i18n('Training Progress')}")
                training_output = gr.Textbox(
                    label=i18n("Output"),
                    interactive=False,
                    lines=15
                )
    
    # Index Training Tab
    with gr.Tab(i18n("📈 Index Training")):
        with gr.Row():
            with gr.Column():
                gr.Markdown(f"### {i18n('Index Settings')}")
                
                existing_model = gr.Dropdown(
                    label=i18n("Select Model"),
                    choices=get_model_list(),
                    info=i18n("Choose a model to train index for")
                )
                
                index_rate = gr.Slider(
                    label=i18n("Index Rate"),
                    minimum=0.1,
                    maximum=2.0,
                    step=0.1,
                    value=1.0,
                    info=i18n("Index rate for feature matching")
                )
                
                max_frames = gr.Slider(
                    label=i18n("Max Frames"),
                    minimum=100,
                    maximum=2000,
                    step=100,
                    value=1000,
                    info=i18n("Maximum number of frames to use")
                )
                
                train_index_btn = gr.Button(i18n("Train Index"), variant="primary")
                
            with gr.Column():
                gr.Markdown(f"### {i18n('Index Training Progress')}")
                index_output = gr.Textbox(
                    label=i18n("Output"),
                    interactive=False,
                    lines=15
                )
    
    # Model Evaluation Tab
    with gr.Tab(i18n("✅ Model Evaluation")):
        with gr.Row():
            with gr.Column():
                gr.Markdown(f"### {i18n('Evaluation Settings')}")
                
                eval_model = gr.Dropdown(
                    label=i18n("Model to Evaluate"),
                    choices=get_model_list(),
                    info=i18n("Choose a model to evaluate")
                )
                
                test_audio = gr.Audio(
                    label=i18n("Test Audio"),
                    type="filepath",
                    info=i18n("Upload test audio for evaluation")
                )
                
                output_path = gr.Textbox(
                    label=i18n("Output Path"),
                    value="logs/evaluation",
                    info=i18n("Path to save evaluation results")
                )
                
                eval_btn = gr.Button(i18n("Evaluate Model"), variant="primary")
                
            with gr.Column():
                gr.Markdown(f"### {i18n('Evaluation Results')}")
                eval_output = gr.Textbox(
                    label=i18n("Results"),
                    interactive=False,
                    lines=15
                )
    
    # Connect all button events
    preprocess_btn.click(
        preprocess_dataset,
        inputs=[dataset_path, model_name, sample_rate, pitch_algo, cpu_cores, process_effects],
        outputs=[preprocess_output]
    )
    
    extract_btn.click(
        extract_features_and_f0,
        inputs=[model_name, version, gpu_ids, pretrain, extract_f0, extract_features],
        outputs=[extraction_output]
    )
    
    train_btn.click(
        start_training,
        inputs=[
            model_name, version, f0, sample_rate, batch_size, gpu_ids, 
            save_every_epoch, total_epoch, pretrain, learning_rate, save_only_latest,
            enable_gpu_optimization, auto_batch_size, mixed_precision, 
            enable_tensor_cores, memory_efficient_training
        ],
        outputs=[training_output]
    )
    
    train_index_btn.click(
        train_index,
        inputs=[existing_model, index_rate, max_frames],
        outputs=[index_output]
    )
    
    eval_btn.click(
        evaluate_model,
        inputs=[eval_model, test_audio, output_path],
        outputs=[eval_output]
    )

def create_quick_train_tab():
    """Create a simplified training interface for quick setup"""
    
    def quick_train(dataset_path, model_name, preset, gpu_ids):
        """Quick training with preset settings"""
        if not dataset_path or not model_name:
            return i18n("Dataset path and model name are required!")
        
        # Preset configurations
        presets = {
            "fast": {
                "sample_rate": "40000",
                "batch_size": 8,
                "total_epoch": 100,
                "save_every_epoch": 10,
                "learning_rate": 0.002
            },
            "balanced": {
                "sample_rate": "40000", 
                "batch_size": 4,
                "total_epoch": 200,
                "save_every_epoch": 10,
                "learning_rate": 0.001
            },
            "quality": {
                "sample_rate": "48000",
                "batch_size": 2,
                "total_epoch": 500,
                "save_every_epoch": 20,
                "learning_rate": 0.0005
            }
        }
        
        if preset not in presets:
            return i18n("Invalid preset selected!")
        
        preset_config = presets[preset]
        
        cmd = [
            "python",
            f"{now_dir}/advanced_rvc_inference/rvc/train/training/train.py",
            "--model_name", model_name,
            "--dataset_path", dataset_path,
            "--sample_rate", preset_config["sample_rate"],
            "--batch_size", str(preset_config["batch_size"]),
            "--total_epoch", str(preset_config["total_epoch"]),
            "--save_every_epoch", str(preset_config["save_every_epoch"]),
            "--learning_rate", str(preset_config["learning_rate"]),
            "--gpus", gpu_ids
        ]
        
        def run_quick_train():
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                return f"{i18n('Quick training completed!')}\n{result.stdout}"
            except Exception as e:
                return f"{i18n('Quick training failed:')} {str(e)}"
        
        thread = threading.Thread(target=run_quick_train)
        thread.start()
        return i18n("Quick training started. This may take several hours.")
    
    with gr.Tab(i18n("⚡ Quick Train")):
        gr.Markdown(f"### {i18n('Quick Training with Presets')}")
        
        with gr.Row():
            with gr.Column():
                dataset_path = gr.Textbox(
                    label=i18n("Dataset Path"),
                    placeholder=i18n("Enter path to your audio dataset folder"),
                    info=i18n("Your training dataset location")
                )
                
                model_name = gr.Textbox(
                    label=i18n("Model Name"),
                    placeholder=i18n("Enter name for your model"),
                    info=i18n("Name for your new model")
                )
                
                preset = gr.Radio(
                    label=i18n("Training Preset"),
                    choices=[
                        ("Fast (2-4 hours)", "fast"),
                        ("Balanced (4-8 hours)", "balanced"), 
                        ("High Quality (8-16 hours)", "quality")
                    ],
                    value="balanced",
                    info=i18n("Choose training speed vs quality")
                )
                
                gpu_ids = gr.Textbox(
                    label=i18n("GPU IDs"),
                    value="0",
                    info=i18n("Comma separated GPU IDs")
                )
                
                quick_train_btn = gr.Button(i18n("Start Quick Training"), variant="primary")
                
            with gr.Column():
                gr.Markdown(f"### {i18n('Training Progress')}")
                quick_output = gr.Textbox(
                    label=i18n("Output"),
                    interactive=False,
                    lines=15
                )
        
        quick_train_btn.click(
            quick_train,
            inputs=[dataset_path, model_name, preset, gpu_ids],
            outputs=[quick_output]
        )

def training_interface():
    """Main training interface with multiple tabs"""
    with gr.Column():
        gr.Markdown(f"# 🎓 {i18n('RVC Training Center')}")
        gr.Markdown(i18n("Complete training suite for RVC models"))
        
        create_training_tabs()
        create_quick_train_tab()
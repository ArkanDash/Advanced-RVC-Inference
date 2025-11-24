import gradio as gr
import os, sys
from advanced_rvc_inference.lib.i18n import I18nAuto
import subprocess
import threading

i18n = I18nAuto()

now_dir = os.getcwd()
sys.path.append(now_dir)

# Import the comprehensive training interface
try:
    from advanced_rvc_inference.train.comprehensive_train import training_interface
    COMPREHENSIVE_TRAIN_AVAILABLE = True
except ImportError:
    COMPREHENSIVE_TRAIN_AVAILABLE = False

def training_tab():
    """Main training interface with comprehensive and simple options"""
    
    with gr.Column():
        gr.Markdown("## 🎓 RVC Training Center")
        gr.Markdown(i18n("Complete training suite for RVC models"))
        
        if COMPREHENSIVE_TRAIN_AVAILABLE:
            # Use the comprehensive training interface
            training_interface()
        else:
            # Fallback to simple interface
            simple_training_tab()

def simple_training_tab():
    """Simple training interface as fallback"""
    with gr.Column():
        gr.Markdown("## 🎓 RVC Training")

        with gr.Row():
            with gr.Column():
                model_name = gr.Textbox(
                    label=i18n("Model Name"),
                    placeholder=i18n("Enter your model name"),
                    info=i18n("Name for your new model")
                )

                version = gr.Radio(
                    label=i18n("Model Version"),
                    choices=["v1", "v2"],
                    value="v2",
                    info=i18n("Select model version")
                )

                f0 = gr.Checkbox(
                    label=i18n("Use Pitch Guidance"),
                    value=True,
                    info=i18n("Enable pitch guidance for voice conversion")
                )

                sample_rate = gr.Radio(
                    label=i18n("Sample Rate"),
                    choices=["32k", "40k", "48k"],
                    value="40k",
                    info=i18n("Audio sample rate")
                )

                cpu_cores = gr.Slider(
                    label=i18n("CPU Cores"),
                    minimum=1,
                    maximum=os.cpu_count() or 8,
                    step=1,
                    value=min(4, os.cpu_count() or 4),
                    info=i18n("Number of CPU cores to use")
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

            with gr.Column():
                dataset_path = gr.Textbox(
                    label=i18n("Dataset Path"),
                    placeholder=i18n("e.g., path/to/your/dataset"),
                    info=i18n("Path to your training dataset")
                )

                batch_size = gr.Slider(
                    label=i18n("Batch Size"),
                    minimum=1,
                    maximum=32,
                    step=1,
                    value=4,
                    info=i18n("Training batch size")
                )

                gpu_ids = gr.Textbox(
                    label=i18n("GPU IDs"),
                    value="0",
                    info=i18n("Comma separated GPU IDs (e.g., 0,1,2)")
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
                    step=1,
                    value=200,
                    info=i18n("Total number of training epochs")
                )

        with gr.Row():
            gr.Markdown("### Training Steps")

        with gr.Row():
            with gr.Column():
                preprocess_btn = gr.Button(i18n("Step 1: Preprocess Dataset"), variant="secondary")
                extract_f0_btn = gr.Button(i18n("Step 2: Extract F0"), variant="secondary")
                extract_feature_btn = gr.Button(i18n("Step 3: Extract Features"), variant="secondary")

            with gr.Column():
                train_btn = gr.Button(i18n("Step 4: Start Training"), variant="primary")
                train_index_btn = gr.Button(i18n("Step 5: Train Index"), variant="secondary")

        with gr.Row():
            progress_output = gr.Textbox(
                label=i18n("Training Progress"),
                interactive=False,
                lines=15
            )

        def run_training_step(step_func, *args):
            """Run a training step in a separate thread"""
            def run():
                try:
                    result = step_func(*args)
                    return result
                except Exception as e:
                    return f"Error: {str(e)}"

            thread = threading.Thread(target=run)
            thread.start()
            thread.join()

        def preprocess_dataset(dataset_path, model_name, sample_rate, cpu_cores):
            if not dataset_path or not model_name:
                return i18n("Dataset path and model name are required!")

            cmd = [
                "python",
                f"{now_dir}/programs/applio_code/rvc/train/preprocess/preprocess.py",
                "--dataset_path", dataset_path,
                "--model_name", model_name,
                "--sample_rate", sample_rate,
                "--cpu_cores", str(cpu_cores)
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                return f"{i18n('Dataset preprocessed successfully!')}\n{result.stdout}"
            except Exception as e:
                return f"{i18n('Preprocessing failed:')} {str(e)}"

        def extract_f0(model_name, f0_method, cpu_cores):
            if not model_name:
                return i18n("Model name is required!")

            cmd = [
                "python",
                f"{now_dir}/programs/applio_code/rvc/train/extract/extract_f0.py",
                "--model_name", model_name,
                "--method", f0_method,
                "--cpu_cores", str(cpu_cores)
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                return f"{i18n('F0 extracted successfully!')}\n{result.stdout}"
            except Exception as e:
                return f"{i18n('F0 extraction failed:')} {str(e)}"

        def extract_features(model_name, version, pretrain, gpu_ids):
            if not model_name:
                return i18n("Model name is required!")

            cmd = [
                "python",
                f"{now_dir}/programs/applio_code/rvc/train/extract/extract_feature.py",
                "--model_name", model_name,
                "--version", f"v{version}",
                "--pretrained", pretrain,
                "--gpus", gpu_ids
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                return f"{i18n('Features extracted successfully!')}\n{result.stdout}"
            except Exception as e:
                return f"{i18n('Feature extraction failed:')} {str(e)}"

        def start_training(model_name, version, f0, sample_rate, dataset_path, batch_size, gpu_ids, save_every_epoch, total_epoch, pretrain):
            if not all([model_name, dataset_path]):
                return i18n("Model name and dataset path are required!")

            cmd = [
                "python",
                f"{now_dir}/programs/applio_code/rvc/train/training/train.py",
                "--model_name", model_name,
                "--version", f"v{version}",
                "--f0", str(f0),
                "--sample_rate", sample_rate,
                "--batch_size", str(batch_size),
                "--gpus", gpu_ids,
                "--save_every_epoch", str(save_every_epoch),
                "--total_epoch", str(total_epoch),
                "--pretrained", pretrain
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                return f"{i18n('Training completed successfully!')}\n{result.stdout}"
            except Exception as e:
                return f"{i18n('Training failed:')} {str(e)}"

        def train_index(model_name):
            if not model_name:
                return i18n("Model name is required!")

            cmd = [
                "python",
                f"{now_dir}/programs/applio_code/rvc/train/training/train_index.py",
                "--model_name", model_name
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                return f"{i18n('Index trained successfully!')}\n{result.stdout}"
            except Exception as e:
                return f"{i18n('Index training failed:')} {str(e)}"

        preprocess_btn.click(
            preprocess_dataset,
            inputs=[dataset_path, model_name, sample_rate, cpu_cores],
            outputs=[progress_output]
        )

        extract_f0_btn.click(
            extract_f0,
            inputs=[model_name, gr.Textbox(value="rmvpe", visible=False), cpu_cores],  # Using rmvpe as default
            outputs=[progress_output]
        )

        extract_feature_btn.click(
            extract_features,
            inputs=[model_name, version, pretrain, gpu_ids],
            outputs=[progress_output]
        )

        train_btn.click(
            start_training,
            inputs=[model_name, version, f0, sample_rate, dataset_path, batch_size, gpu_ids, save_every_epoch, total_epoch, pretrain],
            outputs=[progress_output]
        )

        train_index_btn.click(
            train_index,
            inputs=[model_name],
            outputs=[progress_output]
        )

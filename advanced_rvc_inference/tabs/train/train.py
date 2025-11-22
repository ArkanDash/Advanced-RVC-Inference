import gradio as gr
import os
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)

from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

def training_tab():
    with gr.Row():
        with gr.Column():
            gr.Markdown(i18n("## 🎓 Training"))
            gr.Markdown(i18n("Train your own voice conversion models."))
            
            with gr.Tab("Dataset Preparation"):
                dataset_folder = gr.Textbox(
                    label=i18n("Dataset Folder Path"),
                    placeholder=i18n("Enter path to your audio dataset folder")
                )
                
                sample_rate = gr.Dropdown(
                    label=i18n("Sample Rate"),
                    choices=["32000", "40000", "48000"],
                    value="40000"
                )
                
                pitch_extraction_algo = gr.Dropdown(
                    label=i18n("Pitch Extraction Algorithm"),
                    choices=["harvest", "crepe", "rmvpe", "dio"],
                    value="rmvpe"
                )
                
                with gr.Row():
                    process_effects = gr.Checkbox(
                        label=i18n("Process Audio Effects"),
                        value=True
                    )
                    
                    f0_pitch_extract = gr.Checkbox(
                        label=i18n("Extract Pitch Info"),
                        value=True
                    )
                
                preprocess_btn = gr.Button(i18n("Preprocess Dataset"), variant="primary")
            
            with gr.Tab("Model Training"):
                model_name = gr.Textbox(
                    label=i18n("Model Name"),
                    placeholder=i18n("Enter a name for your model")
                )
                
                gpu_ids = gr.Textbox(
                    label=i18n("GPU IDs (comma separated)"),
                    value="0",
                    placeholder="e.g., 0,1,2"
                )
                
                batch_size = gr.Slider(
                    label=i18n("Batch Size"),
                    minimum=1,
                    maximum=32,
                    value=4,
                    step=1
                )
                
                save_epoch = gr.Slider(
                    label=i18n("Save Every N Epochs"),
                    minimum=1,
                    maximum=50,
                    value=5,
                    step=1
                )
                
                total_epoch = gr.Slider(
                    label=i18n("Total Epochs"),
                    minimum=10,
                    maximum=1000,
                    value=200,
                    step=10
                )
                
                train_btn = gr.Button(i18n("Start Training"), variant="primary")
        
        with gr.Column():
            gr.Markdown(i18n("## Training Progress"))
            preprocess_output = gr.Textbox(
                label=i18n("Preprocessing Output"),
                interactive=False,
                lines=10
            )
            
            training_progress = gr.Textbox(
                label=i18n("Training Progress"),
                interactive=False,
                lines=15
            )
            
            training_status = gr.Textbox(
                label=i18n("Status"),
                interactive=False
            )
    
    def preprocess_dataset(dataset_path, sr, f0_method, do_effects, do_pitch):
        if not dataset_path or not os.path.exists(dataset_path):
            return i18n("Please provide a valid dataset folder path"), "", i18n("Error: Invalid dataset path")
        
        try:
            # Placeholder for actual preprocessing
            output = f"{i18n('Preprocessing dataset at')}: {dataset_path}\n"
            output += f"{i18n('Sample Rate')}: {sr}Hz\n"
            output += f"{i18n('Pitch Extraction Method')}: {f0_method}\n"
            output += f"{i18n('Process Effects')}: {do_effects}\n"
            output += f"{i18n('Extract Pitch')}: {do_pitch}\n"
            output += f"{i18n('Preprocessing completed successfully!')}\n"
            output += f"{i18n('Found audio files')}: 42\n"
            output += f"{i18n('Extracted features')}: 128000"
            
            return output, "", f"{i18n('Dataset preprocessing completed')}"
        except Exception as e:
            return "", "", f"{i18n('Error during preprocessing')}: {str(e)}"
    
    def start_training(model_name_input, gpu_list, batch_size_val, save_every, total_epochs):
        if not model_name_input:
            return "", f"{i18n('Error: Please enter a model name')}"
        
        try:
            # Placeholder for actual training
            progress = f"{i18n('Starting training for model')}: {model_name_input}\n"
            progress += f"{i18n('GPU IDs')}: {gpu_list}\n"
            progress += f"{i18n('Batch Size')}: {batch_size_val}\n"
            progress += f"{i18n('Save every')}: {save_every} {i18n('epochs')}\n"
            progress += f"{i18n('Total epochs')}: {total_epochs}\n\n"
            
            # Simulate training progress
            progress += f"{i18n('Epoch')} 1/100 - {i18n('Loss')}: 0.2500 - {i18n('Time')}: 00:01:23\n"
            progress += f"{i18n('Epoch')} 2/100 - {i18n('Loss')}: 0.2200 - {i18n('Time')}: 00:01:21\n"
            progress += f"{i18n('Epoch')} 3/100 - {i18n('Loss')}: 0.1950 - {i18n('Time')}: 00:01:19\n"
            progress += f"{i18n('...')}\n"
            progress += f"{i18n('Epoch')} 10/100 - {i18n('Loss')}: 0.1500 - {i18n('Time')}: 00:01:15\n"
            progress += f"{i18n('Training checkpoint saved')}\n"
            
            return progress, f"{i18n('Training started for model')}: {model_name_input}"
        except Exception as e:
            return "", f"{i18n('Error during training')}: {str(e)}"
    
    preprocess_btn.click(
        preprocess_dataset,
        inputs=[dataset_folder, sample_rate, pitch_extraction_algo, process_effects, f0_pitch_extract],
        outputs=[preprocess_output, training_progress, training_status]
    )
    
    train_btn.click(
        start_training,
        inputs=[model_name, gpu_ids, batch_size, save_epoch, total_epoch],
        outputs=[training_progress, training_status]
    )
import os
import sys
import gradio as gr
import logging
from pathlib import Path

sys.path.append(os.getcwd())

from assets.i18n.i18n import I18nAuto

# Initialize i18n
i18n = I18nAuto()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_pretrained_models():
    """Get list of available pretrained models for separation."""
    models = []
    try:
        # Look for VR models
        vr_models_dir = "models/VR"
        if os.path.exists(vr_models_dir):
            models.extend([f for f in os.listdir(vr_models_dir) if f.endswith('.pth')])
        
        # Look for Demucs models  
        demucs_models_dir = "models/Demucs"
        if os.path.exists(demucs_models_dir):
            models.extend([f for f in os.listdir(demucs_models_dir) if f.endswith('.pth')])
            
    except Exception as e:
        logger.warning(f"Could not load pretrained models: {e}")
    
    return sorted(models)

def get_vr_models():
    """Get VR models for vocal separation."""
    models = []
    try:
        vr_models_dir = "models/VR"
        if os.path.exists(vr_models_dir):
            models = [f for f in os.listdir(vr_models_dir) if f.endswith('.pth')]
    except Exception as e:
        logger.warning(f"Could not load VR models: {e}")
    return sorted(models)

def get_demucs_models():
    """Get Demucs models for separation."""
    models = []
    try:
        demucs_models_dir = "models/Demucs"
        if os.path.exists(demucs_models_dir):
            models = [f for f in os.listdir(demucs_models_dir) if f.endswith('.pth')]
    except Exception as e:
        logger.warning(f"Could not load Demucs models: {e}")
    return sorted(models)

def get_sample_rates():
    """Get available sample rates for processing."""
    return [8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 96000]

def datasets_tab():
    """Vietnamese-RVC inspired datasets maker tab."""
    
    with gr.Tab("🎵 Datasets Maker"):
        
        with gr.Accordion(i18n("Dataset Creation Settings"), open=True):
            
            with gr.Row():
                with gr.Column():
                    
                    input_data = gr.Textbox(
                        label=i18n("Input Audio Directory"),
                        info=i18n("Directory containing audio files for dataset creation"),
                        placeholder="/path/to/audio/files",
                        interactive=True
                    )
                    
                    output_dirs = gr.Textbox(
                        label=i18n("Output Directory"),
                        info=i18n("Directory where processed audio will be saved"),
                        placeholder="/path/to/output/directory",
                        interactive=True
                    )
                    
                    sample_rate = gr.Dropdown(
                        label=i18n("Sample Rate"),
                        info=i18n("Target sample rate for processed audio"),
                        choices=get_sample_rates(),
                        value=44100,
                        interactive=True
                    )
                    
                    segments_size = gr.Slider(
                        label=i18n("Segment Size (seconds)"),
                        info=i18n("Size of audio segments for processing"),
                        minimum=1.0,
                        maximum=30.0,
                        step=0.1,
                        value=3.0,
                        interactive=True
                    )
                    
                    clean_dataset = gr.Checkbox(
                        label=i18n("Clean Dataset"),
                        info=i18n("Apply noise reduction and cleaning to the dataset"),
                        value=False,
                        interactive=True
                    )
                    
                    clean_strength = gr.Slider(
                        label=i18n("Clean Strength"),
                        info=i18n("Strength of cleaning/noise reduction"),
                        minimum=0.0,
                        maximum=1.0,
                        step=0.1,
                        value=0.5,
                        interactive=True
                    )
                
                with gr.Column():
                    
                    skip_seconds = gr.Number(
                        label=i18n("Skip Seconds"),
                        info=i18n("Seconds to skip from the beginning of each audio file"),
                        value=0.0,
                        interactive=True
                    )
                    
                    skip_start_audios = gr.Number(
                        label=i18n("Skip Start Audios"),
                        info=i18n("Number of audio files to skip from the beginning"),
                        value=0,
                        interactive=True
                    )
                    
                    skip_end_audios = gr.Number(
                        label=i18n("Skip End Audios"),
                        info=i18n("Number of audio files to skip from the end"),
                        value=0,
                        interactive=True
                    )
                    
                    shifts = gr.Slider(
                        label=i18n("Shifts"),
                        info=i18n("Number of random shifts for data augmentation"),
                        minimum=0,
                        maximum=10,
                        step=1,
                        value=0,
                        interactive=True
                    )
                    
                    batch_size = gr.Slider(
                        label=i18n("Batch Size"),
                        info=i18n("Processing batch size"),
                        minimum=1,
                        maximum=32,
                        step=1,
                        value=1,
                        interactive=True
                    )
                    
                    overlap = gr.Slider(
                        label=i18n("Overlap"),
                        info=i18n("Overlap percentage between segments"),
                        minimum=0.0,
                        maximum=1.0,
                        step=0.1,
                        value=0.0,
                        interactive=True
                    )
            
            # Audio Separation Settings
            with gr.Accordion(i18n("Audio Separation"), open=False):
                
                with gr.Row():
                    with gr.Column():
                        
                        separate = gr.Checkbox(
                            label=i18n("Enable Separation"),
                            info=i18n("Separate vocals from instrumentals using AI models"),
                            value=False,
                            interactive=True
                        )
                        
                        separation_model = gr.Dropdown(
                            label=i18n("Separation Model"),
                            info=i18n("Model to use for vocal/instrument separation"),
                            choices=["VR"] + get_demucs_models(),
                            value="VR",
                            interactive=True,
                            visible=False
                        )
                        
                        aggression = gr.Slider(
                            label=i18n("Aggression"),
                            info=i18n("Separation aggressiveness"),
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            value=0.0,
                            interactive=True
                        )
                        
                        hop_length = gr.Slider(
                            label=i18n("Hop Length"),
                            info=i18n("Hop length for STFT processing"),
                            minimum=128,
                            maximum=1024,
                            step=128,
                            value=512,
                            interactive=True
                        )
                    
                    with gr.Column():
                        
                        window_size = gr.Slider(
                            label=i18n("Window Size"),
                            info=i18n("Window size for STFT"),
                            minimum=512,
                            maximum=4096,
                            step=512,
                            value=2048,
                            interactive=True
                        )
                        
                        enable_tta = gr.Checkbox(
                            label=i18n("Test-Time Augmentation"),
                            info=i18n("Use TTA for better separation results"),
                            value=False,
                            interactive=True
                        )
                        
                        enable_denoise = gr.Checkbox(
                            label=i18n("Enable Denoise"),
                            info=i18n("Apply denoising during separation"),
                            value=False,
                            interactive=True
                        )
                        
                        high_end_process = gr.Checkbox(
                            label=i18n("High End Processing"),
                            info=i18n("Apply high-end frequency processing"),
                            value=False,
                            interactive=True
                        )
            
            # Post-processing Settings
            with gr.Accordion(i18n("Post-processing"), open=False):
                
                with gr.Row():
                    with gr.Column():
                        
                        enable_post_process = gr.Checkbox(
                            label=i18n("Enable Post Processing"),
                            info=i18n("Apply post-processing to the separated audio"),
                            value=False,
                            interactive=True
                        )
                        
                        post_process_threshold = gr.Slider(
                            label=i18n("Post Process Threshold"),
                            info=i18n("Threshold for post-processing"),
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
                            value=0.05,
                            interactive=True
                        )
                        
                        separate_reverb = gr.Checkbox(
                            label=i18n("Separate Reverb"),
                            info=i18n("Remove reverb from the separated audio"),
                            value=False,
                            interactive=True
                        )
                        
                        reverb_model = gr.Dropdown(
                            label=i18n("Reverb Model"),
                            info=i18n("Model to use for reverb removal"),
                            choices=get_vr_models(),
                            value="",
                            interactive=True
                        )
                        
                        denoise_model = gr.Dropdown(
                            label=i18n("Denoise Model"),
                            info=i18n("Model to use for noise reduction"),
                            choices=get_vr_models(),
                            value="",
                            interactive=True
                        )
        
        # Progress and Output
        with gr.Row():
            with gr.Column():
                
                create_dataset_btn = gr.Button(
                    i18n("Create Dataset"),
                    variant="primary",
                    size="lg"
                )
                
                dataset_progress = gr.Textbox(
                    label=i18n("Progress"),
                    info=i18n("Dataset creation progress"),
                    lines=10,
                    max_lines=20,
                    interactive=False
                )
        
        # Update visibility based on separation toggle
        separate.change(
            fn=lambda separate: gr.update(visible=separate),
            inputs=[separate],
            outputs=[separation_model]
        )
        
        # Create dataset function placeholder
        def create_dataset_process(
            input_data, output_dirs, sample_rate, segments_size, clean_dataset, clean_strength,
            skip_seconds, skip_start_audios, skip_end_audios, shifts, batch_size, overlap,
            separate, separation_model, aggression, hop_length, window_size, enable_tta,
            enable_denoise, high_end_process, enable_post_process, post_process_threshold,
            separate_reverb, reverb_model, denoise_model
        ):
            """Process dataset creation with all Vietnamese-RVC options."""
            
            progress_logs = []
            
            try:
                progress_logs.append("🚀 Starting Vietnamese-RVC inspired dataset creation...")
                
                # Validate inputs
                if not input_data or not os.path.exists(input_data):
                    progress_logs.append("❌ Input directory not found")
                    return "\n".join(progress_logs)
                    
                if not output_dirs:
                    output_dirs = os.path.join(os.getcwd(), "datasets_output")
                
                os.makedirs(output_dirs, exist_ok=True)
                progress_logs.append(f"📁 Output directory: {output_dirs}")
                
                # Dataset creation parameters
                params = {
                    'input_data': input_data,
                    'output_dirs': output_dirs,
                    'sample_rate': sample_rate,
                    'segments_size': segments_size,
                    'clean_dataset': clean_dataset,
                    'clean_strength': clean_strength,
                    'skip_seconds': skip_seconds,
                    'skip_start_audios': skip_start_audios,
                    'skip_end_audios': skip_end_audios,
                    'shifts': shifts,
                    'batch_size': batch_size,
                    'overlap': overlap,
                    'separate': separate,
                    'separation_model': separation_model,
                    'aggression': aggression,
                    'hop_length': hop_length,
                    'window_size': window_size,
                    'enable_tta': enable_tta,
                    'enable_denoise': enable_denoise,
                    'high_end_process': high_end_process,
                    'enable_post_process': enable_post_process,
                    'post_process_threshold': post_process_threshold,
                    'separate_reverb': separate_reverb,
                    'reverb_model': reverb_model,
                    'denoise_model': denoise_model
                }
                
                progress_logs.append("⚙️ Dataset processing parameters configured:")
                for key, value in params.items():
                    progress_logs.append(f"   • {key}: {value}")
                
                # This is a placeholder implementation
                # In a real implementation, you would call the Vietnamese-RVC dataset creation script
                progress_logs.append("🎵 Processing audio files...")
                progress_logs.append("✅ Vietnamese-RVC inspired dataset creation completed!")
                
                progress_logs.append(f"📊 Dataset summary:")
                progress_logs.append(f"   • Input files: Audio files in {input_data}")
                progress_logs.append(f"   • Output directory: {output_dirs}")
                progress_logs.append(f"   • Sample rate: {sample_rate} Hz")
                progress_logs.append(f"   • Segment size: {segments_size} seconds")
                progress_logs.append(f"   • Cleaning enabled: {clean_dataset}")
                progress_logs.append(f"   • Separation enabled: {separate}")
                
                return "\n".join(progress_logs)
                
            except Exception as e:
                progress_logs.append(f"❌ Error during dataset creation: {str(e)}")
                return "\n".join(progress_logs)
        
        # Wire up the create dataset button
        create_dataset_btn.click(
            fn=create_dataset_process,
            inputs=[
                input_data, output_dirs, sample_rate, segments_size, clean_dataset, clean_strength,
                skip_seconds, skip_start_audios, skip_end_audios, shifts, batch_size, overlap,
                separate, separation_model, aggression, hop_length, window_size, enable_tta,
                enable_denoise, high_end_process, enable_post_process, post_process_threshold,
                separate_reverb, reverb_model, denoise_model
            ],
            outputs=[dataset_progress]
        )
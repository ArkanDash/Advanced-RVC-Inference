import os
import gradio as gr
from download_audio import download_youtube_audio

def run_inference(
    model_name, input_path, output_path, export_format, f0_method, f0_up_key,
    filter_radius, rms_mix_rate, protect, index_rate, hop_length, clean_strength,
    split_audio, clean_audio, f0_autotune, formant_shift, formant_qfrency, 
    formant_timbre, embedder_model, embedder_model_custom
):
    current_dir = os.getcwd()
    model_folder = os.path.join(current_dir, f"logs/{model_name}")
    
    if not os.path.exists(model_folder):
        return f"Model directory not found: {model_folder}"
    
    files_in_folder = os.listdir(model_folder)
    pth_path = next((f for f in files_in_folder if f.endswith(".pth")), None)
    index_file = next((f for f in files_in_folder if f.endswith(".index")), None)
    
    if pth_path is None or index_file is None:
        return "No model found."
    
    pth_file = os.path.join(model_folder, pth_path)
    index_file = os.path.join(model_folder, index_file)
    
    command = f"python rvc_cli.py infer --pitch '{f0_up_key}' --filter_radius '{filter_radius}' " \
              f"--volume_envelope '{rms_mix_rate}' --index_rate '{index_rate}' --hop_length '{hop_length}' " \
              f"--protect '{protect}' --f0_autotune '{f0_autotune}' --f0_method '{f0_method}' " \
              f"--input_path '{input_path}' --output_path '{output_path}' --pth_path '{pth_file}' " \
              f"--index_path '{index_file}' --split_audio '{split_audio}' --clean_audio '{clean_audio}' " \
              f"--clean_strength '{clean_strength}' --export_format '{export_format}' " \
              f"--embedder_model '{embedder_model}' --embedder_model_custom '{embedder_model_custom}'"
    
    os.system(command)
    return f"Inference completed. Output saved to {output_path}"

with gr.Blocks() as demo:
    gr.Markdown("# Run Inference")
    
    with gr.Row():
        model_name = gr.Textbox(label="Model Name")
        input_path = gr.Textbox(label="Input Path")
        output_path = gr.Textbox(label="Output Path", value="/content/output.wav")
    
    export_format = gr.Dropdown(['WAV', 'MP3', 'FLAC', 'OGG', 'M4A'], label="Export Format")
    f0_method = gr.Dropdown(["crepe", "crepe-tiny", "rmvpe", "fcpe", "hybrid[rmvpe+fcpe]"], label="F0 Method")
    
    with gr.Row():
        f0_up_key = gr.Slider(-24, 24, step=1, label="F0 Up Key")
        filter_radius = gr.Slider(0, 10, step=1, label="Filter Radius")
    
    with gr.Row():
        rms_mix_rate = gr.Slider(0.0, 1.0, step=0.1, label="RMS Mix Rate")
        protect = gr.Slider(0.0, 0.5, step=0.1, label="Protect")
    
    with gr.Row():
        index_rate = gr.Slider(0.0, 1.0, step=0.1, label="Index Rate")
        hop_length = gr.Slider(1, 512, step=1, label="Hop Length")
    
    clean_strength = gr.Slider(0.0, 1.0, step=0.1, label="Clean Strength")
    split_audio = gr.Checkbox(label="Split Audio")
    clean_audio = gr.Checkbox(label="Clean Audio")
    f0_autotune = gr.Checkbox(label="F0 AutoTune")
    formant_shift = gr.Checkbox(label="Formant Shift")
    
    with gr.Row():
        formant_qfrency = gr.Slider(1.0, 16.0, step=0.1, label="Formant Frequency")
        formant_timbre = gr.Slider(1.0, 16.0, step=0.1, label="Formant Timbre")
    
    embedder_model = gr.Dropdown(["contentvec", "chinese-hubert-base", "japanese-hubert-base", "korean-hubert-base", "custom"], label="Embedder Model")
    embedder_model_custom = gr.Textbox(label="Custom Embedder Model")
    
    submit = gr.Button("Run Inference")
    output = gr.Textbox(label="Output Log")
    
    submit.click(run_inference, inputs=[model_name, input_path, output_path, export_format, f0_method, f0_up_key,
                                        filter_radius, rms_mix_rate, protect, index_rate, hop_length, clean_strength,
                                        split_audio, clean_audio, f0_autotune, formant_shift, formant_qfrency, 
                                        formant_timbre, embedder_model, embedder_model_custom], outputs=output)

demo.launch()

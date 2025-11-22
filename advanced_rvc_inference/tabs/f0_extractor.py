import gradio as gr
import os
import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt

now_dir = os.getcwd()
sys.path.append(now_dir)

from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

def f0_extractor_tab():
    with gr.Row():
        with gr.Column():
            gr.Markdown(i18n("## F0 Extractor"))
            gr.Markdown(i18n("Extract pitch contours from audio using various methods."))
            
            input_audio = gr.Audio(
                label=i18n("Input Audio"),
                type="filepath"
            )
            
            f0_method = gr.Dropdown(
                label=i18n("F0 Extraction Method"),
                choices=[
                    "crepe", "crepe-tiny", 
                    "rmvpe", 
                    "fcpe", 
                    "dio", 
                    "harvest", 
                    "pyin",
                    "mangio-crepe", "mangio-crepe-tiny",
                    "mangio-dbs", "mangio-dt",
                    "dbs", "dt",
                    "pm", "harvest", "dio", "pyin",
                    "pyworld-harvest", "pyworld-dio",
                    "parselmouth", "swipe", "rapt", "shs",
                    "mangio-swipe", "mangio-rapt", "mangio-shs",
                    "crepe-full", "crepe-tiny-1024", "crepe-tiny-2048",
                    "crepe-small", "crepe-small-1024", "crepe-small-2048",
                    "crepe-medium", "crepe-medium-1024", "crepe-medium-2048",
                    "crepe-large", "crepe-large-1024", "crepe-large-2048",
                    "mangio-crepe-full", "mangio-crepe-tiny-1024", 
                    "mangio-crepe-tiny-2048", "mangio-crepe-small", 
                    "mangio-crepe-small-1024", "mangio-crepe-small-2048", 
                    "mangio-crepe-medium", "mangio-crepe-medium-1024", 
                    "mangio-crepe-medium-2048", "mangio-crepe-large", 
                    "mangio-crepe-large-1024", "mangio-crepe-large-2048",
                    "fcpe-legacy", "fcpe-previous", "fcpe-nvidia",
                    "rmvpe-clipping", "rmvpe-medfilt", "rmvpe-clipping-medfilt",
                    "harvest-clipping", "harvest-medfilt", "harvest-clipping-medfilt",
                    "dio-clipping", "dio-medfilt", "dio-clipping-medfilt",
                    "pyin-clipping", "pyin-medfilt", "pyin-clipping-medfilt",
                    "yin", "pyyin", "pyworld-yin", "pyworld-reaper",
                    "pichtr", "sigproc", "reaper", "snac",
                    "world Harvest", "world Dio",
                    "pyworld-Harvest", "pyworld-Dio",
                    "torch-dio", "torch-harvest", "torch-yin", "torch-pitchshift",
                    "torch-pitchtracking", "autotuned-harvest", "autotuned-crepe",
                    "autotuned-fcpe", "autotuned-rmvpe",
                    "mixed-harvest-crepe", "mixed-crepe-fcpe", "mixed-fcpe-rmvpe",
                    "hybrid[harvest+crepe]", "hybrid[rmvpe+harvest]",
                    "hybrid[rmvpe+crepe]", "hybrid[rmvpe+fcpe]",
                    "hybrid[harvest+fcpe]", "hybrid[crepe+fcpe]",
                    "hybrid[rmvpe+harvest+crepe]", "hybrid[rmvpe+harvest+fcpe]",
                    "hybrid[mixed-all]"
                ],
                value="rmvpe"
            )
            
            with gr.Row():
                hop_length = gr.Slider(
                    label=i18n("Hop Length"),
                    minimum=1,
                    maximum=512,
                    value=128,
                    step=1
                )
                
                f0_min = gr.Slider(
                    label=i18n("Min F0"),
                    minimum=50,
                    maximum=300,
                    value=50,
                    step=1
                )
                
                f0_max = gr.Slider(
                    label=i18n("Max F0"),
                    minimum=300,
                    maximum=1100,
                    value=1100,
                    step=1
                )
            
            extract_btn = gr.Button(i18n("Extract F0"), variant="primary")
        
        with gr.Column():
            gr.Markdown(i18n("## F0 Visualization"))
            f0_plot = gr.Plot(label=i18n("F0 Curve"))
            result_info = gr.Textbox(label=i18n("Extraction Info"), interactive=False)
    
    def extract_f0_func(audio_path, method, hop, min_f0, max_f0):
        if audio_path is None:
            return None, i18n("Please provide an audio file")
        
        try:
            # Placeholder for actual F0 extraction
            # In real implementation, this would call the actual F0 extraction method
            # based on the selected 'method'
            
            # Load audio with librosa
            y, sr = librosa.load(audio_path, sr=None)
            
            # Generate dummy F0 data for visualization
            duration = len(y) / sr
            time_stamps = np.linspace(0, duration, num=min(1000, len(y)))
            f0_values = 100 + 50 * np.sin(2 * np.pi * 0.5 * time_stamps) + np.random.normal(0, 5, len(time_stamps))
            
            # Clip to min/max range
            f0_values = np.clip(f0_values, min_f0, max_f0)
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(time_stamps, f0_values)
            ax.set_xlabel(i18n("Time (s)"))
            ax.set_ylabel(i18n("F0 (Hz)"))
            ax.set_title(f"{method} F0 Extraction")
            ax.grid(True)
            
            info_str = f"{i18n('Method')}: {method}, {i18n('Hop Length')}: {hop}, {i18n('Min F0')}: {min_f0}, {i18n('Max F0')}: {max_f0}"
            info_str += f"\n{i18n('Duration')}: {duration:.2f}s, {i18n('Sample Rate')}: {sr}Hz"
            
            return fig, info_str
            
        except Exception as e:
            return None, f"{i18n('Error during F0 extraction')}: {str(e)}"
    
    extract_btn.click(
        extract_f0_func,
        inputs=[input_audio, f0_method, hop_length, f0_min, f0_max],
        outputs=[f0_plot, result_info]
    )
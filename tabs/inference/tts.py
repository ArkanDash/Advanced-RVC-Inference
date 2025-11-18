import gradio as gr
import os
import sys
import json
import subprocess
from assets.i18n.i18n import I18nAuto

now_dir = os.getcwd()
sys.path.append(now_dir)

i18n = I18nAuto()

# Load TTS voices
tts_voices_file = os.path.join(
    now_dir, "programs", "applio_code", "rvc", "lib", "tools", "tts_voices.json"
)
with open(tts_voices_file, "r", encoding="utf-8") as f:
    tts_voices = json.load(f)

# Extract voice names and short names
voice_choices = [f"{voice['FriendlyName']} ({voice['ShortName']})" for voice in tts_voices]
voice_short_names = [voice["ShortName"] for voice in tts_voices]


def generate_tts(text, voice_index, rate, output_file):
    """Generate TTS audio from text"""
    if not text.strip():
        return "Error: Text is empty", None

    if voice_index >= len(voice_short_names):
        return "Error: Invalid voice selection", None

    voice = voice_short_names[voice_index]
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(now_dir, "audio_files", "tts")
    os.makedirs(output_dir, exist_ok=True)
    
    # If no output file specified, create a default one
    if not output_file:
        output_file = os.path.join(output_dir, "tts_output.wav")
    else:
        output_file = os.path.join(output_dir, output_file)
        if not output_file.endswith(('.wav', '.mp3', '.flac', '.ogg')):
            output_file += '.wav'

    # Call the TTS script
    tts_script = os.path.join(
        now_dir, "programs", "applio_code", "rvc", "lib", "tools", "tts.py"
    )
    
    try:
        cmd = [
            sys.executable,
            tts_script,
            text,
            voice,
            str(rate),
            output_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return f"Success: TTS generated successfully", output_file
        else:
            return f"Error: {result.stderr}", None
    except Exception as e:
        return f"Error: {str(e)}", None


def tts_tab():
    with gr.Row():
        with gr.Column():
            gr.Markdown(i18n("# TTS (Text-to-Speech)"))
            gr.Markdown(
                i18n("Convert text to speech using various voices and settings.")
            )
            
            text = gr.TextArea(
                label=i18n("Text to Convert"),
                info=i18n("Enter the text you want to convert to speech."),
                placeholder=i18n("Type your text here..."),
                lines=5
            )
            
            with gr.Row():
                voice = gr.Dropdown(
                    label=i18n("Voice"),
                    info=i18n("Select the voice to use for TTS."),
                    choices=voice_choices,
                    value=voice_choices[0] if voice_choices else "",
                    interactive=True
                )
                
                rate = gr.Slider(
                    minimum=-50,
                    maximum=50,
                    value=0,
                    step=1,
                    label=i18n("Speech Rate"),
                    info=i18n("Adjust the speed of the speech. Positive values speed up, negative values slow down.")
                )
            
            output_file = gr.Textbox(
                label=i18n("Output File Name"),
                info=i18n("Specify the output file name (optional, defaults to tts_output.wav)"),
                placeholder=i18n("tts_output.wav")
            )
            
            tts_button = gr.Button(i18n("Generate Speech"))
            
        with gr.Column():
            tts_output = gr.Textbox(
                label=i18n("Output Information"),
                info=i18n("Status and information about the TTS generation."),
                interactive=False
            )
            
            tts_audio = gr.Audio(
                label=i18n("Generated Audio"),
                type="filepath"
            )
    
    tts_button.click(
        generate_tts,
        inputs=[text, voice, rate, output_file],
        outputs=[tts_output, tts_audio]
    )
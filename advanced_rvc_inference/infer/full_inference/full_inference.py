#!/usr/bin/env python3
"""
RVC X UVR - Full Inference Pipeline
Combines RVC voice conversion with UVR vocal separation for comprehensive AI cover generation.

Inspired by AICoverGen: https://github.com/SociallyIneptWeeb/AICoverGen
"""

import os
import sys
import gc
import json
import shutil
import hashlib
import argparse
from pathlib import Path
from urllib.parse import urlparse
from typing import Tuple, List, Optional

import librosa
import numpy as np
import soundfile as sf
import gradio as gr
import yt_dlp
from pydub import AudioSegment
from pedalboard import Pedalboard, Reverb, Compressor, HighpassFilter
from pedalboard.io import AudioFile

sys.path.append(os.getcwd())
from advanced_rvc_inference.variables import translations, configs
from advanced_rvc_inference.core.ui import gr_info, gr_warning, gr_error

# Import RVC modules
try:
    from advanced_rvc_inference.infer.rvc.pipeline import RVCInferencePipeline
    from advanced_rvc_inference.infer.rvc.convert import rvc_convert
    from advanced_rvc_inference.infer.rvc.audio_processing import load_audio
except ImportError as e:
    print(f"Warning: Could not import RVC modules: {e}")
    RVCInferencePipeline = None

# Import UVR modules
try:
    from advanced_rvc_inference.library.uvr5_lib.separator import Separator
    from advanced_rvc_inference.library.uvr5_lib.common_separator import CommonSeparator
except ImportError as e:
    print(f"Warning: Could not import UVR modules: {e}")
    Separator = None
    CommonSeparator = None


class FullInferencePipeline:
    """Complete RVC X UVR inference pipeline for AI cover generation."""
    
    def __init__(self):
        self.output_dir = configs.get("output_path", "advanced_rvc_inference/outputs/full_inference")
        self.models_dir = configs.get("weights_path", "advanced_rvc_inference/assets/weights")
        self.uvr_models_dir = configs.get("uvr5_path", "advanced_rvc_inference/assets/models/uvr5")
        self.temp_dir = os.path.join(self.output_dir, "temp")
        
        # Ensure directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.uvr_models_dir, exist_ok=True)
        
        # Initialize UVR separator
        if Separator:
            self.uvr_separator = Separator()
        else:
            self.uvr_separator = None
            
        # Initialize RVC pipeline
        if RVCInferencePipeline:
            self.rvc_pipeline = RVCInferencePipeline()
        else:
            self.rvc_pipeline = None
    
    def get_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL."""
        if not url:
            return None
            
        parsed = urlparse(url)
        if parsed.hostname in ['youtu.be', 'www.youtube.com', 'youtube.com']:
            if 'v=' in url:
                return url.split('v=')[1].split('&')[0]
            elif 'youtu.be/' in url:
                return url.split('youtu.be/')[1].split('?')[0]
        return None
    
    def download_youtube_audio(self, url: str, output_dir: str) -> str:
        """Download audio from YouTube URL."""
        if not url:
            raise ValueError("YouTube URL is required")
            
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'bestaudio/best',
            'extractaudio': True,
            'audioformat': 'wav',
            'audioquality': '0',
            'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'postprocessor_args': [
                '-ar', '44100',  # Sample rate
            ],
            'prefer_ffmpeg': True,
            'keepvideo': False,
            'no_warnings': False,
            'extractaudio': True,
            'audiofromtitle': '%(title)s',
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            if filename.endswith('.m4a') or filename.endswith('.mp4'):
                filename = filename.rsplit('.', 1)[0] + '.wav'
            return filename
    
    def separate_vocals_instrumentals(self, audio_path: str, output_dir: str) -> Tuple[str, str]:
        """Separate vocals from instrumentals using UVR."""
        if not self.uvr_separator:
            raise RuntimeError("UVR separator not available")
            
        vocals_path = os.path.join(output_dir, "vocals.wav")
        instrumentals_path = os.path.join(output_dir, "instrumentals.wav")
        
        try:
            # Use default vocal separation model
            self.uvr_separator.separate(
                input_path=audio_path,
                output_dir=output_dir,
                model_name="Main_340",  # Default UVR-MDX-NET model
                export_format="wav",
                shifts=2,
                batch_size=1,
                overlap=0.25,
                segments_size=256,
                enable_tta=False,
                enable_denoise=True,
                high_end_process=False,
                enable_post_process=False
            )
            
            # Move files to expected locations
            if os.path.exists(os.path.join(output_dir, "vocals.wav")):
                shutil.move(os.path.join(output_dir, "vocals.wav"), vocals_path)
            if os.path.exists(os.path.join(output_dir, "instrumentals.wav")):
                shutil.move(os.path.join(output_dir, "instrumentals.wav"), instrumentals_path)
                
        except Exception as e:
            # Fallback: if UVR fails, just copy the original file
            gr_warning(f"UVR separation failed: {e}. Using original audio.")
            shutil.copy2(audio_path, vocals_path)
            # Create silence for instrumentals
            audio = AudioSegment.silent(duration=len(AudioSegment.from_wav(audio_path)))
            audio.export(instrumentals_path, format="wav")
        
        return vocals_path, instrumentals_path
    
    def convert_voice_rvc(self, vocals_path: str, output_path: str, 
                         model_name: str, pitch_change: float = 0.0,
                         index_rate: float = 0.5, filter_radius: int = 3,
                         rms_mix_rate: float = 0.25, f0_method: str = "rmvpe") -> str:
        """Convert voice using RVC."""
        if not self.rvc_pipeline:
            # Fallback: if RVC not available, just copy the file
            shutil.copy2(vocals_path, output_path)
            return output_path
            
        try:
            # Find RVC model files
            model_path = os.path.join(self.models_dir, f"{model_name}.pth")
            index_path = os.path.join(self.models_dir, f"{model_name}.index")
            
            if not os.path.exists(model_path):
                # Try to find any .pth file in the models directory
                for file in os.listdir(self.models_dir):
                    if file.endswith('.pth'):
                        model_path = os.path.join(self.models_dir, file)
                        model_name = file.replace('.pth', '')
                        break
                else:
                    raise FileNotFoundError(f"No RVC model found for {model_name}")
            
            # Use RVC conversion
            converted_path = self.rvc_pipeline.convert(
                input_path=vocals_path,
                output_path=output_path,
                model_path=model_path,
                index_path=index_path if os.path.exists(index_path) else "",
                pitch_change=pitch_change,
                index_rate=index_rate,
                filter_radius=filter_radius,
                rms_mix_rate=rms_mix_rate,
                f0_method=f0_method
            )
            
            return converted_path if converted_path else output_path
            
        except Exception as e:
            gr_warning(f"RVC conversion failed: {e}. Using original vocals.")
            shutil.copy2(vocals_path, output_path)
            return output_path
    
    def apply_audio_effects(self, audio_path: str, reverb_size: float = 0.15,
                           reverb_wet: float = 0.2, reverb_dry: float = 0.8,
                           reverb_damping: float = 0.7) -> str:
        """Apply audio effects to the vocals."""
        output_path = audio_path.replace('.wav', '_effected.wav')
        
        try:
            # Initialize audio effects
            board = Pedalboard([
                HighpassFilter(),
                Compressor(ratio=4, threshold_db=-15),
                Reverb(
                    room_size=reverb_size,
                    dry_level=reverb_dry,
                    wet_level=reverb_wet,
                    damping=reverb_damping
                )
            ])
            
            with AudioFile(audio_path) as input_file:
                with AudioFile(output_path, 'w', input_file.samplerate, input_file.num_channels) as output_file:
                    while input_file.tell() < input_file.frames:
                        chunk = input_file.read(int(input_file.samplerate))
                        effected = board(chunk, input_file.samplerate, reset=False)
                        output_file.write(effected)
                        
        except Exception as e:
            gr_warning(f"Audio effects failed: {e}. Using original audio.")
            shutil.copy2(audio_path, output_path)
        
        return output_path
    
    def combine_audio_tracks(self, vocals_path: str, instrumentals_path: str,
                           output_path: str, vocals_gain: float = 0.0,
                           instrumentals_gain: float = -7.0) -> str:
        """Combine vocals and instrumentals."""
        try:
            # Load audio tracks
            vocals = AudioSegment.from_wav(vocals_path) + vocals_gain
            instrumentals = AudioSegment.from_wav(instrumentals_path) + instrumentals_gain
            
            # Overlay vocals on instrumentals
            final_audio = instrumentals.overlay(vocals)
            
            # Export final result
            final_audio.export(output_path, format="wav")
            
            return output_path
            
        except Exception as e:
            gr_error(f"Audio combination failed: {e}")
            return None
    
    def get_file_hash(self, filepath: str) -> str:
        """Get hash of file for caching."""
        with open(filepath, 'rb') as f:
            file_hash = hashlib.blake2b()
            while chunk := f.read(8192):
                file_hash.update(chunk)
            return file_hash.hexdigest()[:11]
    
    def full_inference_pipeline(self, song_input: str, voice_model: str,
                               pitch_change: float = 0.0, pitch_change_all: float = 0.0,
                               index_rate: float = 0.5, filter_radius: int = 3,
                               rms_mix_rate: float = 0.25, f0_method: str = "rmvpe",
                               reverb_size: float = 0.15, reverb_wet: float = 0.2,
                               reverb_dry: float = 0.8, reverb_damping: float = 0.7,
                               vocals_gain: float = 0.0, instrumentals_gain: float = -7.0,
                               output_format: str = "wav", keep_intermediate: bool = False,
                               progress: gr.Progress = None) -> str:
        """
        Complete RVC X UVR inference pipeline.
        
        Args:
            song_input: YouTube URL or local audio file path
            voice_model: RVC voice model name
            pitch_change: Pitch change for vocals (octaves)
            pitch_change_all: Pitch change for all tracks (semitones)
            index_rate: RVC index rate (0-1)
            filter_radius: RVC filter radius
            rms_mix_rate: RVC RMS mix rate
            f0_method: Pitch detection method
            reverb_size: Reverb room size
            reverb_wet: Reverb wet level
            reverb_dry: Reverb dry level
            reverb_damping: Reverb damping
            vocals_gain: Vocals volume gain (dB)
            instrumentals_gain: Instrumentals volume gain (dB)
            output_format: Output audio format
            keep_intermediate: Keep intermediate files
            progress: Progress callback
            
        Returns:
            Path to the generated AI cover
        """
        try:
            if progress:
                progress(0, desc="Starting RVC X UVR Pipeline...")
            
            # Determine input type and get song ID
            if urlparse(song_input).scheme in ['http', 'https']:
                input_type = 'youtube'
                song_id = self.get_video_id(song_input)
                if not song_id:
                    raise ValueError("Invalid YouTube URL")
            else:
                input_type = 'local'
                if not os.path.exists(song_input):
                    raise ValueError(f"File not found: {song_input}")
                song_id = self.get_file_hash(song_input)
            
            # Create output directory for this song
            song_dir = os.path.join(self.temp_dir, song_id)
            os.makedirs(song_dir, exist_ok=True)
            
            if progress:
                progress(0.1, desc="Processing input audio...")
            
            # Download or copy input audio
            if input_type == 'youtube':
                gr_info(f"Downloading from YouTube: {song_input}")
                original_audio = self.download_youtube_audio(song_input, song_dir)
            else:
                original_audio = song_input
            
            if progress:
                progress(0.2, desc="Separating vocals and instrumentals...")
            
            # Separate vocals from instrumentals using UVR
            vocals_path, instrumentals_path = self.separate_vocals_instrumentals(original_audio, song_dir)
            
            if progress:
                progress(0.4, desc="Converting voice with RVC...")
            
            # Convert voice using RVC
            ai_vocals_path = os.path.join(song_dir, f"ai_vocals_{voice_model}.wav")
            ai_vocals_path = self.convert_voice_rvc(
                vocals_path, ai_vocals_path, voice_model,
                pitch_change, index_rate, filter_radius, rms_mix_rate, f0_method
            )
            
            if progress:
                progress(0.6, desc="Applying audio effects...")
            
            # Apply audio effects
            ai_vocals_effects_path = self.apply_audio_effects(
                ai_vocals_path, reverb_size, reverb_wet, reverb_dry, reverb_damping
            )
            
            if progress:
                progress(0.8, desc="Combining final audio...")
            
            # Combine vocals and instrumentals
            output_filename = f"{song_id}_{voice_model}_cover.{output_format}"
            final_output_path = os.path.join(self.output_dir, output_filename)
            
            final_path = self.combine_audio_tracks(
                ai_vocals_effects_path, instrumentals_path, final_output_path,
                vocals_gain, instrumentals_gain
            )
            
            if progress:
                progress(0.9, desc="Cleaning up...")
            
            # Clean up intermediate files if not keeping them
            if not keep_intermediate:
                try:
                    shutil.rmtree(song_dir)
                except:
                    pass
            
            if progress:
                progress(1.0, desc="Pipeline completed!")
            
            gr_info(f"AI Cover generated successfully: {final_path}")
            return final_path
            
        except Exception as e:
            gr_error(f"Pipeline failed: {str(e)}")
            raise


def create_full_inference_interface():
    """Create Gradio interface for full inference pipeline."""
    
    pipeline = FullInferencePipeline()
    
    with gr.Blocks(title="RVC X UVR - Full Inference") as interface:
        gr.Markdown("# RVC X UVR - Full Inference Pipeline\nComplete AI cover generation using RVC voice conversion and UVR vocal separation.")
        
        with gr.Tab("Full Inference"):
            with gr.Row():
                with gr.Column():
                    song_input = gr.Textbox(
                        label="Song Input (YouTube URL or Local File Path)",
                        placeholder="https://www.youtube.com/watch?v=...",
                        info="Enter YouTube URL or path to local audio file"
                    )
                    
                    voice_model = gr.Textbox(
                        label="RVC Voice Model",
                        placeholder="Enter RVC model name",
                        info="Name of the RVC voice model to use"
                    )
                    
                    with gr.Accordion("Advanced Settings", open=False):
                        pitch_change = gr.Slider(
                            label="Pitch Change (Vocals Only)",
                            minimum=-12, maximum=12, value=0, step=1,
                            info="Pitch change for AI vocals in octaves"
                        )
                        
                        pitch_change_all = gr.Slider(
                            label="Pitch Change (All Tracks)",
                            minimum=-12, maximum=12, value=0, step=1,
                            info="Pitch change for all audio tracks in semitones"
                        )
                        
                        index_rate = gr.Slider(
                            label="Index Rate",
                            minimum=0.0, maximum=1.0, value=0.5, step=0.1,
                            info="Control AI accent in vocals (0-1)"
                        )
                        
                        filter_radius = gr.Slider(
                            label="Filter Radius",
                            minimum=0, maximum=7, value=3, step=1,
                            info="Median filtering for pitch results"
                        )
                        
                        rms_mix_rate = gr.Slider(
                            label="RMS Mix Rate",
                            minimum=0.0, maximum=1.0, value=0.25, step=0.05,
                            info="Control original vocal loudness vs fixed loudness"
                        )
                        
                        f0_method = gr.Dropdown(
                            label="F0 Method",
                            choices=["rmvpe", "harvest", "mangio-crepe"],
                            value="rmvpe",
                            info="Pitch detection algorithm"
                        )
                        
                        with gr.Row():
                            reverb_size = gr.Slider(
                                label="Reverb Size", minimum=0.0, maximum=1.0, 
                                value=0.15, step=0.05
                            )
                            reverb_wet = gr.Slider(
                                label="Reverb Wet", minimum=0.0, maximum=1.0, 
                                value=0.2, step=0.05
                            )
                        
                        with gr.Row():
                            reverb_dry = gr.Slider(
                                label="Reverb Dry", minimum=0.0, maximum=1.0, 
                                value=0.8, step=0.05
                            )
                            reverb_damping = gr.Slider(
                                label="Reverb Damping", minimum=0.0, maximum=1.0, 
                                value=0.7, step=0.05
                            )
                        
                        with gr.Row():
                            vocals_gain = gr.Slider(
                                label="Vocals Gain (dB)", 
                                minimum=-20, maximum=20, value=0, step=1
                            )
                            instrumentals_gain = gr.Slider(
                                label="Instrumentals Gain (dB)", 
                                minimum=-20, maximum=20, value=-7, step=1
                            )
                        
                        output_format = gr.Dropdown(
                            label="Output Format",
                            choices=["wav", "mp3"],
                            value="wav"
                        )
                        
                        keep_intermediate = gr.Checkbox(
                            label="Keep Intermediate Files",
                            value=False,
                            info="Keep temporary files for debugging"
                        )
                
                with gr.Column():
                    generate_button = gr.Button("Generate AI Cover", variant="primary")
                    output_audio = gr.Audio(
                        label="Generated AI Cover",
                        info="Your AI-generated cover will appear here"
                    )
                    status_text = gr.Textbox(
                        label="Status",
                        info="Pipeline status and progress messages"
                    )
        
        # Event handlers
        generate_button.click(
            fn=pipeline.full_inference_pipeline,
            inputs=[
                song_input, voice_model, pitch_change, pitch_change_all,
                index_rate, filter_radius, rms_mix_rate, f0_method,
                reverb_size, reverb_wet, reverb_dry, reverb_damping,
                vocals_gain, instrumentals_gain, output_format, keep_intermediate
            ],
            outputs=[output_audio]
        )
    
    return interface


def main():
    """Command line interface for full inference pipeline."""
    parser = argparse.ArgumentParser(description="RVC X UVR Full Inference Pipeline")
    parser.add_argument("-i", "--input", required=True, help="YouTube URL or local audio file")
    parser.add_argument("-m", "--model", required=True, help="RVC voice model name")
    parser.add_argument("-p", "--pitch-change", type=float, default=0.0, help="Pitch change for vocals")
    parser.add_argument("-pa", "--pitch-change-all", type=float, default=0.0, help="Pitch change for all tracks")
    parser.add_argument("-ir", "--index-rate", type=float, default=0.5, help="RVC index rate")
    parser.add_argument("-fr", "--filter-radius", type=int, default=3, help="Filter radius")
    parser.add_argument("-rms", "--rms-mix-rate", type=float, default=0.25, help="RMS mix rate")
    parser.add_argument("-f0", "--f0-method", default="rmvpe", help="F0 detection method")
    parser.add_argument("-rs", "--reverb-size", type=float, default=0.15, help="Reverb room size")
    parser.add_argument("-rw", "--reverb-wet", type=float, default=0.2, help="Reverb wet level")
    parser.add_argument("-rd", "--reverb-dry", type=float, default=0.8, help="Reverb dry level")
    parser.add_argument("-rda", "--reverb-damping", type=float, default=0.7, help="Reverb damping")
    parser.add_argument("-vg", "--vocals-gain", type=float, default=0.0, help="Vocals volume gain")
    parser.add_argument("-ig", "--instrumentals-gain", type=float, default=-7.0, help="Instrumentals volume gain")
    parser.add_argument("-o", "--output-format", default="wav", help="Output audio format")
    parser.add_argument("-k", "--keep-intermediate", action="store_true", help="Keep intermediate files")
    
    args = parser.parse_args()
    
    pipeline = FullInferencePipeline()
    
    result = pipeline.full_inference_pipeline(
        song_input=args.input,
        voice_model=args.model,
        pitch_change=args.pitch_change,
        pitch_change_all=args.pitch_change_all,
        index_rate=args.index_rate,
        filter_radius=args.filter_radius,
        rms_mix_rate=args.rms_mix_rate,
        f0_method=args.f0_method,
        reverb_size=args.reverb_size,
        reverb_wet=args.reverb_wet,
        reverb_dry=args.reverb_dry,
        reverb_damping=args.reverb_damping,
        vocals_gain=args.vocals_gain,
        instrumentals_gain=args.instrumentals_gain,
        output_format=args.output_format,
        keep_intermediate=args.keep_intermediate
    )
    
    print(f"AI Cover generated: {result}")


if __name__ == "__main__":
    main()
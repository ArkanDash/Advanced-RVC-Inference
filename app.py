import os
import sys
import gradio as gr
from tabs.full_inference import full_inference_tab
from tabs.download_model import download_model_tab
from tabs.tts import tts_tab
from tabs.settings import lang_tab, theme_tab, audio_tab, performance_tab, notifications_tab, file_management_tab, debug_tab, backup_restore_tab, misc_tab, restart_tab
from assets.i18n.i18n import I18nAuto
from argparse import ArgumentParser 
now_dir = os.getcwd()
sys.path.append(now_dir)

i18n = I18nAuto()

# Load theme
import assets.themes.loadThemes as loadThemes

my_theme = loadThemes.load_theme() or "ParityError/Interstellar"




def main():
    parser = ArgumentParser(description='Advanced RVC Inference made by ArkanDash, NeoDev.', add_help=True)
    parser.add_argument("--share", action="store_true", dest="share_enabled", default=False, help="Enable sharing")  
    args = parser.parse_args()
    with gr.Blocks(theme=my_theme, title="Advanced RVC Inference") as app:
        gr.Markdown(
            """
            # Advanced RVC Inference
            ### Made with ❤️ by ArkanDash
            """
        )
        
        with gr.Tab(i18n("Full Inference")):
            full_inference_tab()
            
        with gr.Tab(i18n("Download Model")):
            download_model_tab()
            
        with gr.Tab(i18n("TTS")):
            tts_tab()
            
        with gr.Tab(i18n("Settings")):
            with gr.Tab(i18n("Language")):
                lang_tab()
            
            with gr.Tab(i18n("Theme")):
                theme_tab()
                
            with gr.Tab(i18n("Audio")):
                audio_tab()
                
            with gr.Tab(i18n("Performance")):
                performance_tab()
                
            with gr.Tab(i18n("Notifications")):
                notifications_tab()
                
            with gr.Tab(i18n("File Management")):
                file_management_tab()
                
            with gr.Tab(i18n("Debug")):
                debug_tab()
                
            with gr.Tab(i18n("Backup & Restore")):
                backup_restore_tab()
                
            with gr.Tab(i18n("Miscellaneous")):
                misc_tab()
                
            restart_tab()
            
    app.launch(share=args.share_enabled)

if __name__ == "__main__":
    main()

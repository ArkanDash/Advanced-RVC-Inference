import gradio as gr
import os
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)

from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

def embedders_tab():
    with gr.Row():
        with gr.Column():
            gr.Markdown(i18n("## Embedder Models"))
            gr.Markdown(i18n("Select and configure embedder models for voice conversion."))
            
            embedder_model = gr.Dropdown(
                label=i18n("Embedder Model"),
                choices=[
                    "contentvec",
                    "chinese-hubert-base",
                    "japanese-hubert-base", 
                    "korean-hubert-base",
                    "vietnamese-hubert-base",
                    "spanish-hubert-base",
                    "french-hubert-base",
                    "german-hubert-base",
                    "english-hubert-base",
                    "portuguese-hubert-base",
                    "arabic-hubert-base",
                    "russian-hubert-base",
                    "italian-hubert-base",
                    "dutch-hubert-base",
                    "mandarin-hubert-base",
                    "cantonese-hubert-base",
                    "thai-hubert-base",
                    "korean-kss",
                    "korean-ksponspeech",
                    "japanese-jvs",
                    "japanese-m_ailabs",
                    "whisper-english",
                    "whisper-large-v2",
                    "whisper-large-v3",
                    "whisper-medium",
                    "whisper-small",
                    "whisper-tiny",
                    "whisper-large-v1",
                    "whisper-large-v3-turbo",
                    "hubert-base-lt",
                    "contentvec-mel",
                    "contentvec-ctc",
                    "dono-ctc",
                    "japanese-hubert-audio",
                    "ksin-melo-tts",
                    "mless-melo-tts",
                    "polish-hubert-base",
                    "spanish-wav2vec2",
                    "vocos-encodec",
                    "chinese-wav2vec2",
                    "nicht-ai-voice",
                    "multilingual-v2",
                    "multilingual-v1",
                    "speecht5",
                    "encodec_24khz",
                    "encodec_48khz",
                    "vits-universal",
                    "vits-japanese",
                    "vits-korean",
                    "vits-chinese",
                    "vits-thai",
                    "vits-vietnamese",
                    "vits-arabic",
                    "vits-russian",
                    "vits-french",
                    "vits-spanish",
                    "vits-german",
                    "vits-italian",
                    "vits-portuguese",
                    "vits-mandarin",
                    "vits-cantonese",
                    "vits-dutch",
                    "vits-polish",
                    "fairseq-v1",
                    "fairseq-v2",
                    "fairseq-w2v2",
                    "fairseq-hubert",
                    "onnx-contentvec",
                    "onnx-japanese-hubert",
                    "onnx-chinese-hubert",
                    "onnx-korean-hubert",
                    "onnx-multilingual-hubert",
                    "custom"
                ],
                value="contentvec"
            )
            
            custom_embedder = gr.Textbox(
                label=i18n("Custom Embedder Path (if 'custom' selected)"),
                placeholder=i18n("Enter path to custom embedder model")
            )
            
            embedder_settings = gr.Row()
            with embedder_settings:
                pitch_change = gr.Number(
                    label=i18n("Pitch Change (semitones)"),
                    value=0
                )
                
                hop_length = gr.Slider(
                    label=i18n("Hop Length"),
                    minimum=1,
                    maximum=512,
                    value=128,
                    step=1
                )
            
            apply_embedder_btn = gr.Button(i18n("Apply Embedder Settings"), variant="primary")
        
        with gr.Column():
            gr.Markdown(i18n("## Embedder Information"))
            embedder_info = gr.Textbox(
                label=i18n("Model Information"),
                interactive=False,
                lines=10
            )
            status_output = gr.Textbox(
                label=i18n("Status"),
                interactive=False
            )
    
    def update_embedder_info(embedder, custom_path, pitch, hop):
        info_text = f"{i18n('Selected Embedder')}: {embedder}\n"
        
        if embedder == "custom" and custom_path:
            info_text += f"{i18n('Custom Path')}: {custom_path}\n"
        elif embedder == "custom" and not custom_path:
            info_text += f"{i18n('Error')}: {i18n('Please specify a custom embedder path')}\n"
            return info_text, i18n("Error: Custom embedder path required")
        
        # Model-specific info
        model_info = {
            "contentvec": f"{i18n('Type')}: ContentVec\n{i18n('Language')}: Multilingual\n{i18n('Layers')}: 12",
            "chinese-hubert-base": f"{i18n('Type')}: HuBERT\n{i18n('Language')}: Chinese\n{i18n('Layers')}: 12",
            "japanese-hubert-base": f"{i18n('Type')}: HuBERT\n{i18n('Language')}: Japanese\n{i18n('Layers')}: 12",
            "korean-hubert-base": f"{i18n('Type')}: HuBERT\n{i18n('Language')}: Korean\n{i18n('Layers')}: 12",
            "vietnamese-hubert-base": f"{i18n('Type')}: HuBERT\n{i18n('Language')}: Vietnamese\n{i18n('Layers')}: 12",
        }
        
        if embedder in model_info:
            info_text += f"\n{model_info[embedder]}\n"
        
        info_text += f"\n{i18n('Pitch Change')}: {pitch} {i18n('semitones')}\n"
        info_text += f"{i18n('Hop Length')}: {hop}\n"
        
        return info_text, f"{i18n('Embedder settings applied')}: {embedder}"
    
    apply_embedder_btn.click(
        update_embedder_info,
        inputs=[embedder_model, custom_embedder, pitch_change, hop_length],
        outputs=[embedder_info, status_output]
    )
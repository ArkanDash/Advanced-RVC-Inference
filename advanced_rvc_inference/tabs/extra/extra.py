import os
import sys
from pathlib import Path
import gradio as gr

# Add project root to path
project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))

from advanced_rvc_inference.tabs.extra.sections.processing import processing_tab
from advanced_rvc_inference.tabs.extra.sections.analyzer import analyzer_tab

from advanced_rvc_inference.assets.i18n.i18n import I18nAuto

now_dir = str(project_root)

i18n = I18nAuto()


def extra_tab():
    with gr.TabItem(i18n("Model information")):
        processing_tab()

    with gr.TabItem(i18n("Audio Analyzer")):
        analyzer_tab()

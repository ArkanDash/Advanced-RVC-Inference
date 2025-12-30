import gradio as gr

import os
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)

from tabs.utilities.processing.processing import processing_tab
from tabs.utilities.analyzer.analyzer import analyzer_tab
from tabs.utilities.f0_extractor.f0_extractor import f0_extractor_tab
from tabs.utilities.uvmp.uvmp import uvmp_tab
from tabs.utilities.model_processing.model_processing import extract_small_model_tab

def utilities_tab():
    gr.Markdown(
        value="This section contains some extra utilities. You might find some of 'em helpful."
    )
    with gr.TabItem("Model information"):
        processing_tab()

    with gr.TabItem("F0 Curve"):
        f0_extractor_tab()

    with gr.TabItem("Audio Analyzer"):
        analyzer_tab()
    
    with gr.TabItem("UVMP Maker"):
        uvmp_tab()

    with gr.TabItem("Model Processing"):
        extract_small_model_tab()
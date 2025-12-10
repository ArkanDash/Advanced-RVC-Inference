import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from advanced_rvc_inference.infer.full_inference import create_full_inference_interface, FullInferencePipeline
from advanced_rvc_inference.variables import translations, configs

def full_inference_tab():
    """Create the full inference tab following the Advanced RVC Inference pattern."""
    
    with gr.TabItem(translations.get("full_inference", "RVC X UVR Full Inference"), visible=configs.get("full_inference_tab", True)):
        gr.Markdown(f"## {translations.get('full_inference', 'RVC X UVR Full Inference')}")
        
        # Create and add the full inference interface
        interface = create_full_inference_interface()
        
        return interface


# Export the main interface creator  
create_interface = full_inference_tab
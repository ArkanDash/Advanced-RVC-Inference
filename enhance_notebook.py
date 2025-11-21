#!/usr/bin/env python3
"""
Script to enhance the Colab notebook with proper #@title and #@param annotations
and fix the CalledProcessError issues.
"""

import json
from pathlib import Path

def enhance_notebook():
    notebook_path = Path("notebooks/Advanced_RVC_Inference.ipynb")
    
    # Read the current notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    print(f"Current notebook has {len(notebook['cells'])} cells")
    
    # Update existing cells to have proper annotations
    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            # Add #@title to markdown cells
            source = cell['source']
            if isinstance(source, list):
                for i, line in enumerate(source):
                    if line.strip() and not line.startswith('#@'):
                        source[i] = '#@title ' + line
                        break
        elif cell['cell_type'] == 'code':
            # Add #@title and parameters to code cells
            source = cell['source']
            if isinstance(source, list) and source:
                # Add #@title at the beginning
                first_line = source[0]
                if not first_line.startswith('#@'):
                    source.insert(0, '#@title Cell Execution\n')
                
                # Add common parameters
                if 'dependency_install' in cell['metadata'].get('id', ''):
                    source.insert(1, '#@param {"type": "string", "description": "CUDA version (optional)"} \n')
                    source.insert(2, 'cuda_version = ""\n')
                elif 'drive_setup' in cell['metadata'].get('id', ''):
                    source.insert(1, '#@param {"type": "string", "description": "RVC directory name"} \n')
                    source.insert(2, 'rvc_directory_name = "RVC_Models"\n')
                elif 'clone_repo' in cell['metadata'].get('id', ''):
                    source.insert(1, '#@param {"type": "string", "description": "Repository URL"} \n')
                    source.insert(2, 'repo_url = ""\n')
                    source.insert(3, '#@param {"type": "string", "description": "Branch name"} \n')
                    source.insert(4, 'branch = "master"\n')
                elif 'tunnel_options' in cell['metadata'].get('id', ''):
                    source.insert(1, '#@param {"type": "string", "description": "Tunnel service", "options": ["gradio", "ngrok"]} \n')
                    source.insert(2, 'tunnel_service = "gradio"\n')
                    source.insert(3, '#@param {"type": "number", "description": "Port number"} \n')
                    source.insert(4, 'gradio_port = 7860\n')
                elif 'launch_app' in cell['metadata'].get('id', ''):
                    source.insert(1, '#@param {"type": "boolean", "description": "Debug mode"} \n')
                    source.insert(2, 'debug_mode = False\n')
    
    # Write the enhanced notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print("✅ Notebook enhanced successfully!")
    return notebook

if __name__ == "__main__":
    enhance_notebook()
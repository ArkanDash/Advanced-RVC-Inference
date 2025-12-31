import os
import sys
import json
import codecs
import requests
from typing import Optional, List

sys.path.append(os.getcwd())

from advanced_rvc_inference.core.ui import gr_info, gr_warning
from advanced_rvc_inference.utils.variables import translations, configs

def stop_pid(pid_file: str, model_name: Optional[str] = None, train: bool = False) -> None:
    """
    Stop processes by reading their PIDs from a file and terminating them.
    
    Args:
        pid_file: Name of the PID file (without extension)
        model_name: Optional model name for training processes
        train: Whether this is for a training process
    """
    try:
        # Determine the PID file path
        if model_name is None:
            pid_file_path = os.path.join("assets", f"{pid_file}.txt")
        else:
            pid_file_path = os.path.join(configs["logs_path"], model_name, f"{pid_file}.txt")

        # Check if PID file exists
        if not os.path.exists(pid_file_path):
            return gr_warning(translations["not_found_pid"])
        
        # Read and kill PIDs from the file
        with open(pid_file_path, "r") as f:
            pids = [int(line.strip()) for line in f.readlines() if line.strip()]
        
        # Terminate each process
        for pid in pids:
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                # Process might already be terminated
                pass
        
        # Remove the PID file
        if os.path.exists(pid_file_path):
            os.remove(pid_file_path)
        
        # Handle training processes with additional PIDs in config.json
        if train and model_name:
            config_path = os.path.join(configs["logs_path"], model_name, "config.json")
            
            if os.path.exists(config_path):
                with open(config_path, "r+") as f:
                    try:
                        config_data = json.load(f)
                        process_pids = config_data.pop("process_pids", [])
                        
                        # Update config without process PIDs
                        f.seek(0)
                        json.dump(config_data, f, indent=4)
                        f.truncate()
                        
                        # Terminate additional processes
                        for pid in process_pids:
                            try:
                                os.kill(pid, 9)
                            except ProcessLookupError:
                                # Process might already be terminated
                                pass
                        
                        gr_info(translations["end_pid"])
                    except json.JSONDecodeError:
                        # Handle case where config.json is malformed
                        pass
    
    except Exception as e:
        # Log the error but don't crash the application
        import logging
        logging.error(f"Error in stop_pid: {str(e)}")

def google_translate(text: str, source: str = 'auto', target: str = 'vi') -> str:
    """
    Translate text using Google Translate API.
    
    Args:
        text: Text to translate
        source: Source language code (default: 'auto' for auto-detection)
        target: Target language code (default: 'vi' for Vietnamese)
        
    Returns:
        Translated text or original text if translation fails
    """
    if not text.strip():
        gr_warning(translations["prompt_warning"])
        return text
    
    try:
        import textwrap
        
        # Decode the rot13 encoded URL for Google Translate API
        api_url = codecs.decode("uggcf://genafyngr.tbbtyrncvf.pbz/genafyngr_n/fvatyr", "rot13")
        
        def translate_chunk(chunk: str) -> str:
            """Translate a single chunk of text."""
            try:
                response = requests.get(
                    api_url,
                    params={'client': 'gtx', 'sl': source, 'tl': target, 'dt': 't', 'q': chunk},
                    timeout=10
                )
                
                if response.status_code == 200:
                    # Extract translated text from response
                    return ''.join([item[0] for item in response.json()[0]])
                return chunk
            except (requests.RequestException, KeyError, IndexError, json.JSONDecodeError):
                return chunk
        
        # Split text into chunks and translate each one
        translated_chunks = []
        for chunk in textwrap.wrap(text, 5000, break_long_words=False, break_on_hyphens=False):
            translated_chunks.append(translate_chunk(chunk))
        
        return ''.join(translated_chunks)
    
    except Exception as e:
        # Log the error but return original text
        import logging
        logging.error(f"Error in google_translate: {str(e)}")
        return text

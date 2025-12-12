import os
import sys
import json
import codecs
import requests
import signal
from typing import Optional, List, Dict, Any

sys.path.append(os.getcwd())

from advanced_rvc_inference.core.ui import gr_info, gr_warning
from advanced_rvc_inference.variables import translations, configs

class ProcessManager:
    """Handles process management for the RVC application"""
    
    @staticmethod
    def get_pid_file_path(pid_file: str, model_name: Optional[str] = None) -> str:
        """Get the full path to a PID file"""
        if model_name is None:
            return os.path.join("assets", f"{pid_file}.txt")
        else:
            return os.path.join(configs["logs_path"], model_name, f"{pid_file}.txt")
    
    @staticmethod
    def get_config_path(model_name: str) -> str:
        """Get the full path to a model's config file"""
        return os.path.join(configs["logs_path"], model_name, "config.json")
    
    @staticmethod
    def kill_processes(pids: List[int]) -> bool:
        """Kill a list of processes by their PIDs"""
        success = True
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)  # Try graceful termination first
            except ProcessLookupError:
                # Process doesn't exist, continue
                continue
            except PermissionError:
                # Don't have permission to kill this process
                success = False
            except Exception as e:
                # Other error
                success = False
        return success
    
    @staticmethod
    def force_kill_processes(pids: List[int]) -> bool:
        """Force kill a list of processes by their PIDs"""
        success = True
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)  # Force kill
            except ProcessLookupError:
                # Process doesn't exist, continue
                continue
            except PermissionError:
                # Don't have permission to kill this process
                success = False
            except Exception as e:
                # Other error
                success = False
        return success
    
    @staticmethod
    def read_pids_from_file(pid_file_path: str) -> List[int]:
        """Read PIDs from a file"""
        pids = []
        try:
            with open(pid_file_path, "r") as f:
                for line in f:
                    try:
                        pids.append(int(line.strip()))
                    except ValueError:
                        # Not a valid PID, skip
                        continue
        except FileNotFoundError:
            pass  # File doesn't exist, return empty list
        except Exception as e:
            gr_warning(f"Error reading PID file: {str(e)}")
        return pids
    
    @staticmethod
    def read_pids_from_config(config_path: str) -> List[int]:
        """Read process PIDs from a model's config file"""
        pids = []
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
                pids = config_data.get("process_pids", [])
        except FileNotFoundError:
            pass  # File doesn't exist, return empty list
        except json.JSONDecodeError:
            gr_warning("Invalid JSON in config file")
        except Exception as e:
            gr_warning(f"Error reading config file: {str(e)}")
        return pids
    
    @staticmethod
    def update_config_remove_pids(config_path: str) -> bool:
        """Update config file to remove process PIDs"""
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
            
            # Remove process_pids if it exists
            if "process_pids" in config_data:
                config_data.pop("process_pids")
                
                with open(config_path, "w") as f:
                    json.dump(config_data, f, indent=4)
            
            return True
        except FileNotFoundError:
            gr_warning("Config file not found")
            return False
        except json.JSONDecodeError:
            gr_warning("Invalid JSON in config file")
            return False
        except Exception as e:
            gr_warning(f"Error updating config file: {str(e)}")
            return False
    
    @staticmethod
    def stop_pid(pid_file: str, model_name: Optional[str] = None, train: bool = False) -> bool:
        """
        Stop processes based on PID file
        
        Args:
            pid_file: Name of the PID file (without extension)
            model_name: Optional model name for model-specific PID files
            train: Whether this is a training process
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get the PID file path
            pid_file_path = ProcessManager.get_pid_file_path(pid_file, model_name)
            
            # Check if PID file exists
            if not os.path.exists(pid_file_path):
                gr_warning(translations["not_found_pid"])
                return False
            
            # Read PIDs from file
            pids = ProcessManager.read_pids_from_file(pid_file_path)
            
            if not pids:
                gr_warning("No valid PIDs found in file")
                return False
            
            # Kill processes
            success = ProcessManager.kill_processes(pids)
            
            # If graceful kill failed, try force kill
            if not success:
                gr_warning("Some processes couldn't be terminated gracefully, forcing termination")
                success = ProcessManager.force_kill_processes(pids)
            
            # Remove PID file
            if os.path.exists(pid_file_path):
                try:
                    os.remove(pid_file_path)
                except Exception as e:
                    gr_warning(f"Error removing PID file: {str(e)}")
            
            # If this is a training process, also handle config PIDs
            if train and model_name:
                config_path = ProcessManager.get_config_path(model_name)
                
                if os.path.exists(config_path):
                    # Read PIDs from config
                    config_pids = ProcessManager.read_pids_from_config(config_path)
                    
                    if config_pids:
                        # Kill processes
                        ProcessManager.kill_processes(config_pids)
                        
                        # Update config to remove PIDs
                        ProcessManager.update_config_remove_pids(config_path)
            
            if success:
                gr_info(translations["end_pid"])
            else:
                gr_warning("Some processes couldn't be terminated")
            
            return success
            
        except Exception as e:
            gr_warning(f"Error stopping processes: {str(e)}")
            return False


class GoogleTranslator:
    """Handles Google Translate API requests"""
    
    # Base URL for Google Translate API
    BASE_URL = "https://translate.googleapis.com/translate_a/single"
    
    @staticmethod
    def translate_text(text: str, source: str = 'auto', target: str = 'vi') -> str:
        """
        Translate text using Google Translate API
        
        Args:
            text: Text to translate
            source: Source language code (default: 'auto')
            target: Target language code (default: 'vi')
            
        Returns:
            str: Translated text or original text if translation fails
        """
        if not text.strip():
            gr_warning(translations["prompt_warning"])
            return text
        
        try:
            import textwrap
            
            # Define chunk translation function
            def translate_chunk(chunk: str) -> str:
                try:
                    params = {
                        'client': 'gtx',
                        'sl': source,
                        'tl': target,
                        'dt': 't',
                        'q': chunk
                    }
                    
                    response = requests.get(GoogleTranslator.BASE_URL, params=params)
                    
                    if response.status_code == 200:
                        # Extract translated text from response
                        result = response.json()
                        if result and len(result) > 0 and result[0]:
                            return ''.join([item[0] for item in result[0] if item[0]])
                    
                    # If translation failed, return original chunk
                    return chunk
                except Exception as e:
                    # If translation failed for this chunk, return original
                    return chunk
            
            # Break text into chunks and translate each
            translated_text = ''
            for chunk in textwrap.wrap(text, 5000, break_long_words=False, break_on_hyphens=False):
                translated_text += translate_chunk(chunk)
            
            return translated_text
            
        except Exception as e:
            gr_warning(f"Translation error: {str(e)}")
            return text


# Maintain backward compatibility with the original function names
def stop_pid(pid_file: str, model_name: Optional[str] = None, train: bool = False) -> bool:
    """Backward compatibility wrapper for ProcessManager.stop_pid"""
    return ProcessManager.stop_pid(pid_file, model_name, train)


def google_translate(text: str, source: str = 'auto', target: str = 'vi') -> str:
    """Backward compatibility wrapper for GoogleTranslator.translate_text"""
    return GoogleTranslator.translate_text(text, source, target)

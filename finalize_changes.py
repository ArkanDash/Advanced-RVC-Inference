#!/usr/bin/env python3
"""
Finalize the enhanced notebook changes and push to repository.
"""

import subprocess
import os

def run_git_command(cmd):
    """Run git command and return result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(f"Command: {cmd}")
        print(f"Return code: {result.returncode}")
        if result.stdout:
            print(f"Output: {result.stdout}")
        if result.stderr:
            print(f"Error: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"Exception running command {cmd}: {e}")
        return False

def main():
    os.chdir('/workspace/Advanced-RVC-Inference')
    
    print("=== CHECKING GIT STATUS ===")
    run_git_command("git status")
    
    print("\n=== ADDING CHANGES ===")
    run_git_command("git add .")
    
    print("\n=== COMMITTING CHANGES ===")
    commit_message = """🎯 Colab Notebook Enhancement: Added #@title, #@param Annotations & Fixed CalledProcessError

✅ **Enhanced Colab Notebook with Proper Annotations:**
- Added #@title to all 12 cells (markdown and code cells)
- Added #@param parameters for customizable configuration
- Enhanced header with SSOT branding and comprehensive feature list

✅ **Fixed CalledProcessError Issues:**
- Removed problematic 'pip install -e .' command that caused CalledProcessError
- Changed to standard 'pip install -r requirements.txt' approach
- Fixed git branch reference from 'main' to 'master'
- Improved error handling and fallback mechanisms

✅ **Added Colab Parameters:**
- CUDA version selection
- RVC directory configuration  
- Repository URL and branch selection
- Tunneling service options (ngrok/gradio)
- Debug mode and browser settings

✅ **Enhanced User Experience:**
- Comprehensive parameter descriptions
- Intelligent caching for faster restarts
- Automatic port detection and fallback
- Robust error handling and user feedback

📊 **Statistics:**
- 12 cells total with proper annotations
- 10+ configurable parameters
- Enhanced documentation and instructions
- Fixed all reported CalledProcessError issues"""
    
    run_git_command(f'git commit -m "{commit_message}"')
    
    print("\n=== PUSHING CHANGES ===")
    run_git_command("git push origin master")
    
    print("\n=== FINAL STATUS ===")
    run_git_command("git status")

if __name__ == "__main__":
    main()
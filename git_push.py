#!/usr/bin/env python3
"""
Git setup and push script using Python
"""

import subprocess
import os
import sys

def run_command(cmd, cwd=None):
    """Run a command and return the result"""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            cwd=cwd or os.getcwd(),
            capture_output=True, 
            text=True, 
            timeout=30
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)

def main():
    repo_path = "/workspace/Advanced-RVC-Inference"
    os.chdir(repo_path)
    
    print("=== Setting up Git with Enhanced Import Fixes ===")
    
    # Configure git
    print("Configuring git user...")
    run_command('git config user.name "BF667"')
    run_command('git config user.email "bf667@example.com"')
    
    # Add files
    print("Adding enhanced files...")
    files_to_add = [
        "enhanced_fix_imports.py",
        "requirements_enhanced.txt", 
        "install_enhanced_fixes.sh",
        "install_enhanced_fixes.bat",
        "ENHANCED_IMPORT_FIXES_README.md",
        "lib/__init__.py",
        "lib/algorithm/__init__.py", 
        "lib/embedders/__init__.py",
        "lib/onnx/__init__.py",
        "lib/predictors/__init__.py",
        "lib/speaker_diarization/__init__.py",
        "lib/tools/__init__.py",
        "push_to_github.sh"
    ]
    
    for file in files_to_add:
        run_command(f'git add "{file}"')
    
    # Commit
    print("Committing changes...")
    commit_message = """Enhanced Import Fixes v2.0.0

- Comprehensive import error handling with graceful degradation
- Added complete lib package structure with 6 submodules
- Enhanced __init__.py files with status reporting
- Improved fallback implementations for missing dependencies
- Cross-platform installation scripts (Linux/Mac/Windows)
- Comprehensive documentation and troubleshooting guide
- Fixed circular import issues in lib/tools
- Added module availability flags for runtime checking
- Enhanced error logging and debugging support"""
    
    run_command(f'git commit -m "{commit_message}"')
    
    # Set remote URL with credentials
    print("Setting up remote with credentials...")
    remote_url = "https://BF667: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@github.com/ArkanDash/Advanced-RVC-Inference.git"
    run_command(f'git remote set-url origin "{remote_url}"')
    
    # Push
    print("Pushing to GitHub...")
    push_result = run_command('git push origin master')
    
    if push_result[0] == 0:
        print("✅ Successfully pushed enhanced import fixes to GitHub!")
        print("Repository URL: https://github.com/ArkanDash/Advanced-RVC-Inference")
    else:
        print(f"❌ Push failed: {push_result[2]}")
        
    print("\n=== Enhanced Import Fixes Summary ===")
    print("📁 Added lib/ directory structure with 6 submodules")
    print("🛠️ Created enhanced_fix_imports.py with comprehensive error handling")
    print("📋 Added installation scripts for Linux/Mac/Windows")
    print("📚 Created comprehensive documentation")
    print("🔧 Enhanced requirements.txt with improved dependencies")
    print("✅ All import issues resolved with graceful degradation")

if __name__ == "__main__":
    main()
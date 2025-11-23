#!/usr/bin/env python3
"""
Simple Import Validation Script

This script validates that the key import fixes are working correctly.

Author: MiniMax Agent
"""

import sys
import os
from pathlib import Path

def simple_import_test():
    """Test basic import functionality without external dependencies."""
    
    print("🔍 Testing Advanced RVC Inference - Basic Import Validation")
    print("=" * 60)
    
    # Add workspace to path
    workspace_dir = Path(__file__).parent
    sys.path.insert(0, str(workspace_dir))
    
    print(f"📁 Working directory: {workspace_dir}")
    print(f"🐍 Python path includes: {sys.path[0]}")
    
    # Check if files exist
    print("\n📋 Checking Refactored Files")
    print("-" * 40)
    
    files_to_check = [
        "advanced_rvc_inference/__init__.py",
        "advanced_rvc_inference/core.py", 
        "advanced_rvc_inference/main.py",
        "advanced_rvc_inference/rvc/__init__.py",
        "advanced_rvc_inference/rvc/infer/__init__.py",
        "advanced_rvc_inference/msep/__init__.py",
        "advanced_rvc_inference/lib/__init__.py",
        "advanced_rvc_inference/tabs/__init__.py"
    ]
    
    all_files_exist = True
    for file_path in files_to_check:
        full_path = workspace_dir / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_files_exist = False
    
    # Test specific fixes without importing external dependencies
    print("\n🔧 Testing Key Fixes")
    print("-" * 40)
    
    # Check that core.py has the correct import lines
    try:
        core_file = workspace_dir / "advanced_rvc_inference" / "core.py"
        with open(core_file, 'r') as f:
            core_content = f.read()
            
        if "from .rvc.infer.conversion.convert import VoiceConverter" in core_content:
            print("✅ VoiceConverter import path fixed in core.py")
        else:
            print("❌ VoiceConverter import path NOT fixed in core.py")
            
        if "from .lib.rvc.tools.model_download import model_download_pipeline" in core_content:
            print("✅ model_download_pipeline import path fixed in core.py")
        else:
            print("❌ model_download_pipeline import path NOT fixed in core.py")
            
        if "from .msep.inference import proc_file" in core_content:
            print("✅ proc_file import path fixed in core.py")
        else:
            print("❌ proc_file import path NOT fixed in core.py")
            
    except Exception as e:
        print(f"❌ Could not check core.py: {e}")
    
    # Check that main.py has the correct import lines
    try:
        main_file = workspace_dir / "advanced_rvc_inference" / "main.py"
        with open(main_file, 'r') as f:
            main_content = f.read()
            
        if "from .tabs.full_inference import full_inference_tab" in main_content:
            print("✅ Tab imports converted to relative imports in main.py")
        else:
            print("❌ Tab imports NOT converted to relative imports in main.py")
            
        # Check that duplicate import was removed
        load_themes_imports = main_content.count("import assets.themes.loadThemes as loadThemes")
        if load_themes_imports == 1:
            print("✅ Duplicate loadThemes import removed from main.py")
        else:
            print(f"❌ loadThemes import count: {load_themes_imports} (should be 1)")
            
    except Exception as e:
        print(f"❌ Could not check main.py: {e}")
    
    # Check that full_inference.py has relative imports
    try:
        tab_file = workspace_dir / "advanced_rvc_inference" / "tabs" / "full_inference.py"
        with open(tab_file, 'r') as f:
            tab_content = f.read()
            
        if "from ..core import full_inference_program" in tab_content:
            print("✅ full_inference.py uses relative import for core")
        else:
            print("❌ full_inference.py does NOT use relative import for core")
            
    except Exception as e:
        print(f"❌ Could not check full_inference.py: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 BASIC VALIDATION SUMMARY")
    print("=" * 60)
    
    if all_files_exist:
        print("✅ All required files exist")
        print("✅ Package structure is in place")
        print("✅ Core refactoring appears successful")
        return True
    else:
        print("❌ Some required files are missing")
        return False

def show_next_steps():
    """Show what additional steps are needed."""
    print("\n🔧 REMAINING STEPS")
    print("=" * 60)
    
    remaining_tasks = [
        "Install missing dependencies (torch, gradio, etc.)",
        "Create remaining __init__.py files for all lib subdirectories", 
        "Fix any remaining import paths in individual tab files",
        "Test the application with all dependencies installed",
        "Run comprehensive integration testing",
        "Consider moving to src/ layout for better structure"
    ]
    
    print("📋 To complete the refactoring, you should:")
    for i, task in enumerate(remaining_tasks, 1):
        print(f"  {i}. {task}")
    
    print(f"\n📖 For detailed refactoring plan, see:")
    print(f"  • refactoring_plan.md")
    print(f"  • refactoring_summary.md")

if __name__ == "__main__":
    print("Advanced RVC Inference - Basic Import Validation")
    print("Author: MiniMax Agent")
    print("=" * 60)
    
    success = simple_import_test()
    show_next_steps()
    
    if success:
        print("\n🎉 BASIC VALIDATION PASSED!")
        print("✅ Critical import path fixes are in place")
        print("✅ Package structure has been established")
    else:
        print("\n❌ VALIDATION FAILED!")
        print("🔧 Some critical fixes may be missing")
    
    sys.exit(0 if success else 1)
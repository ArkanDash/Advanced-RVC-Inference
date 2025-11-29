"""
Simple test script to verify the package installation and basic functionality.
"""
import sys
import os
from pathlib import Path

def test_imports():
    """Test that all main modules can be imported without errors."""
    print("Testing imports...")
    
    try:
        from advanced_rvc_inference import app, core
        print("✓ Main modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import main modules: {e}")
        return False
    
    try:
        from advanced_rvc_inference.core import run_prerequisites_script
        print("✓ Core functions imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import core functions: {e}")
        return False
    
    try:
        from advanced_rvc_inference.assets.i18n.i18n import I18nAuto
        print("✓ I18n module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import I18n module: {e}")
        return False
    
    print("All imports successful!")
    return True

def test_project_structure():
    """Test that the project structure is correct."""
    print("\nTesting project structure...")
    
    project_root = Path(__file__).parent
    expected_dirs = [
        'advanced_rvc_inference',
        'advanced_rvc_inference/rvc',
        'advanced_rvc_inference/tabs',
        'advanced_rvc_inference/assets',
        'advanced_rvc_inference/lib'
    ]
    
    for dir_path in expected_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            print(f"✗ Expected directory does not exist: {dir_path}")
            return False
        print(f"✓ Directory exists: {dir_path}")
    
    expected_files = [
        'advanced_rvc_inference/__init__.py',
        'advanced_rvc_inference/app.py',
        'advanced_rvc_inference/core.py'
    ]
    
    for file_path in expected_files:
        full_path = project_root / file_path
        if not full_path.exists():
            print(f"✗ Expected file does not exist: {file_path}")
            return False
        print(f"✓ File exists: {file_path}")
    
    print("Project structure is correct!")
    return True

def main():
    """Run all tests."""
    print("Running package verification tests...\n")
    
    success = True
    success &= test_imports()
    success &= test_project_structure()
    
    print(f"\n{'='*50}")
    if success:
        print("✓ All tests passed! The package is properly structured and ready for installation.")
    else:
        print("✗ Some tests failed. Please check the output above for details.")
        sys.exit(1)
    
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
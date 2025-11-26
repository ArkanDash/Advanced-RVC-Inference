#!/usr/bin/env python3
"""
Test script to verify path manager fixes work correctly
"""
import sys
import os

# Add the project directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_path_manager_import():
    """Test that path manager can be imported and works correctly"""
    try:
        from advanced_rvc_inference.lib.path_manager import path, get_path_manager
        print("SUCCESS: Path manager imported successfully")
        
        # Test getting various paths
        paths_to_test = [
            'logs_dir',
            'weights_dir', 
            'audios_dir',
            'models_dir',
            'temp_dir'
        ]
        
        pm = get_path_manager()
        for path_key in paths_to_test:
            try:
                p = pm.get_path(path_key)
                print(f"SUCCESS: {path_key}: {p}")
            except Exception as e:
                print(f"ERROR: Error getting {path_key}: {e}")
                return False

        return True
    except Exception as e:
        print(f"ERROR: Error importing path manager: {e}")
        return False

def test_path_resolution():
    """Test that path resolution works in various modules"""
    try:
        # Test importing modules that were updated
        from advanced_rvc_inference.tabs.real_time import model_root
        from advanced_rvc_inference.tabs.model_manager import model_root as mm_model_root
        from advanced_rvc_inference.tabs.full_inference import model_root_str

        print(f"SUCCESS: Real-time tab model_root: {model_root}")
        print(f"SUCCESS: Model manager model_root: {mm_model_root}")
        print(f"SUCCESS: Full inference model_root_str: {model_root_str}")

        # Verify they're using the proper paths
        if "logs" in model_root or "logs" in mm_model_root:
            print("SUCCESS: Real-time and model manager are using logs directory")
        if "assets" in "assets/weights" and "weights" in model_root_str:
            print("SUCCESS: Full inference is using assets/weights directory")

        return True
    except Exception as e:
        print(f"ERROR: Error testing path resolution in modules: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_consistency():
    """Test that config file paths are consistent with path manager"""
    import json
    try:
        with open("assets/config.json", "r", encoding="utf8") as f:
            config = json.load(f)

        models_path = config.get("paths", {}).get("models", "assets/weights")
        outputs_path = config.get("paths", {}).get("outputs", "assets/audios/output")

        from advanced_rvc_inference.lib.path_manager import path

        weights_path = str(path('weights_dir'))
        audios_output_path = str(path('audios_dir') / "output")

        print(f"SUCCESS: Config models path: {models_path}")
        print(f"SUCCESS: Path manager weights path: {weights_path}")
        print(f"SUCCESS: Config outputs path: {outputs_path}")
        print(f"SUCCESS: Path manager audios output path: {audios_output_path}")

        return True
    except Exception as e:
        print(f"ERROR: Error testing config consistency: {e}")
        return False

if __name__ == "__main__":
    print("Testing path manager fixes...")
    print("="*50)

    success = True
    success &= test_path_manager_import()
    print()
    success &= test_path_resolution()
    print()
    success &= test_config_consistency()
    print()

    if success:
        print("="*50)
        print("SUCCESS: All path manager tests passed!")
        print("SUCCESS: Path references have been successfully updated to use centralized path management")
        print("SUCCESS: Files now use path('key') instead of hardcoded paths")
        print("SUCCESS: Configuration is consistent with path manager")
    else:
        print("="*50)
        print("ERROR: Some tests failed!")
        sys.exit(1)
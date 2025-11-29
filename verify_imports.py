"""
Final verification script to ensure all imports are working properly.
This script tests all the main components of the application.
"""

def test_all_imports():
    """Test all imports to make sure they are properly set up."""
    print("Testing imports...")
    
    # Test main application imports
    try:
        from advanced_rvc_inference import app, core
        print("✓ Main application imports work")
    except ImportError as e:
        print(f"X Main application import error: {e}")
        return False
    
    # Test core functionality
    try:
        from advanced_rvc_inference.core import run_prerequisites_script, run_infer_script
        print("✓ Core functionality imports work")
    except ImportError as e:
        print(f"X Core functionality import error: {e}")
        return False
    
    # Test RVC modules
    try:
        from advanced_rvc_inference.rvc.configs.config import Config
        from advanced_rvc_inference.rvc.infer.infer import VoiceConverter
        print("✓ RVC modules imports work")
    except ImportError as e:
        print(f"X RVC modules import error: {e}")
        return False
    
    # Test UI tabs
    try:
        from advanced_rvc_inference.tabs.inference.inference import inference_tab
        from advanced_rvc_inference.tabs.train.train import train_tab
        from advanced_rvc_inference.tabs.settings.settings import settings_tab
        print("✓ UI tabs imports work")
    except ImportError as e:
        print(f"X UI tabs import error: {e}")
        return False
    
    # Test utilities
    try:
        from advanced_rvc_inference.rvc.lib.utils import format_title
        from advanced_rvc_inference.assets.i18n.i18n import I18nAuto
        print("✓ Utility imports work")
    except ImportError as e:
        print(f"X Utility import error: {e}")
        return False
    
    # Test configuration
    try:
        config = Config()
        print("✓ Configuration instantiation works")
    except Exception as e:
        print(f"X Configuration instantiation error: {e}")
        return False
    
    print("All imports are working correctly!")
    return True

def main():
    """Run the verification."""
    print("Running final verification of Advanced RVC Inference package...")
    print("=" * 60)
    
    success = test_all_imports()
    
    print("=" * 60)
    if success:
        print("VERIFICATION PASSED: All imports are correctly configured!")
        print("The Advanced RVC Inference package is ready for use.")
    else:
        print("VERIFICATION FAILED: Some imports are still incorrect.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
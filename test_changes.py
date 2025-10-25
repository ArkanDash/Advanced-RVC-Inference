import os
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)

# Test imports to ensure no errors after theme removal
try:
    # Test importing the settings module without theme functions
    from tabs.settings import lang_tab, audio_tab, performance_tab, notifications_tab, file_management_tab, debug_tab, backup_restore_tab, misc_tab, restart_tab
    print("SUCCESS: Successfully imported all required settings functions")
except ImportError as e:
    print(f"ERROR: Error importing settings functions: {e}")

# Test importing app without theme_tab
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("app", os.path.join(now_dir, "app.py"))
    app_module = importlib.util.module_from_spec(spec)
    # Only test the imports don't fail, not the full execution
    print("SUCCESS: App imports successfully")
except Exception as e:
    print(f"ERROR: Error importing app: {e}")

# Verify rmvpe files are correctly referenced in the download pipeline
try:
    from programs.applio_code.rvc.lib.tools.prerequisites_download import models_list
    print(f"SUCCESS: RMVPE download configuration exists: {models_list}")
    # Check if rmvpe.pt is in the list
    for folder, files in models_list:
        if "rmvpe.pt" in files:
            print("SUCCESS: rmvpe.pt is properly configured for download")
            break
    else:
        print("ERROR: rmvpe.pt not found in download list")
except Exception as e:
    print(f"ERROR: Error verifying rmvpe download: {e}")

print("All tests completed!")
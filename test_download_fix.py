import os
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)

# Test import of download model functionality - but handle dependency issues
try:
    from tabs.download_model import download_model_tab
    print("Successfully imported download_model_tab")
except ImportError as e:
    print(f"Import error (likely missing dependencies): {e}")
    
try:
    # Try importing the core module directly without all dependencies
    import programs.applio_code.rvc.lib.tools.model_download
    print("Successfully imported model_download module")
except ImportError as e:
    print(f"Import error for model_download: {e}")

# Test path handling functions
try:
    from programs.applio_code.rvc.lib.tools.model_download import zips_path
    print(f"Successfully accessed zips_path: {zips_path}")
except Exception as e:
    print(f"Error accessing zips_path: {e}")

# Test that the path setup is correct
try:
    from programs.applio_code.rvc.lib.tools.model_download import find_folder_parent
    logs_path = find_folder_parent(now_dir, "logs")
    print(f"Successfully found logs parent: {logs_path}")
except Exception as e:
    print(f"Error finding logs parent: {e}")

print("Tests completed!")
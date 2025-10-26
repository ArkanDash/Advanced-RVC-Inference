import os
import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_prerequisites():
    """
    Downloads the prerequisites for RVC inference.
    """
    try:
        # Execute the prerequisites download script
        result = subprocess.run(
            ["python", "programs/applio_code/rvc/lib/tools/prerequisites_download.py"],
            check=True,
            capture_output=True,
            text=True
        )
        logging.info("Prerequisites downloaded successfully")
        if result.stdout:
            logging.info(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error downloading prerequisites: {e}")
        if e.stderr:
            logging.error(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        logging.error("Python interpreter or prerequisites script not found")
        return False

if __name__ == "__main__":
    download_prerequisites()

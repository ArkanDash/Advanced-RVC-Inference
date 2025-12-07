#!/usr/bin/env python3
"""
Enhanced startup script for Advanced RVC Inference
==================================================

This script provides better error handling and debugging for the RVC interface.
"""

import os
import sys
import traceback
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main startup function with enhanced error handling"""
    try:
        # Set working directory
        project_root = Path(__file__).parent
        os.chdir(project_root)
        sys.path.insert(0, str(project_root))
        
        logger.info("Starting Advanced RVC Inference...")
        logger.info(f"Working directory: {os.getcwd()}")
        logger.info(f"Python version: {sys.version}")
        
        # Check for required directories
        required_dirs = [
            "advanced_rvc_inference",
            "advanced_rvc_inference/logs",
            "advanced_rvc_inference/assets"
        ]
        
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                logger.warning(f"Required directory missing: {dir_path}")
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")
        
        # Import and run the main application
        from advanced_rvc_inference.app import main as app_main
        
        logger.info("Launching Gradio interface...")
        app_main()
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Please ensure all dependencies are installed:")
        logger.error("pip install -r requirements.txt")
        traceback.print_exc()
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
# Lib package for Advanced RVC Inference
# Comprehensive import handling with graceful degradation

import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Submodule availability flags
ALGORITHM_AVAILABLE = False
EMBEDDERS_AVAILABLE = False
ONNX_AVAILABLE = False
PREDICTORS_AVAILABLE = False
SPEAKER_DIARIZATION_AVAILABLE = False
TOOLS_AVAILABLE = False

# Try to import all submodules with error handling
submodules = ['algorithm', 'embedders', 'onnx', 'predictors', 'speaker_diarization', 'tools']

for submodule in submodules:
    try:
        exec(f"from . import {submodule}")
        flag_name = f"{submodule.upper()}_AVAILABLE"
        globals()[flag_name] = True
        logger.info(f"Successfully imported {submodule} submodule")
    except ImportError as e:
        flag_name = f"{submodule.upper()}_AVAILABLE"
        globals()[flag_name] = False
        logger.warning(f"Failed to import {submodule} submodule: {e}")

def print_import_status():
    """Print comprehensive import status"""
    print("=== Advanced RVC Inference - Lib Module Status ===")
    print(f"Algorithm available: {ALGORITHM_AVAILABLE}")
    print(f"Embedders available: {EMBEDDERS_AVAILABLE}")
    print(f"ONNX available: {ONNX_AVAILABLE}")
    print(f"Predictors available: {PREDICTORS_AVAILABLE}")
    print(f"Speaker Diarization available: {SPEAKER_DIARIZATION_AVAILABLE}")
    print(f"Tools available: {TOOLS_AVAILABLE}")
    
    available_count = sum([
        ALGORITHM_AVAILABLE, EMBEDDERS_AVAILABLE, ONNX_AVAILABLE,
        PREDICTORS_AVAILABLE, SPEAKER_DIARIZATION_AVAILABLE, TOOLS_AVAILABLE
    ])
    print(f"Total available modules: {available_count}/6")
    print("=" * 50)

def get_available_modules():
    """Get list of available modules"""
    available = []
    if ALGORITHM_AVAILABLE: available.append('algorithm')
    if EMBEDDERS_AVAILABLE: available.append('embedders')
    if ONNX_AVAILABLE: available.append('onnx')
    if PREDICTORS_AVAILABLE: available.append('predictors')
    if SPEAKER_DIARIZATION_AVAILABLE: available.append('speaker_diarization')
    if TOOLS_AVAILABLE: available.append('tools')
    return available
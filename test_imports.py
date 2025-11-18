#!/usr/bin/env python3
"""
Test script to verify all imports work correctly after our fixes
"""

import sys
import os

# Add current directory to path
sys.path.append('.')

def test_core_imports():
    """Test core module imports"""
    try:
        from core import full_inference_program, download_music
        print("✅ Core imports successful!")
        print(f"✅ full_inference_program: {full_inference_program}")
        print(f"✅ download_music: {download_music}")
        return True
    except Exception as e:
        print(f"❌ Core import error: {e}")
        return False

def test_tabs_imports():
    """Test tabs imports"""
    try:
        from tabs.inference.full_inference import full_inference_tab
        print("✅ Tabs inference import successful!")
        return True
    except Exception as e:
        print(f"❌ Tabs import error: {e}")
        return False

def test_app_imports():
    """Test main app imports"""
    try:
        from app import *
        print("✅ App imports successful!")
        return True
    except Exception as e:
        print(f"❌ App import error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing all imports after fixes...")
    
    success_count = 0
    total_tests = 3
    
    if test_core_imports():
        success_count += 1
    
    if test_tabs_imports():
        success_count += 1
    
    if test_app_imports():
        success_count += 1
    
    print(f"\n📊 Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All imports working correctly!")
    else:
        print("⚠️ Some imports still have issues")
#!/usr/bin/env python3
"""
Simple test to verify core functions exist without importing dependencies
"""

import re

def check_function_exists(file_path, function_name):
    """Check if a function exists in the file by parsing the source code"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for function definition
        pattern = rf'def\s+{function_name}\s*\('
        if re.search(pattern, content):
            print(f"✓ Function '{function_name}' found in core.py")
            return True
        else:
            print(f"✗ Function '{function_name}' NOT found in core.py")
            return False
            
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        return False

def check_in_all_list(file_path, function_name):
    """Check if function is in __all__ list"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for __all__ list
        all_match = re.search(r'__all__\s*=\s*\[(.*?)\]', content, re.DOTALL)
        if all_match:
            all_list = all_match.group(1)
            if f"'{function_name}'" in all_list or f'"{function_name}"' in all_list:
                print(f"✓ Function '{function_name}' is in __all__ list")
                return True
            else:
                print(f"✗ Function '{function_name}' is NOT in __all__ list")
                return False
        
        print("✗ No __all__ list found")
        return False
        
    except Exception as e:
        print(f"✗ Error checking __all__ list: {e}")
        return False

def main():
    print("Testing core.py function availability...")
    print("=" * 50)
    
    core_file = "/workspace/Advanced-RVC-Inference/core.py"
    
    # Check if core.py exists
    try:
        with open(core_file, 'r') as f:
            content = f.read()
        print("✓ core.py file exists and is readable")
    except Exception as e:
        print(f"✗ Cannot read core.py: {e}")
        return
    
    print("\nChecking for required functions:")
    functions_to_check = ['full_inference_program', 'download_music']
    
    for func_name in functions_to_check:
        print(f"\n--- Checking '{func_name}' ---")
        func_exists = check_function_exists(core_file, func_name)
        in_all = check_in_all_list(core_file, func_name)
        
        if func_exists and in_all:
            print(f"✓ '{func_name}' is properly defined and exported")
        elif func_exists:
            print(f"⚠ '{func_name}' exists but may not be properly exported")
        else:
            print(f"✗ '{func_name}' is missing")
    
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    main()
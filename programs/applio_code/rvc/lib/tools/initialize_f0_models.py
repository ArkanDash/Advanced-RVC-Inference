#!/usr/bin/env python3
"""
F0 Models Initialization Script for Advanced-RVC-Inference
Downloads essential F0 models and sets up the models directory structure
"""

import os
import sys
import argparse
from pathlib import Path

# Add the tools directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from f0_models_manager import F0ModelsManager
from f0_model_auto_loader import get_auto_loader

def initialize_f0_models(quick: bool = False, comprehensive: bool = False):
    """
    Initialize F0 models for Advanced-RVC-Inference
    
    Args:
        quick: Download only essential models (rmvpe, fcpe)
        comprehensive: Download all available models
    """
    print("=" * 60)
    print("F0 Models Initialization for Advanced-RVC-Inference")
    print("=" * 60)
    
    # Initialize the manager
    manager = F0ModelsManager()
    
    # Create models directory structure
    print("📁 Creating models directory structure...")
    models_dir = Path(__file__).parent.parent.parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    subdirs = ["predictors", "embedders/fairseq", "embedders/onnx", 
               "embedders/spin", "embedders/whisper", 
               "pretraineds/pretrained_v1", "pretraineds/pretrained_v2", "formant"]
    
    for subdir in subdirs:
        (models_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Models directory structure created at: {models_dir}")
    
    # Show available F0 methods
    print("\n📋 Available F0 Methods:")
    methods = manager.list_available_f0_methods()
    for category, method_list in methods.items():
        print(f"\n  {category}:")
        for method in method_list[:5]:  # Show first 5 methods
            print(f"    • {method}")
        if len(method_list) > 5:
            print(f"    ... and {len(method_list) - 5} more")
    
    # Determine which models to download
    if comprehensive:
        f0_methods = (manager.builtin_methods + 
                     manager.model_based_methods + 
                     manager.hybrid_methods)
        print(f"\n🚀 Downloading comprehensive F0 model set ({len(f0_methods)} methods)...")
    elif quick:
        # Essential models for basic RVC functionality
        f0_methods = ["rmvpe", "fcpe", "harvest", "yin"]
        print(f"\n⚡ Downloading essential F0 models ({len(f0_methods)} methods)...")
    else:
        # Default: Download popular models
        f0_methods = ["rmvpe", "fcpe", "harvest", "yin", "crepe-tiny", "crepe-medium"]
        print(f"\n📦 Downloading recommended F0 models ({len(f0_methods)} methods)...")
    
    # Check what's already available
    print("\n🔍 Checking current model availability...")
    availability = manager.check_model_availability(f0_methods)
    
    available_count = sum(1 for info in availability.values() if info["available"])
    total_count = len(f0_methods)
    
    print(f"  • {available_count}/{total_count} methods already available")
    
    # Download missing models
    missing_models = manager.get_missing_models_summary(f0_methods)
    
    if missing_models["missing"] > 0:
        print(f"\n⬇️  Downloading {missing_models['missing']} missing models...")
        print(f"  Total size to download: {missing_models['total_size_missing']/1024/1024:.1f} MB")
        
        # Download models
        auto_loader = get_auto_loader()
        download_results = auto_loader.manager.download_models_for_f0_methods(f0_methods)
        
        # Show results
        success_count = sum(1 for success in download_results.values() if success)
        print(f"\n📊 Download Results:")
        print(f"  • {success_count}/{len(download_results)} models downloaded successfully")
        
        if success_count < len(download_results):
            failed_models = [model for model, success in download_results.items() if not success]
            print(f"  • Failed downloads: {failed_models}")
    else:
        print("\n✅ All required models are already available!")
    
    # Validate downloaded models
    print("\n🔬 Validating downloaded models...")
    valid_count = 0
    
    for method in f0_methods:
        if method in manager.builtin_methods:
            continue  # Skip built-in methods
            
        # Import the function directly since it's not a method of the manager
        from tools.prerequisites_download import get_modelname_from_f0_method
        modelname = get_modelname_from_f0_method(method)
        if modelname and manager.model_exists(modelname):
            if manager.validate_model(modelname):
                valid_count += 1
                print(f"  ✅ {method} ({modelname})")
            else:
                print(f"  ❌ {method} ({modelname}) - validation failed")
        else:
            print(f"  ⚠️  {method} - model file missing")
    
    # Show final status
    print("\n" + "=" * 60)
    print("📋 F0 Models Initialization Summary")
    print("=" * 60)
    print(f"  • Models directory: {models_dir}")
    print(f"  • Total F0 methods: {len(f0_methods)}")
    print(f"  • Built-in methods: {len([m for m in f0_methods if m in manager.builtin_methods])}")
    print(f"  • Model-based methods: {len([m for m in f0_methods if m not in manager.builtin_methods])}")
    print(f"  • Valid models: {valid_count}")
    print(f"  • Total size: {sum(manager.get_model_size(m) for m in manager.model_urls.keys() if manager.model_exists(m))/1024/1024:.1f} MB")
    
    # Test auto-loader
    print("\n🧪 Testing auto-loader functionality...")
    try:
        auto_loader = get_auto_loader()
        
        # Test with a simple method
        test_methods = [m for m in f0_methods[:3] if m not in manager.builtin_methods]
        
        for method in test_methods:
            if auto_loader.ensure_model_available(method):
                print(f"  ✅ Auto-loader test passed for {method}")
            else:
                print(f"  ❌ Auto-loader test failed for {method}")
                
    except Exception as e:
        print(f"  ⚠️  Auto-loader test error: {e}")
    
    print("\n🎉 F0 Models initialization completed!")
    
    # Usage instructions
    print("\n📖 Usage Instructions:")
    print("  • F0 models are automatically downloaded when needed")
    print("  • Use F0Extractor with any of the downloaded methods")
    print("  • Hybrid methods combine multiple F0 estimators")
    print("  • Models are cached for repeated use")
    
    return True


def show_models_status():
    """Show detailed status of all F0 models"""
    print("=" * 60)
    print("F0 Models Status Report")
    print("=" * 60)
    
    manager = F0ModelsManager()
    info = manager.get_model_info()
    
    print(f"\n📊 Overall Statistics:")
    print(f"  • Total models available: {info['total_models']}")
    print(f"  • Downloaded: {info['downloaded_models']}")
    print(f"  • Valid: {info['valid_models']}")
    print(f"  • Total size: {info['total_size']/1024/1024:.1f} MB")
    
    print(f"\n📋 Model Details:")
    for model_name, model_info in info['models'].items():
        status = "✅" if model_info['valid'] else "❌" if model_info['exists'] else "⭕"
        size_str = f"{model_info['size']/1024/1024:.1f}MB" if model_info['size'] > 0 else "0MB"
        print(f"  {status} {model_name:<20} ({size_str})")
    
    print(f"\n🔍 Available F0 Methods:")
    methods = manager.list_available_f0_methods()
    for category, method_list in methods.items():
        print(f"\n  {category}:")
        for method in method_list:
            builtin = "🏠" if method in manager.builtin_methods else "⬇️"
            print(f"    {builtin} {method}")


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="F0 Models Initialization for Advanced-RVC-Inference")
    parser.add_argument("--quick", action="store_true", help="Download only essential models")
    parser.add_argument("--comprehensive", action="store_true", help="Download all available models")
    parser.add_argument("--status", action="store_true", help="Show models status without downloading")
    parser.add_argument("--test", action="store_true", help="Test auto-loader functionality")
    
    args = parser.parse_args()
    
    if args.status:
        show_models_status()
    elif args.test:
        print("Testing F0 auto-loader functionality...")
        auto_loader = get_auto_loader()
        
        # Test essential methods
        test_methods = ["rmvpe", "fcpe", "harvest"]
        
        for method in test_methods:
            print(f"\nTesting {method}:")
            if auto_loader.ensure_model_available(method):
                print(f"  ✅ Model availability: OK")
                model = auto_loader.load_f0_model(method)
                if model:
                    print(f"  ✅ Model loading: OK")
                else:
                    print(f"  ❌ Model loading: Failed")
            else:
                print(f"  ❌ Model availability: Failed")
        
        info = auto_loader.get_cache_info()
        print(f"\nCache info: {info}")
        
    else:
        initialize_f0_models(quick=args.quick, comprehensive=args.comprehensive)


if __name__ == "__main__":
    main()
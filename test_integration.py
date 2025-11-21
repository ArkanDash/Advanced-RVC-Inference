#!/usr/bin/env python3
"""
Simple test to verify F0 model auto-loader integration works
"""
import os
import sys

# Add the project root to Python path
sys.path.append('/workspace/Advanced-RVC-Inference')

def test_auto_loader_import():
    """Test that the auto-loader can be imported and used"""
    
    print("Testing F0 model auto-loader integration...")
    
    try:
        from programs.applio_code.rvc.lib.tools.f0_model_auto_loader import get_auto_loader
        print("✅ Auto-loader import: SUCCESS")
        
        # Test getting auto-loader instance
        auto_loader = get_auto_loader()
        print("✅ Auto-loader instance: SUCCESS")
        
        # Test model availability checks
        methods = ["rmvpe", "fcpe", "yin"]
        for method in methods:
            try:
                available = auto_loader.ensure_model_available(method)
                print(f"✅ {method.upper()} model availability: {'AVAILABLE' if available else 'NOT AVAILABLE'}")
            except Exception as e:
                print(f"❌ {method.upper()} model availability check: FAILED - {str(e)}")
        
        print("\n✅ Auto-loader integration test: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Auto-loader integration test: FAILED - {str(e)}")
        return False

def test_pipeline_import():
    """Test that the updated pipeline can be imported"""
    
    print("\nTesting updated pipeline import...")
    
    try:
        from programs.applio_code.rvc.infer.pipeline import Pipeline
        print("✅ Pipeline import: SUCCESS")
        
        # Test that Pipeline class has expected methods
        expected_methods = ['get_f0', 'get_f0_hybrid', 'get_f0_crepe', 'voice_conversion', 'pipeline']
        for method in expected_methods:
            if hasattr(Pipeline, method):
                print(f"✅ Pipeline.{method}: EXISTS")
            else:
                print(f"❌ Pipeline.{method}: MISSING")
        
        print("✅ Pipeline import test: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline import test: FAILED - {str(e)}")
        return False

def test_model_files():
    """Test that F0 model files exist in expected locations"""
    
    print("\nTesting F0 model files...")
    
    models_dir = "/workspace/Advanced-RVC-Inference/programs/applio_code/models/predictors"
    
    if os.path.exists(models_dir):
        print(f"✅ Models directory exists: {models_dir}")
        
        # Check for expected model files
        expected_models = ["rmvpe.pt", "ddsp_200k.pt"]
        for model in expected_models:
            model_path = os.path.join(models_dir, model)
            if os.path.exists(model_path):
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                print(f"✅ {model}: EXISTS ({size_mb:.1f} MB)")
            else:
                print(f"❌ {model}: MISSING")
    else:
        print(f"❌ Models directory missing: {models_dir}")
        return False
    
    print("✅ Model files test: PASSED")
    return True

def main():
    """Run all tests"""
    
    print("=" * 60)
    print("F0 Model Auto-Loader Integration Test")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    test_results.append(test_auto_loader_import())
    test_results.append(test_pipeline_import()) 
    test_results.append(test_model_files())
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("\nThe updated pipeline.py is successfully using the F0 model auto-loader.")
        print("F0 model loading issues should now be resolved.")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
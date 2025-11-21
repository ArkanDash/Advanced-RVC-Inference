#!/usr/bin/env python3
"""
F0 Model Auto-Loader for Advanced-RVC-Inference
Automatically downloads and loads F0 models when needed
"""

import os
import sys
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
import importlib.util

# Import the F0 models manager
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from f0_models_manager import F0ModelsManager

logger = logging.getLogger(__name__)

class F0ModelAutoLoader:
    """
    Auto-loader for F0 models that handles downloading and loading
    """
    
    def __init__(self):
        self.manager = F0ModelsManager()
        self.loaded_models = {}  # Cache for loaded models
        
    def ensure_model_available(self, f0_method: str, force_download: bool = False) -> bool:
        """
        Ensure the required model for an F0 method is available
        
        Args:
            f0_method: The F0 method name
            force_download: Force re-download even if model exists
            
        Returns:
            bool: True if model is available, False otherwise
        """
        # Built-in methods don't need downloads
        if f0_method in self.manager.builtin_methods:
            return True
        
        # Handle hybrid methods
        if f0_method in self.manager.hybrid_methods:
            import re
            match = re.search(r'\[(.+)\]', f0_method)
            if match:
                individual_methods = [m.strip() for m in match.group(1).split('+')]
                results = {}
                for method in individual_methods:
                    results[method] = self.ensure_model_available(method, force_download)
                return all(results.values())
        
        # Get required model file name
        modelname = self.manager.get_modelname_from_f0_method(f0_method)
        if not modelname:
            logger.error(f"No model mapping found for F0 method: {f0_method}")
            return False
        
        # Check if model exists and is valid
        if self.manager.model_exists(modelname) and self.manager.validate_model(modelname) and not force_download:
            logger.info(f"Model {modelname} is already available and valid")
            return True
        
        # Download the model
        logger.info(f"Downloading model for F0 method: {f0_method}")
        success = self.manager.download_model(modelname, force=force_download)
        
        if success:
            logger.info(f"Successfully downloaded model for {f0_method}")
        else:
            logger.error(f"Failed to download model for {f0_method}")
        
        return success

    def load_f0_model(self, f0_method: str, device: str = "cpu", **kwargs) -> Optional[Any]:
        """
        Load an F0 model for inference
        
        Args:
            f0_method: The F0 method name
            device: Device to load the model on
            **kwargs: Additional arguments for model loading
            
        Returns:
            Loaded model instance or None if failed
        """
        # Check if model is already loaded
        cache_key = f"{f0_method}_{device}"
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]
        
        # Ensure model is available
        if not self.ensure_model_available(f0_method):
            return None
        
        try:
            model = None
            
            # Load specific model types
            if f0_method == "rmvpe":
                from programs.applio_code.rvc.lib.predictors.RMVPE import RMVPE0Predictor
                
                # Get model path
                modelname = self.manager.get_modelname_from_f0_method(f0_method)
                model_path = self.manager.get_model_path(modelname)
                
                # Determine precision
                is_half = False if device == "cpu" else kwargs.get('is_half', False)
                
                model = RMVPE0Predictor(
                    str(model_path),
                    is_half=is_half,
                    device=device
                )
                
            elif f0_method in ["fcpe", "fcpe-legacy", "fcpe-previous", "ddsp_200k"]:
                import torchfcpe
                model = torchfcpe.spawn_bundled_infer_model(device=device)
                
            elif "crepe" in f0_method:
                import torchcrepe
                model = torchcrepe.loadmodel(
                    device=device,
                    model=kwargs.get('model_name', 'tiny')  # Default to tiny model
                )
                
            elif f0_method == "penn":
                from programs.applio_code.rvc.lib.predictors.PENN.PENN import PENNPredictor
                modelname = self.manager.get_modelname_from_f0_method(f0_method)
                model_path = self.manager.get_model_path(modelname)
                model = PENNPredictor(str(model_path), device=device)
                
            elif f0_method == "djcm":
                from programs.applio_code.rvc.lib.predictors.DJCM.DJCM import DJCMPredictor
                modelname = self.manager.get_modelname_from_f0_method(f0_method)
                model_path = self.manager.get_model_path(modelname)
                model = DJCMPredictor(str(model_path), device=device)
                
            elif f0_method == "swift":
                # SWIFT is typically ONNX-based
                import onnxruntime as ort
                modelname = self.manager.get_modelname_from_f0_method(f0_method)
                model_path = self.manager.get_model_path(modelname)
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
                model = ort.InferenceSession(str(model_path), providers=providers)
                
            elif f0_method == "pesto":
                from programs.applio_code.rvc.lib.predictors.PESTO.PESTO import PESTOPredictor
                modelname = self.manager.get_modelname_from_f0_method(f0_method)
                model_path = self.manager.get_model_path(modelname)
                model = PESTOPredictor(str(model_path), device=device)
                
            else:
                logger.error(f"Unsupported F0 method for model loading: {f0_method}")
                return None
            
            # Cache the loaded model
            if model:
                self.loaded_models[cache_key] = model
                logger.info(f"Successfully loaded {f0_method} model")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load {f0_method} model: {e}")
            return None

    def preload_models(self, f0_methods: list, device: str = "cpu", **kwargs) -> Dict[str, bool]:
        """
        Preload multiple F0 models
        
        Args:
            f0_methods: List of F0 method names to preload
            device: Device to load models on
            **kwargs: Additional arguments for model loading
            
        Returns:
            Dict mapping method names to success status
        """
        results = {}
        
        for method in f0_methods:
            results[method] = self.load_f0_model(method, device, **kwargs) is not None
        
        return results

    def unload_model(self, f0_method: str, device: str = "cpu") -> bool:
        """
        Unload a specific F0 model from cache
        
        Args:
            f0_method: The F0 method name
            device: Device the model was loaded on
            
        Returns:
            bool: True if unloaded successfully
        """
        cache_key = f"{f0_method}_{device}"
        
        if cache_key in self.loaded_models:
            del self.loaded_models[cache_key]
            logger.info(f"Unloaded {f0_method} model from cache")
            return True
        
        return False

    def clear_cache(self):
        """Clear all loaded models from cache"""
        self.loaded_models.clear()
        logger.info("Cleared all models from cache")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached models"""
        return {
            "cached_models": list(self.loaded_models.keys()),
            "cache_size": len(self.loaded_models),
            "total_models_available": len(self.manager.model_urls),
            "models_directory": str(self.manager.models_dir)
        }


# Global auto-loader instance
_auto_loader = F0ModelAutoLoader()

def get_auto_loader() -> F0ModelAutoLoader:
    """Get the global auto-loader instance"""
    return _auto_loader

def ensure_f0_model_available(f0_method: str, force_download: bool = False) -> bool:
    """
    Convenience function to ensure F0 model is available
    
    Args:
        f0_method: The F0 method name
        force_download: Force re-download even if model exists
        
    Returns:
        bool: True if model is available, False otherwise
    """
    return _auto_loader.ensure_model_available(f0_method, force_download)

def load_f0_model(f0_method: str, device: str = "cpu", **kwargs) -> Optional[Any]:
    """
    Convenience function to load F0 model
    
    Args:
        f0_method: The F0 method name
        device: Device to load the model on
        **kwargs: Additional arguments for model loading
        
    Returns:
        Loaded model instance or None if failed
    """
    return _auto_loader.load_f0_model(f0_method, device, **kwargs)

def preload_f0_models(f0_methods: list, device: str = "cpu", **kwargs) -> Dict[str, bool]:
    """
    Convenience function to preload multiple F0 models
    
    Args:
        f0_methods: List of F0 method names to preload
        device: Device to load models on
        **kwargs: Additional arguments for model loading
        
    Returns:
        Dict mapping method names to success status
    """
    return _auto_loader.preload_models(f0_methods, device, **kwargs)


if __name__ == "__main__":
    # Test the auto-loader
    import argparse
    
    parser = argparse.ArgumentParser(description="Test F0 Model Auto-Loader")
    parser.add_argument("--method", required=True, help="F0 method to test")
    parser.add_argument("--device", default="cpu", help="Device to load model on")
    parser.add_argument("--force-download", action="store_true", help="Force model download")
    
    args = parser.parse_args()
    
    print(f"Testing F0 model auto-loader for method: {args.method}")
    
    # Ensure model is available
    if ensure_f0_model_available(args.method, args.force_download):
        print(f"✓ Model for {args.method} is available")
        
        # Try to load the model
        model = load_f0_model(args.method, args.device)
        if model:
            print(f"✓ Successfully loaded {args.method} model on {args.device}")
        else:
            print(f"✗ Failed to load {args.method} model")
    else:
        print(f"✗ Model for {args.method} is not available")
    
    # Show cache info
    info = _auto_loader.get_cache_info()
    print(f"\nCache info: {info}")
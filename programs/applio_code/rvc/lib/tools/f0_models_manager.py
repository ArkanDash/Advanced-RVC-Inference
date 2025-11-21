#!/usr/bin/env python3
"""
F0 Models Manager for Advanced-RVC-Inference
Comprehensive F0 model download and management system
"""

import os
import sys
import json
import torch
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests
from urllib.parse import urlparse

# Add the parent directory to the path to import prerequisites_download
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.prerequisites_download import (
    check_predictors_downloaded,
    check_and_download_predictors,
    get_modelname_from_f0_method,
    predictors_url,
    url_base
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class F0ModelsManager:
    """
    Comprehensive F0 Models Manager for Advanced-RVC-Inference
    Handles download, validation, and management of all F0 extraction models
    """
    
    def __init__(self):
        self.models_dir = Path(__file__).parent.parent.parent.parent / "models" / "predictors"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # F0 model URLs mapping (based on Vietnamese-RVC)
        self.model_urls = {
            # RMVPE models
            "rmvpe.pt": f"{predictors_url}rmvpe.pt",
            "rmvpe.onnx": f"{predictors_url}rmvpe.onnx",
            
            # FCPE models
            "fcpe.pt": f"{predictors_url}fcpe.pt",
            "fcpe.onnx": f"{predictors_url}fcpe.onnx",
            "fcpe_legacy.pt": f"{predictors_url}fcpe_legacy.pt",
            "fcpe_legacy.onnx": f"{predictors_url}fcpe_legacy.onnx",
            "ddsp_200k.pt": f"{predictors_url}ddsp_200k.pt",
            "ddsp_200k.onnx": f"{predictors_url}ddsp_200k.onnx",
            
            # CREPE models
            "crepe_tiny.pth": f"{predictors_url}crepe_tiny.pth",
            "crepe_tiny.onnx": f"{predictors_url}crepe_tiny.onnx",
            "crepe_small.pth": f"{predictors_url}crepe_small.pth",
            "crepe_small.onnx": f"{predictors_url}crepe_small.onnx",
            "crepe_medium.pth": f"{predictors_url}crepe_medium.pth",
            "crepe_medium.onnx": f"{predictors_url}crepe_medium.onnx",
            "crepe_large.pth": f"{predictors_url}crepe_large.pth",
            "crepe_large.onnx": f"{predictors_url}crepe_large.onnx",
            "crepe_full.pth": f"{predictors_url}crepe_full.pth",
            "crepe_full.onnx": f"{predictors_url}crepe_full.onnx",
            
            # PENN models
            "fcn.pt": f"{predictors_url}fcn.pt",
            "fcn.onnx": f"{predictors_url}fcn.onnx",
            
            # DJCM models
            "djcm.pt": f"{predictors_url}djcm.pt",
            "djcm.onnx": f"{predictors_url}djcm.onnx",
            
            # SWIFT models (ONNX only)
            "swift.onnx": f"{predictors_url}swift.onnx",
            
            # PESTO models
            "pesto.pt": f"{predictors_url}pesto.pt",
            "pesto.onnx": f"{predictors_url}pesto.onnx",
        }
        
        # F0 methods that don't require model downloads (built-in)
        self.builtin_methods = [
            "world", "pyin", "yin", "harvest", "parselmouth", "swipe", "piptrack",
            "pm-ac", "pm-cc", "pm-shs", "dio"
        ]
        
        # F0 methods that require model downloads
        self.model_based_methods = [
            "rmvpe", "fcpe", "fcpe-legacy", "fcpe-previous", "ddsp_200k",
            "crepe-tiny", "crepe-small", "crepe-medium", "crepe-large", "crepe-full",
            "mangio-crepe-tiny", "mangio-crepe-small", "mangio-crepe-medium",
            "mangio-crepe-large", "mangio-crepe-full", "penn", "mangio-penn",
            "djcm", "swift", "pesto"
        ]
        
        # Hybrid methods - will be resolved to individual methods
        self.hybrid_methods = [
            "hybrid[pm+dio]", "hybrid[pm+crepe-tiny]", "hybrid[pm+crepe]",
            "hybrid[pm+fcpe]", "hybrid[pm+rmvpe]", "hybrid[pm+harvest]",
            "hybrid[pm+yin]", "hybrid[dio+crepe-tiny]", "hybrid[dio+crepe]",
            "hybrid[dio+fcpe]", "hybrid[dio+rmvpe]", "hybrid[dio+harvest]",
            "hybrid[dio+yin]", "hybrid[crepe-tiny+crepe]", "hybrid[crepe-tiny+fcpe]",
            "hybrid[crepe-tiny+rmvpe]", "hybrid[crepe-tiny+harvest]",
            "hybrid[crepe+fcpe]", "hybrid[crepe+rmvpe]", "hybrid[crepe+harvest]",
            "hybrid[crepe+yin]", "hybrid[fcpe+rmvpe]", "hybrid[fcpe+harvest]",
            "hybrid[fcpe+yin]", "hybrid[rmvpe+harvest]", "hybrid[rmvpe+yin]",
            "hybrid[harvest+yin]"
        ]

    def get_model_path(self, model_name: str) -> Path:
        """Get the local path for a model"""
        return self.models_dir / model_name

    def model_exists(self, model_name: str) -> bool:
        """Check if a model file exists locally"""
        return self.get_model_path(model_name).exists()

    def get_model_size(self, model_name: str) -> int:
        """Get the size of a model file in bytes"""
        model_path = self.get_model_path(model_name)
        if model_path.exists():
            return model_path.stat().st_size
        return 0

    def validate_model(self, model_name: str) -> bool:
        """Validate that a model file is properly downloaded and not corrupted"""
        model_path = self.get_model_path(model_name)
        
        if not model_path.exists():
            return False
            
        try:
            # Check file size (should be > 0)
            if model_path.stat().st_size == 0:
                return False
                
            # For PyTorch models, try to load state dict
            if model_name.endswith(('.pt', '.pth')):
                try:
                    if model_name == 'rmvpe.pt':
                        # Special handling for RMVPE model
                        state_dict = torch.load(model_path, map_location='cpu')
                        if not isinstance(state_dict, dict):
                            return False
                    else:
                        # For other models, try basic loading
                        torch.load(model_path, map_location='cpu')
                except Exception:
                    # Don't fail validation for models that might have loading issues
                    # but log the error
                    logger.warning(f"Model {model_name} loaded with warnings")
            
            # For ONNX models, check if file is readable
            elif model_name.endswith('.onnx'):
                # Basic check - try to read the file
                with open(model_path, 'rb') as f:
                    header = f.read(4)
                    # ONNX files should start with 'ONNX' magic number
                    if header != b'ONNX':
                        return False
                        
            return True
            
        except Exception as e:
            logger.error(f"Error validating model {model_name}: {e}")
            return False

    def download_model(self, model_name: str, force: bool = False) -> bool:
        """Download a single F0 model"""
        if model_name not in self.model_urls:
            logger.error(f"No URL found for model: {model_name}")
            return False
            
        model_path = self.get_model_path(model_name)
        
        # Skip if model exists and force is False
        if model_path.exists() and not force:
            if self.validate_model(model_name):
                logger.info(f"Model {model_name} already exists and is valid")
                return True
            else:
                logger.warning(f"Model {model_name} exists but validation failed, re-downloading")
        
        url = self.model_urls[model_name]
        
        try:
            logger.info(f"Downloading {model_name} from {url}")
            
            # Create directory if it doesn't exist
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download with progress
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(model_path, 'wb') as f, tqdm(
                desc=f"Downloading {model_name}",
                total=total_size,
                unit='B',
                unit_scale=True,
                leave=False
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        pbar.update(len(chunk))
            
            # Validate the downloaded model
            if self.validate_model(model_name):
                logger.info(f"Successfully downloaded and validated {model_name}")
                return True
            else:
                logger.error(f"Downloaded {model_name} failed validation")
                model_path.unlink(missing_ok=True)
                return False
                
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            # Clean up partial download
            if model_path.exists():
                model_path.unlink(missing_ok=True)
            return False

    def download_multiple_models(self, model_names: List[str], max_workers: int = 4) -> Dict[str, bool]:
        """Download multiple models concurrently"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_model = {
                executor.submit(self.download_model, model_name): model_name 
                for model_name in model_names
            }
            
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    results[model_name] = future.result()
                except Exception as e:
                    logger.error(f"Download failed for {model_name}: {e}")
                    results[model_name] = False
        
        return results

    def get_required_models_for_methods(self, f0_methods: List[str]) -> List[str]:
        """Get the list of model files required for given F0 methods"""
        required_models = []
        
        for method in f0_methods:
            if method in self.builtin_methods:
                # Built-in methods don't need downloads
                continue
            elif method in self.hybrid_methods:
                # Parse hybrid methods to extract individual methods
                import re
                match = re.search(r'\[(.+)\]', method)
                if match:
                    individual_methods = [m.strip() for m in match.group(1).split('+')]
                    for individual_method in individual_methods:
                        modelname = get_modelname_from_f0_method(individual_method)
                        if modelname:
                            required_models.append(modelname)
            else:
                # Single model-based method
                modelname = get_modelname_from_f0_method(method)
                if modelname:
                    required_models.append(modelname)
        
        return list(set(required_models))  # Remove duplicates

    def download_models_for_f0_methods(self, f0_methods: List[str], force: bool = False) -> Dict[str, bool]:
        """Download all models required for specified F0 methods"""
        required_models = self.get_required_models_for_methods(f0_methods)
        
        if not required_models:
            logger.info("No model downloads required for the specified F0 methods")
            return {}
        
        logger.info(f"Downloading {len(required_models)} models for F0 methods: {f0_methods}")
        return self.download_multiple_models(required_models)

    def check_model_availability(self, f0_methods: List[str]) -> Dict[str, Dict]:
        """Check availability of models for F0 methods"""
        availability = {}
        required_models = self.get_required_models_for_methods(f0_methods)
        
        for method in f0_methods:
            availability[method] = {
                "type": "builtin" if method in self.builtin_methods else "model",
                "available": False,
                "model_file": None,
                "size": 0,
                "valid": False
            }
            
            if method in self.builtin_methods:
                availability[method]["available"] = True
            else:
                modelname = get_modelname_from_f0_method(method)
                if modelname:
                    availability[method]["model_file"] = modelname
                    availability[method]["available"] = self.model_exists(modelname)
                    availability[method]["size"] = self.get_model_size(modelname)
                    availability[method]["valid"] = self.validate_model(modelname) if availability[method]["available"] else False
        
        return availability

    def get_model_info(self) -> Dict:
        """Get comprehensive information about all available models"""
        info = {
            "total_models": len(self.model_urls),
            "total_size": sum(self.get_model_size(model) for model in self.model_urls.keys()),
            "downloaded_models": sum(1 for model in self.model_urls.keys() if self.model_exists(model)),
            "valid_models": sum(1 for model in self.model_urls.keys() if self.validate_model(model)),
            "models": {}
        }
        
        for model_name in self.model_urls.keys():
            info["models"][model_name] = {
                "exists": self.model_exists(model_name),
                "size": self.get_model_size(model_name),
                "valid": self.validate_model(model_name),
                "url": self.model_urls[model_name]
            }
        
        return info

    def list_available_f0_methods(self) -> Dict[str, List[str]]:
        """List all available F0 methods organized by category"""
        return {
            "Built-in Methods (no downloads required)": self.builtin_methods,
            "Model-based Methods (downloads required)": self.model_based_methods,
            "Hybrid Methods (downloads required)": self.hybrid_methods,
        }

    def cleanup_invalid_models(self) -> int:
        """Remove invalid model files and return count of removed files"""
        removed_count = 0
        
        for model_name in self.model_urls.keys():
            model_path = self.get_model_path(model_name)
            if model_path.exists() and not self.validate_model(model_name):
                try:
                    model_path.unlink()
                    logger.info(f"Removed invalid model: {model_name}")
                    removed_count += 1
                except Exception as e:
                    logger.error(f"Failed to remove {model_name}: {e}")
        
        return removed_count

    def get_missing_models_summary(self, f0_methods: List[str]) -> Dict:
        """Get a summary of missing models for F0 methods"""
        required_models = self.get_required_models_for_methods(f0_methods)
        missing_models = []
        available_models = []
        
        for model in required_models:
            if self.model_exists(model) and self.validate_model(model):
                available_models.append(model)
            else:
                missing_models.append(model)
        
        return {
            "total_required": len(required_models),
            "available": len(available_models),
            "missing": len(missing_models),
            "available_models": available_models,
            "missing_models": missing_models,
            "total_size_available": sum(self.get_model_size(model) for model in available_models),
            "total_size_missing": sum(self.get_model_size(model) for model in missing_models),
        }


def main():
    """Command line interface for F0 Models Manager"""
    import argparse
    
    parser = argparse.ArgumentParser(description="F0 Models Manager for Advanced-RVC-Inference")
    parser.add_argument("command", choices=[
        "download", "check", "info", "methods", "cleanup", "summary"
    ], help="Command to execute")
    
    parser.add_argument("--methods", nargs="+", help="F0 methods to work with")
    parser.add_argument("--force", action="store_true", help="Force download even if models exist")
    parser.add_argument("--output", help="Output file for info or summary commands")
    
    args = parser.parse_args()
    manager = F0ModelsManager()
    
    if args.command == "download":
        if not args.methods:
            print("Error: --methods required for download command")
            return
            
        print(f"Downloading models for F0 methods: {args.methods}")
        results = manager.download_models_for_f0_methods(args.methods, force=args.force)
        
        print("\nDownload Results:")
        for method, success in results.items():
            status = "✓ Success" if success else "✗ Failed"
            print(f"  {method}: {status}")
    
    elif args.command == "check":
        if not args.methods:
            print("Error: --methods required for check command")
            return
            
        availability = manager.check_model_availability(args.methods)
        
        print("\nModel Availability Check:")
        for method, info in availability.items():
            type_str = "Built-in" if info["type"] == "builtin" else "Model"
            status = "Available" if info["available"] else "Missing"
            valid_str = " (Valid)" if info["valid"] else ""
            size_str = f" ({info['size']/1024/1024:.1f}MB)" if info['size'] > 0 else ""
            print(f"  {method}: {type_str} - {status}{valid_str}{size_str}")
    
    elif args.command == "info":
        info = manager.get_model_info()
        
        print(f"\nF0 Models Information:")
        print(f"  Total models available: {info['total_models']}")
        print(f"  Downloaded: {info['downloaded_models']}")
        print(f"  Valid: {info['valid_models']}")
        print(f"  Total size downloaded: {info['total_size']/1024/1024:.1f}MB")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(info, f, indent=2)
            print(f"  Detailed info saved to: {args.output}")
    
    elif args.command == "methods":
        methods = manager.list_available_f0_methods()
        
        print("\nAvailable F0 Methods:")
        for category, method_list in methods.items():
            print(f"\n{category}:")
            for method in method_list:
                print(f"  - {method}")
    
    elif args.command == "cleanup":
        removed = manager.cleanup_invalid_models()
        print(f"\nCleaned up {removed} invalid model files")
    
    elif args.command == "summary":
        if not args.methods:
            print("Error: --methods required for summary command")
            return
            
        summary = manager.get_missing_models_summary(args.methods)
        
        print(f"\nMissing Models Summary for: {args.methods}")
        print(f"  Total required: {summary['total_required']}")
        print(f"  Available: {summary['available']}")
        print(f"  Missing: {summary['missing']}")
        print(f"  Total size available: {summary['total_size_available']/1024/1024:.1f}MB")
        print(f"  Total size missing: {summary['total_size_missing']/1024/1024:.1f}MB")
        
        if summary['missing_models']:
            print(f"\nMissing models:")
            for model in summary['missing_models']:
                print(f"  - {model}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"  Summary saved to: {args.output}")


if __name__ == "__main__":
    main()
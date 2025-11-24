#!/usr/bin/env python3
"""
Enhanced RVC Training System - GPU Optimization
T4/A100 GPU optimized training with automatic configuration
Version 3.5.3
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    from ..gpu_optimization import get_gpu_optimizer, get_opencl_processor
    GPU_OPTIMIZATION_AVAILABLE = True
except ImportError:
    GPU_OPTIMIZATION_AVAILABLE = False
    logger.warning("GPU optimization not available")

class EnhancedRVCTrainer:
    """Enhanced RVC trainer with GPU optimization for T4/A100"""
    
    def __init__(self):
        self.gpu_optimizer = None
        self.training_settings = {}
        self.gpu_info = {}
        
        if GPU_OPTIMIZATION_AVAILABLE:
            try:
                self.gpu_optimizer = get_gpu_optimizer()
                self.gpu_info = self.gpu_optimizer.gpu_info
                self.training_settings = self._get_optimal_training_settings()
                logger.info(f"Enhanced trainer initialized for {self.gpu_info['type']}")
            except Exception as e:
                logger.error(f"Failed to initialize enhanced trainer: {e}")
                self._set_default_settings()
        else:
            self._set_default_settings()
    
    def _set_default_settings(self):
        """Set default training settings"""
        self.gpu_info = {"type": "cpu", "memory_gb": 0, "tensor_cores": False}
        self.training_settings = {
            'batch_size': 4,
            'gradient_accumulation_steps': 8,
            'mixed_precision': 'fp32',
            'memory_efficient': True,
            'max_audio_length': 30,
            'enable_amp': False
        }
    
    def _get_optimal_training_settings(self) -> Dict[str, Any]:
        """Get optimal training settings based on GPU hardware"""
        gpu_type = self.gpu_info.get('type', 'cpu')
        memory_gb = self.gpu_info.get('memory_gb', 0)
        
        if gpu_type == "T4":
            # T4 optimization for training
            settings = {
                'batch_size': 1 if memory_gb < 16 else 2,
                'gradient_accumulation_steps': 8,
                'mixed_precision': 'fp16',
                'memory_efficient': True,
                'max_audio_length': 20,
                'enable_amp': True,
                'enable_tensor_cores': False,  # T4 doesn't have tensor cores
                'memory_growth': True,
                'gradient_checkpointing': True
            }
        elif gpu_type == "A100":
            # A100 optimization for training
            settings = {
                'batch_size': 2 if memory_gb < 40 else 4,
                'gradient_accumulation_steps': 2,
                'mixed_precision': 'bf16',
                'memory_efficient': False,
                'max_audio_length': 60,
                'enable_amp': True,
                'enable_tensor_cores': True,
                'memory_growth': False,
                'gradient_checkpointing': False,
                'compile_model': True
            }
        elif gpu_type.startswith("RTX"):
            # RTX series optimization
            settings = {
                'batch_size': 1 if memory_gb < 24 else 2,
                'gradient_accumulation_steps': 4,
                'mixed_precision': 'fp16',
                'memory_efficient': True,
                'max_audio_length': 40,
                'enable_amp': True,
                'enable_tensor_cores': True,
                'memory_growth': True,
                'gradient_checkpointing': True
            }
        else:
            # CPU or unknown GPU
            settings = {
                'batch_size': 1,
                'gradient_accumulation_steps': 16,
                'mixed_precision': 'fp32',
                'memory_efficient': True,
                'max_audio_length': 10,
                'enable_amp': False,
                'enable_tensor_cores': False,
                'memory_growth': True,
                'gradient_checkpointing': True
            }
        
        return settings
    
    def get_training_command(self, model_name: str, **kwargs) -> list:
        """Generate optimized training command"""
        settings = self.training_settings.copy()
        
        # Override with provided kwargs
        for key, value in kwargs.items():
            if value is not None:
                settings[key] = value
        
        cmd = [
            "python",
            "-m", "advanced_rvc_inference.rvc.train.training.train",
            "--train",
            "--model_name", model_name,
            "--enable_gpu_optimization", "true",
            "--auto_batch_size", str(settings.get('auto_batch_size', True)),
            "--mixed_precision", settings['mixed_precision'],
            "--enable_tensor_cores", str(settings['enable_tensor_cores']),
            "--memory_efficient_training", str(settings['memory_efficient']),
            "--gradient_accumulation_steps", str(settings['gradient_accumulation_steps']),
            "--max_audio_length", str(settings['max_audio_length'])
        ]
        
        # Add other training parameters
        if 'rvc_version' in kwargs:
            cmd.extend(["--rvc_version", kwargs['rvc_version']])
        if 'batch_size' in kwargs and kwargs['batch_size']:
            cmd.extend(["--batch_size", str(kwargs['batch_size'])])
        if 'gpu' in kwargs:
            cmd.extend(["--gpu", kwargs['gpu']])
        if 'total_epoch' in kwargs:
            cmd.extend(["--total_epoch", str(kwargs['total_epoch'])])
        if 'save_every_epoch' in kwargs:
            cmd.extend(["--save_every_epoch", str(kwargs['save_every_epoch'])])
        
        return cmd
    
    def optimize_training_environment(self):
        """Optimize training environment settings"""
        if not GPU_OPTIMIZATION_AVAILABLE:
            return
        
        try:
            # Set environment variables for GPU optimization
            if self.gpu_info.get('type', '').startswith(('T4', 'A100', 'RTX')):
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
                os.environ['TORCH_CUDA_ARCH_LIST'] = '7.0;7.5;8.6'  # T4, V100, A100
                
                # Enable CUDA optimizations
                if hasattr(os, 'environ'):
                    # Set memory allocation strategy
                    if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
                        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            
            # Optimize memory
            if self.gpu_optimizer:
                self.gpu_optimizer.optimize_memory()
                logger.info("Training environment optimized")
                
        except Exception as e:
            logger.warning(f"Failed to optimize training environment: {e}")
    
    def get_memory_requirements(self) -> Dict[str, float]:
        """Get memory requirements for current GPU"""
        if self.gpu_optimizer:
            return self.gpu_optimizer.get_memory_info()
        
        return {"total_gb": 0, "allocated_gb": 0, "cached_gb": 0, "utilization": 0}
    
    def get_training_recommendations(self) -> Dict[str, str]:
        """Get training recommendations based on GPU"""
        gpu_type = self.gpu_info.get('type', 'cpu')
        memory_gb = self.gpu_info.get('memory_gb', 0)
        
        recommendations = {
            'batch_size': f"Recommended: {self.training_settings['batch_size']}",
            'mixed_precision': f"Use: {self.training_settings['mixed_precision']}",
            'gradient_accumulation': f"Steps: {self.training_settings['gradient_accumulation_steps']}",
            'audio_length': f"Max: {self.training_settings['max_audio_length']}s"
        }
        
        if gpu_type == "T4":
            recommendations['notes'] = (
                "T4 GPU detected - Memory efficient training recommended. "
                "Use FP16 precision and gradient checkpointing for best results."
            )
        elif gpu_type == "A100":
            recommendations['notes'] = (
                "A100 GPU detected - High performance training available. "
                "Use BF16 precision and tensor cores for maximum throughput."
            )
        elif gpu_type == "cpu":
            recommendations['notes'] = (
                "CPU training detected - Use small batches and FP32 precision. "
                "Training will be slower but functional."
            )
        
        return recommendations

def create_enhanced_training_config(model_name: str, **kwargs) -> Dict[str, Any]:
    """Create enhanced training configuration"""
    trainer = EnhancedRVCTrainer()
    trainer.optimize_training_environment()
    
    config = {
        'model_name': model_name,
        'gpu_info': trainer.gpu_info,
        'training_settings': trainer.training_settings,
        'memory_requirements': trainer.get_memory_requirements(),
        'recommendations': trainer.get_training_recommendations(),
        'training_command': trainer.get_training_command(model_name, **kwargs)
    }
    
    return config

def save_training_config(config: Dict[str, Any], output_path: str):
    """Save training configuration to file"""
    try:
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Training configuration saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save training config: {e}")
        return False

def main():
    """CLI interface for enhanced training configuration"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced RVC Training Configuration")
    parser.add_argument("--model_name", required=True, help="Model name")
    parser.add_argument("--output", default="training_config.json", help="Output config file")
    parser.add_argument("--rvc_version", default="v2", help="RVC version (v1/v2)")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--gpu", default="0", help="GPU device ID")
    parser.add_argument("--total_epoch", type=int, default=300, help="Total epochs")
    parser.add_argument("--save_every_epoch", type=int, default=10, help="Save every N epochs")
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_enhanced_training_config(
        model_name=args.model_name,
        rvc_version=args.rvc_version,
        batch_size=args.batch_size,
        gpu=args.gpu,
        total_epoch=args.total_epoch,
        save_every_epoch=args.save_every_epoch
    )
    
    # Save configuration
    if save_training_config(config, args.output):
        print(f"Enhanced training configuration created: {args.output}")
        print(f"GPU: {config['gpu_info']['type']}")
        print(f"Recommended batch size: {config['training_settings']['batch_size']}")
        print(f"Mixed precision: {config['training_settings']['mixed_precision']}")
        print(f"Training command: {' '.join(config['training_command'])}")
    else:
        print("Failed to create training configuration")

if __name__ == "__main__":
    main()
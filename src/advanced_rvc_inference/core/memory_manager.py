"""
Professional Memory Management for Advanced RVC Inference
Handles GPU memory optimization and cleanup
"""

import gc
import logging
import psutil
import torch
import threading
import time
from typing import Optional, Dict, Any, Callable
from functools import wraps
from contextlib import contextmanager
from pathlib import Path


class MemoryManager:
    """
    Professional Memory Management System
    Provides automatic memory optimization and cleanup
    """
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._enabled = torch.cuda.is_available()
        self._memory_threshold = 0.85  # 85% memory usage threshold
        self._auto_cleanup_enabled = True
        self._monitoring = False
        self._monitor_thread = None
        self._cleanup_callbacks = []
        self._memory_history = []
        self._max_history = 100
        
        if self._enabled:
            self._initialize_cuda()
        
        self._logger.info(f"MemoryManager initialized (GPU: {self._enabled})")
    
    def _initialize_cuda(self) -> None:
        """Initialize CUDA memory management."""
        try:
            # Set memory fraction if specified
            torch.cuda.set_per_process_memory_fraction(0.8)
            
            # Enable memory efficient algorithms
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Clear any existing allocations
            torch.cuda.empty_cache()
            
            self._logger.info("CUDA memory management initialized")
            
        except Exception as e:
            self._logger.warning(f"Failed to initialize CUDA memory management: {e}")
            self._enabled = False
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get comprehensive memory information."""
        info = {
            'timestamp': time.time(),
            'system': {}
        }
        
        # System memory
        system_memory = psutil.virtual_memory()
        info['system'] = {
            'total_gb': system_memory.total / (1024**3),
            'available_gb': system_memory.available / (1024**3),
            'used_gb': system_memory.used / (1024**3),
            'percent': system_memory.percent
        }
        
        # GPU memory (if available)
        if self._enabled:
            try:
                gpu_memory = torch.cuda.memory_stats(0)
                gpu_allocated = torch.cuda.memory_allocated(0)
                gpu_cached = torch.cuda.memory_reserved(0)
                gpu_total = torch.cuda.get_device_properties(0).total_memory
                
                info['gpu'] = {
                    'allocated_gb': gpu_allocated / (1024**3),
                    'cached_gb': gpu_cached / (1024**3),
                    'total_gb': gpu_total / (1024**3),
                    'utilization_percent': (gpu_cached / gpu_total) * 100,
                    'memory_stats': gpu_memory
                }
            except Exception as e:
                self._logger.warning(f"Could not get GPU memory info: {e}")
                info['gpu'] = None
        
        return info
    
    def get_memory_usage_percent(self) -> float:
        """Get current memory usage percentage."""
        info = self.get_memory_info()
        
        if info.get('gpu'):
            return info['gpu']['utilization_percent']
        else:
            return info['system']['percent']
    
    def should_cleanup(self) -> bool:
        """Check if memory cleanup is needed."""
        usage_percent = self.get_memory_usage_percent()
        return usage_percent >= (self._memory_threshold * 100)
    
    def cleanup_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """
        Perform memory cleanup.
        
        Args:
            aggressive: If True, perform more aggressive cleanup
            
        Returns:
            Dictionary with cleanup statistics
        """
        cleanup_stats = {
            'before': self.get_memory_info(),
            'after': None,
            'freed_gb': 0,
            'operations': []
        }
        
        try:
            # Python garbage collection
            collected = gc.collect()
            cleanup_stats['operations'].append(f"gc.collect(): {collected} objects")
            
            # GPU cleanup if available
            if self._enabled:
                before_allocated = torch.cuda.memory_allocated(0)
                
                # Clear cache
                torch.cuda.empty_cache()
                cleanup_stats['operations'].append("torch.cuda.empty_cache()")
                
                # Aggressive cleanup: reset memory pools
                if aggressive:
                    try:
                        torch.cuda.reset_peak_memory_stats(0)
                        torch.cuda.reset_accumulated_memory_stats(0)
                        cleanup_stats['operations'].append("reset memory stats")
                    except:
                        pass
                
                after_allocated = torch.cuda.memory_allocated(0)
                freed_bytes = before_allocated - after_allocated
                cleanup_stats['freed_gb'] = freed_bytes / (1024**3)
            
            # Run custom cleanup callbacks
            for callback in self._cleanup_callbacks:
                try:
                    callback_result = callback()
                    if callback_result:
                        cleanup_stats['operations'].append(f"callback: {callback_result}")
                except Exception as e:
                    self._logger.warning(f"Cleanup callback failed: {e}")
            
            cleanup_stats['after'] = self.get_memory_info()
            
            # Update memory history
            self._memory_history.append(cleanup_stats)
            if len(self._memory_history) > self._max_history:
                self._memory_history.pop(0)
            
            self._logger.info(f"Memory cleanup completed: {cleanup_stats['freed_gb']:.2f} GB freed")
            
        except Exception as e:
            self._logger.error(f"Memory cleanup failed: {e}")
        
        return cleanup_stats
    
    def add_cleanup_callback(self, callback: Callable[[], str]) -> None:
        """Add custom cleanup callback."""
        self._cleanup_callbacks.append(callback)
        self._logger.debug(f"Added cleanup callback: {callback.__name__}")
    
    def start_monitoring(self, interval: int = 30) -> None:
        """Start automatic memory monitoring."""
        if self._monitoring:
            return
        
        def monitor_loop():
            while self._monitoring:
                try:
                    if self.should_cleanup():
                        self._logger.info("Auto cleanup triggered by memory monitoring")
                        self.cleanup_memory()
                    time.sleep(interval)
                except Exception as e:
                    self._logger.error(f"Memory monitoring error: {e}")
                    time.sleep(interval)
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        self._logger.info(f"Memory monitoring started (interval: {interval}s)")
    
    def stop_monitoring(self) -> None:
        """Stop automatic memory monitoring."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        self._logger.info("Memory monitoring stopped")
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get memory optimization report."""
        memory_info = self.get_memory_info()
        history = self._memory_history[-10:] if self._memory_history else []
        
        report = {
            'current_usage': memory_info,
            'monitoring_enabled': self._monitoring,
            'cleanup_threshold': self._memory_threshold,
            'recent_cleanups': len(history),
            'total_memory_freed_gb': sum(h['freed_gb'] for h in history),
            'recommendations': []
        }
        
        # Add recommendations
        if memory_info.get('system', {}).get('percent', 0) > 90:
            report['recommendations'].append("System memory usage is very high (>90%)")
        
        if memory_info.get('gpu'):
            gpu_usage = memory_info['gpu'].get('utilization_percent', 0)
            if gpu_usage > 90:
                report['recommendations'].append("GPU memory usage is very high (>90%)")
            if gpu_usage > 85:
                report['recommendations'].append("Consider reducing batch size or enabling aggressive cleanup")
        
        if not self._monitoring:
            report['recommendations'].append("Enable memory monitoring for automatic cleanup")
        
        return report
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup_memory(aggressive=True)


# Global memory manager instance
memory_manager = MemoryManager()


def memory_optimized(func: Callable) -> Callable:
    """
    Decorator for memory-optimized function execution.
    Automatically cleans up memory before and after function execution.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Pre-execution cleanup
        memory_manager.cleanup_memory()
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            # Post-execution cleanup
            memory_manager.cleanup_memory()
            
            return result
            
        except Exception as e:
            # Emergency cleanup on exception
            memory_manager.cleanup_memory(aggressive=True)
            raise
    
    return wrapper


@contextmanager
def memory_context(threshold: float = 0.85):
    """
    Context manager for memory-intensive operations.
    
    Args:
        threshold: Memory usage threshold for auto-cleanup
    """
    original_threshold = memory_manager._memory_threshold
    memory_manager._memory_threshold = threshold
    
    try:
        with memory_manager:
            yield memory_manager
    finally:
        memory_manager._memory_threshold = original_threshold


def get_memory_info() -> Dict[str, Any]:
    """Get current memory information."""
    return memory_manager.get_memory_info()


def cleanup_memory(aggressive: bool = False) -> Dict[str, Any]:
    """Perform memory cleanup."""
    return memory_manager.cleanup_memory(aggressive)


def monitor_memory(interval: int = 30) -> None:
    """Start memory monitoring."""
    memory_manager.start_monitoring(interval)


def stop_monitoring() -> None:
    """Stop memory monitoring."""
    memory_manager.stop_monitoring()


def should_cleanup() -> bool:
    """Check if cleanup is needed."""
    return memory_manager.should_cleanup()


# Export memory management utilities
__all__ = [
    'MemoryManager', 'memory_manager', 'memory_optimized', 
    'memory_context', 'get_memory_info', 'cleanup_memory',
    'monitor_memory', 'stop_monitoring', 'should_cleanup'
]
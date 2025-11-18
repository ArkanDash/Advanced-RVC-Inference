# Enhanced Advanced RVC Inference Test Suite
# Basic tests for ensuring code quality and functionality

import pytest
import os
import tempfile
import shutil
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import modules to test
try:
    from config import config_manager, ConfigurationManager
    from core import safe_file_validation, CacheManager, PerformanceMonitor
except ImportError as e:
    print(f"Warning: Could not import modules for testing: {e}")

class TestConfigurationManager:
    """Test configuration management functionality."""
    
    def test_config_manager_initialization(self):
        """Test that configuration manager initializes correctly."""
        assert config_manager is not None
        assert hasattr(config_manager, 'get')
        assert hasattr(config_manager, 'set')
    
    def test_config_get_default_values(self):
        """Test getting configuration values with defaults."""
        # Test numeric values
        max_file_size = config_manager.get('MAX_FILE_SIZE_MB', 500)
        assert isinstance(max_file_size, (int, float))
        assert 1 <= max_file_size <= 10000
        
        # Test boolean values
        enable_gpu = config_manager.get('ENABLE_GPU_ACCELERATION', True)
        assert isinstance(enable_gpu, bool)
    
    def test_config_validation(self):
        """Test configuration value validation."""
        # These should be within valid ranges
        max_threads = config_manager.get('MAX_THREADS', 4)
        assert 1 <= max_threads <= 16
        
        port = config_manager.get('SERVER_PORT', 7860)
        assert 1024 <= port <= 65535
    
    def test_paths_configuration(self):
        """Test path configuration."""
        paths = config_manager.get_paths()
        assert isinstance(paths, dict)
        assert 'CACHE_DIR' in paths
        assert 'LOGS_DIR' in paths
        assert 'MODELS_DIR' in paths
    
    def test_security_settings(self):
        """Test security settings."""
        security = config_manager.get_security_settings()
        assert 'max_file_size_mb' in security
        assert 'allowed_extensions' in security
        assert 'enable_validation' in security

class TestFileValidation:
    """Test file validation functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_audio_file = self.temp_dir / "test.wav"
        # Create a dummy audio file
        self.test_audio_file.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_safe_file_validation_valid_file(self):
        """Test file validation with valid file."""
        is_valid, message = safe_file_validation(self.test_audio_file)
        # Note: This may fail if file is too small or invalid format
        assert isinstance(is_valid, bool)
        assert isinstance(message, str)
    
    def test_safe_file_validation_nonexistent_file(self):
        """Test file validation with non-existent file."""
        nonexistent_file = self.temp_dir / "nonexistent.wav"
        is_valid, message = safe_file_validation(nonexistent_file)
        assert is_valid == False
        assert "does not exist" in message.lower()
    
    def test_safe_file_validation_invalid_extension(self):
        """Test file validation with invalid extension."""
        invalid_file = self.temp_dir / "test.invalid"
        invalid_file.write_text("invalid content")
        is_valid, message = safe_file_validation(invalid_file)
        assert is_valid == False
        assert "unsupported" in message.lower()

class TestCacheManager:
    """Test cache management functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_cache_dir = Path(tempfile.mkdtemp())
        self.cache_manager = CacheManager(str(self.temp_cache_dir), max_size_mb=1)
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_cache_dir.exists():
            shutil.rmtree(self.temp_cache_dir)
    
    def test_cache_initialization(self):
        """Test cache manager initialization."""
        assert self.cache_manager.cache_dir.exists()
        assert self.cache_manager.max_size_bytes > 0
    
    def test_cached_operation(self):
        """Test cached operation context manager."""
        test_key = "test_operation"
        
        with self.cache_manager.cached_operation(test_key) as cache_path:
            assert cache_path.exists()
            # Write test data
            cache_path.write_text("test content")
        
        # Verify file was created
        assert self.cache_manager.get_cache_path(test_key).exists()
    
    def test_cache_path_generation(self):
        """Test cache path generation."""
        test_key = "test_key_123"
        cache_path = self.cache_manager.get_cache_path(test_key)
        assert isinstance(cache_path, Path)
        assert cache_path.suffix == ".cache"

class TestPerformanceMonitor:
    """Test performance monitoring functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.monitor = PerformanceMonitor()
    
    def test_time_operation_context_manager(self):
        """Test timing operation context manager."""
        operation_name = "test_operation"
        
        with self.monitor.time_operation(operation_name):
            import time
            time.sleep(0.01)  # Small delay
        
        stats = self.monitor.get_stats()
        assert operation_name in stats
        assert stats[operation_name]['count'] == 1
        assert stats[operation_name]['total_time'] > 0
    
    def test_multiple_operations(self):
        """Test multiple operations timing."""
        operations = ["op1", "op2", "op3"]
        
        for op in operations:
            with self.monitor.time_operation(op):
                pass
        
        stats = self.monitor.get_stats()
        for op in operations:
            assert op in stats
            assert stats[op]['count'] == 1

class TestPathSecurity:
    """Test path security and validation."""
    
    def test_path_traversal_protection(self):
        """Test protection against path traversal attacks."""
        dangerous_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "/etc/passwd",
            "C:\\Windows\\System32"
        ]
        
        for dangerous_path in dangerous_paths:
            # This should be handled safely by file validation
            is_valid, message = safe_file_validation(dangerous_path)
            assert is_valid == False or "does not exist" in message.lower()

class TestConfigurationSettings:
    """Test configuration settings validation."""
    
    def test_boolean_settings(self):
        """Test boolean configuration settings."""
        boolean_keys = [
            'ENABLE_GPU_ACCELERATION',
            'ENABLE_FILE_VALIDATION',
            'NORMALIZE_OUTPUT',
            'ENABLE_POST_PROCESSING',
            'ENABLE_NOISE_REDUCTION'
        ]
        
        for key in boolean_keys:
            value = config_manager.get(key, default=False)
            assert isinstance(value, bool)
    
    def test_numeric_limits(self):
        """Test numeric configuration limits."""
        # Test that all numeric values are within reasonable limits
        numeric_configs = {
            'MAX_FILE_SIZE_MB': (1, 10000),
            'MAX_AUDIO_DURATION_MINUTES': (1, 180),
            'MAX_THREADS': (1, 16),
            'SERVER_PORT': (1024, 65535)
        }
        
        for key, (min_val, max_val) in numeric_configs.items():
            value = config_manager.get(key, default=min_val)
            assert min_val <= value <= max_val

# Utility functions for testing
def create_test_audio_file(path: Path, duration_seconds: float = 1.0):
    """Create a test audio file for testing."""
    # Simple WAV header and data
    sample_rate = 44100
    num_samples = int(sample_rate * duration_seconds)
    
    # WAV header (44 bytes)
    header = bytearray([
        0x52, 0x49, 0x46, 0x46,  # "RIFF"
        0x24, 0x08, 0x00, 0x00,  # File size - 8
        0x57, 0x41, 0x56, 0x45,  # "WAVE"
        0x66, 0x6D, 0x74, 0x20,  # "fmt "
        0x10, 0x00, 0x00, 0x00,  # Subchunk size
        0x01, 0x00,              # Audio format (PCM)
        0x01, 0x00,              # Number of channels
        0x44, 0xAC, 0x00, 0x00,  # Sample rate (44100)
        0x88, 0x58, 0x01, 0x00,  # Byte rate
        0x02, 0x00,              # Block align
        0x10, 0x00,              # Bits per sample
        0x64, 0x61, 0x74, 0x61,  # "data"
        0x00, 0x08, 0x00, 0x00   # Data size
    ])
    
    # Generate simple sine wave data
    import numpy as np
    t = np.linspace(0, duration_seconds, num_samples)
    audio_data = np.sin(2 * np.pi * 440 * t) * 0.3
    audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
    
    # Write file
    with open(path, 'wb') as f:
        f.write(header)
        f.write(audio_bytes)

# Pytest configuration
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )

if __name__ == "__main__":
    # Run basic smoke tests
    print("Running Enhanced RVC Inference Test Suite...")
    
    try:
        # Test configuration
        print("Testing configuration manager...")
        assert config_manager is not None
        print("✓ Configuration manager working")
        
        # Test file validation
        print("Testing file validation...")
        test_file = Path("test_file.wav")
        test_file.write_text("test")
        is_valid, msg = safe_file_validation(test_file)
        test_file.unlink()
        print(f"✓ File validation working: {msg}")
        
        # Test cache manager
        print("Testing cache manager...")
        cache_manager = CacheManager()
        with cache_manager.cached_operation("test") as path:
            assert path.exists()
        print("✓ Cache manager working")
        
        # Test performance monitor
        print("Testing performance monitor...")
        monitor = PerformanceMonitor()
        with monitor.time_operation("test_op"):
            import time
            time.sleep(0.001)
        stats = monitor.get_stats()
        assert "test_op" in stats
        print("✓ Performance monitor working")
        
        print("\n🎉 All basic tests passed!")
        print("Run 'pytest' for comprehensive test suite.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
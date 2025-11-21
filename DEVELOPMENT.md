# Development Guide - Advanced RVC Inference

## 📋 **Table of Contents**
- [Prerequisites](#prerequisites)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Package Development](#package-development)
- [CLI Development](#cli-development)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Documentation](#documentation)
- [Building & Distribution](#building--distribution)
- [Contributing](#contributing)

---

## 🛠️ **Prerequisites**

### **System Requirements**
- **Python 3.8 or higher**
- **Git** for version control
- **CUDA-compatible GPU** (optional, for GPU acceleration)
- **FFmpeg** for audio processing
- **Node.js** (optional, for documentation generation)

### **Required Software**
```bash
# Python development tools
pip install --upgrade pip wheel setuptools

# Version control
git --version

# Audio processing
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
choco install ffmpeg
```

---

## 🚀 **Development Setup**

### **1. Clone Repository**
```bash
git clone https://github.com/ArkanDash/Advanced-RVC-Inference.git
cd Advanced-RVC-Inference
```

### **2. Create Virtual Environment**
```bash
# Using venv
python -m venv rvc_dev_env
source rvc_dev_env/bin/activate  # On Windows: rvc_dev_env\Scripts\activate

# Or using conda
conda create -n rvc-dev python=3.11
conda activate rvc-dev
```

### **3. Install Development Dependencies**
```bash
# Install in development mode
pip install -e ".[dev]"

# Or install specific groups
pip install -e ".[cuda118,dev]"  # For CUDA development
pip install -e ".[apple,dev]"    # For Apple Silicon development
```

### **4. Install Pre-commit Hooks**
```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

---

## 📁 **Project Structure**

```
Advanced-RVC-Inference/
├── src/advanced_rvc_inference/          # Main package
│   ├── __init__.py                      # Package exports and metadata
│   ├── cli.py                          # CLI interface tools
│   ├── core/                           # Core processing modules
│   │   ├── __init__.py                 # Core exports
│   │   └── f0_extractor.py             # F0 extraction implementations
│   ├── audio/                          # Audio processing
│   │   ├── __init__.py                 # Audio exports
│   │   ├── separation.py               # Audio separation algorithms
│   │   └── voice_changer.py            # Real-time voice changing
│   ├── models/                         # Model management
│   │   ├── __init__.py                 # Model exports
│   │   └── manager.py                  # Model loading and management
│   ├── ui/                             # User interface
│   │   ├── __init__.py                 # UI exports
│   │   └── components.py               # Gradio UI components
│   └── utils/                          # Utilities
│       └── __init__.py                 # Utility exports
├── programs/                           # Additional programs
│   ├── applio_code/                    # Applio compatibility
│   ├── kernels/                        # KADVC optimization kernels
│   ├── music_separation_code/          # Audio separation models
│   └── training/                       # Training utilities
├── tabs/                               # Web UI tabs
├── assets/                             # UI assets and resources
├── docs/                               # Documentation source
├── tests/                              # Test suite
├── pyproject.toml                      # Package configuration
├── MANIFEST.in                         # Package manifest
├── requirements.txt                    # Development dependencies
├── setup.cfg                           # Setup configuration
├── Dockerfile                          # Docker definition
├── docker-compose.yml                  # Docker compose
├── README.md                           # Main documentation
├── CHANGELOG.md                        # Version changelog
├── DEVELOPMENT.md                      # This file
├── CONTRIBUTING.md                     # Contribution guidelines
└── CODE_OF_CONDUCT.md                  # Code of conduct
```

---

## 📦 **Package Development**

### **Adding New Modules**

#### **1. Create Module Structure**
```bash
# Create new module directory
mkdir -p src/advanced_rvc_inference/new_module

# Create __init__.py
touch src/advanced_rvc_inference/new_module/__init__.py

# Create module implementation
touch src/advanced_rvc_inference/new_module/implementation.py
```

#### **2. Implement Module**
```python
# src/advanced_rvc_inference/new_module/implementation.py
"""New module implementation."""

import numpy as np
from typing import List, Optional, Dict, Any


class NewFeature:
    """Example implementation of a new feature."""
    
    def __init__(self, param1: str = "default", param2: int = 42):
        """Initialize the new feature.
        
        Args:
            param1: First parameter description
            param2: Second parameter description
        """
        self.param1 = param1
        self.param2 = param2
    
    def process(self, data: np.ndarray) -> np.ndarray:
        """Process input data.
        
        Args:
            data: Input data array
            
        Returns:
            Processed data array
        """
        # Implementation here
        return data
    
    def get_info(self) -> Dict[str, Any]:
        """Get feature information.
        
        Returns:
            Dictionary with feature information
        """
        return {
            "param1": self.param1,
            "param2": self.param2,
            "class_name": self.__class__.__name__
        }
```

#### **3. Update Package Exports**
```python
# src/advanced_rvc_inference/new_module/__init__.py
"""New module package."""

from .implementation import NewFeature

__all__ = ["NewFeature"]
```

#### **4. Update Main Package**
```python
# src/advanced_rvc_inference/__init__.py
# Add new import
from .new_module import NewFeature

# Add to __all__ list
__all__ = [
    # ... existing exports
    "NewFeature",
]
```

### **Version Management**

#### **Update Version in Multiple Places**
```bash
# Update in pyproject.toml
# Update in src/advanced_rvc_inference/__init__.py
# Update in CLI tools
# Update in documentation
```

#### **Automated Version Update Script**
```python
# scripts/update_version.py
#!/usr/bin/env python3
"""Script to update version across all files."""

import re
import sys
from pathlib import Path

def update_version(version: str):
    """Update version in all relevant files."""
    files_to_update = [
        "src/advanced_rvc_inference/__init__.py",
        "pyproject.toml",
        "src/advanced_rvc_inference/cli.py"
    ]
    
    for file_path in files_to_update:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Update version patterns
        content = re.sub(r'__version__ = ".*?"', f'__version__ = "{version}"', content)
        content = re.sub(r'version = ".*?"', f'version = "{version}"', content)
        content = re.sub(r'version=".*?"', f'version="{version}"', content)
        
        with open(file_path, 'w') as f:
            f.write(content)
    
    print(f"Version updated to {version}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/update_version.py <version>")
        sys.exit(1)
    
    update_version(sys.argv[1])
```

---

## 🔧 **CLI Development**

### **Adding New CLI Commands**

#### **1. Extend Main CLI**
```python
# src/advanced_rvc_inference/cli.py

def new_feature_cli():
    """CLI for new feature operation."""
    parser = argparse.ArgumentParser(
        description="New Feature CLI - Description of feature"
    )
    
    parser.add_argument(
        "--input",
        required=True,
        help="Input file path"
    )
    
    parser.add_argument(
        "--output",
        required=True,
        help="Output file path"
    )
    
    parser.add_argument(
        "--option",
        default="default",
        help="Optional parameter"
    )
    
    args = parser.parse_args()
    
    # Implementation
    from .new_module import NewFeature
    
    feature = NewFeature(param1=args.option)
    result = feature.process(args.input)
    
    print(f"Processing completed: {args.input} -> {args.output}")

# Add to main function
def main():
    # ... existing code ...
    
    if args.mode == "new-feature":
        new_feature_cli()
```

#### **2. Update pyproject.toml**
```toml
[project.scripts]
# ... existing scripts ...
new-feature = "advanced_rvc_inference.cli:new_feature_cli"
```

### **CLI Best Practices**

#### **1. Consistent Interface**
```python
# Use consistent argument patterns
parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
parser.add_argument("--quiet", "-q", action="store_true", help="Enable quiet mode")
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
```

#### **2. Progress Indication**
```python
from tqdm import tqdm

def process_with_progress(items):
    """Process items with progress bar."""
    for item in tqdm(items, desc="Processing"):
        # Process item
        pass
```

#### **3. Error Handling**
```python
def safe_execute(func, *args, **kwargs):
    """Safely execute function with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
```

---

## 🧪 **Testing**

### **Test Structure**
```
tests/
├── unit/                          # Unit tests
│   ├── test_f0_extractor.py      # F0 extractor tests
│   ├── test_audio_separator.py   # Audio separator tests
│   ├── test_model_manager.py     # Model manager tests
│   └── test_cli.py               # CLI tests
├── integration/                   # Integration tests
│   ├── test_full_pipeline.py     # Full pipeline tests
│   └── test_web_interface.py     # Web interface tests
├── fixtures/                      # Test fixtures
│   ├── audio_samples/            # Sample audio files
│   └── model_samples/            # Sample model files
└── conftest.py                    # pytest configuration
```

### **Writing Tests**

#### **Unit Test Example**
```python
# tests/unit/test_f0_extractor.py
import pytest
import numpy as np
from advanced_rvc_inference import EnhancedF0Extractor


class TestEnhancedF0Extractor:
    """Test cases for EnhancedF0Extractor."""
    
    @pytest.fixture
    def f0_extractor(self):
        """Create F0 extractor instance for testing."""
        return EnhancedF0Extractor(method="rmvpe")
    
    @pytest.fixture
    def sample_audio(self):
        """Create sample audio for testing."""
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        frequency = 440.0  # A4 note
        audio = 0.5 * np.sin(2 * np.pi * frequency * t)
        return audio, sr
    
    def test_initialization(self, f0_extractor):
        """Test F0 extractor initialization."""
        assert f0_extractor.method == "rmvpe"
        assert f0_extractor.sample_rate == 44100
    
    def test_extract_f0(self, f0_extractor, sample_audio):
        """Test F0 extraction from audio."""
        audio, sr = sample_audio
        f0_values = f0_extractor.extract(audio, sr)
        
        assert len(f0_values) > 0
        assert all(f0 > 0 for f0 in f0_values if not np.isnan(f0))
    
    def test_extract_batch(self, f0_extractor):
        """Test batch F0 extraction."""
        audio_files = ["test1.wav", "test2.wav"]
        # Mock audio files or use fixtures
        results = f0_extractor.extract_batch(audio_files)
        assert len(results) == len(audio_files)
```

#### **Integration Test Example**
```python
# tests/integration/test_full_pipeline.py
import pytest
import tempfile
import os
from advanced_rvc_inference import (
    EnhancedF0Extractor,
    EnhancedAudioSeparator,
    EnhancedModelManager
)


class TestFullPipeline:
    """Test full voice conversion pipeline."""
    
    @pytest.mark.integration
    def test_end_to_end_conversion(self):
        """Test complete voice conversion workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup
            input_audio = os.path.join(temp_dir, "input.wav")
            output_audio = os.path.join(temp_dir, "output.wav")
            model_path = "tests/fixtures/models/test_model.pth"
            
            # Create sample input audio
            self.create_sample_audio(input_audio)
            
            # Initialize components
            f0_extractor = EnhancedF0Extractor()
            model_manager = EnhancedModelManager()
            
            # Load model
            model = model_manager.load_model(model_path)
            
            # Perform conversion (mock implementation)
            result = self.mock_conversion(input_audio, model, f0_extractor)
            
            # Verify output
            assert os.path.exists(output_audio)
            assert self.get_audio_duration(output_audio) > 0
    
    def create_sample_audio(self, path):
        """Create sample audio file for testing."""
        import soundfile as sf
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # A4 note
        sf.write(path, audio, sr)
    
    def mock_conversion(self, input_path, model, f0_extractor):
        """Mock conversion function for testing."""
        # Implementation would go here
        pass
```

### **Running Tests**

#### **All Tests**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/advanced_rvc_inference

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_f0_extractor.py

# Run tests matching pattern
pytest -k "test_extract"

# Run integration tests only
pytest -m integration

# Run tests without integration tests
pytest -m "not integration"
```

#### **Continuous Testing**
```bash
# Install pytest-watch
pip install pytest-watch

# Run tests on file changes
ptw  # or pytest-watch
```

---

## 📏 **Code Quality**

### **Code Formatting**

#### **Black (Code Formatter)**
```bash
# Format code
black src/ tests/

# Format with line length
black --line-length=88 src/ tests/

# Check without formatting
black --check src/ tests/

# Diff mode
black --diff src/ tests/
```

#### **isort (Import Sorter)**
```bash
# Sort imports
isort src/ tests/

# Check import sorting
isort --check-only src/ tests/

# Diff mode
isort --diff src/ tests/
```

#### **Combined Formatting**
```bash
# Format imports then code
isort src/ tests/ && black src/ tests/
```

### **Linting**

#### **Flake8**
```bash
# Run flake8
flake8 src/ tests/

# With specific rules
flake8 src/ --extend-ignore=E203,W503

# Output to file
flake8 src/ > linting_results.txt
```

#### **mypy (Type Checking)**
```bash
# Run type checking
mypy src/

# With specific configuration
mypy src/ --config-file=pyproject.toml

# Ignore missing imports
mypy src/ --ignore-missing-imports
```

### **Pre-commit Configuration**

#### **.pre-commit-config.yaml**
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: debug-statements

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

---

## 📚 **Documentation**

### **Documentation Structure**
```
docs/
├── source/
│   ├── index.rst                  # Main documentation index
│   ├── installation.rst           # Installation guide
│   ├── usage.rst                  # Usage guide
│   ├── api/                       # API documentation
│   │   ├── f0_extractor.rst
│   │   ├── audio_separator.rst
│   │   └── model_manager.rst
│   ├── cli/                       # CLI documentation
│   ├── development/               # Development guide
│   └── examples/                  # Usage examples
├── Makefile                       # Documentation build commands
├── requirements.txt               # Documentation dependencies
└── conf.py                        # Sphinx configuration
```

### **Sphinx Documentation Setup**

#### **Initialize Sphinx**
```bash
# Install Sphinx
pip install sphinx sphinx-rtd-theme

# Initialize documentation
sphinx-quickstart docs

# Configure conf.py
```

#### **conf.py Configuration**
```python
# docs/conf.py
import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

project = 'Advanced RVC Inference'
copyright = '2025, BF667'
author = 'BF667'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
```

#### **Building Documentation**
```bash
# Build HTML documentation
cd docs
make html

# Build and serve locally
make html && python -m http.server 8000 -d _build/html

# Clean build
make clean
```

### **API Documentation**

#### **Docstring Format**
```python
def example_function(param1: str, param2: int, param3: Optional[List[str]] = None) -> Dict[str, Any]:
    """Brief description of the function.
    
    This is a more detailed description that explains what the function does,
    its purpose, and any important considerations.
    
    Args:
        param1: Description of the first parameter.
        param2: Description of the second parameter.
        param3: Description of the optional parameter. Defaults to None.
        
    Returns:
        Dictionary containing the results with keys 'status', 'data', and 'errors'.
        
    Raises:
        ValueError: If param1 is empty or param2 is negative.
        TypeError: If param3 contains non-string items.
        
    Example:
        >>> result = example_function("test", 42, ["item1", "item2"])
        >>> print(result['status'])
        'success'
        
    Note:
        This function requires specific dependencies to be installed.
        See the installation guide for more information.
    """
    pass
```

---

## 🏗️ **Building & Distribution**

### **Package Building**

#### **Build Distribution Packages**
```bash
# Install build tools
pip install build twine

# Build source distribution and wheel
python -m build

# Check package
twine check dist/*

# Upload to Test PyPI
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*
```

#### **Build Script**
```bash
# scripts/build.sh
#!/bin/bash
set -e

echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/

echo "Running tests..."
pytest tests/

echo "Checking code quality..."
flake8 src/ tests/
mypy src/

echo "Building package..."
python -m build

echo "Checking package..."
twine check dist/*

echo "Package built successfully!"
ls -la dist/
```

### **Docker Building**

#### **Build Docker Image**
```bash
# Build CPU version
docker build -f Dockerfile.cpu -t advanced-rvc-inference:cpu .

# Build CUDA version
docker build -f Dockerfile.cuda -t advanced-rvc-inference:gpu .

# Build and tag latest
docker build -t advanced-rvc-inference:latest .

# Multi-platform build
docker buildx build --platform linux/amd64,linux/arm64 -t advanced-rvc-inference:latest .
```

#### **Docker Compose**
```bash
# Development environment
docker-compose up -d

# Production environment
docker-compose -f docker-compose.prod.yml up -d

# GPU support
docker-compose -f docker-compose.gpu.yml up -d
```

### **Release Process**

#### **Release Checklist**
1. **Update version** in all relevant files
2. **Update CHANGELOG.md** with new version
3. **Run all tests** and ensure passing
4. **Check code quality** with linting tools
5. **Update documentation** if needed
6. **Build package** and verify
7. **Test installation** in clean environment
8. **Create release** on GitHub
9. **Upload to PyPI**
10. **Announce release** in community channels

#### **Release Script**
```python
# scripts/release.py
#!/usr/bin/env python3
"""Automated release script."""

import subprocess
import sys
from pathlib import Path

def run_command(cmd):
    """Run shell command and check result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, check=True)
    return result.returncode == 0

def main():
    """Main release process."""
    print("Starting release process...")
    
    # Update version
    version = input("Enter version number (e.g., 3.4.1): ")
    update_version_script = f"python scripts/update_version.py {version}"
    run_command(update_version_script)
    
    # Run tests
    if not run_command("pytest tests/"):
        print("Tests failed! Aborting release.")
        sys.exit(1)
    
    # Run quality checks
    if not run_command("flake8 src/ tests/"):
        print("Linting failed! Aborting release.")
        sys.exit(1)
    
    # Build package
    run_command("python -m build")
    
    # Upload to PyPI
    upload = input("Upload to PyPI? (y/N): ")
    if upload.lower() == 'y':
        run_command("twine upload dist/*")
    
    print(f"Release {version} completed successfully!")

if __name__ == "__main__":
    main()
```

---

## 🤝 **Contributing**

### **Contribution Workflow**

1. **Fork** the repository
2. **Clone** your fork locally
3. **Create** a feature branch
4. **Make** your changes
5. **Run** tests and quality checks
6. **Commit** your changes
7. **Push** to your fork
8. **Create** a Pull Request

### **Contribution Guidelines**

#### **Code Style**
- Follow PEP 8 style guidelines
- Use Black for code formatting
- Use isort for import sorting
- Add type hints for all functions
- Write comprehensive docstrings

#### **Testing**
- Write tests for all new features
- Ensure all tests pass
- Maintain test coverage above 80%
- Use appropriate test fixtures

#### **Documentation**
- Update documentation for new features
- Add docstrings for all public functions
- Include usage examples
- Update CHANGELOG.md

#### **Pull Request Process**
1. Update README.md if needed
2. Update CHANGELOG.md
3. Run all quality checks
4. Ensure tests pass
5. Create descriptive PR description

### **Issue Guidelines**

#### **Bug Reports**
Include:
- Description of the bug
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details
- Error logs

#### **Feature Requests**
Include:
- Description of the feature
- Use case explanation
- Proposed implementation
- Alternative solutions considered

---

## 📞 **Support & Community**

### **Getting Help**
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/ArkanDash/Advanced-RVC-Inference/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/ArkanDash/Advanced-RVC-Inference/discussions)
- 💬 **Discord**: [Community Server](https://discord.gg/arkandash)
- 📧 **Email**: [bf667@example.com](mailto:bf667@example.com)

### **Developer Resources**
- 📖 **API Documentation**: [Read the Docs](https://advanced-rvc-inference.readthedocs.io/)
- 🎥 **Video Tutorials**: [YouTube Channel](https://youtube.com/@arkan-dash)
- 📝 **Blog Posts**: [Developer Blog](https://blog.arkan-dash.com)
- 🔧 **Tools**: [Development Tools](./tools/)

---

## 📋 **Common Development Tasks**

### **Adding New F0 Method**
```python
# 1. Implement in src/advanced_rvc_inference/core/f0_extractor.py
class NewF0Method:
    def extract(self, audio, sr):
        # Implementation
        pass

# 2. Register method in f0_methods dictionary
# 3. Add to __init__.py exports
# 4. Write tests
# 5. Update documentation
```

### **Adding New Audio Effect**
```python
# 1. Create in src/advanced_rvc_inference/audio/effects.py
class NewEffect:
    def apply(self, audio, **params):
        # Implementation
        return processed_audio

# 2. Register in effects registry
# 3. Add to UI components
# 4. Write tests
# 5. Update documentation
```

### **Extending CLI**
```python
# 1. Add new function in cli.py
def new_cli_command():
    # Implementation
    pass

# 2. Add to argument parser
# 3. Register in pyproject.toml scripts
# 4. Write tests
# 5. Update documentation
```

---

**Happy coding! 🚀✨**

For questions about development, please open an issue or join our Discord community.
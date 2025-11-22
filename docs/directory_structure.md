# Directory Structure Guide

This document provides a detailed explanation of the Advanced RVC Inference project structure and how to organize your files effectively.

## Main Directory Structure

```
Advanced-RVC-Inference/
├── src/advanced_rvc_inference/          # Core package
├── notebooks/                           # Google Colab notebooks
├── weights/                             # Model weights storage
├── indexes/                             # Index files storage
├── logs/                                # Training and application logs
├── docs/                                # Documentation files
└── app.py                               # Application launcher
```

## Source Package: `src/advanced_rvc_inference/`

### Core Modules

```
src/advanced_rvc_inference/
├── __init__.py                          # Package exports and metadata
├── cli.py                              # Command-line interface
├── config.py                           # Centralized configuration (Singleton)
└── core/                               # Core processing modules
    ├── __init__.py
    ├── memory_manager.py               # Memory optimization and cleanup
    ├── app_launcher.py                 # Application launcher logic
    └── f0_extractor.py                 # F0 extraction (40+ methods)
```

### GUI Components: `tabs/`

```
src/advanced_rvc_inference/tabs/
├── __init__.py
├── inference/                          # Voice conversion interfaces
│   ├── __init__.py
│   ├── full_inference.py               # Main inference interface
│   ├── realtime.py                     # Real-time voice changing
│   ├── tts.py                         # Text-to-speech integration
│   └── variable.py                     # Shared variables
├── training/                           # Training interfaces
│   ├── __init__.py
│   └── training_tab.py                 # Model training interface
├── settings/                           # Configuration interfaces
│   ├── __init__.py
│   ├── settings.py                     # Main settings interface
│   └── settinginf.py                   # Information settings
├── utilities/                           # Utility interfaces
│   ├── __init__.py
│   └── download_model.py               # Model downloading utilities
├── datasets/                            # Dataset management
│   ├── __init__.py
│   └── datasets_tab.py                 # Dataset management interface
├── downloads/                           # Download tools
│   ├── __init__.py
│   └── downloads_tab.py                # Download interface
├── extra/                               # Additional features
│   ├── __init__.py
│   └── extra_tab.py                    # Extra features interface
└── credits/                             # Information tabs
    ├── __init__.py
    └── credits_tab.py                  # Credits and information
```

### Asset Management: `assets/`

```
src/advanced_rvc_inference/assets/
├── config.json                         # Default configuration
├── ytdlstuff.txt                       # YouTube downloader configuration
├── i18n/                              # Internationalization
│   ├── i18n.py                        # Translation engine
│   ├── scan.py                        # Translation scanner
│   ├── update_i18n.py                 # Translation updater
│   └── languages/                     # Language files
├── themes/                             # UI themes
│   ├── Grheme.py                      # Theme engine
│   ├── loadThemes.py                  # Theme loader
│   └── themes_list.json               # Available themes
└── presence/                           # Discord Rich Presence
    └── discord_presence.py             # Discord integration
```

### Specialized Modules

```
├── audio/                              # Audio processing
│   ├── __init__.py
│   ├── separation.py                   # Audio separation algorithms
│   └── voice_changer.py                # Real-time voice changing
├── models/                             # Model management
│   ├── __init__.py
│   └── manager.py                      # Enhanced model manager
├── training/                           # Training pipeline
│   ├── __init__.py
│   ├── trainer.py                      # RVC trainer
│   ├── simple_trainer.py               # Simplified trainer
│   ├── core/                           # Training core
│   │   └── training_config.py          # Training configuration
│   ├── data/                           # Data handling
│   │   └── dataset.py                  # RVC dataset utilities
│   └── utils/                          # Training utilities
│       ├── audio_utils.py              # Audio processing utilities
│       └── feature_extraction.py       # Feature extraction tools
├── applio_code/                        # Applio compatibility
│   ├── __init__.py
│   └── rvc/                            # Applio RVC integration
│       ├── __init__.py                 # VoiceConverter, RVC_Inference_Pipeline
│       ├── configs/                    # Model configurations
│       ├── infer/                      # Inference implementations
│       └── realtime/                   # Real-time processing
├── kernels/                            # KADVC optimization
│   ├── __init__.py
│   ├── kadvc_config.py                 # KADVC configuration
│   ├── kadvc_integration.py            # KADVC integration
│   └── kadvc_kernels.py                # Custom CUDA kernels
├── music_separation_code/              # Audio separation models
│   ├── __init__.py
│   ├── inference.py                    # Separation inference
│   ├── utils.py                        # Separation utilities
│   ├── ensemble.py                     # Model ensemble methods
│   └── models/                         # Separation model implementations
│       ├── demucs4ht.py                # Demucs 4HT model
│       ├── mdx23c_tfc_tdf_v3.py        # MDX23C model
│       ├── bs_roformer/                # BS-Roformer models
│       ├── scnet/                      # SCNet models
│       └── bandit/                     # Bandit models
└── utils/                              # General utilities
    └── __init__.py
```

## User Data Directories

### Model Storage: `weights/`

```
weights/
├── your_model.pth                       # RVC model files (PyTorch)
├── your_model.onnx                      # RVC model files (ONNX)
└── model_metadata.json                  # Model information
```

**File Organization Tips:**
- Name files descriptively: `female_singer_v2.pth`
- Use subdirectories: `weights/japanese/`, `weights/english/`
- Include metadata: Create JSON files with model descriptions

### Index Files: `indexes/`

```
indexes/
├── your_model.index                     # Feature extraction indices
└── index_metadata.json                  # Index file information
```

**Best Practices:**
- Always pair .index files with corresponding .pth/.onnx models
- Use consistent naming: `model_name.index` should match `model_name.pth`
- Include sampling rate and other metadata

### Application Logs: `logs/`

```
logs/
├── app.log                              # Application runtime logs
├── training.log                         # Training session logs
├── error.log                            # Error and exception logs
└── performance.log                      # Performance metrics
```

**Log Management:**
- Logs are automatically rotated to prevent disk fill
- Debug level logging for troubleshooting
- Training logs include loss curves and validation metrics

## Google Colab: `notebooks/`

```
notebooks/
└── Advanced_RVC_Inference.ipynb         # Master Colab notebook
```

**Colab Strategy:**
- **Single Source of Truth**: Only this notebook contains installation code
- **Documentation References**: Other docs link to this master notebook
- **Badge Linking**: README uses badge pointing to this exact file
- **Version Control**: Always updated with main branch changes

## Documentation: `docs/`

```
docs/
├── directory_structure.md               # This file
├── api_usage.md                         # Python API documentation
├── troubleshooting.md                   # Common issues guide
└── configuration.md                     # Advanced configuration
```

**Documentation Principles:**
- Text-only content (no images or screenshots)
- Clear code examples and explanations
- Cross-references between documents
- Regular updates synchronized with code changes

## File Organization Best Practices

### Model Files

**Naming Convention:**
```
{voice_type}_{language}_{version}.pth
female_japanese_v2.pth
male_english_v1.onnx
neutral_chinese_v3.pth
```

**Directory Structure:**
```
weights/
├── japanese/
│   ├── female_singer_v2.pth
│   └── male_voice_v1.pth
├── english/
│   └── celebrity_voices/
└── experimental/
    └── test_models/
```

### Index Files

**Naming Convention:**
```
{model_name}_{sample_rate}.index
female_japanese_v2_44100.index
```

**Organization:**
- Always store in `indexes/` directory
- Use same naming structure as model files
- Include metadata in JSON format

### Configuration Files

**Default Locations:**
- Application config: `src/advanced_rvc_inference/assets/config.json`
- User preferences: Stored in user's home directory
- Training configs: `src/advanced_rvc_inference/training/core/training_config.py`

### Audio Files

**Input Files:**
```
input_audio/
├── source_voice.wav
├── target_voice.wav
└── reference_audio.wav
```

**Output Files:**
```
output_audio/
├── converted_voice.wav
├── processed_audio.wav
└── batch_processed/
```

## Import Path Structure

### Absolute Imports (Recommended)

```python
from src.advanced_rvc_inference.core.memory_manager import MemoryManager
from src.advanced_rvc_inference.tabs.inference.full_inference import InferenceTab
from src.advanced_rvc_inference.models.manager import EnhancedModelManager
```

### Relative Imports (Within Package)

```python
# Within tabs/inference/full_inference.py
from ..core.memory_manager import MemoryManager
from ...models.manager import EnhancedModelManager
```

### Installation for Development

```bash
# Install in development mode for absolute imports to work
pip install -e .
```

## Configuration Management

### Singleton Config Pattern

```python
from src.advanced_rvc_inference.config import Config

# Access global configuration
config = Config.get_instance()

# Modify settings
config.set_device("cuda")
config.set_batch_size(8)
config.set_memory_threshold(85)

# Access current settings
device = config.get_device()
batch_size = config.get_batch_size()
```

### Configuration File Structure

```json
{
  "device": "cuda",
  "batch_size": 8,
  "memory_threshold": 85,
  "audio": {
    "sample_rate": 44100,
    "bit_depth": 16,
    "channels": 1
  },
  "models": {
    "weights_path": "weights/",
    "index_path": "indexes/",
    "auto_load": true
  },
  "ui": {
    "theme": "dark",
    "language": "en_US"
  }
}
```

## Memory Management

### Automatic Cleanup

The `MemoryManager` automatically:
- Monitors GPU memory usage
- Cleans up unused tensors
- Prevents Out of Memory errors
- Provides context managers for manual control

### Manual Memory Control

```python
from src.advanced_rvc_inference.core.memory_manager import MemoryManager

# Get current memory usage
usage = MemoryManager.get_memory_usage()

# Force cleanup
MemoryManager.cleanup_memory()

# Use context manager for large operations
with MemoryManager.memory_optimized():
    # Your memory-intensive code here
    result = process_large_audio()
```

## Troubleshooting Structure

### Log File Locations

- **Application Logs**: `logs/app.log`
- **Error Logs**: `logs/error.log`
- **Training Logs**: `logs/training.log`
- **Performance Logs**: `logs/performance.log`

### Debug Mode

```bash
# Enable debug logging
python app.py --debug

# Or set environment variable
export RVC_DEBUG=1
python app.py
```

### Common File Locations

| Purpose | Location | Notes |
|---------|----------|-------|
| Models | `weights/` | .pth and .onnx files |
| Indices | `indexes/` | Feature extraction indices |
| Logs | `logs/` | Application and training logs |
| Config | `src/advanced_rvc_inference/assets/config.json` | Default configuration |
| Cache | `~/.cache/advanced_rvc_inference/` | Downloaded files cache |

This structure ensures:
- **Clear Separation**: Each component has a dedicated location
- **Scalability**: Easy to add new modules and features
- **Maintainability**: Organized code structure for development
- **User Experience**: Intuitive file organization for end users
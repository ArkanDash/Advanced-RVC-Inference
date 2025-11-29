# Python Version Support Update for Advanced RVC Inference

## Changes Made

### 1. Updated Python Version Requirements
- Changed minimum Python requirement from `>=3.8` to `>=3.9`
- Added maximum Python version constraint `<3.14` to maintain compatibility
- Updated classifiers to include Python 3.13 support

### 2. Dependency Updates
- Removed upper limit constraint on `numpy` (was `<2.0.0`, now only `>=1.25.2`)
- Removed upper limit constraint on `pycryptodome` (was `<4.0.0`)
- Updated PyTorch and related packages to use version ranges in installer scripts

### 3. Files Modified
- `pyproject.toml` - Updated project metadata and Python version requirements
- `setup.py` - Updated Python version requirements and classifiers
- `requirements.txt` - Removed restrictive upper bounds on key dependencies
- `advanced_rvc_inference/pyproject.toml` - Updated inner package requirements
- `installer.bat` - Updated PyTorch installation to use version ranges
- `installer.sh` - Updated PyTorch installation to use version ranges

### 4. Testing Scripts Added
- `verify_compatibility.py` - Script to check if all dependencies can be imported
- `test_training_inference.py` - Script to specifically test training and inference functionality

## Compatibility Notes

### Python 3.9-3.13 Support
The project now officially supports Python versions 3.9 through 3.13, with the following considerations:

- **NumPy**: Removed <2.0.0 constraint to allow newer versions compatible with Python 3.12+
- **PyTorch**: Maintained >=2.0.0 requirement which has Python 3.13 support
- **Gradio**: Version 5.x should maintain compatibility with newer Python versions

### Key Dependencies Maintained
- PyTorch ecosystem (torch, torchvision, torchaudio) - All >=2.0.0
- Audio processing libraries (librosa, pydub, pedalboard)
- Machine learning libraries (transformers, scikit-learn, huggingface-hub)
- Web interface (gradio >=5.23.3)

## Testing
Run the following commands to verify the installation:

```bash
# Check basic dependency compatibility
python verify_compatibility.py

# Test training/inference functionality
python test_training_inference.py
```

## Notes for Users
- The project maintains backward compatibility with Python 3.9-3.11
- Python 3.12 and 3.13 compatibility has been enabled
- Some dependencies may still have their own version constraints that could affect compatibility with the newest Python versions
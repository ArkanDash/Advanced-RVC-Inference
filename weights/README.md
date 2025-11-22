# Weights Directory

This directory stores model weights for RVC (Real-time Voice Conversion) models.

## Structure
- `hubert/`: HuBERT model weights
- `rmvpe/`: RMVPE F0 extraction model weights  
- `rvc/`: RVC model weights and configurations
- `pretrained/`: Pre-trained model weights

## Usage
Model weights are automatically downloaded when needed by the application, but can also be manually placed here for offline usage.

## Supported Formats
- .pth (PyTorch models)
- .onnx (ONNX models)
- .bin (Binary weights)
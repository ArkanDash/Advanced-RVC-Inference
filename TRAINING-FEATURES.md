# RVC Training Features - Advanced RVC Inference

This document describes the training capabilities added to Advanced RVC Inference.

## 🎓 Training Features

The Advanced RVC Inference application now includes comprehensive training capabilities that allow users to train their own voice conversion models.

### Main Features

#### 🎤 Model Training
- **Full RVC Training Pipeline**: Complete end-to-end model training from audio data
- **Multiple Architectures**: Support for v1 and v2 model architectures
- **Flexible Configuration**: Configurable epochs, batch size, learning rate, and more
- **GPU Acceleration**: Automatic GPU detection and utilization
- **Mixed Precision Training**: Support for faster training on modern GPUs

#### 🧹 Data Preprocessing
- **Audio Normalization**: Automatic audio level normalization
- **Silence Trimming**: Intelligent silence removal from training data
- **Format Support**: Support for WAV, MP3, FLAC, and other audio formats
- **Quality Analysis**: Dataset quality assessment and recommendations

#### 🔍 Feature Extraction
- **F0 Extraction**: Multiple F0 extraction methods (librosa, RMVPE, CREPE, hybrid)
- **Embedding Models**: Support for Hubert Base, ContentVec, and custom embedding models
- **Mel Spectrograms**: Automatic mel spectrogram extraction for training
- **Batch Processing**: Efficient batch feature extraction

#### 📊 Training Monitoring
- **Real-time Progress**: Live training progress with visual progress bars
- **Loss Tracking**: Comprehensive loss monitoring and logging
- **Model Checkpoints**: Automatic checkpoint saving during training
- **Training Logs**: Detailed training logs and summaries

### Usage

1. **Navigate to Training Tab**: Click the "🎓 Training" tab in the main interface

2. **Configure Model**:
   - Set model name and architecture (v1/v2)
   - Choose sample rate (32k, 40k, 48k)
   - Configure training parameters (epochs, batch size, learning rate)

3. **Prepare Data**:
   - Place audio files in the `dataset` folder
   - Click "🔍 Analyze Dataset" to check data quality
   - Click "1. Preprocess Data" to prepare audio files

4. **Start Training**:
   - Configure advanced settings if needed
   - Click "🎓 Start Training" to begin training
   - Monitor progress in real-time

5. **Download Results**:
   - Trained models are saved in the `weights` folder
   - Training logs and summaries are in the `logs` folder
   - Feature indices are created automatically

### Requirements

#### Training Dependencies Added to `requirements.txt`:
- `tensorboard>=2.13.0` - Training progress monitoring
- `onnx>=1.14.0` - Model format support
- `onnxslim>=0.4.13` - ONNX optimization
- `onnx2torch>=1.5.15` - ONNX to PyTorch conversion
- `webrtcvad-wheels>=2.0.14` - Voice activity detection
- `pysrt>=1.1.2` - Subtitle processing
- `tabulate>=0.9.0` - Progress tables
- `colorama>=0.4.6` - Colored terminal output
- `psutil>=5.9.0` - System monitoring
- `accelerate>=0.20.0` - Training optimization
- `deepspeed>=0.12.0` - Efficient large model training
- `apex>=0.1.0` - NVIDIA Apex for mixed precision

#### Data Requirements:
- **Minimum**: 10 minutes of clean, single-speaker audio
- **Recommended**: 30+ minutes of high-quality, noise-free recordings
- **Format**: WAV, MP3, FLAC, M4A, AAC, OGG
- **Quality**: Low background noise, consistent volume

### Training Configuration

#### Basic Settings:
- **Model Name**: Unique identifier for the trained model
- **Sample Rate**: Audio sample rate (32k, 40k, 48k)
- **Architecture**: v1 (older) or v2 (recommended)
- **Total Epochs**: Number of training iterations (100-300 recommended)
- **Batch Size**: Training batch size (adjust based on GPU memory)
- **Learning Rate**: Optimizer learning rate (0.001 recommended)

#### Advanced Settings:
- **Use Pretrained Models**: Start from pretrained weights
- **Cache in GPU**: Keep training data in GPU memory
- **Save Only Latest**: Save disk space by keeping only latest checkpoint
- **Save Small Weights**: Save lightweight model files during training
- **Custom Dataset Path**: Use custom dataset directory

#### Feature Extraction:
- **F0 Method**: Pitch extraction algorithm
  - `librosa`: Basic F0 extraction (fast)
  - `rmvpe`: High-quality F0 extraction (recommended)
  - `crepe`: Neural network-based F0 extraction
  - `hybrid`: Combination of multiple methods

- **Embedding Model**: Speaker embedding extraction
  - `hubert_base`: Facebook's Hubert model
  - `contentvec`: Advanced content-based embeddings
  - `custom`: Use custom embedding model

### Training Output

#### Generated Files:
- **Model Weights**: `.pth` files in `weights/` directory
- **Training Config**: JSON configuration file
- **Feature Index**: `.json` file in `index/` directory
- **Training Logs**: Text summary in `logs/` directory
- **TensorBoard Logs**: Detailed training metrics (if tensorboard enabled)

#### File Structure:
```
project/
├── dataset/              # Input audio files
├── preprocessed/         # Processed audio files
├── weights/              # Trained model files
├── index/                # Feature index files
├── logs/                 # Training logs and summaries
└── features/             # Extracted features (cache)
```

### GPU Requirements

#### Recommended Hardware:
- **GPU**: NVIDIA RTX 3060 or better
- **VRAM**: 8GB+ for comfortable training
- **RAM**: 16GB+ system memory
- **Storage**: 50GB+ free space

#### Training Performance:
- **RTX 3060 (12GB)**: ~2-4 hours for 100 epochs with 30min dataset
- **RTX 4070 (12GB)**: ~1-2 hours for 100 epochs with 30min dataset
- **RTX 4090 (24GB)**: ~30-60 minutes for 100 epochs with 30min dataset

### Troubleshooting

#### Common Issues:

**Training Fails to Start**:
- Check audio file format and quality
- Ensure sufficient disk space
- Verify GPU drivers and CUDA installation

**Out of Memory Errors**:
- Reduce batch size
- Enable mixed precision training
- Use gradient accumulation

**Poor Training Results**:
- Check dataset quality (noise, consistency)
- Increase training epochs
- Adjust learning rate
- Try different F0 extraction methods

**Slow Training**:
- Enable GPU acceleration
- Use mixed precision training
- Optimize batch size for your GPU
- Consider using smaller sample rates

### Advanced Usage

#### Custom Training Scripts:
The training system can also be used programmatically:
```python
from programs.training.simple_trainer import simple_train_rvc_model, create_training_config

# Create configuration
config = create_training_config(
    model_name="my_custom_model",
    total_epochs=200,
    batch_size=16,
    learning_rate=0.0005
)

# Start training
success = simple_train_rvc_model(config)
```

#### Integration with Inference:
Trained models are automatically available in the inference tab and can be used immediately for voice conversion.

### Support

For training-related issues:
1. Check the training logs in the `logs/` directory
2. Verify dataset quality using the analyze feature
3. Review the troubleshooting section above
4. Consult the original RVC documentation for technical details

This training system provides a complete solution for training custom voice conversion models with the Advanced RVC Inference platform.

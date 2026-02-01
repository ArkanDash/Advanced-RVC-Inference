<div align="center">

# Advanced RVC Inference

Advanced RVC Inference presents itself as a state-of-the-art web UI crafted to streamline rapid and effortless inference. This comprehensive toolset encompasses a model downloader, a voice splitter, training and more.

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArkanDash/Advanced-RVC-Inference/blob/master/Advanced-RVC.ipynb)

</div>


> [!NOTE]  
> Advanced RVC Inference will no longer receive frequent updates. Going forward, development will focus mainly on security patches, dependency updates, and occasional feature improvements. This is because the project is already stable and mature with limited room for further improvements. Pull requests are still welcome and will be reviewed.




# Getting Started

## Installation

```bash
pip install git+https://github.com/ArkanDash/Advanced-RVC-Inference.git
```

### With GPU Support

For CUDA-enabled GPUs:

```bash
pip install git+https://github.com/ArkanDash/Advanced-RVC-Inference.git#egg=advanced-rvc-inference[gpu]
```

### From Source

```bash
git clone https://github.com/ArkanDash/Advanced-RVC-Inference.git
cd Advanced-RVC-Inference
pip install -e .
```


## Quick Start

### Web Interface

Launch the Gradio web UI:

```bash
rvc-gui
```

```
# or
python -m advanced_rvc_inference.app.gui
```

The web interface will be available at `http://localhost:7860`

### Command Line Interface

see guides more at [Wiki](https://github.com/ArkanDash/Advanced-RVC-Inference/wiki/Cli-Guide)!




### Quick Inference

Run voice conversion on a single audio file:

```bash
rvc-cli infer --model path/to/model.pth --input audio.wav --output converted.wav
```

With pitch shift (one octave up):

```bash
rvc-cli infer --model vocals.pth --input audio.wav --pitch 12 --output output.wav
```

### Batch Processing

Process multiple audio files at once:

```bash
rvc-cli infer-batch --model model.pth --input_dir ./songs --output_dir ./converted
```

### Music Separation

Separate vocals from instrumental tracks:

```bash
rvc-cli separate --input song.mp3 --output_dir ./separated
```

### Web Interface

Launch the Gradio web UI:

```bash
rvc-cli serve --port 7860
```

View help for any command:

```bash
rvc-cli --help
rvc-cli infer --help
rvc-cli separate --help
```




## Documentation

- [API Reference](https://github.com/ArkanDash/Advanced-RVC-Inference#api-reference)
- [Usage Guide](https://github.com/ArkanDash/Advanced-RVC-Inference#usage)
- [Contributing](CONTRIBUTING.md)

## Troubleshooting

### GPU Not Detected

Ensure you have CUDA installed and PyTorch with CUDA support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```


## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Terms of Use

The use of the converted voice for the following purposes is prohibited:

- Criticizing or attacking individuals
- Advocating for or opposing specific political positions, religions, or ideologies
- Publicly displaying strongly stimulating expressions without proper zoning
- Selling of voice models and generated voice clips
- Impersonation of the original owner of the voice with malicious intentions
- Fraudulent purposes that lead to identity theft or fraudulent phone calls

## Credits

| Repository | Owner |
|------------|-------|
| [Vietnamese-RVC](https://github.com/PhamHuynhAnh16/Vietnamese-RVC) | Phạm Huỳnh Anh |
| [Applio](https://github.com/IAHispano/Applio) | IAHispano |
| [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) | Nomad Karaoke |
| [whisper](https://github.com/openai/whisper) | OpenAI |

## Support

For issues and feature requests, please use the [GitHub Issues](https://github.com/ArkanDash/Advanced-RVC-Inference/issues) page.

# Vocoder Reference Guide

> Advanced RVC Inference supports **12 vocoders** for audio synthesis, each with different architectures, strengths, and quality characteristics. This guide provides detailed descriptions, ratings, and recommendations.

## Quick Reference

| Rating | Vocoder | Category | Key Feature |
|--------|---------|----------|-------------|
| ⭐⭐⭐⭐⭐ | **BigVGAN** | Anti-Aliased GAN | SnakeBeta + AMP blocks, highest quality |
| ⭐⭐⭐⭐⭐ | **Default (HiFi-GAN NSF)** | HiFi-GAN | Most compatible, best tested |
| ⭐⭐⭐⭐½ | **HiFi-GAN-v3** | Enhanced HiFi-GAN | SnakeBeta activations, wider dilations |
| ⭐⭐⭐⭐½ | **MRF-HiFi-GAN** | Multi-Receptive Field | MRF blocks for richer features |
| ⭐⭐⭐⭐½ | **Vocos** | Fourier-based | ISTFT output, lightweight |
| ⭐⭐⭐⭐½ | **RingFormer** | Conformer-based | iSTFT + attention, phase prediction |
| ⭐⭐⭐⭐ | **JVSF-HiFi-GAN** | Source-Filter | Separate source/filter modeling |
| ⭐⭐⭐⭐ | **RefineGAN** | U-Net GAN | Skip connections, parallel ResBlocks |
| ⭐⭐⭐⭐ | **NSF-APNet** | Hybrid NSF | All-pass phase correction |
| ⭐⭐⭐⭐ | **FullBand-MRF** | Enhanced MRF | Wider dilations, channel attention |
| ⭐⭐⭐½ | **WaveGlow** | Flow-based | Invertible convolutions, WaveNet layers |
| ⭐⭐⭐½ | **PCPH-GAN** | Phase-Corrected | PCHIP interpolation, harmonic generator |

---

## Tier 1: Best Quality (⭐⭐⭐⭐⭐)

### BigVGAN
- **Rating:** ⭐⭐⭐⭐⭐ (5.0/5)
- **Category:** Anti-Aliased GAN
- **Paper:** "BigVGAN: A Universal Neural Vocoder with Large-Scale Training" (2023)

BigVGAN is the highest-quality vocoder available in the system. It introduces two key innovations: Snake activations with Anti-Aliasing (SnakeBeta and Anti-Aliased Multi-Period/AMP blocks) and data-augmented adversarial training. The Snake activation function provides a periodic, non-monotonic nonlinearity that is naturally suited for audio signals, while the anti-aliased design prevents high-frequency artifacts during upsampling. BigVGAN uses kaiser-sinc filters for both upsampling and downsampling, achieving state-of-the-art audio quality across multiple benchmarks. Its architecture includes extensive AMP blocks with parallel branches at different periods, capturing both fine and coarse spectral details.

**Key Features:**
- SnakeBeta learnable periodic activations
- Anti-Aliased Multi-Period (AMP) blocks
- Kaiser-sinc filters for upsampling/downsampling
- Data-augmented adversarial training
- Superior high-frequency reconstruction

**Recommended for:** Maximum audio quality. Best for singing voice conversion and high-fidelity speech synthesis.

---

### Default (HiFi-GAN NSF)
- **Rating:** ⭐⭐⭐⭐⭐ (5.0/5)
- **Category:** HiFi-GAN
- **Source:** `library/generators/nsf_hifigan.py`

The Default vocoder is the HiFi-GAN with Neural Sine Filter (NSF). This is the most widely tested and compatible vocoder in the RVC ecosystem. It combines HiFi-GAN's transposed convolution upsampling with a Neural Sine Filter that injects harmonic information directly into each upsampling layer. The NSF source module generates sine waves conditioned on F0, which are mixed with the upsampled features through noise convolution layers. This vocoder strikes the best balance between audio quality, training stability, and compatibility with existing pre-trained models.

**Key Features:**
- Most compatible with existing RVC models
- Neural Sine Filter for harmonic injection
- Noise convolutions at each upsampling layer
- Supports gradient checkpointing
- Well-tested across thousands of models

**Recommended for:** Default choice for all training. Best compatibility with existing pre-trained weights and community models.

---

## Tier 2: Excellent Quality (⭐⭐⭐⭐½)

### HiFi-GAN-v3
- **Rating:** ⭐⭐⭐⭐½ (4.5/5)
- **Category:** Enhanced HiFi-GAN
- **Source:** `library/generators/hifigan_v3.py`

HiFi-GAN-v3 is an enhanced variant of the standard HiFi-GAN architecture that incorporates SnakeBeta activations and wider dilation patterns in its residual blocks. Unlike the original HiFi-GAN which uses LeakyReLU activations, v3 uses the same Snake activation that made BigVGAN successful, providing periodic nonlinearity that better matches audio signal characteristics. The wider dilation pattern in residual blocks captures longer-range temporal dependencies, improving the modeling of speech formant transitions and vocal tract resonances.

**Key Features:**
- SnakeBeta learnable activations
- Enhanced residual blocks with wider dilations
- Better formant transition modeling
- Gradient checkpointing support
- Drop-in upgrade over standard HiFi-GAN

**Recommended for:** Users who want a quality upgrade over Default without the computational cost of BigVGAN.

---

### MRF-HiFi-GAN
- **Rating:** ⭐⭐⭐⭐½ (4.5/5)
- **Category:** Multi-Receptive Field
- **Source:** `library/generators/mrf_hifigan.py`

MRF-HiFi-GAN replaces the standard residual blocks with Multi-Receptive Field (MRF) blocks. Each MRF block contains a sequence of MRFLayers with different dilation stacks, allowing the network to capture features at multiple temporal scales simultaneously. This multi-scale approach is particularly effective for speech synthesis because speech contains information at multiple time scales — from fine-grained spectral details to broader prosodic patterns. The SineGenerator provides harmonic conditioning with `harmonic_num=8`.

**Key Features:**
- Multi-Receptive Field fusion blocks
- Multiple dilation stacks per block
- SineGenerator with 8 harmonics
- Richer multi-scale feature extraction
- Gradient checkpointing support

**Recommended for:** Speech with complex spectral characteristics. Good for multi-speaker models where diverse voice qualities need to be captured.

---

### Vocos
- **Rating:** ⭐⭐⭐⭐½ (4.5/5)
- **Category:** Fourier-based
- **Source:** `library/generators/vocos.py`

Vocos takes a fundamentally different approach to audio synthesis by using inverse Short-Time Fourier Transform (iSTFT) for waveform reconstruction instead of traditional transposed convolution upsampling. This architecture is inherently more lightweight because it operates in the frequency domain, where the waveform can be reconstructed analytically. The backbone uses 1D convolutions with Snake activations for feature extraction, and the iSTFT head predicts both magnitude and phase for clean waveform output. This Fourier-based approach naturally handles pitch and harmonic structure.

**Key Features:**
- iSTFT-based waveform reconstruction (no transposed convolutions)
- Lightweight and computationally efficient
- Snake activations in backbone
- Magnitude + phase prediction
- Naturally suited for harmonic signals

**Recommended for:** Users who want fast inference and lightweight models. Excellent for real-time applications.

---

### RingFormer
- **Rating:** ⭐⭐⭐⭐½ (4.5/5)
- **Category:** Conformer-based
- **Source:** `library/algorithm/generators/ringformer.py`

RingFormer is the most architecturally advanced vocoder in the system, using Conformer blocks with both time and frequency attention mechanisms. It employs `einops`-based rearrange operations for efficient multi-head attention across both dimensions. The output layer uses iSTFT (TorchSTFT) for waveform generation from predicted magnitude and phase, similar to Vocos but with the added power of Conformer attention. RingFormer uses ResBlock_SnakeBeta for its convolutional components and SourceModuleHnNSF with `harmonic_num=8` for harmonic conditioning.

**Key Features:**
- Conformer attention (time + frequency)
- iSTFT output with magnitude + phase prediction
- SnakeBeta activations
- Most complex architecture
- Unique 3-output return (audio, spec, phase)

**Recommended for:** Advanced users seeking maximum quality through attention-based processing. Higher memory usage than other vocoders.

---

## Tier 3: Very Good Quality (⭐⭐⭐⭐)

### JVSF-HiFi-GAN
- **Rating:** ⭐⭐⭐⭐ (4.0/5)
- **Category:** Source-Filter
- **Source:** `library/generators/jvsf_hifigan.py`

Joint-Variable Source-Filter HiFi-GAN explicitly separates the voice generation process into source (excitation) and filter (vocal tract) components, mirroring the human speech production model. The source module generates learnable harmonic components with adjustable weights, while the filter module uses a pyramid of dilated convolutions to model the vocal tract transfer function. This separation allows the model to independently control pitch-related and timbre-related features, leading to more natural voice conversion results.

**Key Features:**
- Explicit source-filter separation
- Learnable harmonic weights
- Filter module with dilation pyramid
- Physically motivated architecture
- Good for voice identity preservation

**Recommended for:** Voice conversion tasks where preserving speaker identity is critical. The source-filter separation helps maintain natural vocal characteristics.

---

### RefineGAN
- **Rating:** ⭐⭐⭐⭐ (4.0/5)
- **Category:** U-Net GAN
- **Source:** `library/generators/refinegan.py`

RefineGAN uses a U-Net architecture with skip connections, a significant departure from the purely feedforward design of HiFi-GAN. The harmonic downsampling path processes F0 through sine generation, pre-convolution, and progressive downsampling using torchaudio's resample function. The upsampling path uses ParallelResBlocks with three parallel branches (kernel sizes 3, 7, 11) combined through AdaIN noise injection. Skip connections from the encoder to decoder preserve fine spectral details that might otherwise be lost during the compression-expansion process.

**Key Features:**
- U-Net architecture with skip connections
- Parallel ResBlocks (3/7/11 kernel branches)
- AdaIN noise injection
- Anti-aliased harmonic downsampling
- Progressive refinement through skip connections

**Recommended for:** High-fidelity audio where spectral detail preservation is important. Good for singing and complex vocal passages.

---

### NSF-APNet
- **Rating:** ⭐⭐⭐⭐ (4.0/5)
- **Category:** Hybrid NSF
- **Source:** `library/generators/nsf_apnet.py`

NSF-APNet combines the Neural Sine Filter with an All-Pass Network for improved phase modeling. The All-Pass Network applies magnitude-preserving phase corrections at each residual block, addressing a common weakness of GAN-based vocoders: inaccurate phase reconstruction. By refining the phase component separately from the magnitude, NSF-APNet achieves more natural-sounding output, particularly in the higher frequency ranges where phase errors are most perceptible. This hybrid approach maintains the harmonic conditioning of NSF while adding sophisticated phase correction.

**Key Features:**
- Neural Sine Filter for harmonic generation
- All-Pass Network for phase correction
- Magnitude-preserving processing
- Better high-frequency phase accuracy
- Gradient checkpointing support

**Recommended for:** Audio with prominent high-frequency content. The phase correction reduces metallic artifacts in synthesized speech.

---

### FullBand-MRF
- **Rating:** ⭐⭐⭐⭐ (4.0/5)
- **Category:** Enhanced MRF
- **Source:** `library/generators/fullband_mrf.py`

FullBand-MRF extends the MRF-HiFi-GAN concept with significantly wider dilation patterns (up to dilation rate 15) and channel attention mechanisms. The wider dilations allow the network to capture longer-range dependencies in the speech signal, which is particularly important for modeling vocal formant transitions and coarticulation effects. Channel attention helps the network focus on the most informative frequency bands dynamically. Snake activations provide periodic nonlinearity that matches the quasi-periodic nature of speech signals.

**Key Features:**
- Extended Multi-Receptive Field (dilations up to 15)
- Channel attention mechanism
- Snake activations
- Longer-range temporal modeling
- Better coarticulation handling

**Recommended for:** Speech with rapid formant transitions or wide pitch ranges. The extended receptive field captures longer temporal context.

---

## Tier 4: Good Quality (⭐⭐⭐½)

### WaveGlow
- **Rating:** ⭐⭐⭐½ (3.5/5)
- **Category:** Flow-based
- **Source:** `library/generators/waveglow.py`

WaveGlow is a flow-based vocoder that uses invertible 1×1 convolutions and WaveNet-style layers with gated activations (tanh × sigmoid) for audio generation. Unlike GAN-based vocoders that learn through adversarial training, WaveGlow learns the exact probability distribution of audio waveforms through maximum likelihood estimation with invertible transformations. This provides more stable training but typically requires more computation. The WaveNet layers capture temporal dependencies through dilated causal convolutions with skip connections.

**Key Features:**
- Flow-based (invertible) architecture
- Invertible 1×1 convolutions
- WaveNet layers with gated activations
- Maximum likelihood training (no GAN instability)
- Deterministic and reproducible

**Recommended for:** Users who prefer deterministic models. WaveGlow's flow-based approach avoids adversarial training instability.

---

### PCPH-GAN
- **Rating:** ⭐⭐⭐½ (3.5/5)
- **Category:** Phase-Corrected
- **Source:** `library/algorithm/generators/pcph_gan.py`

Phase-Corrected Parallel HiFi-GAN uses PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) interpolation for F0 upsampling, providing smoother pitch contour transitions compared to nearest-neighbor or linear interpolation. The PCPH generator models pitch harmonics with a specialized source module, while SiLU (Sigmoid Linear Unit) activations and SnakeBeta provide smooth nonlinearity. A progressive upsampling path with downsampling pre-processing creates a coarse-to-fine synthesis pipeline that naturally handles multi-scale speech features.

**Key Features:**
- PCHIP interpolation for smooth F0 upsampling
- Phase-Corrected Pulse Harmonic generator
- Coarse-to-fine synthesis pipeline
- SiLU + SnakeBeta activations
- Progressive upsampling with pre-processing

**Recommended for:** Singing voice conversion where smooth pitch transitions are important. The PCHIP interpolation preserves melodic contour accuracy.

---

## Recommendations for RVC Training

### Beginner
Use **Default (HiFi-GAN NSF)**. It's the most tested, most compatible, and provides excellent quality. Works with all pre-trained weights and community models.

### Intermediate
Try **BigVGAN** for the highest quality, or **HiFi-GAN-v3** for a lighter upgrade. Both provide noticeable quality improvements over the default.

### Advanced
Experiment with **RingFormer** for attention-based synthesis, **Vocos** for lightweight inference, or **MRF-HiFi-GAN** for multi-scale feature extraction.

### Real-time / Low Latency
Use **Vocos** — its Fourier-based architecture is inherently faster at inference since it avoids expensive transposed convolution upsampling.

### Maximum Quality
Use **BigVGAN** — it consistently achieves the highest objective and subjective quality scores across all benchmarks.

---

## Technical Notes

- All non-Default vocoders require **v2 + pitch guidance** (enforced by UI)
- Non-Default vocoders use **fp32** at inference (fp16 disabled for stability)
- Vocoder choice is saved in the model checkpoint and used automatically during inference
- Pre-trained weights for non-Default vocoders follow the naming pattern: `{VocoderName}_f0G48k.pth`
- All vocoders support gradient checkpointing except BigVGAN and WaveGlow
- The vocoder registry is in `advanced_rvc_inference/library/generators/__init__.py`

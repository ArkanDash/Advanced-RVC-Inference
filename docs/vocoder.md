# Vocoder Reference Guide

> Advanced RVC Inference supports **4 vocoders** for audio synthesis, matching the vocoder support from Vietnamese-RVC (VRVC). Each vocoder has different architectures, strengths, and quality characteristics. This guide provides detailed descriptions, ratings, and recommendations.

## Quick Reference

| Rating | Vocoder | Category | Key Feature |
|--------|---------|----------|-------------|
| ⭐⭐⭐⭐⭐ | **Default** (HiFi-GAN NSF) | HiFi-GAN | Neural Sine Filter, harmonic injection |
| ⭐⭐⭐⭐⭐ | **BigVGAN** | Anti-Aliased GAN | SnakeBeta + AMP blocks, highest quality |
| ⭐⭐⭐⭐½ | **MRF-HiFi-GAN** | Multi-Receptive Field | MRF blocks for richer features |
| ⭐⭐⭐⭐ | **RefineGAN** | U-Net GAN | Skip connections, parallel ResBlocks |

---

## Default (HiFi-GAN NSF)
- **Rating:** ⭐⭐⭐⭐⭐ (5.0/5)
- **Category:** HiFi-GAN
- **Registry Key:** `"Default"`
- **Source:** `models/generators/nsf_hifigan.py`
- **Class:** `HiFiGANNRFGenerator`

The Default vocoder is the HiFi-GAN with Neural Sine Filter (NSF), and the recommended vocoder for best compatibility. It combines HiFi-GAN's transposed convolution upsampling with a Neural Sine Filter that injects harmonic information directly into each upsampling layer. The NSF source module generates sine waves conditioned on F0, which are mixed with the upsampled features through noise convolution layers. This vocoder provides improved pitch accuracy compared to standard HiFi-GAN due to the explicit harmonic conditioning. It is the default vocoder selected in both the UI and CLI, and the only vocoder available for V1 models.

**Key Features:**
- Neural Sine Filter for harmonic injection at each upsampling layer
- Noise convolutions for mixing harmonic and learned features
- Improved pitch accuracy from explicit harmonic conditioning
- Supports gradient checkpointing
- Requires pitch guidance (f0)
- Default selection in UI and CLI

**Recommended for:** Best compatibility. The default choice for all training. Works best when pitch accuracy is critical, such as singing voice and tonal languages.

---

## BigVGAN
- **Rating:** ⭐⭐⭐⭐⭐ (5.0/5)
- **Category:** Anti-Aliased GAN
- **Registry Key:** `"BigVGAN"`
- **Source:** `models/generators/bigvgan.py`
- **Class:** `BigVGANGenerator`
- **Paper:** "BigVGAN: A Universal Neural Vocoder with Large-Scale Training" (2023)

BigVGAN is the highest-quality vocoder available in the system. It introduces two key innovations: Snake activations with Anti-Aliasing (SnakeBeta and Anti-Aliased Multi-Period/AMP blocks) and data-augmented adversarial training. The Snake activation function provides a periodic, non-monotonic nonlinearity that is naturally suited for audio signals, while the anti-aliased design prevents high-frequency artifacts during upsampling. BigVGAN uses kaiser-sinc filters for both upsampling and downsampling, achieving state-of-the-art audio quality across multiple benchmarks. Its architecture includes extensive AMP blocks with parallel branches at different periods, capturing both fine and coarse spectral details. During training, BigVGAN uses the v3 discriminator for improved adversarial signal.

**Key Features:**
- SnakeBeta learnable periodic activations
- Anti-Aliased Multi-Period (AMP) blocks
- Kaiser-sinc filters for upsampling/downsampling
- Data-augmented adversarial training
- Superior high-frequency reconstruction
- Uses v3 discriminator during training
- Uses fp32 at inference (fp16 disabled for stability)

**Recommended for:** Maximum audio quality. Best for singing voice conversion and high-fidelity speech synthesis where quality is the top priority.

---

## MRF-HiFi-GAN
- **Rating:** ⭐⭐⭐⭐½ (4.5/5)
- **Category:** Multi-Receptive Field
- **Registry Key:** `"MRF-HiFi-GAN"`
- **Source:** `models/generators/mrf_hifigan.py`
- **Class:** `HiFiGANMRFGenerator`

MRF-HiFi-GAN replaces the standard residual blocks with Multi-Receptive Field (MRF) blocks. Each MRF block contains a sequence of MRFLayers with different dilation stacks, allowing the network to capture features at multiple temporal scales simultaneously. This multi-scale approach is particularly effective for speech synthesis because speech contains information at multiple time scales — from fine-grained spectral details to broader prosodic patterns. The SineGenerator provides harmonic conditioning with `harmonic_num=8`. The synthesizer also accepts `"MRF HiFi-GAN"` (with space instead of hyphen) as an alias for backward compatibility.

**Key Features:**
- Multi-Receptive Field fusion blocks
- Multiple dilation stacks per block
- SineGenerator with 8 harmonics
- Richer multi-scale feature extraction
- Supports gradient checkpointing
- Uses fp32 at inference (fp16 disabled for stability)

**Recommended for:** Speech with complex spectral characteristics. Good for multi-speaker models where diverse voice qualities need to be captured across different temporal scales.

---

## RefineGAN
- **Rating:** ⭐⭐⭐⭐ (4.0/5)
- **Category:** U-Net GAN
- **Registry Key:** `"RefineGAN"`
- **Source:** `models/generators/refinegan.py`
- **Class:** `RefineGANGenerator`

RefineGAN uses a U-Net architecture with skip connections, a significant departure from the purely feedforward design of HiFi-GAN. The harmonic downsampling path processes F0 through sine generation, pre-convolution, and progressive downsampling using torchaudio's resample function. The upsampling path uses ParallelResBlocks with three parallel branches (kernel sizes 3, 7, 11) combined through AdaIN noise injection. Skip connections from the encoder to decoder preserve fine spectral details that might otherwise be lost during the compression-expansion process. During training, RefineGAN uses the v3 discriminator for improved adversarial signal.

**Key Features:**
- U-Net architecture with skip connections
- Parallel ResBlocks (3/7/11 kernel branches)
- AdaIN noise injection
- Anti-aliased harmonic downsampling
- Progressive refinement through skip connections
- Uses v3 discriminator during training
- Uses fp32 at inference (fp16 disabled for stability)

**Recommended for:** High-fidelity audio where spectral detail preservation is important. Good for singing and complex vocal passages where fine-grained detail matters.

---

## Non-f0 Mode (Plain HiFi-GAN)

When training **without pitch guidance** (`pitch_guidance=False`), the synthesizer automatically uses a plain HiFi-GAN generator (`HiFiGANGenerator` from `models/generators/hifigan.py`) regardless of the vocoder name selected. This is a separate, simpler HiFi-GAN without the Neural Sine Filter — it uses standard transposed convolution upsampling with weight-normalized residual blocks. The vocoder selection in the UI is locked to "Default" when pitch guidance is disabled.

---

## UI Business Rules

The following rules are enforced by the UI (`arvc/ui/feedback.py`):

- **V1 models** can only use the **Default** vocoder (locked via `unlock_vocoder()`)
- **V2 models** can use all 4 vocoders
- **No pitch guidance (f0=False)** → vocoder is locked to **Default** (via `vocoders_lock()`)
- **Non-Default vocoders** force pitch guidance ON (via `pitch_guidance_lock()`)
- **Non-Default vocoders** use **fp32** at inference (fp16 disabled for stability)

---

## Recommendations for RVC Training

### Beginner
Use **Default (HiFi-GAN NSF)**. It's the default for a reason — best compatibility, good quality, and works reliably across all scenarios. The harmonic injection improves pitch accuracy out of the box.

### Intermediate
Try **BigVGAN** for the highest audio quality. It consistently achieves the best objective and subjective quality scores across all benchmarks. The Snake activations and anti-aliased design produce noticeably cleaner output.

### Advanced
Experiment with **MRF-HiFi-GAN** for multi-scale feature extraction, or **RefineGAN** for spectral detail preservation through its U-Net skip connections. Both offer unique quality characteristics for specific use cases.

### Maximum Quality
Use **BigVGAN** — it consistently achieves the highest objective and subjective quality scores across all benchmarks.

---

## Technical Notes

- V1 models are locked to the Default vocoder (enforced by UI)
- Non-Default vocoders require **v2 + pitch guidance** (enforced by UI)
- Non-Default vocoders use **fp32** at inference (fp16 disabled for stability)
- Vocoder choice is saved in the model checkpoint and used automatically during inference
- Pre-trained weights for non-Default vocoders follow the naming pattern: `{VocoderName}_f0G48k.pth`
- BigVGAN and RefineGAN use the v3 discriminator during training
- The vocoder registry is in `arvc/engine/models/generators/__init__.py`

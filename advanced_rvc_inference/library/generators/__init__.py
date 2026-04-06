"""
Vocoder Registry for Advanced RVC Inference.

Provides a centralized registry of all available vocoder generators,
including metadata (rating, category, description) and utility functions
for querying available vocoders.
"""

from typing import Dict, List, Optional, Any


VOCODER_REGISTRY: Dict[str, Dict[str, Any]] = {
    "HiFi-GAN": {
        "module": "advanced_rvc_inference.library.generators.hifigan",
        "class": "HiFiGANGenerator",
        "display_name": "HiFi-GAN",
        "rating": 100,
        "category": "HiFi-GAN",
        "description": "Standard HiFi-GAN vocoder. The classic neural vocoder using transposed convolution upsampling with weight-normalized residual blocks. Lightweight, fast, and well-tested.",
        "requires_f0": False,
        "is_default": True,
    },
    "Default": {
        "module": "advanced_rvc_inference.library.generators.nsf_hifigan",
        "class": "HiFiGANNRFGenerator",
        "display_name": "Default (HiFi-GAN NSF)",
        "rating": 98,
        "category": "HiFi-GAN",
        "description": "HiFi-GAN with Neural Sine Filter (NSF). Adds harmonic sine wave injection at each upsampling layer for improved pitch accuracy.",
        "requires_f0": True,
        "is_default": False,
    },
    "MRF-HiFi-GAN": {
        "module": "advanced_rvc_inference.library.generators.mrf_hifigan",
        "class": "HiFiGANMRFGenerator",
        "display_name": "MRF-HiFi-GAN",
        "rating": 90,
        "category": "HiFi-GAN",
        "description": "HiFi-GAN with Multi-Receptive Field (MRF) fusion. Uses MRF blocks instead of standard residual blocks for richer feature extraction.",
        "requires_f0": True,
        "is_default": False,
    },
    "RefineGAN": {
        "module": "advanced_rvc_inference.library.generators.refinegan",
        "class": "RefineGANGenerator",
        "display_name": "RefineGAN",
        "rating": 85,
        "category": "RefineGAN",
        "description": "U-Net based vocoder with parallel residual blocks and anti-aliased resampling. Produces high-fidelity audio with good spectral detail.",
        "requires_f0": True,
        "is_default": False,
    },
    "BigVGAN": {
        "module": "advanced_rvc_inference.library.generators.bigvgan",
        "class": "BigVGANGenerator",
        "display_name": "BigVGAN",
        "rating": 95,
        "category": "BigVGAN",
        "description": "Uses Snake activations with Anti-Aliasing (SnakeBeta + AMP blocks). State-of-the-art neural vocoder with superior audio quality.",
        "requires_f0": True,
        "is_default": False,
    },
    "RingFormer": {
        "module": "advanced_rvc_inference.library.algorithm.generators",
        "class": "RingFormerGenerator",
        "display_name": "RingFormer",
        "rating": 88,
        "category": "RingFormer",
        "description": "RingFormer-based vocoder with ring-structured attention mechanisms. Good balance between quality and computational efficiency.",
        "requires_f0": True,
        "is_default": False,
    },
    "PCPH-GAN": {
        "module": "advanced_rvc_inference.library.algorithm.generators",
        "class": "PCPH_GAN_Generator",
        "display_name": "PCPH-GAN",
        "rating": 82,
        "category": "PCPH-GAN",
        "description": "Phase-Corrected Parallel HiFi-GAN variant with improved phase modeling for more natural sounding output.",
        "requires_f0": True,
        "is_default": False,
    },
    "Vocos": {
        "module": "advanced_rvc_inference.library.generators.vocos",
        "class": "VocosGenerator",
        "display_name": "Vocos",
        "rating": 87,
        "category": "Fourier-based",
        "description": "Fourier-based vocoder using inverse STFT for waveform reconstruction instead of transposed convolutions. Lightweight and efficient with Snake activations.",
        "requires_f0": True,
        "is_default": False,
    },
    "HiFi-GAN-v3": {
        "module": "advanced_rvc_inference.library.generators.hifigan_v3",
        "class": "HiFiGANV3Generator",
        "display_name": "HiFi-GAN v3",
        "rating": 83,
        "category": "HiFi-GAN",
        "description": "Improved HiFi-GAN with SnakeBeta activations and enhanced residual blocks with wider dilated convolutions. Better spectral fidelity than vanilla HiFi-GAN.",
        "requires_f0": True,
        "is_default": False,
    },
    "JVSF-HiFi-GAN": {
        "module": "advanced_rvc_inference.library.generators.jvsf_hifigan",
        "class": "JVSFHiFiGANGenerator",
        "display_name": "JVSF-HiFi-GAN",
        "rating": 80,
        "category": "Source-Filter",
        "description": "Joint-Variable Source-Filter HiFi-GAN that models source (harmonics) and filter (spectral envelope) separately for more controllable synthesis.",
        "requires_f0": True,
        "is_default": False,
    },
    "WaveGlow": {
        "module": "advanced_rvc_inference.library.generators.waveglow",
        "class": "WaveGlowGenerator",
        "display_name": "WaveGlow",
        "rating": 75,
        "category": "Flow-based",
        "description": "Flow-based vocoder using invertible 1x1 convolutions and WaveNet-like layers with gated activations. Fully invertible architecture.",
        "requires_f0": True,
        "is_default": False,
    },
    "NSF-APNet": {
        "module": "advanced_rvc_inference.library.generators.nsf_apnet",
        "class": "NSFAPNetGenerator",
        "display_name": "NSF-APNet",
        "rating": 78,
        "category": "Hybrid",
        "description": "Hybrid of Neural Sine Filter and All-Pass Network for improved phase correction. Combines harmonic generation with phase-aware processing.",
        "requires_f0": True,
        "is_default": False,
    },
    "FullBand-MRF": {
        "module": "advanced_rvc_inference.library.generators.fullband_mrf",
        "class": "FullBandMRFGenerator",
        "display_name": "FullBand-MRF",
        "rating": 86,
        "category": "MRF",
        "description": "Enhanced MRF vocoder with wider receptive fields and full-band mel processing. Snake activations for improved high-frequency detail.",
        "requires_f0": True,
        "is_default": False,
    },
}


def get_vocoder_choices() -> List[str]:
    """Return all vocoder names sorted by rating (highest first)."""
    sorted_vocoders = sorted(
        VOCODER_REGISTRY.items(),
        key=lambda item: item[1]["rating"],
        reverse=True,
    )
    return [name for name, _ in sorted_vocoders]


def get_vocoder_info(name: str) -> Optional[Dict[str, Any]]:
    """Return metadata for a specific vocoder by name."""
    return VOCODER_REGISTRY.get(name)


def get_vocoders_by_category(category: str) -> List[str]:
    """Return vocoder names belonging to a specific category."""
    return [
        name
        for name, info in VOCODER_REGISTRY.items()
        if info["category"] == category
    ]


def get_default_vocoder() -> str:
    """Return the default vocoder name."""
    for name, info in VOCODER_REGISTRY.items():
        if info.get("is_default", False):
            return name
    return "HiFi-GAN"


def get_f0_vocoders() -> List[str]:
    """Return all vocoders that require f0 (pitch) input."""
    return [
        name
        for name, info in VOCODER_REGISTRY.items()
        if info.get("requires_f0", False)
    ]

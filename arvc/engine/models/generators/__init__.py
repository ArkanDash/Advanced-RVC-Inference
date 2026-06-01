"""
Vocoder Registry for Advanced RVC Inference.

Provides a centralized registry of all available vocoder generators,
matching the vocoder support from Vietnamese-RVC (VRVC).
"""

from typing import Dict, List, Optional, Any


VOCODER_REGISTRY: Dict[str, Dict[str, Any]] = {
    "Default": {
        "module": "arvc.engine.models.generators.nsf_hifigan",
        "class": "HiFiGANNRFGenerator",
        "display_name": "Default (HiFi-GAN NSF)",
        "rating": 100,
        "category": "HiFi-GAN",
        "description": "HiFi-GAN with Neural Sine Filter (NSF). Adds harmonic sine wave injection at each upsampling layer for improved pitch accuracy. Recommended for best compatibility.",
        "requires_f0": True,
        "is_default": True,
    },
    "BigVGAN": {
        "module": "arvc.engine.models.generators.bigvgan",
        "class": "BigVGANGenerator",
        "display_name": "BigVGAN",
        "rating": 95,
        "category": "BigVGAN",
        "description": "Uses Snake activations with Anti-Aliasing (SnakeBeta + AMP blocks). State-of-the-art neural vocoder with superior audio quality.",
        "requires_f0": True,
        "is_default": False,
    },
    "MRF-HiFi-GAN": {
        "module": "arvc.engine.models.generators.mrf_hifigan",
        "class": "HiFiGANMRFGenerator",
        "display_name": "MRF-HiFi-GAN",
        "rating": 90,
        "category": "HiFi-GAN",
        "description": "HiFi-GAN with Multi-Receptive Field (MRF) fusion. Uses MRF blocks instead of standard residual blocks for richer feature extraction.",
        "requires_f0": True,
        "is_default": False,
    },
    "RefineGAN": {
        "module": "arvc.engine.models.generators.refinegan",
        "class": "RefineGANGenerator",
        "display_name": "RefineGAN",
        "rating": 85,
        "category": "RefineGAN",
        "description": "U-Net based vocoder with parallel residual blocks and anti-aliased resampling. Produces high-fidelity audio with good spectral detail.",
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
    return "Default"


def get_f0_vocoders() -> List[str]:
    """Return all vocoders that require f0 (pitch) input."""
    return [
        name
        for name, info in VOCODER_REGISTRY.items()
        if info.get("requires_f0", False)
    ]

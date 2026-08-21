"""
Advanced RVC Inference - Enhanced Optimizer Registry
====================================================

A centralized registry for supported optimizers with metadata
including ratings, categories, and default hyperparameters.

Includes 8+ optimizers proven to work well for RVC/SVC training:
- PyTorch built-ins: AdamW, RAdam
- Mixed-precision: AnyPrecisionAdamW  
- Belief-based: AdaBelief, AdaBeliefV2, PolOpt
- Advanced (from Codename RVC Fork v4): Ranger2020, Prodigy

Usage:
    from arvc.rvc.models.optimizers import get_optimizer_class, OPTIMIZER_REGISTRY

    # Get optimizer class by name
    opt_class = get_optimizer_class("Ranger2020")

    # List all available optimizers
    for name, info in OPTIMIZER_REGISTRY.items():
        print(f"{name}: {info['rating']}/5 - {info['category']}")
"""

import torch

# Import all supported optimizers
from arvc.rvc.models.optimizers.anyprecision_optimizer import AnyPrecisionAdamW
from arvc.rvc.models.optimizers.adabelief import AdaBelief
from arvc.rvc.models.optimizers.adabeliefv2 import AdaBeliefV2
from arvc.rvc.models.optimizers.polopt import PolOpt
from arvc.rvc.models.optimizers.ranger2020 import Ranger2020
from arvc.rvc.models.optimizers.prodigy import Prodigy


# ============================================================
# ENHANCED OPTIMIZER REGISTRY (v2 - with Fork Backports)
# ============================================================
# 8 optimizers — including advanced options from Codename RVC Fork v4

OPTIMIZER_REGISTRY = {
    "AdamW": {
        "class": torch.optim.AdamW,
        "rating": 5.0,
        "category": "PyTorch Built-in",
        "supports_betas": True,
        "supports_eps": True,
        "supports_fused": True,
        "supports_weight_decay": True,
        "description": "Adam with decoupled weight decay. The gold standard optimizer for deep learning training. Provides reliable convergence with adaptive learning rates and L2 regularization.",
    },
    "RAdam": {
        "class": torch.optim.RAdam,
        "rating": 4.0,
        "category": "PyTorch Built-in",
        "supports_betas": True,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Rectified Adam with variance rectification. Eliminates the need for warmup by dynamically adjusting the update rule during early training stages.",
    },
    "AnyPrecisionAdamW": {
        "class": AnyPrecisionAdamW,
        "rating": 4.0,
        "category": "Mixed-Precision",
        "supports_betas": True,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "AdamW variant with configurable precision for momentum/variance buffers. Supports Kahan summation for numerical precision. Best used with bfloat16.",
    },
    "AdaBelief": {
        "class": AdaBelief,
        "rating": 3.0,
        "category": "Belief-Based",
        "supports_betas": True,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Adjusts the step size according to the 'belief' in the current gradient direction. Uses gradient residual for better conditioning of the adaptive LR.",
    },
    "AdaBeliefV2": {
        "class": AdaBeliefV2,
        "rating": 3.0,
        "category": "Belief-Based",
        "supports_betas": True,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Improved AdaBelief with AMSGrad support and better bias correction. More stable variance estimates for training deep generative models.",
    },
    "PolOpt": {
        "class": PolOpt,
        "rating": 4.5,
        "category": "Belief-Based",
        "supports_betas": True,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Yogi + AdaBelief hybrid (from PolTrain by Politrees). Uses sign-of-diff second-moment update to prevent denominator collapse during GAN oscillations. Includes trust-region clamping to protect acoustic filters from gradient spikes. Recommended for v3/RefineGAN vocoders.",
    },
    # ═════════════════════════════════════════════════════════
    # NEW OPTIMIZERS FROM CODENAME RVC FORK v4
    # ═════════════════════════════════════════════════════════
    "Ranger2020": {
        "class": Ranger2020,
        "rating": 4.5,
        "category": "Advanced Hybrid",
        "supports_betas": True,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "RAdam + Lookahead + Gradient Centralization (lessw2020). 12+ FastAI leaderboard records. Combines variance rectification, slow weight interpolation, and gradient centering for superior GAN training stability. Excellent for RVC discriminator/generator training.",
    },
    "Prodigy": {
        "class": Prodigy,
        "rating": 4.5,
        "category": "D-Adaptation",
        "supports_betas": True,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "D-adaptation based optimizer with automatic LR tuning. Leave lr=1.0! Built-in warmup via growth_rate parameter. Memory-efficient slice_p option. Ideal when you don't want to manually tune learning rates. From D-Adaptation paper (arXiv:2301.07733).",
    },
}


def get_optimizer_class(name: str):
    """Get the optimizer class by name.

    Args:
        name: Name of the optimizer (case-sensitive)

    Returns:
        The optimizer class

    Raises:
        ValueError: If the optimizer name is not found in the registry
    """
    if name not in OPTIMIZER_REGISTRY:
        available = ", ".join(sorted(OPTIMIZER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown optimizer '{name}'. Available optimizers: {available}"
        )
    return OPTIMIZER_REGISTRY[name]["class"]


def get_optimizer_choices() -> list:
    """Get the list of all available optimizer names, sorted by rating descending.

    Returns:
        List of optimizer names sorted by rating (highest first)
    """
    sorted_optimizers = sorted(
        OPTIMIZER_REGISTRY.items(),
        key=lambda x: x[1]["rating"],
        reverse=True
    )
    return [name for name, _ in sorted_optimizers]


def create_optimizer(name: str, params, lr: float, betas=None, eps=None,
                     weight_decay=0.0, fused=False, **kwargs):
    """Factory function to create an optimizer instance.

    Args:
        name: Name of the optimizer from the registry
        params: Model parameters
        lr: Learning rate
        betas: Momentum coefficients (beta1, beta2) if supported
        eps: Epsilon for numerical stability if supported
        weight_decay: Weight decay coefficient if supported
        fused: Whether to use fused CUDA kernels if supported
        **kwargs: Additional optimizer-specific arguments

    Returns:
        Optimizer instance
    """
    config = OPTIMIZER_REGISTRY[name]
    opt_class = config["class"]

    # Build keyword arguments
    opt_kwargs = {"lr": lr}
    opt_kwargs.update(kwargs)

    if config["supports_betas"] and betas is not None:
        opt_kwargs["betas"] = betas
    if config["supports_eps"] and eps is not None:
        opt_kwargs["eps"] = eps
    if config["supports_weight_decay"]:
        opt_kwargs["weight_decay"] = weight_decay
    if config["supports_fused"] and fused:
        opt_kwargs["fused"] = fused

    return opt_class(params, **opt_kwargs)


def get_optimizer_info(name: str) -> dict:
    """Get metadata for an optimizer.

    Args:
        name: Name of the optimizer

    Returns:
        Dictionary with optimizer metadata
    """
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer: {name}")
    return OPTIMIZER_REGISTRY[name]

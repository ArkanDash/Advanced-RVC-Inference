"""
Advanced RVC Inference - Optimizer Registry

A centralized registry for supported optimizers with metadata
including ratings, categories, and default hyperparameters.

Only includes the 5 optimizers proven to work well for RVC training,
matching the UI selection. Additional optimizers are not included
to avoid confusion and keep the codebase clean.

Usage:
    from arvc.engine.models.optimizers import get_optimizer_class, OPTIMIZER_REGISTRY

    # Get optimizer class by name
    opt_class = get_optimizer_class("AdamW")

    # List all available optimizers
    for name, info in OPTIMIZER_REGISTRY.items():
        print(f"{name}: {info['rating']}/5 - {info['category']}")
"""

import torch

# Import the 5 supported optimizers
from arvc.engine.models.optimizers.anyprecision_optimizer import AnyPrecisionAdamW
from arvc.engine.models.optimizers.adabelief import AdaBelief
from arvc.engine.models.optimizers.adabeliefv2 import AdaBeliefV2


# ============================================================
# OPTIMIZER REGISTRY
# ============================================================
# Only 5 optimizers — the ones proven to work well for RVC
# and exposed in the UI.

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

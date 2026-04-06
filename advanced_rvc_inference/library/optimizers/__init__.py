"""
Advanced RVC Inference - Optimizer Registry

A centralized registry for all supported optimizers with metadata
including ratings, categories, and default hyperparameters.

Usage:
    from advanced_rvc_inference.library.optimizers import get_optimizer_class, OPTIMIZER_REGISTRY

    # Get optimizer class by name
    opt_class = get_optimizer_class("AdamW")

    # List all available optimizers
    for name, info in OPTIMIZER_REGISTRY.items():
        print(f"{name}: {info['rating']}/5 - {info['category']}")
"""

import torch

# Import all custom optimizers
from advanced_rvc_inference.library.optimizers.anyprecision_optimizer import AnyPrecisionAdamW
from advanced_rvc_inference.library.optimizers.adabelief import AdaBelief
from advanced_rvc_inference.library.optimizers.adabeliefv2 import AdaBeliefV2
from advanced_rvc_inference.library.optimizers.lion import Lion
from advanced_rvc_inference.library.optimizers.prodigy import Prodigy
from advanced_rvc_inference.library.optimizers.sophia import Sophia
from advanced_rvc_inference.library.optimizers.adan import Adan
from advanced_rvc_inference.library.optimizers.lookahead import Lookahead
from advanced_rvc_inference.library.optimizers.ranger import Ranger21
from advanced_rvc_inference.library.optimizers.schedule_free import (
    ScheduleFreeAdamW, ScheduleFreeAdam, ScheduleFreeSGD
)
from advanced_rvc_inference.library.optimizers.dadapt import (
    DAdaptAdam, DAdaptAdaGrad, DAdaptSGD
)
from advanced_rvc_inference.library.optimizers.adafactor import AdaFactor
from advanced_rvc_inference.library.optimizers.misc_optimizers import (
    NovoGrad, PAdam, Apollo, CAME, LAMB, LARS,
    QHAdam, SWATS, AggMo, PID, Yogi, Fromage, SM3, Nero, A2Grad,
    Shampoo, SOAP, Muon
)


# ============================================================
# OPTIMIZER REGISTRY
# ============================================================
# Each entry contains:
#   class: The optimizer class (callable)
#   rating: Rating from 1.0 to 5.0 (for RVC/audio training)
#   category: Source category
#   supports_betas: Whether the optimizer accepts betas parameter
#   supports_eps: Whether the optimizer accepts eps parameter
#   supports_fused: Whether the optimizer supports fused CUDA kernels
#   supports_weight_decay: Whether the optimizer supports weight decay
#   description: Short description

OPTIMIZER_REGISTRY = {
    # ===== Tier 1: Best for RVC/Audio Training (Rating 5.0) =====
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
    "ScheduleFreeAdamW": {
        "class": ScheduleFreeAdamW,
        "rating": 5.0,
        "category": "Schedule-Free",
        "supports_betas": True,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Schedule-Free AdamW with built-in warmup and decay. Eliminates the need for learning rate scheduling entirely, providing automatic adaptation throughout training.",
    },
    "Muon": {
        "class": Muon,
        "rating": 5.0,
        "category": "Second-Order",
        "supports_betas": False,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Momentum Orthogonalized via Newton-Schulz iteration. Popularized for training large language models with superior conditioning and faster convergence on deep networks.",
    },
    "Sophia": {
        "class": Sophia,
        "rating": 5.0,
        "category": "Second-Order",
        "supports_betas": True,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Stochastic second-order optimizer using diagonal Hessian estimates with clipping. Achieves significantly faster convergence than Adam on large-scale models.",
    },

    # ===== Tier 2: Excellent (Rating 4.5) =====
    "Lion": {
        "class": Lion,
        "rating": 4.5,
        "category": "Sign-Based",
        "supports_betas": True,
        "supports_eps": False,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "EvoLved Sign Momentum discovered via program search. Uses sign operations for simpler, more memory-efficient updates. Strong performance with high learning rates.",
    },
    "Prodigy": {
        "class": Prodigy,
        "rating": 4.5,
        "category": "LR-Free",
        "supports_betas": True,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Parameter-free optimizer that auto-tunes learning rate. Eliminates manual LR tuning by estimating distance to the solution dynamically during training.",
    },
    "NAdam": {
        "class": torch.optim.NAdam,
        "rating": 4.5,
        "category": "PyTorch Built-in",
        "supports_betas": True,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Adam with Nesterov momentum. Combines Adam's adaptive learning rates with Nesterov's accelerated gradient for improved convergence speed.",
    },

    # ===== Tier 3: Very Good (Rating 4.0) =====
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
    "Adan": {
        "class": Adan,
        "rating": 4.0,
        "category": "Nesterov",
        "supports_betas": True,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Adaptive Nesterov Momentum estimator using gradient differences. Combines Nesterov momentum with adaptive LR for faster convergence and better generalization.",
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
    "Ranger21": {
        "class": Ranger21,
        "rating": 4.0,
        "category": "Combined",
        "supports_betas": True,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Synergistic optimizer combining RAdam + Lookahead. Provides warmup-free training with improved stability from periodic slow-weight synchronization.",
    },
    "AdaFactor": {
        "class": AdaFactor,
        "rating": 4.0,
        "category": "Memory-Efficient",
        "supports_betas": True,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Adaptive optimizer with sublinear memory cost. Factors second-moment estimator into row/column statistics. Used extensively in T5 and large model training.",
    },
    "DAdaptAdam": {
        "class": DAdaptAdam,
        "rating": 4.0,
        "category": "LR-Free",
        "supports_betas": True,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Adam with automatic learning rate from D-Adaptation. Estimates distance to solution from gradient statistics and sets provably optimal learning rate.",
    },
    "Adam": {
        "class": torch.optim.Adam,
        "rating": 4.0,
        "category": "PyTorch Built-in",
        "supports_betas": True,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Adaptive Moment Estimation. The classic adaptive optimizer combining momentum (first moment) and adaptive learning rates (second moment). Widely used foundation.",
    },
    "PAdam": {
        "class": PAdam,
        "rating": 4.0,
        "category": "Partial Adaptive",
        "supports_betas": True,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Partially Adaptive Momentum Estimator. Uses a partial power of the second moment, providing smooth interpolation between Adam and SGD for better generalization.",
    },
    "Apollo": {
        "class": Apollo,
        "rating": 4.0,
        "category": "Quasi-Newton",
        "supports_betas": False,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Adaptive Parameter-wise Diagonal Quasi-Newton method. Approximates diagonal Hessian using curvature information for L-BFGS-like convergence with Adam-like cost.",
    },

    # ===== Tier 4: Good (Rating 3.5) =====
    "CAME": {
        "class": CAME,
        "rating": 3.5,
        "category": "Unified",
        "supports_betas": True,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Closes the gap between Adam-style and SGD-style optimizers. Tracks both magnitude and sign of gradients for a unified adaptive framework.",
    },
    "NovoGrad": {
        "class": NovoGrad,
        "rating": 3.5,
        "category": "Normalized",
        "supports_betas": True,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Normalizes gradient by its RMS before computing second moment. Memory efficient and well-conditioned, with per-layer adaptive learning rates.",
    },
    "ScheduleFreeAdam": {
        "class": ScheduleFreeAdam,
        "rating": 3.5,
        "category": "Schedule-Free",
        "supports_betas": True,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Schedule-Free variant of standard Adam. Provides built-in warmup and decay without requiring external LR scheduling.",
    },
    "DAdaptAdaGrad": {
        "class": DAdaptAdaGrad,
        "rating": 3.5,
        "category": "LR-Free",
        "supports_betas": True,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "AdaGrad with D-Adaptation for automatic learning rate. Combines AdaGrad's cumulative second moments with distance-based LR estimation.",
    },

    # ===== Tier 5: Solid (Rating 3.0) =====
    "SGD": {
        "class": torch.optim.SGD,
        "rating": 3.0,
        "category": "PyTorch Built-in",
        "supports_betas": False,
        "supports_eps": False,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Stochastic Gradient Descent. The foundational optimizer. Simple but effective, especially with momentum and learning rate schedules. Best generalization on simple tasks.",
    },
    "RMSprop": {
        "class": torch.optim.RMSprop,
        "rating": 3.0,
        "category": "PyTorch Built-in",
        "supports_betas": False,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Root Mean Square Propagation. Maintains moving average of squared gradients. Popular in reinforcement learning and recurrent networks.",
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
    "LAMB": {
        "class": LAMB,
        "rating": 3.0,
        "category": "Layer-Adaptive",
        "supports_betas": True,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Layer-wise Adaptive Moments optimizer. Applies per-layer trust ratio to Adam updates. Used for large-batch BERT pre-training.",
    },
    "LARS": {
        "class": LARS,
        "rating": 3.0,
        "category": "Layer-Adaptive",
        "supports_betas": False,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Layer-wise Adaptive Rate Scaling. Scales learning rate per-layer based on weight/gradient norm ratio. Enables effective large-batch training.",
    },

    # ===== Tier 6: Moderate (Rating 2.5) =====
    "Adagrad": {
        "class": torch.optim.Adagrad,
        "rating": 2.5,
        "category": "PyTorch Built-in",
        "supports_betas": False,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Adaptive Gradient Algorithm. Accumulates squared gradients over time. Learning rate decreases monotonically. Good for sparse data but can be too aggressive.",
    },
    "Adadelta": {
        "class": torch.optim.Adadelta,
        "rating": 2.5,
        "category": "PyTorch Built-in",
        "supports_betas": False,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Extension of Adagrad that restricts accumulation window to recent gradients. Eliminates the need for manually setting learning rate.",
    },
    "Adamax": {
        "class": torch.optim.Adamax,
        "rating": 2.5,
        "category": "PyTorch Built-in",
        "supports_betas": True,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Adam variant using infinity norm (max) instead of L2 norm for second moment. More robust to outliers in gradient data.",
    },
    "ASGD": {
        "class": torch.optim.ASGD,
        "rating": 2.5,
        "category": "PyTorch Built-in",
        "supports_betas": False,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Averaged Stochastic Gradient Descent. Averages parameter vectors over time for better generalization. Converges to optimal with convex objectives.",
    },
    "DAdaptSGD": {
        "class": DAdaptSGD,
        "rating": 2.5,
        "category": "LR-Free",
        "supports_betas": False,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "SGD with D-Adaptation for automatic learning rate. Provides SGD's generalization benefits without manual LR tuning.",
    },
    "QHAdam": {
        "class": QHAdam,
        "rating": 2.5,
        "category": "Quasi-Hyperbolic",
        "supports_betas": True,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Quasi-Hyperbolic Adam generalizes Adam via discounting parameters. Provides smooth interpolation between SGD and Adam behavior.",
    },
    "SWATS": {
        "class": SWATS,
        "rating": 2.5,
        "category": "Hybrid",
        "supports_betas": True,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Switches from Adam to SGD during training. Starts with Adam for fast convergence, then switches to SGD for better generalization.",
    },
    "Shampoo": {
        "class": Shampoo,
        "rating": 2.5,
        "category": "Preconditioned",
        "supports_betas": True,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Stochastic Tensor Optimization with layer-wise preconditioning. Uses Kronecker-factored approximation of the Hessian for faster convergence.",
    },
    "SOAP": {
        "class": SOAP,
        "rating": 2.5,
        "category": "Second-Order",
        "supports_betas": True,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Second-Order Adam-like Preconditioner. Uses distributed second-order information for better conditioned updates in large-scale training.",
    },

    # ===== Tier 7: Specialized/Niche (Rating 2.0) =====
    "A2Grad": {
        "class": A2Grad,
        "rating": 2.0,
        "category": "Optimal Averaging",
        "supports_betas": False,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": False,
        "description": "Stochastic Gradient Descent with optimal averaging and second-order information. Theoretically motivated convergence guarantees.",
    },
    "AggMo": {
        "class": AggMo,
        "rating": 2.0,
        "category": "Aggregate Momentum",
        "supports_betas": True,
        "supports_eps": False,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Aggregate Momentum uses multiple momentum buffers at different betas. Combines fast (low beta) and slow (high beta) momentum simultaneously.",
    },
    "PID": {
        "class": PID,
        "rating": 2.0,
        "category": "Control Theory",
        "supports_betas": False,
        "supports_eps": False,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Applies PID controller concepts (Proportional, Integral, Derivative) to gradient descent. Novel control-theoretic approach to optimization.",
    },
    "Yogi": {
        "class": Yogi,
        "rating": 2.0,
        "category": "Controlled Growth",
        "supports_betas": True,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Controls growth of second moment estimate to prevent LR explosion. More stable than Adam in scenarios with variable gradient scales.",
    },
    "Fromage": {
        "class": Fromage,
        "rating": 2.0,
        "category": "Functional Regularization",
        "supports_betas": False,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Normalizes updates by Frobenius norm. Simple, robust, and parameter-efficient. Good baseline for understanding gradient dynamics.",
    },
    "SM3": {
        "class": SM3,
        "rating": 2.0,
        "category": "Memory-Efficient",
        "supports_betas": False,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Squared Method of Moments for memory-efficient adaptive optimization. Scales sublinearly with parameter count.",
    },
    "ScheduleFreeSGD": {
        "class": ScheduleFreeSGD,
        "rating": 2.0,
        "category": "Schedule-Free",
        "supports_betas": False,
        "supports_eps": False,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Schedule-Free variant of SGD with momentum. Provides built-in warmup and decay for SGD without external LR scheduling.",
    },
    "Nero": {
        "class": Nero,
        "rating": 2.0,
        "category": "Normalized",
        "supports_betas": False,
        "supports_eps": True,
        "supports_fused": False,
        "supports_weight_decay": True,
        "description": "Normalizes weight matrices at each step for natural regularization. Provides built-in weight normalization for better conditioning.",
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

# Optimizer Reference Guide

> Advanced RVC Inference supports **5 optimizers** for model training, carefully selected from the broader optimizer landscape to cover the most effective choices for RVC/audio model training. This guide provides detailed descriptions, ratings, and recommendations.

## Quick Reference

| Rating | Optimizer | Category | Best For |
|--------|-----------|----------|----------|
| ⭐⭐⭐⭐⭐ | **AdamW** | PyTorch Built-in | General-purpose, most reliable (default) |
| ⭐⭐⭐⭐ | **RAdam** | PyTorch Built-in | Warmup-free training, short training runs |
| ⭐⭐⭐⭐ | **AnyPrecisionAdamW** | Mixed-Precision | Bfloat16 training, long runs with Kahan summation |
| ⭐⭐⭐ | **AdaBelief** | Belief-Based | Better conditioned adaptive learning rates |
| ⭐⭐⭐ | **AdaBeliefV2** | Belief-Based | Stable deep training with AMSGrad + InverseSqrt scheduler |

---

## AdamW
- **Rating:** ⭐⭐⭐⭐⭐ (5.0/5)
- **Category:** PyTorch Built-in
- **Source:** `torch.optim.AdamW`
- **Paper:** "Decoupled Weight Decay Regularization" (2019)

Adam with decoupled weight decay is the gold standard optimizer for deep learning training. It combines the adaptive learning rate of Adam with proper L2 regularization by decoupling weight decay from the gradient update. This is the **default and recommended optimizer** for RVC model training. It provides reliable convergence across a wide range of model architectures, dataset sizes, and training configurations. The weight decay is applied directly to the weights rather than through the gradient, which leads to more consistent regularization behavior regardless of the learning rate. AdamW also supports fused CUDA kernels for faster training on NVIDIA GPUs, which can provide significant speedups when training on a single GPU.

**Key Features:**
- Adaptive learning rates per parameter
- Decoupled weight decay (proper L2 regularization)
- Fused CUDA kernel support for faster training
- Proven track record across all of deep learning
- Well-understood behavior and debugging

**Recommended for:** All RVC training scenarios as the default choice. Works well with learning rates between 1e-4 and 1e-3, batch sizes 4-32, and 100-1000 epochs. The fused CUDA variant is automatically enabled when training on CUDA for best performance.

---

## RAdam
- **Rating:** ⭐⭐⭐⭐ (4.0/5)
- **Category:** PyTorch Built-in
- **Source:** `torch.optim.RAdam`
- **Paper:** "On the Variance of the Adaptive Learning Rate" (2020)

Rectified Adam addresses a fundamental issue with Adam: during the first few training steps, the variance estimate is unreliable because it's computed from very few samples. RAdam dynamically rectifies this by switching between SGD-like updates (when variance is unreliable) and Adam-like updates (when variance becomes trustworthy). This eliminates the need for warmup steps that Adam typically requires, making training more robust to initialization and early-stage instabilities. For RVC training, this means you can start with higher learning rates without the risk of divergence in the first few hundred steps, which is especially valuable for short training runs (under 200 epochs) where warmup would otherwise consume a significant portion of the total training budget.

**Key Features:**
- Eliminates warmup requirement
- Automatic variance rectification
- Smooth transition from SGD-like to Adam-like behavior
- Built into PyTorch (no external dependencies)
- Proven stability benefits across many training scenarios

**Recommended for:** Short training runs where warmup would consume a significant fraction of total steps. Also good when using aggressive learning rates or when you want training to be robust regardless of initialization. A solid alternative to AdamW when you don't want to configure warmup.

---

## AnyPrecisionAdamW
- **Rating:** ⭐⭐⭐⭐ (4.0/5)
- **Category:** Mixed-Precision
- **Source:** `arvc/engine/models/optimizers/anyprecision_optimizer.py`
- **Paper:** Based on "AdamW with configurable precision buffers"

AnyPrecisionAdamW is an AdamW variant with configurable data types for its internal momentum and variance buffers. This allows fine-grained control over numerical precision during mixed-precision training. When using bfloat16, this optimizer can maintain its statistics in bfloat16 (matching the computation precision) or optionally use Kahan summation for enhanced numerical accuracy. Kahan summation is a compensated summation algorithm that significantly reduces floating-point rounding errors, which is particularly valuable for very long training runs where the accumulation of small errors can gradually degrade model quality. The optimizer reduces memory usage compared to standard AdamW when using lower-precision buffers, while maintaining training quality through the Kahan compensation mechanism.

**Key Features:**
- Configurable buffer dtypes (float32, bfloat16, float16)
- Optional Kahan summation for precision
- Best used with bfloat16 training
- Reduces memory with lower-precision buffers
- Important: requires `brain` (bfloat16) config enabled for optimal results

**Recommended for:** Users training with bfloat16 who want maximum numerical stability, especially for very long training runs (500+ epochs). The Kahan summation feature prevents gradual precision loss that can accumulate over thousands of optimizer steps, which makes it ideal for extended training sessions where standard AdamW might silently degrade in quality.

---

## AdaBelief
- **Rating:** ⭐⭐⭐ (3.0/5)
- **Category:** Belief-Based
- **Source:** `arvc/engine/models/optimizers/adabelief.py`
- **Paper:** "AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients" (2020)

AdaBelief adjusts the step size based on the "belief" in the current gradient direction, computed as the difference between the current gradient and the exponential moving average of past gradients. When the current gradient is close to the moving average (high belief), the optimizer takes larger steps; when the gradient deviates significantly from the average (low belief), the optimizer takes smaller steps. This provides better conditioning of the adaptive learning rate compared to standard Adam, which always normalizes by the second moment regardless of gradient consistency. In practice, AdaBelief tends to produce smoother training curves and can converge to better minima in some scenarios, particularly when the loss landscape has varying curvature across different parameter groups.

**Key Features:**
- Gradient "belief" mechanism for adaptive step sizes
- Better conditioned updates than standard Adam
- Smoother training curves in many scenarios
- Works well with cosine annealing LR schedule
- Compatible with standard Adam hyperparameters

**Recommended for:** Users who want to experiment beyond AdamW and are willing to tune the learning rate schedule. Pairs well with cosine annealing LR, which is automatically applied when using AdaBelief. May provide better final model quality on some datasets, but results can vary.

---

## AdaBeliefV2
- **Rating:** ⭐⭐⭐ (3.0/5)
- **Category:** Belief-Based
- **Source:** `arvc/engine/models/optimizers/adabeliefv2.py`
- **Paper:** Based on "AdaBelief Optimizer" with AMSGrad extension + InverseSqrt scheduler

AdaBeliefV2 is an improved version of AdaBelief that incorporates two significant enhancements: AMSGrad support and an InverseSqrt learning rate scheduler. The AMSGrad variant maintains the maximum of the variance estimates to prevent the learning rate from increasing, providing more stable training dynamics especially in later epochs. The InverseSqrt scheduler (inspired by Vietnamese-RVC) decays the learning rate proportionally to the inverse square root of the step count, which provides a principled and gradual learning rate reduction without requiring manual schedule tuning. This combination makes AdaBeliefV2 more robust for deep generative model training, where both stable variance estimates and appropriate learning rate decay are critical for achieving high-quality results.

**Key Features:**
- AMSGrad support for stable variance estimates
- Built-in InverseSqrt learning rate scheduler (from Vietnamese-RVC)
- Better bias correction than AdaBelief
- More stable training dynamics in later epochs
- Automatic LR decay without manual schedule configuration

**Recommended for:** Users who want an "all-in-one" optimizer with built-in scheduling. The InverseSqrt scheduler eliminates the need for manual LR schedule tuning while providing principled decay. Best for training runs where you want the optimizer to handle both the update rule and the learning rate schedule automatically.

---

## Recommendations for RVC Training

### Beginner
Start with **AdamW** (default). It's the most tested and reliable optimizer for RVC training. Use learning rate 1e-3 with 300 epochs and batch size 8. The fused CUDA variant is automatically enabled for best performance.

### Intermediate
Try **RAdam** for warmup-free training, or **AnyPrecisionAdamW** for bfloat16 training with better numerical stability. Both are solid upgrades that require minimal additional configuration.

### Advanced
Experiment with **AdaBelief** paired with cosine annealing LR, or **AdaBeliefV2** with its built-in InverseSqrt scheduler for automatic learning rate decay. These belief-based optimizers can sometimes find better minima but may require more careful hyperparameter tuning.

### Long Training Runs (500+ Epochs)
Use **AnyPrecisionAdamW** with Kahan summation to prevent gradual precision loss, or **AdaBeliefV2** with its built-in InverseSqrt scheduler for automatic LR decay throughout extended training.

---

## Technical Notes

- The optimizer registry is in `arvc/engine/models/optimizers/__init__.py`
- The training engine (`arvc/engine/training/runner/train.py`) uses the registry for dynamic optimizer selection
- Each optimizer automatically receives appropriate kwargs (betas, eps, weight_decay) based on its capabilities
- Fused CUDA kernels are automatically enabled when supported (currently only AdamW)
- For optimizers that don't support `betas` or `eps`, these parameters are silently omitted
- AdaBelief automatically pairs with CosineAnnealingLR scheduler
- AdaBeliefV2 automatically pairs with InverseSqrt scheduler (from Vietnamese-RVC)
- 8-bit Adam is available as a separate option (requires `bitsandbytes`) and overrides the selected optimizer with `AdamW8bit`

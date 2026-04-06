# Optimizer Reference Guide

> Advanced RVC Inference supports **43 optimizers** for model training, each with different characteristics, strengths, and use cases. This guide provides detailed descriptions, ratings, and recommendations for RVC/audio model training.

## Quick Reference

| Rating | Optimizer | Category | Best For |
|--------|-----------|----------|----------|
| ⭐⭐⭐⭐⭐ | **AdamW** | PyTorch Built-in | General-purpose, most reliable |
| ⭐⭐⭐⭐⭐ | **ScheduleFreeAdamW** | Schedule-Free | No LR schedule needed |
| ⭐⭐⭐⭐⭐ | **Muon** | Second-Order | Large models, fast convergence |
| ⭐⭐⭐⭐⭐ | **Sophia** | Second-Order | Large-scale training |
| ⭐⭐⭐⭐½ | **Lion** | Sign-Based | Memory-efficient training |
| ⭐⭐⭐⭐½ | **Prodigy** | LR-Free | No LR tuning needed |
| ⭐⭐⭐⭐½ | **NAdam** | PyTorch Built-in | Faster than standard Adam |
| ⭐⭐⭐⭐ | **RAdam** | PyTorch Built-in | Warmup-free training |
| ⭐⭐⭐⭐ | **Adan** | Nesterov | Vision and audio tasks |
| ⭐⭐⭐⭐ | **AnyPrecisionAdamW** | Mixed-Precision | Bfloat16 training |
| ⭐⭐⭐⭐ | **Ranger21** | Combined | RAdam + Lookahead synergy |
| ⭐⭐⭐⭐ | **AdaFactor** | Memory-Efficient | Large model training |
| ⭐⭐⭐⭐ | **DAdaptAdam** | LR-Free | Automatic LR from gradients |
| ⭐⭐⭐⭐ | **Adam** | PyTorch Built-in | Classic adaptive optimizer |
| ⭐⭐⭐⭐ | **PAdam** | Partial Adaptive | Adam-SGD interpolation |
| ⭐⭐⭐⭐ | **Apollo** | Quasi-Newton | L-BFGS-like convergence |
| ⭐⭐⭐½ | **CAME** | Unified | Adam+SGD benefits combined |
| ⭐⭐⭐½ | **NovoGrad** | Normalized | Well-conditioned gradients |
| ⭐⭐⭐½ | **ScheduleFreeAdam** | Schedule-Free | Adam without LR schedule |
| ⭐⭐⭐½ | **DAdaptAdaGrad** | LR-Free | Auto LR with AdaGrad |
| ⭐⭐⭐ | **SGD** | PyTorch Built-in | Best generalization |
| ⭐⭐⭐ | **RMSprop** | PyTorch Built-in | RL and recurrent networks |
| ⭐⭐⭐ | **AdaBelief** | Belief-Based | Better conditioned updates |
| ⭐⭐⭐ | **AdaBeliefV2** | Belief-Based | Stable deep training |
| ⭐⭐⭐ | **LAMB** | Layer-Adaptive | Large-batch training |
| ⭐⭐⭐ | **LARS** | Layer-Adaptive | Distributed training |
| ⭐⭐½ | **Adagrad** | PyTorch Built-in | Sparse data |
| ⭐⭐½ | **Adadelta** | PyTorch Built-in | No manual LR needed |
| ⭐⭐½ | **Adamax** | PyTorch Built-in | Robust to outliers |
| ⭐⭐½ | **ASGD** | PyTorch Built-in | Convex optimization |
| ⭐⭐½ | **DAdaptSGD** | LR-Free | SGD with auto LR |
| ⭐⭐½ | **QHAdam** | Quasi-Hyperbolic | Adam-SGD continuum |
| ⭐⭐½ | **SWATS** | Hybrid | Adam→SGD switching |
| ⭐⭐½ | **Shampoo** | Preconditioned | Layer preconditioning |
| ⭐⭐½ | **SOAP** | Second-Order | Distributed 2nd order |
| ⭐⭐ | **A2Grad** | Optimal Averaging | Theoretical guarantees |
| ⭐⭐ | **AggMo** | Aggregate Momentum | Multi-scale momentum |
| ⭐⭐ | **PID** | Control Theory | Novel control approach |
| ⭐⭐ | **Yogi** | Controlled Growth | Stable variance |
| ⭐⭐ | **Fromage** | Functional Regularization | Simple baseline |
| ⭐⭐ | **SM3** | Memory-Efficient | Sublinear memory |
| ⭐⭐ | **ScheduleFreeSGD** | Schedule-Free | SGD without schedule |
| ⭐⭐ | **Nero** | Normalized | Weight normalization |

---

## Tier 1: Best for RVC/Audio Training (⭐⭐⭐⭐⭐)

### AdamW
- **Rating:** ⭐⭐⭐⭐⭐ (5.0/5)
- **Category:** PyTorch Built-in
- **Source:** `torch.optim.AdamW`
- **Paper:** "Decoupled Weight Decay Regularization" (2019)

Adam with decoupled weight decay is the gold standard optimizer for deep learning training. It combines the adaptive learning rate of Adam with proper L2 regularization by decoupling weight decay from the gradient update. This is the **default and recommended optimizer** for RVC model training. It provides reliable convergence across a wide range of model architectures, dataset sizes, and training configurations. The weight decay is applied directly to the weights rather than through the gradient, which leads to more consistent regularization behavior regardless of the learning rate.

**Key Features:**
- Adaptive learning rates per parameter
- Decoupled weight decay (proper L2 regularization)
- Fused CUDA kernel support for faster training
- Proven track record across all of deep learning
- Well-understood behavior and debugging

**Recommended for:** All RVC training scenarios as the default choice. Works well with learning rates between 1e-4 and 1e-3, batch sizes 4-32, and 100-1000 epochs.

---

### ScheduleFreeAdamW
- **Rating:** ⭐⭐⭐⭐⭐ (5.0/5)
- **Category:** Schedule-Free
- **Source:** Custom implementation
- **Paper:** "Schedule-Free: Learning Rate Free Training in Adam and SGD" (2024)

Schedule-Free AdamW eliminates the need for any learning rate scheduling by maintaining a dual set of parameters. The "z" parameters serve as a lookahead while "y" parameters follow standard AdamW updates. The optimizer dynamically adjusts its effective learning rate based on the distance between z and y, providing built-in warmup at the start of training and natural decay as convergence approaches. This means you never need to worry about warmup steps, cosine annealing, or step decay schedules again.

**Key Features:**
- No learning rate schedule needed whatsoever
- Built-in warmup phase (first ~5% of training)
- Automatic decay as training converges
- Drop-in replacement for AdamW
- Stable across different model sizes

**Recommended for:** Users who want to avoid learning rate schedule tuning. Especially useful when training with varying dataset sizes or when you're unsure what schedule to use.

---

### Muon
- **Rating:** ⭐⭐⭐⭐⭐ (5.0/5)
- **Category:** Second-Order
- **Source:** Custom implementation
- **Paper:** "Muon: Momentum Orthogonalized Gradient Descent" (2025)

Muon applies Newton-Schulz iteration to orthogonalize the momentum vector at each step. This normalization provides significantly better conditioning for the optimization landscape, similar in spirit to preconditioning in second-order methods but at a much lower computational cost. Muon has gained popularity for training large language models, where it demonstrates faster convergence compared to AdamW, particularly in later training stages. The orthogonalization ensures that updates move in well-conditioned directions, reducing the chance of oscillation or stagnation.

**Key Features:**
- Momentum orthogonalization via Newton-Schulz iteration
- Better conditioned optimization landscape
- Faster convergence on deep models
- Popularized for large-scale language model training
- Works well with high learning rates

**Recommended for:** Advanced users training large RVC models (v2, 48k) who want faster convergence. Particularly effective with 300+ epoch training runs.

---

### Sophia
- **Rating:** ⭐⭐⭐⭐⭐ (5.0/5)
- **Category:** Second-Order
- **Source:** Custom implementation
- **Paper:** "Sophia: A Scalable Stochastic Second-order Optimizer" (2023)

Sophia is a second-order optimizer that uses a diagonal Hessian estimate combined with a stochastic clipping mechanism. Unlike Adam which only uses first-order gradient information, Sophia incorporates curvature information from the Hessian (second derivatives) to make more informed update decisions. The diagonal approximation keeps memory usage manageable while still providing significant convergence benefits. The clipping mechanism prevents excessively large updates in high-curvature directions, ensuring training stability.

**Key Features:**
- Diagonal Hessian estimation for curvature awareness
- Stochastic clipping for stability
- Faster convergence than first-order methods
- Memory-efficient diagonal approximation
- Update frequency control via k parameter

**Recommended for:** Users with sufficient GPU memory who want maximum convergence speed. Best with larger batch sizes (8+) and longer training runs.

---

## Tier 2: Excellent Optimizers (⭐⭐⭐⭐½)

### Lion
- **Rating:** ⭐⭐⭐⭐½ (4.5/5)
- **Category:** Sign-Based
- **Source:** Custom implementation
- **Paper:** "Symbolic Discovery of Optimization Algorithms" (2023)

Lion (EvoLved Sign Momentum) was discovered through automated program search rather than manual design. Its key innovation is using the **sign of the momentum** rather than the momentum itself for the update direction. This dramatically simplifies the computation: instead of dividing by the square root of the variance, Lion just takes the sign. This results in significantly lower memory usage (only one state tensor vs. two in Adam) and often matches or exceeds AdamW's performance, particularly with higher learning rates.

**Key Features:**
- Uses sign(momentum) instead of momentum / sqrt(variance)
- ~50% less memory than AdamW (single state buffer)
- Works well with high learning rates
- Discovered via neural architecture search
- Strong performance across vision, NLP, and audio tasks

**Recommended for:** Memory-constrained training scenarios or when you want to try a higher learning rate than AdamW allows without diverging.

---

### Prodigy
- **Rating:** ⭐⭐⭐⭐½ (4.5/5)
- **Category:** LR-Free
- **Source:** Custom implementation
- **Paper:** "Prodigy: An Expeditiously Adaptive Parameter-Free Learner" (2023)

Prodigy automatically determines the optimal learning rate by estimating the distance to the solution (D0) using gradient statistics. You only need to set one intuitive parameter: `d_coef` (what fraction of D0 to traverse per epoch). The optimizer continuously adapts its effective learning rate during training based on the ratio of parameter change to gradient magnitude. This eliminates the most common failure mode in training — choosing the wrong learning rate — while still allowing the optimizer to benefit from Adam's adaptive per-parameter updates.

**Key Features:**
- Learning rate is automatically determined
- Only requires setting d_coef (distance coefficient)
- Adapts LR dynamically during training
- Based on proven AdamW foundation
- Works across different model scales

**Recommended for:** Users who struggle with learning rate tuning or are training multiple models with different architectures and need a "set it and forget it" optimizer.

---

### NAdam
- **Rating:** ⭐⭐⭐⭐½ (4.5/5)
- **Category:** PyTorch Built-in
- **Source:** `torch.optim.NAdam`
- **Paper:** "Incorporating Nesterov Momentum into Adam" (2015)

NAdam combines Adam's adaptive learning rates with Nesterov accelerated gradient. The Nesterov aspect means the optimizer looks ahead by computing the gradient at the anticipated next position rather than the current position. This lookahead provides a form of implicit momentum correction that often leads to faster convergence, especially in the early stages of training. NAdam is particularly well-suited for RVC training because audio model loss landscapes tend to benefit from the accelerated convergence that Nesterov momentum provides.

**Key Features:**
- Adam + Nesterov momentum combination
- Faster early-stage convergence
- Lookahead gradient computation
- Available directly in PyTorch (no custom code)
- Good stability characteristics

**Recommended for:** Users who want a slight upgrade over AdamW without the complexity of newer optimizers. Good default alternative to AdamW.

---

## Tier 3: Very Good Optimizers (⭐⭐⭐⭐)

### RAdam
- **Rating:** ⭐⭐⭐⭐ (4.0/5)
- **Category:** PyTorch Built-in
- **Source:** `torch.optim.RAdam`
- **Paper:** "On the Variance of the Adaptive Learning Rate" (2020)

Rectified Adam addresses a fundamental issue with Adam: during the first few training steps, the variance estimate is unreliable because it's computed from very few samples. RAdam dynamically rectifies this by switching between SGD-like updates (when variance is unreliable) and Adam-like updates (when variance becomes trustworthy). This eliminates the need for warmup steps that Adam typically requires, making training more robust to initialization and early-stage instabilities.

**Key Features:**
- Eliminates warmup requirement
- Automatic variance rectification
- Smooth transition from SGD-like to Adam-like behavior
- Built into PyTorch
- Proven stability benefits

**Recommended for:** Short training runs where warmup would consume a significant fraction of total steps. Also good when using aggressive learning rates.

---

### Adan
- **Rating:** ⭐⭐⭐⭐ (4.0/5)
- **Category:** Nesterov
- **Source:** Custom implementation
- **Paper:** "Adan: Adaptive Nesterov Momentum Algorithm" (2022)

Adan introduces a unique third moment that tracks the **difference between consecutive gradients**. This gradient difference captures information about the curvature of the loss landscape, effectively providing second-order information at first-order cost. The Nesterov-style momentum estimation further enhances convergence speed. Adan has shown particularly strong results on vision and audio tasks where gradient smoothness is important, making it a natural fit for RVC voice model training.

**Key Features:**
- Uses gradient differences as a third moment estimate
- Implicit curvature information
- Nesterov momentum estimation
- Strong performance on generative models
- Works well with standard learning rates

**Recommended for:** Audio/vision training tasks where gradient smoothness matters. A solid alternative to AdamW for users wanting to experiment with different optimization dynamics.

---

### AnyPrecisionAdamW
- **Rating:** ⭐⭐⭐⭐ (4.0/5)
- **Category:** Mixed-Precision
- **Source:** Custom implementation
- **Paper:** Based on "AdamW with configurable precision buffers"

AnyPrecisionAdamW is an AdamW variant with configurable data types for its internal momentum and variance buffers. This allows fine-grained control over numerical precision during mixed-precision training. When using bfloat16, this optimizer can maintain its statistics in bfloat16 (matching the computation precision) or optionally use Kahan summation for enhanced numerical accuracy. This is particularly valuable for very long training runs where accumulation of floating-point errors can become problematic.

**Key Features:**
- Configurable buffer dtypes (float32, bfloat16, float16)
- Optional Kahan summation for precision
- Best used with bfloat16 training
- Reduces memory with lower-precision buffers
- Important: requires `brain` (bfloat16) config enabled

**Recommended for:** Users training with bfloat16 who want maximum numerical stability, especially for very long training runs (500+ epochs).

---

### Ranger21
- **Rating:** ⭐⭐⭐⭐ (4.0/5)
- **Category:** Combined
- **Source:** Custom implementation
- **Paper:** Based on "Lookahead Optimizer" (2019) + "On the Variance of the Adaptive Learning Rate" (2020)

Ranger21 synergistically combines RAdam's variance rectification with Lookahead's slow-fast weight synchronization. Every k steps, the optimizer interpolates between the current "fast" weights (updated by RAdam) and "slow" weights (updated less frequently). This periodic synchronization acts as a regularizer that prevents the optimizer from overshooting minima, leading to flatter minima and better generalization. The combination eliminates the need for warmup while providing stability improvements from Lookahead.

**Key Features:**
- RAdam + Lookahead in a single optimizer
- Periodic slow-fast weight synchronization
- Built-in regularization effect
- Warmup-free training
- Good default: k=6, alpha=0.5

**Recommended for:** Users who want a "best of both worlds" optimizer with RAdam's stability and Lookahead's generalization benefits without managing two separate components.

---

### AdaFactor
- **Rating:** ⭐⭐⭐⭐ (4.0/5)
- **Category:** Memory-Efficient
- **Source:** Custom implementation
- **Paper:** "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost" (2018)

AdaFactor dramatically reduces memory usage by factoring the second-moment estimator into row-wise and column-wise statistics instead of storing the full per-element variance tensor. For a parameter matrix of shape (m, n), Adam stores m×n variance values while AdaFactor only stores m + n values. It also uses a relative step size based on the RMS of the parameters themselves, which provides better scaling across layers of different sizes. This optimizer was instrumental in training the T5 model and is well-suited for memory-constrained environments.

**Key Features:**
- Sublinear memory cost (scales with parameters, not their square)
- Factored second-moment approximation
- Relative step size for better cross-layer scaling
- Proven at scale (T5, BERT large)
- Memory savings increase with model size

**Recommended for:** Training large RVC models on GPUs with limited memory. The memory savings are more significant for models with large weight matrices.

---

### DAdaptAdam
- **Rating:** ⭐⭐⭐⭐ (4.0/5)
- **Category:** LR-Free
- **Source:** Custom implementation
- **Paper:** "Learning-Rate-Free Learning by D-Adaptation" (2023)

DAdaptAdam automatically determines the learning rate by estimating the distance to the optimal solution from accumulated gradient statistics. The key insight is that the sum of squared gradients provides information about this distance. D-Adapt uses this to compute a provably optimal (in a certain sense) learning rate that adapts during training. The optimizer maintains the Adam update rule but automatically adjusts the effective learning rate, so you get Adam's per-parameter adaptation plus automatic global LR tuning.

**Key Features:**
- Automatically determines learning rate from gradient statistics
- Theoretical convergence guarantees
- Maintains Adam's per-parameter adaptation
- Adapts LR throughout training
- Set LR to 1.0 and let it figure out the rest

**Recommended for:** Users who want automatic learning rate tuning while keeping the familiar Adam behavior. Set `lr=1.0` and let D-Adapt handle the rest.

---

### Adam
- **Rating:** ⭐⭐⭐⭐ (4.0/5)
- **Category:** PyTorch Built-in
- **Source:** `torch.optim.Adam`
- **Paper:** "Adam: A Method for Stochastic Optimization" (2015)

The original Adam optimizer remains one of the most widely used optimizers in deep learning. It combines first moment (mean) and second moment (uncentered variance) estimates with bias correction to provide per-parameter adaptive learning rates. While AdamW has largely replaced it due to better weight decay handling, Adam still performs well in many scenarios and is the optimizer that many practitioners are most familiar with. The bias correction is particularly important during early training steps.

**Key Features:**
- First and second moment estimates with bias correction
- Per-parameter adaptive learning rates
- Widely supported and well-documented
- Good default performance
- Foundation for most modern adaptive optimizers

**Recommended for:** Users who want the classic Adam experience, or when comparing against existing results that used Adam.

---

### PAdam
- **Rating:** ⭐⭐⭐⭐ (4.0/5)
- **Category:** Partial Adaptive
- **Source:** Custom implementation
- **Paper:** "Closing the Generalization Gap of Adaptive Gradient Methods" (2020)

PAdam introduces a `p_partial` parameter that controls how much of the second moment's power to use. When `p_partial=0`, PAdam behaves like SGD; when `p_partial=1`, it behaves like Adam. The default `p_partial=0.25` provides a balance that retains some of Adam's adaptivity while gaining some of SGD's generalization benefits. This partial adaptation allows fine-tuning the optimizer's behavior between the Adam-SGD spectrum based on the specific task.

**Key Features:**
- Partial power of second moment (configurable p_partial)
- Smooth interpolation between Adam and SGD
- Better generalization than full Adam
- More adaptive than pure SGD
- Default p_partial=0.25 works well for most tasks

**Recommended for:** Users who want a balance between Adam's fast convergence and SGD's good generalization, with a single tunable parameter.

---

### Apollo
- **Rating:** ⭐⭐⭐⭐ (4.0/5)
- **Category:** Quasi-Newton
- **Source:** Custom implementation
- **Paper:** "Apollo: An Adaptive Parameter-wise Diagonal Quasi-Newton Method" (2020)

Apollo approximates diagonal Hessian information using the ratio of consecutive gradients, similar to how L-BFGS builds up curvature information over time. This quasi-Newton approach provides second-order convergence benefits without the computational cost of full Hessian computation. The optimizer starts with Adam-like behavior and progressively incorporates more curvature information as training proceeds, leading to faster convergence in later stages.

**Key Features:**
- Diagonal Hessian approximation from gradient ratios
- L-BFGS-like convergence at Adam-like cost
- Progressive curvature incorporation
- Built-in warmup phase
- Good for smooth loss landscapes

**Recommended for:** Users who want quasi-Newton convergence speed without the complexity and memory cost of full second-order methods.

---

## Tier 4: Good Optimizers (⭐⭐⭐½)

### CAME
Closes the gap between Adam-style and SGD-style optimizers by tracking both the magnitude and sign consistency of gradients. It computes a "sign scale" that upweights updates when the gradient direction is consistent across steps and downweights when the direction oscillates. This provides a natural adaptive mechanism that combines Adam's per-parameter learning rates with SGD's generalization benefits.

### NovoGrad
Normalizes the gradient by its RMS before computing the second moment, which provides better conditioning across layers. The second moment is computed on the normalized gradient rather than the raw gradient, leading to more stable and predictable behavior.

### ScheduleFreeAdam
Schedule-Free variant of standard Adam (without decoupled weight decay). Provides built-in warmup and decay for Adam without requiring external LR scheduling. Slightly different behavior than ScheduleFreeAdamW due to coupled weight decay.

### DAdaptAdaGrad
Combines AdaGrad's cumulative second moment with D-Adaptation's automatic learning rate estimation. The cumulative nature of AdaGrad provides good performance on sparse or noisy gradient landscapes while D-Adapt handles the global LR tuning.

---

## Tier 5: Solid Optimizers (⭐⭐⭐)

### SGD
The foundational stochastic gradient descent optimizer. While simple, SGD with momentum and proper learning rate scheduling often provides the best generalization, especially on smaller datasets. Its simplicity means it's well-understood and easy to debug, but it requires more careful learning rate tuning than adaptive methods.

### RMSprop
Maintains a moving average of squared gradients (unlike Adagrad which accumulates all past gradients). Popular in reinforcement learning and recurrent network training where the non-stationary gradient statistics benefit from the decayed averaging.

### AdaBelief
Adjusts the step size based on the "belief" in the current gradient direction, computed as the difference between the current gradient and the exponential moving average of past gradients. This provides better conditioning of the adaptive learning rate.

### AdaBeliefV2
Improved version of AdaBelief with AMSGrad support and better bias correction. The AMSGrad variant maintains the maximum of the variance estimates to prevent the learning rate from increasing, providing more stable training.

### LAMB
Layer-wise Adaptive Moments optimizer that applies a per-layer trust ratio to Adam updates. The trust ratio scales each layer's update by the ratio of the layer's weight norm to its update norm. Essential for large-batch distributed training (BERT pre-training at scale).

### LARS
Layer-wise Adaptive Rate Scaling computes a local learning rate for each layer based on the ratio of the layer's weight norm to its gradient norm. This allows layers with larger gradients to use proportionally smaller learning rates, preventing any single layer from dominating the update.

---

## Tier 6: Moderate Optimizers (⭐⭐½)

### Adagrad
Accumulates the sum of squared gradients over all training steps. The learning rate for each parameter decreases as its accumulated gradient grows, providing larger updates for infrequent parameters. However, the monotonic decrease can cause the learning rate to become too small for continued effective training.

### Adadelta
Addresses Adagrad's monotonically decreasing learning rate by restricting the accumulation window to a fixed number of recent gradients. This allows the effective learning rate to adapt to changing gradient distributions throughout training.

### Adamax
Adam variant that uses the infinity norm (maximum absolute value) instead of the L2 norm for the second moment. This makes the optimizer more robust to outliers in the gradient data, as a single large gradient value won't disproportionately affect the denominator.

### ASGD (Averaged SGD)
Averaged Stochastic Gradient Descent maintains a running average of all past parameter vectors. The final averaged parameters often generalize better than the last iterate, especially for convex objectives. Provides theoretical convergence guarantees.

### DAdaptSGD
SGD with momentum combined with D-Adaptation's automatic learning rate. Provides SGD's generalization benefits without the need for manual learning rate tuning.

### QHAdam
Quasi-Hyperbolic Adam generalizes Adam via two discounting parameters (nu1, nu2) that control the interpolation between SGD and Adam. At (0,0) it behaves like SGD; at (1,1) it's standard Adam. This provides a principled continuum between the two optimizers.

### SWATS
Starts training with Adam for fast initial convergence, then switches to SGD when the adaptive learning rate's variance drops below a threshold. The idea is to get the best of both worlds: Adam's speed early on and SGD's generalization later.

### Shampoo
Uses layer-wise preconditioning by approximating the Hessian with Kronecker products of smaller matrices. This provides much better conditioning than diagonal-only methods while keeping memory usage tractable through the Kronecker factorization.

### SOAP
Second-Order Adam-like Preconditioner uses distributed second-order information for better conditioned updates. Designed for large-scale distributed training where collecting global curvature information is feasible.

---

## Tier 7: Specialized/Niche Optimizers (⭐⭐)

### A2Grad
Stochastic Gradient Descent with optimal averaging of iterates. Uses second-order information to compute theoretically optimal step sizes and iterate averaging. Provides strong theoretical convergence guarantees.

### AggMo
Aggregate Momentum maintains multiple momentum buffers simultaneously at different decay rates (betas). The final update averages across all momentum buffers, combining fast adaptation (low beta) with long-term memory (high beta).

### PID
Applies Proportional-Integral-Derivative control theory concepts to gradient descent. The P term responds to the current gradient, the I term accumulates past gradients, and the D term responds to changes in the gradient direction.

### Yogi
Controls the growth rate of the second moment estimate to prevent the effective learning rate from increasing uncontrollably. Uses a sign-based update rule that ensures the variance estimate is monotonically non-decreasing, providing more stability than Adam.

### Fromage
Normalizes each parameter update by the Frobenius norm of its gradient and clamps it by the parameter norm. Very simple optimizer that provides natural regularization through its normalization scheme.

### SM3
Squared Method of Moments maintains element-wise maximum of squared gradients for memory-efficient adaptation. Scales sublinearly with the number of parameters, making it suitable for very large models.

### ScheduleFreeSGD
Schedule-Free variant of SGD with momentum. Provides built-in warmup and decay for SGD-based training without requiring external learning rate scheduling.

### Nero
Normalizes weight matrices at each step, providing built-in weight normalization. The normalized gradient is scaled by the parameter norm before applying the update, which acts as a natural regularizer.

---

## Recommendations for RVC Training

### Beginner
Start with **AdamW** (default). It's the most tested and reliable optimizer for RVC training. Use learning rate 1e-3 with 300 epochs and batch size 8.

### Intermediate
Try **ScheduleFreeAdamW** to eliminate LR schedule tuning, or **NAdam** for slightly faster convergence. These are drop-in replacements that require no additional configuration.

### Advanced
Experiment with **Sophia** or **Muon** for faster convergence on larger models. **Prodigy** and **DAdaptAdam** are excellent choices if you want to eliminate learning rate tuning entirely.

### Memory-Constrained
Use **Lion** (50% less memory than Adam) or **AdaFactor** (sublinear memory scaling). Both provide good performance while reducing memory footprint.

### Large-Batch Training
Use **LAMB** or **LARS** for their per-layer adaptive learning rate scaling, which prevents gradient explosion in large-batch scenarios.

---

## Technical Notes

- All custom optimizers are implemented in `advanced_rvc_inference/library/optimizers/`
- The central registry in `__init__.py` maps optimizer names to their classes
- The training engine (`rvc/train/training/train.py`) uses the registry for dynamic optimizer selection
- Each optimizer automatically receives appropriate kwargs (betas, eps, weight_decay) based on its capabilities
- Fused CUDA kernels are automatically enabled when supported (currently only AdamW)
- For optimizers that don't support `betas` or `eps`, these parameters are silently omitted

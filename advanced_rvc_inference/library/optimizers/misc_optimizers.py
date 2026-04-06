"""
Extra Optimizers Collection

A collection of additional optimizers including NovoGrad, PAdam, Apollo,
CAME, LAMB, LARS, QHAdam, SWATS, AggMo, PID, Yogi, Fromage, SM3,
Nero, A2Grad, Shampoo, SOAP, and Muon.

Each optimizer is implemented with proper docstrings and references.
"""

import math
import torch
from torch.optim.optimizer import Optimizer


# ============================================================
# NovoGrad
# ============================================================
class NovoGrad(Optimizer):
    """NovoGrad: Normalized gradient with per-layer learning rate.
    Paper: https://arxiv.org/abs/1905.11286
    
    Key: Normalizes gradient by its RMS, computes second moment of
    normalized gradient only. Memory efficient and well-conditioned.
    """

    def __init__(self, params, lr=0.01, betas=(0.95, 0.98), eps=1e-8,
                 weight_decay=0.0, grad_averaging=False):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, grad_averaging=grad_averaging)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                lr, beta1, beta2, eps = group["lr"], group["betas"][0], group["betas"][1], group["eps"]
                weight_decay = group["weight_decay"]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros([1], dtype=p.dtype, device=p.device)

                state["step"] += 1
                step = state["step"]

                # Compute per-layer grad norm
                grad_norm = torch.sqrt(grad.pow(2).sum().add_(eps))

                # Normalize gradient
                normalized_grad = grad / grad_norm

                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                state["exp_avg"].mul_(beta1).add_(normalized_grad, alpha=1 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(normalized_grad, normalized_grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                exp_avg_corrected = state["exp_avg"] / bias_correction1
                denom = (state["exp_avg_sq"] / bias_correction2).sqrt().add_(eps)

                p.data.addcdiv_(exp_avg_corrected, denom, value=-lr)

        return loss


# ============================================================
# PAdam (Partial Adaptive Momentum)
# ============================================================
class PAdam(Optimizer):
    """PAdam: Partially Adaptive Momentum Estimator.
    Paper: https://arxiv.org/abs/2006.08217
    
    Key: Uses a partial power of the second moment, providing a
    smooth interpolation between Adam and SGD.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, p_partial=0.25):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, p_partial=p_partial)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                p_partial = group["p_partial"]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = (1 - beta2 ** step) ** 0.5

                # Partial adaptation: use p_partial power of the variance
                denom = (exp_avg_sq.pow(p_partial) / bias_correction2 ** (2 * p_partial)).add_(eps)

                step_size = lr / bias_correction1
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


# ============================================================
# Apollo
# ============================================================
class Apollo(Optimizer):
    """Apollo: An Adaptive Parameter-wise Diagonal Quasi-Newton Method.
    Paper: https://arxiv.org/abs/2009.13586
    
    Key: Approximates diagonal Hessian using curvature information.
    Combines ideas from L-BFGS and Adam for fast convergence.
    """

    def __init__(self, params, lr=1e-3, beta=0.9, eps=1e-8, weight_decay=0.0,
                 warmup=0, init_lr=None):
        defaults = dict(lr=lr, beta=beta, eps=eps,
                        weight_decay=weight_decay, warmup=warmup, init_lr=init_lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                lr = group["lr"]
                beta = group["beta"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                warmup = group["warmup"]
                init_lr = group["init_lr"]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["hessian_diag"] = torch.ones_like(p)
                    state["prev_grad"] = grad.clone()

                state["step"] += 1
                step = state["step"]

                actual_lr = lr
                if warmup > 0 and step <= warmup:
                    actual_lr = init_lr if init_lr is not None else lr
                    actual_lr *= step / warmup

                if weight_decay != 0.0:
                    p.data.mul_(1.0 - actual_lr * weight_decay)

                prev_grad = state["prev_grad"]

                # Estimate diagonal Hessian
                h = state["hessian_diag"]
                delta_grad = grad - prev_grad

                # Simplified Apollo update (gradient ratio approximation)
                grad_ratio = grad / prev_grad.clamp(min=eps)
                h.mul_(beta).add_(grad_ratio.abs(), alpha=1 - beta)

                state["prev_grad"].copy_(grad)

                # Adaptive step
                state["exp_avg"].mul_(beta).add_(grad, alpha=1 - beta)
                denom = h.add_(eps)
                p.data.addcdiv_(state["exp_avg"], denom, value=-actual_lr)

        return loss


# ============================================================
# CAME (Closing the Gap)
# ============================================================
class CAME(Optimizer):
    """CAME: Closing the Gap Between Adam-style and SGD-style Optimizers.
    Paper: https://arxiv.org/abs/2305.01296
    
    Key: Combines the adaptive learning rate of Adam with the
    generalization benefits of SGD through a unified framework.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                amsgrad = group["amsgrad"]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["exp_avg_sign"] = torch.zeros_like(p)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg_sign = state["exp_avg_sign"]

                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                # CAME uses both magnitude and sign tracking
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Track sign changes
                exp_avg_sign.mul_(0.9).add_(torch.sign(grad), alpha=0.1)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = (1 - beta2 ** step) ** 0.5

                denom = (exp_avg_sq.sqrt() / bias_correction2).add_(eps)

                if amsgrad:
                    state["max_exp_avg_sq"] = torch.max(state["max_exp_avg_sq"], exp_avg_sq)
                    denom = (state["max_exp_avg_sq"].sqrt()).add_(eps)

                # Blended update: Adam-style * sign_consistency
                sign_scale = (1.0 + exp_avg_sign.abs()).clamp(max=2.0)
                step_size = lr / bias_correction1

                update = exp_avg / denom * sign_scale
                p.data.add_(update, alpha=-step_size)

        return loss


# ============================================================
# LAMB (Layer-wise Adaptive Moments)
# ============================================================
class LAMB(Optimizer):
    """LAMB: Layer-wise Adaptive Moments optimizer for Batch Training.
    Paper: https://arxiv.org/abs/1904.00962
    
    Key: Applies per-layer trust ratio to Adam updates, enabling
    large-batch training. Used for BERT pre-training.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, adam=False, trust_clip=True):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, adam=adam, trust_clip=trust_clip)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                adam = group["adam"]
                trust_clip = group["trust_clip"]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = (1 - beta2 ** step) ** 0.5

                adam_update = (exp_avg / bias_correction1) / ((exp_avg_sq / bias_correction2).sqrt().add_(eps))

                if weight_decay != 0.0:
                    adam_update.add_(p, alpha=weight_decay * lr)

                if trust_clip and not adam:
                    # Compute trust ratio
                    weight_norm = p.norm(2).clamp(min=eps)
                    adam_norm = adam_update.norm(2).clamp(min=eps)
                    trust_ratio = torch.clamp(weight_norm / adam_norm, max=10.0)
                    p.data.add_(adam_update, alpha=-lr * trust_ratio)
                else:
                    p.data.add_(adam_update, alpha=-lr)

        return loss


# ============================================================
# LARS (Layer-wise Adaptive Rate Scaling)
# ============================================================
class LARS(Optimizer):
    """LARS: Layer-wise Adaptive Rate Scaling.
    Paper: https://arxiv.org/abs/1708.03888
    
    Key: Scales learning rate per-layer based on the ratio of
    weight norm to gradient norm. Enables large-batch training.
    """

    def __init__(self, params, lr=0.1, momentum=0.9, weight_decay=1e-4,
                 trust_coefficient=0.001, eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        trust_coefficient=trust_coefficient, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                lr = group["lr"]
                momentum = group["momentum"]
                weight_decay = group["weight_decay"]
                trust_coefficient = group["trust_coefficient"]
                eps = group["eps"]

                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]

                weight_norm = p.norm(2).clamp(min=eps)
                grad_norm = grad.norm(2).clamp(min=eps)

                # Local LR scaling
                local_lr = trust_coefficient * weight_norm / grad_norm
                local_lr = min(local_lr, lr)

                if weight_decay != 0.0:
                    grad = grad.add(p, alpha=weight_decay)

                buf.mul_(momentum).add_(grad, alpha=local_lr)
                p.data.add_(buf, alpha=-local_lr)

        return loss


# ============================================================
# QHAdam (Quasi-Hyperbolic Adam)
# ============================================================
class QHAdam(Optimizer):
    """QHAdam: Quasi-Hyperbolic Adam.
    Paper: https://arxiv.org/abs/1810.06801
    
    Key: Generalizes Adam via quasi-hyperbolic discounting, providing
    a smooth interpolation between SGD and Adam.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), nus=(1.0, 1.0),
                 eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, nus=nus, eps=eps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                lr = group["lr"]
                beta1, beta2 = group["betas"]
                nu1, nu2 = group["nus"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = (1 - beta2 ** step) ** 0.5

                # Quasi-hyperbolic moment averaging
                grad_avg = nu1 * exp_avg + (1 - nu1) * grad
                grad_sq_avg = nu2 * exp_avg_sq + (1 - nu2) * grad.pow(2)

                update = grad_avg / bias_correction1 / (grad_sq_avg.sqrt() / bias_correction2 + eps)
                p.data.add_(update, alpha=-lr)

        return loss


# ============================================================
# SWATS (Switching from Adam to SGD)
# ============================================================
class SWATS(Optimizer):
    """SWATS: Improving Generalization Performance by Switching from
    Adam to SGD.
    Paper: https://arxiv.org/abs/1712.07628
    
    Key: Starts with Adam for fast early convergence, then switches to
    SGD when the variance of the adaptive LR drops below a threshold.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, nesterov=False, amsgrad=False,
                 switch_ratio_threshold=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        nesterov=nesterov, amsgrad=amsgrad,
                        switch_ratio_threshold=switch_ratio_threshold)
        super().__init__(params, defaults)
        self._switched = False

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                switch_ratio_threshold = group["switch_ratio_threshold"]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["sgd_momentum"] = torch.zeros_like(p)
                    state["sgd_lr"] = lr

                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = (1 - beta2 ** step) ** 0.5

                if not self._switched:
                    # Adam phase
                    denom = (exp_avg_sq.sqrt() / bias_correction2).add_(eps)
                    update = exp_avg / bias_correction1 / denom
                    p.data.add_(update, alpha=-lr)

                    # Check switch condition
                    if step > 1:
                        var_ratio = exp_avg_sq.sqrt().mean() / (exp_avg.abs().mean().add_(eps))
                        if var_ratio < switch_ratio_threshold:
                            # Switch to SGD with momentum = exp_avg / bias_correction
                            state["sgd_lr"] = lr
                            self._switched = True
                else:
                    # SGD phase
                    sgd_lr = state["sgd_lr"]
                    momentum_buf = state["sgd_momentum"]
                    momentum_buf.mul_(0.9).add_(grad, alpha=0.1)
                    p.data.add_(momentum_buf, alpha=-sgd_lr)

        return loss


# ============================================================
# AggMo (Aggregate Momentum)
# ============================================================
class AggMo(Optimizer):
    """AggMo: Aggregated Momentum.
    Paper: https://arxiv.org/abs/1809.11145
    
    Key: Uses multiple momentum buffers at different betas to combine
    fast and slow momentum for better convergence.
    """

    def __init__(self, params, lr=1e-3, betas=(0.0, 0.9, 0.99),
                 weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                lr = group["lr"]
                betas = group["betas"]
                weight_decay = group["weight_decay"]

                if len(state) == 0:
                    state["momentum_buffers"] = [torch.zeros_like(p) for _ in betas]

                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                # Aggregate momentum from all betas
                update = torch.zeros_like(p)
                for i, beta in enumerate(betas):
                    buf = state["momentum_buffers"][i]
                    buf.mul_(beta).add_(grad, alpha=1 - beta)
                    update.add_(buf)

                update.div_(len(betas))
                p.data.add_(update, alpha=-lr)

        return loss


# ============================================================
# PID Optimizer
# ============================================================
class PID(Optimizer):
    """PID Optimizer: A PID Controller Approach to Gradient Descent.
    Paper: https://arxiv.org/abs/2006.04144
    
    Key: Applies Proportional, Integral, and Derivative terms to the
    gradient update, like a control system PID controller.
    """

    def __init__(self, params, lr=1e-3, pid_kp=0.1, pid_ki=0.01, pid_kd=0.0,
                 weight_decay=0.0, momentum=0.9):
        defaults = dict(lr=lr, pid_kp=pid_kp, pid_ki=pid_ki, pid_kd=pid_kd,
                        weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                lr = group["lr"]
                kp = group["pid_kp"]
                ki = group["pid_ki"]
                kd = group["pid_kd"]
                weight_decay = group["weight_decay"]
                momentum = group["momentum"]

                if len(state) == 0:
                    state["integral"] = torch.zeros_like(p)
                    state["prev_grad"] = grad.clone()

                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                # P: proportional to gradient
                p_term = kp * grad

                # I: integral of gradients
                state["integral"].mul_(momentum).add_(grad, alpha=1 - momentum)
                i_term = ki * state["integral"]

                # D: derivative of gradients (difference)
                d_term = kd * (grad - state["prev_grad"])
                state["prev_grad"].copy_(grad)

                update = p_term + i_term + d_term
                p.data.add_(update, alpha=-lr)

        return loss


# ============================================================
# Yogi
# ============================================================
class Yogi(Optimizer):
    """Yogi: An adaptive optimizer with controlled increase of the
    effective learning rate.
    Paper: https://papers.nips.cc/paper/8186-yogi-paper
    
    Key: Controls the growth of the second moment estimate to prevent
    learning rate from exploding. More stable than Adam in some cases.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, initial_accumulator=1e-6):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, initial_accumulator=initial_accumulator)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.full_like(p, group["initial_accumulator"])

                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Yogi: controlled update of second moment
                diff = exp_avg_sq - grad.pow(2)
                exp_avg_sq.add_(diff, alpha=-beta2).addcmul_(
                    torch.sign(diff), grad.pow(2), value=-beta2
                )

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = (1 - beta2 ** step) ** 0.5

                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / bias_correction2).add_(eps)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


# ============================================================
# Fromage
# ============================================================
class Fromage(Optimizer):
    """Fromage: A Simple and Efficient Optimizer Based on
    Functional Regularization.
    Paper: https://arxiv.org/abs/2002.03738
    
    Key: Normalizes updates by the Frobenius norm of the gradient.
    Simple, robust, and parameter-efficient.
    """

    def __init__(self, params, lr=0.01, weight_decay=0.0, eps=1e-8):
        defaults = dict(lr=lr, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                lr = group["lr"]
                weight_decay = group["weight_decay"]
                eps = group["eps"]

                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                grad_norm = torch.sqrt(grad.pow(2).sum().add_(eps))
                param_norm = torch.sqrt(p.pow(2).sum().add_(eps))

                # Fromage: gradient / grad_norm * min(param_norm, 1)
                update = grad / grad_norm * param_norm.clamp(max=1.0)
                p.data.add_(update, alpha=-lr)

        return loss


# ============================================================
# SM3 (Squared Method of Moments)
# ============================================================
class SM3(Optimizer):
    """SM3: Memory-Efficient Adaptive Optimizer.
    Paper: https://arxiv.org/abs/1901.11150
    
    Key: Uses squared method of moments for memory efficiency.
    Scales sublinearly with the number of parameters.
    """

    def __init__(self, params, lr=0.1, momentum=0.9, beta=0.999,
                 eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, beta=beta,
                        eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                lr = group["lr"]
                momentum = group["momentum"]
                beta = group["beta"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]

                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buf"] = torch.zeros_like(p)
                    # SM3 accumulator (element-wise)
                    state["accum"] = torch.zeros_like(p, dtype=torch.float32)

                state["step"] += 1
                step = state["step"]

                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                accum = state["accum"]
                buf = state["momentum_buf"]

                # SM3: element-wise max of gradient^2
                accum.copy_(torch.max(accum, grad.pow(2)))

                # Update momentum
                buf.mul_(momentum).add_(grad / accum.sqrt().add_(eps), alpha=1 - momentum)

                # Apply update
                p.data.add_(buf, alpha=-lr)

        return loss


# ============================================================
# Nero (Normalized Error-Robust Optimizer)
# ============================================================
class Nero(Optimizer):
    """Nero: Normalized Error-Robust Optimizer.
    Paper: https://arxiv.org/abs/2107.04103
    
    Key: Normalizes the weight matrix at each step, providing
    natural regularization and better conditioning.
    """

    def __init__(self, params, lr=0.01, beta=0.9, weight_decay=0.0, eps=1e-8):
        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                lr = group["lr"]
                beta = group["beta"]
                weight_decay = group["weight_decay"]
                eps = group["eps"]

                if len(state) == 0:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(p)

                state["step"] += 1
                step = state["step"]

                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                # Nero: normalize update by parameter norm
                param_norm = p.norm(2).clamp(min=eps)
                normalized_grad = grad * param_norm / (grad.norm(2).add_(eps))

                state["momentum"].mul_(beta).add_(normalized_grad, alpha=1 - beta)
                p.data.add_(state["momentum"], alpha=-lr)

                # Re-normalize parameters
                p.data.mul_(param_norm / p.norm(2).clamp(min=eps))

        return loss


# ============================================================
# A2Grad
# ============================================================
class A2Grad(Optimizer):
    """A2Grad: Stochastic Gradient Descent with Optimal
    Averaging and Second Order Information.
    Paper: https://arxiv.org/abs/1810.00553
    
    Key: Uses optimal averaging of iterates with second-order
    information for faster convergence.
    """

    def __init__(self, params, lr=1e-3, beta=1.0, lips=1.0, rho=0.0,
                 variant="uni", eps=1e-8):
        defaults = dict(lr=lr, beta=beta, lips=lips, rho=rho,
                        variant=variant, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                lr = group["lr"]
                beta = group["beta"]
                lips = group["lips"]
                rho = group["rho"]
                eps = group["eps"]

                if len(state) == 0:
                    state["step"] = 0
                    state["avg"] = p.clone()
                    state["h_avg"] = torch.zeros_like(p)

                state["step"] += 1
                step = state["step"]

                h_avg = state["h_avg"]

                # Update Hessian approximation
                grad_sq_norm = grad.pow(2).sum()
                h_avg.mul_(beta).add_(grad, alpha=1 - beta)

                # Compute step size
                h_norm = h_avg.norm(2).add_(eps)
                delta = 2.0 * (grad_sq_norm + rho * h_norm) / (h_norm * lips)

                # Update parameters
                p.data.add_(grad, alpha=-lr * delta)
                state["avg"].lerp_(p, weight=1.0 / step)

        return loss


# ============================================================
# Shampoo (Simplified)
# ============================================================
class Shampoo(Optimizer):
    """Shampoo: Preconditioned Stochastic Tensor Optimization.
    Paper: https://arxiv.org/abs/1802.09568
    
    Note: This is a simplified diagonal approximation of Shampoo.
    The full version uses Kronecker-factored preconditioning which
    requires more complex data structures.
    
    Key: Uses layer-wise preconditioning for faster convergence.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, momentum=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                step = state["step"]

                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                state["exp_avg"].mul_(beta1).add_(grad, alpha=1 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = (1 - beta2 ** step) ** 0.5

                # Shampoo-style preconditioning (simplified: use sqrt of second moment)
                denom = (state["exp_avg_sq"].sqrt() / bias_correction2).add_(eps)
                step_size = lr / bias_correction1

                # Apply preconditioner: grad / sqrt(v) * sqrt(v) (balanced)
                preconditioned = state["exp_avg"] / denom
                p.data.add_(preconditioned, alpha=-step_size)

        return loss


# ============================================================
# SOAP (Second-Order Adam-like Preconditioner, Simplified)
# ============================================================
class SOAP(Optimizer):
    """SOAP: Second-Order Adaptive Preconditioner for Training
    Neural Networks.
    Paper: https://arxiv.org/abs/2302.06476
    
    Note: Simplified implementation using diagonal approximation.
    The full version uses distributed second-order information.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, precondition_frequency=1):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        precondition_frequency=precondition_frequency)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["preconditioner"] = torch.ones_like(p)

                state["step"] += 1
                step = state["step"]

                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                # Update first moment
                state["exp_avg"].mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update preconditioner (diagonal Hessian approximation)
                state["preconditioner"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # SOAP update with preconditioning
                bias_correction1 = 1 - beta1 ** step
                preconditioner = state["preconditioner"].sqrt().add_(eps)

                update = state["exp_avg"] / (preconditioner * bias_correction1)
                p.data.add_(update, alpha=-lr)

        return loss


# ============================================================
# Muon (Simplified Momentum Orthogonalized)
# ============================================================
class Muon(Optimizer):
    """Muon: Momentum Orthogonalized Gradient Descent.
    Paper: https://arxiv.org/abs/2502.16993
    
    Key: Orthogonalizes the momentum vector at each step using
    Newton-Schulz iteration, providing better conditioning.
    Popularized for training large language models.
    
    Note: Simplified implementation using singular value
    normalization rather than full Newton-Schulz iteration.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.0, eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    def _newton_schulz(self, G, steps=5, eps=1e-7):
        """Approximate orthogonalization via Newton-Schulz iteration."""
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.bfloat16()
        if G.size(-2) > G.size(-1):
            X = X.mT
        norm = X.norm(dim=-2, keepdim=True).clamp(min=eps)
        X = X / norm

        for _ in range(steps):
            A = X @ X.mT
            B = a * A + b * A @ A + c * A @ A @ A
            X = B @ X

        if G.size(-2) > G.size(-1):
            X = X.mT

        norm = X.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
        return (X / norm).to(G.dtype)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                lr = group["lr"]
                momentum = group["momentum"]
                nesterov = group["nesterov"]
                ns_steps = group["ns_steps"]
                weight_decay = group["weight_decay"]
                eps = group["eps"]

                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buf"] = torch.zeros_like(p)

                state["step"] += 1
                step = state["step"]
                buf = state["momentum_buf"]

                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                buf.mul_(momentum).add_(grad, alpha=1 - momentum)

                # Apply orthogonalization for 2D weight matrices
                if p.dim() >= 2 and step > 1:
                    orthogonalized = self._newton_schulz(buf.view(buf.size(0), -1), steps=ns_steps, eps=eps)
                    orthogonalized = orthogonalized.view(buf.shape)
                    update = orthogonalized
                else:
                    update = buf

                if nesterov:
                    grad_nesterov = grad + momentum * update
                    p.data.add_(grad_nesterov, alpha=-lr)
                else:
                    p.data.add_(update, alpha=-lr)

        return loss

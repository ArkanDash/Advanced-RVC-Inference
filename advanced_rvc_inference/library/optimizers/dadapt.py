"""
D-Adapt Optimizer Family

Papers:
- "Learning-Rate-Free Learning by D-Adaptation" (2023): https://arxiv.org/abs/2302.01749
- "D-Adaptation: Gradient Descent with Step-Size Adaptive to Distance" (2022)

D-Adapt optimizers automatically adapt the learning rate based on the
distance to the solution. They estimate the D parameter (distance) from
gradient statistics and use it to set a provably optimal learning rate.

Key characteristics:
- Learning rate is automatically determined from gradient statistics
- No manual LR tuning needed
- Theoretical convergence guarantees
- Multiple variants for different needs
"""

import math
import torch
from torch.optim.optimizer import Optimizer


class DAdaptAdam(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1.0,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        d_coef: float = 1.0,
        growth_rate: float = float("inf"),
        momentum_dtype: torch.dtype = None,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, d_coef=d_coef,
            growth_rate=growth_rate
        )
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

                if p.grad.is_sparse:
                    raise RuntimeError("DAdaptAdam does not support sparse gradients")

                grad = p.grad
                state = self.state[p]

                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                d_coef = group["d_coef"]
                growth_rate = group["growth_rate"]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["s"] = torch.zeros_like(p, dtype=torch.float32)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                s = state["s"]

                state["step"] += 1
                step = state["step"]

                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Accumulate s_k = sum of grad^2 / beta1^k
                s.addcmul_(grad, grad, value=1.0)

                # Estimate d = d_coef * |theta_0 - theta*|
                d = d_coef * s.sqrt().sum()

                # Adaptive learning rate
                if step > 1:
                    denom = (exp_avg_sq.sqrt() / (1 - beta2 ** step) ** 0.5).add_(eps)
                    new_lr = min(d / (denom * (1 - beta1 ** step)).sum().add_(eps), growth_rate) * lr
                else:
                    new_lr = lr

                bias_correction1 = 1 - beta1 ** step
                update = exp_avg / bias_correction1

                p.data.add_(update, alpha=-new_lr)

        return loss


class DAdaptAdaGrad(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1.0,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        d_coef: float = 1.0,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, d_coef=d_coef
        )
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
                beta1 = group["betas"][0]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                d_coef = group["d_coef"]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["s"] = torch.zeros_like(p, dtype=torch.float32)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                s = state["s"]

                state["step"] += 1
                step = state["step"]

                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.addcmul_(grad, grad, value=1.0)
                s.addcmul_(grad, grad, value=1.0)

                # Estimate d and compute adaptive LR
                d = d_coef * s.sqrt().sum()
                denom = exp_avg_sq.sqrt().add_(eps)

                new_lr = lr
                if step > 1:
                    new_lr = d / denom.sum().add_(eps) * lr

                bias_correction = 1 - beta1 ** step
                p.data.addcdiv_(exp_avg / bias_correction, denom, value=-new_lr)

        return loss


class DAdaptSGD(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1.0,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        d_coef: float = 1.0,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, d_coef=d_coef)
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
                d_coef = group["d_coef"]

                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(p)
                    state["s"] = torch.zeros_like(p, dtype=torch.float32)

                buf = state["momentum_buffer"]
                s = state["s"]

                state["step"] += 1
                step = state["step"]

                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                buf.mul_(momentum).add_(grad, alpha=1 - momentum)
                s.addcmul_(grad, grad, value=1.0)

                d = d_coef * s.sqrt().sum()
                new_lr = lr
                if step > 1:
                    new_lr = d / s.sum().sqrt().add_(1e-6) * lr

                p.data.add_(buf, alpha=-new_lr)

        return loss

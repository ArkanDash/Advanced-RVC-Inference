"""
Adan (Adaptive Nesterov Momentum Estimator) Optimizer

Paper: "Adan: Adaptive Nesterov Momentum Algorithm for Faster
        Optimizing Deep Models" (2022)
Reference: https://arxiv.org/abs/2208.06677

Adan combines Nesterov momentum with adaptive learning rates and
gradient difference (the difference between current and previous gradients).
This combination provides faster convergence and better generalization
compared to Adam and AdamW.

Key characteristics:
- Nesterov momentum estimation
- Uses gradient differences for adaptive LR
- Orthogonal regularization for better conditioning
- Strong performance on vision and NLP tasks
"""

import math
import torch
from torch.optim.optimizer import Optimizer


class Adan(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.98, 0.92, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.02,
        no_prox: bool = False,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 2: {betas[2]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, no_prox=no_prox
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
                    raise RuntimeError("Adan does not support sparse gradients")

                grad = p.grad
                state = self.state[p]

                lr = group["lr"]
                beta1, beta2, beta3 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                no_prox = group["no_prox"]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["prev_grad"] = grad.clone()

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                prev_grad = state["prev_grad"]

                # Gradient difference
                diff = grad - prev_grad

                # Update previous gradient
                prev_grad.copy_(grad)

                # Update moments
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1).add_(diff, alpha=(1 - beta1) * (1 - beta2))
                exp_avg_sq.mul_(beta3).addcmul_(exp_avg, exp_avg, value=1 - beta3)

                # Bias correction
                bias_correction = (1 - beta3) ** (state.get("step", 1))
                denom = (exp_avg_sq / bias_correction).sqrt().add_(eps)

                # Weight decay
                if weight_decay != 0.0:
                    if no_prox:
                        p.data.add_(p.data, alpha=-weight_decay * lr)
                    else:
                        p.data.mul_(1.0 - lr * weight_decay)

                # Update parameters (Nesterov-style)
                p.data.addcdiv_(exp_avg, denom, value=-lr)

                state["step"] = state.get("step", 1) + 1

        return loss

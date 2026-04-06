"""
Prodigy Optimizer

Paper: "Prodigy: An Expeditiously Adaptive Parameter-Free Learner" (2023)
Reference: https://arxiv.org/abs/2306.06101

Prodigy is a parameter-free optimizer that automatically sets the learning
rate based on the distance to the solution (D0) and a specified fraction
of that distance to traverse (D). It adapts the learning rate during
training without requiring manual tuning.

Key characteristics:
- Learning rate free (auto-tuned from D0 and d_coef)
- Works with any model and task
- Adapts during training
- Based on Adam with dynamic LR scheduling
"""

import math
import torch
from torch.optim.optimizer import Optimizer


class Prodigy(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1.0,
        betas: tuple = (0.9, 0.999),
        beta3: float = None,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        d_coef: float = 1.0,
        growth_rate: float = float("inf"),
        use_bias_correction: bool = False,
        safetensors: bool = True,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if d_coef < 0.0:
            raise ValueError(f"Invalid d_coef value: {d_coef}")

        if beta3 is None:
            beta3 = betas[1]

        defaults = dict(
            lr=lr,
            betas=betas,
            beta3=beta3,
            eps=eps,
            weight_decay=weight_decay,
            d_coef=d_coef,
            growth_rate=growth_rate,
            use_bias_correction=use_bias_correction,
            safetensors=safetensors,
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
                    raise RuntimeError("Prodigy does not support sparse gradients")

                grad = p.grad
                state = self.state[p]

                lr, beta1, beta2, beta3 = (
                    group["lr"],
                    group["betas"][0],
                    group["betas"][1],
                    group["beta3"],
                )
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                d_coef = group["d_coef"]
                growth_rate = group["growth_rate"]
                use_bias_correction = group["use_bias_correction"]
                safetensors = group["safetensors"]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["exp_avg_3"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq, exp_avg_3 = (
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    state["exp_avg_3"],
                )

                if safetensors and torch.is_floating_point(p) and torch.isfinite(p).all():
                    state["safetensors_p"] = p.clone()

                # Weight decay
                if weight_decay != 0.0:
                    p.data.add_(p.data, alpha=-weight_decay * lr)

                # Update moments
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg_3.mul_(beta3).add_(grad, alpha=1 - beta3)

                state["step"] = state.get("step", 0) + 1
                step = state["step"]

                # Bias correction
                bias_correction1 = 1 - beta1 ** step if use_bias_correction else 1.0
                bias_correction2 = (1 - beta2 ** step) ** 0.5 if use_bias_correction else 1.0

                # Compute the denominator
                denom = (exp_avg_sq.sqrt() / bias_correction2).add_(eps)

                # Prodigy LR adjustment
                exp_avg_corrected = exp_avg / bias_correction1

                if step > 1:
                    # Estimate distance to solution
                    p_old = state.get("safetensors_p", p.clone())
                    d_param = (p_old - p).abs().max()
                    d_grad = grad.abs().max()
                    if d_param != 0 and d_grad != 0:
                        d = d_coef * d_param / d_grad
                        lr = min(d / lr, growth_rate) * lr

                p.data.addcdiv_(exp_avg_corrected, denom, value=-lr)

        return loss

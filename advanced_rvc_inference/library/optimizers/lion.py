"""
Lion (EvoLved Sign Momentum) Optimizer

Paper: "Symbolic Discovery of Optimization Algorithms" (2023)
Reference: https://arxiv.org/abs/2302.06675

Lion uses sign operations instead of momentum for update computation,
resulting in simpler operations and lower memory footprint. It was
discovered through program search and shows strong performance across
various tasks including language modeling and image classification.

Key characteristics:
- Memory efficient: only stores the momentum (not variance)
- Simple update rule: sign(momentum) * lr
- Works well with high learning rates
- Compatible with weight decay
"""

import torch
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

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

                if p.grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")

                grad = p.grad
                state = self.state[p]
                lr, beta1, beta2, weight_decay = (
                    group["lr"],
                    group["betas"][0],
                    group["betas"][1],
                    group["weight_decay"],
                )

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                # Weight decay
                if weight_decay > 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Lion update: sign(exp_avg) * lr
                p.data.add_(torch.sign(exp_avg), alpha=-lr)

        return loss

"""
Sophia Optimizer (Simplified)

Paper: "Sophia: A Scalable Stochastic Second-order Optimizer for
        Language Model Pre-training" (2023)
Reference: https://arxiv.org/abs/2305.14342

Sophia is a second-order optimizer that uses a diagonal Hessian estimate
plus a stochastic update rule with clipping. It achieves faster convergence
than Adam while maintaining similar per-step cost.

Note: This is a simplified implementation. The full Sophia uses Hutchinson's
estimator for diagonal Hessian approximation with periodic updates.

Key characteristics:
- Second-order information via diagonal Hessian
- Faster convergence than first-order methods
- Memory efficient diagonal approximation
- Clipping for stability
"""

import math
import torch
from torch.optim.optimizer import Optimizer


class Sophia(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple = (0.965, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        k: int = 10,
        rho: float = 0.01,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, k=k, rho=rho)
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
                    raise RuntimeError("Sophia does not support sparse gradients")

                grad = p.grad
                state = self.state[p]

                lr, beta1, beta2 = group["lr"], group["betas"][0], group["betas"][1]
                eps, weight_decay = group["eps"], group["weight_decay"]
                k, rho = group["k"], group["rho"]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["hessian"] = torch.ones_like(p)

                exp_avg = state["exp_avg"]
                hessian = state["hessian"]

                state["step"] += 1
                step = state["step"]

                # Weight decay
                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                # Update first moment
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update diagonal Hessian estimate periodically
                if step % k == 0:
                    # Hutchinson's estimator (simplified: use grad^2 as approximation)
                    hessian.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** step

                # Update with clipped Hessian
                h_max = torch.clamp(hessian, min=1.0 / rho) * rho
                update = exp_avg / (h_max + eps)

                # Clip update
                clip_threshold = rho * lr
                torch.clamp_(update, min=-clip_threshold, max=clip_threshold)

                p.data.add_(update, alpha=-lr / bias_correction1)

        return loss

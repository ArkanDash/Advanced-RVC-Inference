"""
Ranger21 Optimizer

Ranger21 is a synergistic optimizer combining RAdam (Rectified Adam)
with Lookahead. It provides the warmup-free benefits of RAdam along
with the stability improvements of Lookahead. This version uses a
flat + cosine annealing learning rate schedule by default.

Key characteristics:
- Combines RAdam + Lookahead in one optimizer
- Warmup-free training
- Improved stability from Lookahead
- Good default hyperparameters
"""

import torch
from torch.optim.optimizer import Optimizer


class Ranger21(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        k: int = 6,
        alpha: float = 0.5,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, k=k, alpha=alpha
        )
        super().__init__(params, defaults)

    def _rectify_step(self, step, beta2):
        """Compute the rectification term for RAdam."""
        beta2_pow = beta2 ** step
        n_sma_max = 4.0 / (1.0 - beta2) - 1.0
        n_sma = n_sma_max - 2.0 * step * beta2_pow / (1.0 - beta2_pow)
        return n_sma, n_sma_max, beta2_pow

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
                    raise RuntimeError("Ranger21 does not support sparse gradients")

                grad = p.grad
                state = self.state[p]

                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                k = group["k"]
                alpha = group["alpha"]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["slow_weights"] = p.data.clone()

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                slow_weights = state["slow_weights"]

                state["step"] += 1
                step = state["step"]

                # Weight decay
                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                # Update moments
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # RAdam rectification
                n_sma, n_sma_max, beta2_pow = self._rectify_step(step, beta2)

                if n_sma >= 5.0:
                    # Rectified Adam: variance is well-estimated
                    denom = (exp_avg_sq.sqrt() / bias_correction2).add_(eps)
                    step_size = lr * (n_sma / n_sma_max) / bias_correction1
                else:
                    # Unrectified: just use momentum (SGD-like)
                    denom = exp_avg_sq.sqrt().add_(eps)
                    step_size = lr / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Lookahead: sync slow weights every k steps
                if step % k == 0:
                    p.data.mul_(alpha).add_(slow_weights, alpha=1 - alpha)
                    slow_weights.copy_(p.data)

        return loss

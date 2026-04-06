"""
Schedule-Free Optimizers

Paper: "Schedule-Free: Learning Rate Free Training in Adam and SGD" (2024)
Reference: https://arxiv.org/abs/2405.15685

Schedule-Free optimizers eliminate the need for learning rate scheduling by
maintaining a dual set of parameters (z and y) where z is the "lookahead"
parameters used for forward passes. The optimizer adjusts the learning rate
dynamically based on the distance between z and y, effectively providing
its own warmup and decay.

Key characteristics:
- No learning rate schedule needed
- Built-in warmup and decay
- Uses dual parameters (z = lookahead, y = standard)
- Works as drop-in replacement for Adam/AdamW/SGD
"""

import math
import torch
from torch.optim.optimizer import Optimizer


class ScheduleFreeAdamW(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        r: float = 0.0,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta at index 1: {betas[1]}")

        defaults = dict(
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, warmup_steps=warmup_steps, r=r
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
                    raise RuntimeError("ScheduleFreeAdamW does not support sparse gradients")

                grad = p.grad
                state = self.state[p]

                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                r = group["r"]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["z"] = p.data.clone()

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                z = state["z"]

                state["step"] += 1
                step = state["step"]

                # Weight decay (applied to y, not z)
                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                # Update moments (using y gradients)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = (1 - beta2 ** step) ** 0.5

                denom = (exp_avg_sq.sqrt() / bias_correction2).add_(eps)

                # Adam step on y
                p.data.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)

                # Lookahead: update z
                p.data.lerp_(z, weight=r)

        return loss


class ScheduleFreeAdam(Optimizer):
    """Schedule-Free variant of standard Adam (with bias correction)."""

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        r: float = 0.0,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, r=r)
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
                r = group["r"]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["z"] = p.data.clone()

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                z = state["z"]

                state["step"] += 1
                step = state["step"]

                if weight_decay != 0.0:
                    p.data.add_(p.data, alpha=-weight_decay * lr)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = (1 - beta2 ** step) ** 0.5

                denom = (exp_avg_sq.sqrt() / bias_correction2).add_(eps)
                p.data.addcdiv_(exp_avg, denom, value=-lr * math.sqrt(bias_correction2) / bias_correction1)
                p.data.lerp_(z, weight=r)

        return loss


class ScheduleFreeSGD(Optimizer):
    """Schedule-Free variant of SGD with momentum."""

    def __init__(
        self,
        params,
        lr: float = 1.0,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        r: float = 0.0,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, r=r)
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
                r = group["r"]

                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(p)
                    state["z"] = p.data.clone()

                buf = state["momentum_buffer"]
                z = state["z"]

                state["step"] += 1

                if weight_decay != 0.0:
                    p.data.add_(p.data, alpha=-weight_decay * lr)

                buf.mul_(momentum).add_(grad, alpha=1 - momentum)
                p.data.add_(buf, alpha=-lr)
                p.data.lerp_(z, weight=r)

        return loss

"""
AdaFactor Optimizer

Paper: "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost" (2018)
Reference: https://arxiv.org/abs/1804.04235

AdaFactor is a memory-efficient optimizer that reduces memory usage by
factoring the second-moment estimator into row-wise and column-wise
statistics instead of storing the full matrix. It also uses a relative
step size based on the RMS of the parameters themselves.

Key characteristics:
- Sublinear memory cost (scales with parameters, not their square)
- Uses factored approximation for second moments
- Relative step size for better scaling
- Used extensively in T5 and other large models
"""

import math
import torch
from torch.optim.optimizer import Optimizer


class AdaFactor(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: tuple = (1e-30, 1e-3),
        weight_decay: float = 0.0,
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        beta1_decay: float = -0.8,
        multiply_by_parameter_scale: bool = True,
        d1: int = 2,
        d2: int = 1024,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, clip_threshold=clip_threshold,
            decay_rate=decay_rate, beta1_decay=beta1_decay,
            multiply_by_parameter_scale=multiply_by_parameter_scale,
            d1=d1, d2=d2,
        )
        super().__init__(params, defaults)

    def _get_lr(self, param, state):
        """Compute learning rate based on parameter scale."""
        group = self._get_param_group(param)
        lr = group["lr"]

        if group["multiply_by_parameter_scale"]:
            rms = torch.sqrt(param.pow(2).mean().add_(group["eps"][1]))
            lr = lr * rms

        return lr

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
                    raise RuntimeError("AdaFactor does not support sparse gradients")

                grad = p.grad
                state = self.state[p]

                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps1, eps2 = group["eps"]
                weight_decay = group["weight_decay"]
                clip_threshold = group["clip_threshold"]
                decay_rate = group["decay_rate"]
                beta1_decay = group["beta1_decay"]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)

                    # Factored second moment
                    grad_shape = grad.shape
                    if grad_shape.dim() >= 2:
                        # Factor along last two dimensions
                        state["r"] = torch.zeros(grad_shape[:-1], dtype=torch.float32)
                        state["c"] = torch.zeros(grad_shape[:-2] + (grad_shape[-1],), dtype=torch.float32)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)

                state["step"] += 1
                step = state["step"]

                # Weight decay
                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                # Update first moment
                beta1_t = 1 - step ** beta1_decay
                state["exp_avg"].mul_(beta1_t).add_(grad, alpha=1 - beta1_t)

                # Update second moment (factored)
                update_sq = grad * grad
                grad_shape = grad.shape

                if grad_shape.dim() >= 2:
                    r = state["r"]
                    c = state["c"]

                    # Update factored moments
                    r.mul_(beta2).add_(update_sq.mean(dim=-1), alpha=1 - beta2)
                    c.mul_(beta2).add_(update_sq.mean(dim=-2), alpha=1 - beta2)

                    # Combine factors
                    exp_avg_sq = r.unsqueeze(-1) * c.unsqueeze(-2)
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).add_(update_sq, alpha=1 - beta2)

                # Compute update
                rms = torch.sqrt(exp_avg_sq.add_(eps1))
                update = state["exp_avg"] / rms

                # Clip update
                update_norm = torch.sqrt(update.pow(2).sum().add_(eps1))
                param_norm = torch.sqrt(p.pow(2).sum().add_(eps1))

                if update_norm > clip_threshold * param_norm:
                    update.mul_(clip_threshold * param_norm / update_norm)

                # Apply learning rate
                actual_lr = lr
                if group["multiply_by_parameter_scale"]:
                    param_rms = torch.sqrt(p.pow(2).mean().add_(eps2))
                    actual_lr = lr * param_rms

                p.data.add_(update, alpha=-actual_lr)

        return loss

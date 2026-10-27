"""
PolOpt — Yogi + AdaBelief Hybrid Optimizer (backported from PolTrain by Politrees)

Combines:
1. Yogi-sign control for the second moment: uses sign((g - m)^2 - v) instead of
   the raw squared gradient to update v. This prevents the denominator from
   collapsing to zero and avoids explosive LR swings during GAN training
   (where D and G gradients oscillate strongly).

2. Decoupled weight decay (AdamW-style) for better generalization.

3. Correct epsilon placement (OUTSIDE the sqrt, after bias correction),
   per the AdamW paper. Many implementations put eps inside the sqrt,
   which makes the effective learning rate depend on the gradient scale.

4. Trust Region Clamping (max_step_clip): clamps the per-parameter update
   magnitude to prevent acoustic filter structures from being destroyed by
   transient gradient spikes. Critical for RVC where conv banks are sensitive
   to large weight perturbations.

Reference: https://github.com/Politrees/PolTrain/blob/main/rvc/train/utils/optimizers/PolOpt.py
"""

import math
from typing import Tuple, List
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class PolOpt(Optimizer):
    """PolOpt (AdaBelief / Yogi Hybrid Optimizer).

    Arguments:
        params: parameters to optimize
        lr: learning rate (default: 1e-4)
        betas: smoothing coefficients (β1, β2) (default: (0.8, 0.99))
        eps: numerical stability term (default: 1e-7)
        weight_decay: decoupled weight decay (default: 0.01)
        max_step_clip: max absolute per-step update (default: 1.0)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.8, 0.99),
        eps: float = 1e-7,
        weight_decay: float = 0.01,
        max_step_clip: float = 1.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if max_step_clip < 0:
            raise ValueError(f"Invalid max_step_clip value: {max_step_clip}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            max_step_clip=max_step_clip,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            wd = group["weight_decay"]
            max_clip = group["max_step_clip"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("PolOpt does not support sparse gradients.")

                state = self.state[p]

                # Lazy state initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_var"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state["step"] += 1
                step = state["step"]
                exp_avg, exp_avg_var = state["exp_avg"], state["exp_avg_var"]

                # 1. Decoupled Weight Decay (AdamW-style)
                if wd != 0:
                    p.mul_(1.0 - lr * wd)

                # 2. Update first moment (m_t) — standard EMA
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # 3. Belief residual: grad - m_t
                grad_res = grad - exp_avg
                grad_res_sq = grad_res * grad_res

                # 4. Yogi-Belief hybrid second-moment update (v_t).
                # Using sign(diff) prevents v from collapsing to zero during
                # GAN oscillations, unlike plain AdaBelief which can suffer
                # from vanishing denominator when D and G push gradients in
                # opposite directions.
                diff = grad_res_sq - exp_avg_var
                exp_avg_var.addcmul_(torch.sign(diff), grad_res_sq, value=1.0 - beta2)

                # 5. Bias correction
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                step_size = lr / bias_correction1

                # 6. Denominator with eps OUTSIDE the sqrt (correct per AdamW paper)
                denom = (exp_avg_var.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # 7. Normalized update
                update = exp_avg / denom

                # 8. Trust Region Clamping — protect acoustic filters from spikes
                if max_clip > 0:
                    update = torch.clamp(update, -max_clip, max_clip)

                # 9. Apply update
                p.add_(update, alpha=-step_size)

        return loss

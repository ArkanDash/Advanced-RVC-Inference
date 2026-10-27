"""
Prodigy Optimizer - D-Adaptation Based Step Size
==================================================

Implements Adam with Prodigy step-sizes based on D-adaptation algorithm.
Automatically adapts learning rates during training - leave LR at 1.0!

Key advantages for RVC training:
- Automatic LR adaptation (no manual tuning needed)
- Built-in warmup effect via growth_rate parameter
- Stable convergence for GAN training scenarios
- Memory-efficient with slice_p option for large models

Paper reference: "Learning Rate Free Training with D-Adaptation"
https://arxiv.org/abs/2301.07733

Backported from Codename RVC Fork v4 for Advanced-RVC-Inference.
"""

import math
from typing import Any, Optional
import torch
import torch.optim
import logging

if hasattr(torch.distributed, 'dist'):
    _has_dist = True
else:
    _has_dist = False


class Prodigy(torch.optim.Optimizer):
    r"""
    Implements Adam with Prodigy step-sizes (D-adaptation).
    
    Leave LR set to 1 unless you encounter instability.
    The optimizer automatically determines the appropriate step size.

    Arguments:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float): Learning rate adjustment parameter. Increases or decreases
            the Prodigy learning rate (default: 1.0).
        betas (Tuple[float, float]): Coefficients used for computing running
            averages of gradient and its square (default: (0.9, 0.999)).
        beta3 (float): Coefficient for computing the Prodigy stepsize using
            running averages. If set to None, uses sqrt(beta2) (default: None).
        eps (float): Term added to the denominator outside of the root operation
            to improve numerical stability (default: 1e-8).
        weight_decay (float): Weight decay, i.e. L2 penalty (default: 0).
        decouple (boolean): Use AdamW style decoupled weight decay (default: True).
        use_bias_correction (boolean): Turn on Adam's bias correction (default: False).
        safeguard_warmup (boolean): Remove lr from denominator of D estimate to
            avoid issues during warm-up stage (default: False).
        d0 (float): Initial D estimate for D-adaptation (default: 1e-6).
        d_coef (float): Coefficient in the expression for the estimate of d
            (default: 1.0). Values like 0.5 and 2.0 typically work well.
        growth_rate (float): Prevent D estimate from growing faster than this
            multiplicative rate. Default is inf (unrestricted). Values like 1.02
            give a kind of learning rate warmup effect.
        fsdp_in_use (bool): Set to True if using sharded parameters for proper
            gradient synchronization (default: False, auto-detected).
        slice_p (int): Reduce memory usage by calculating LR adaptation statistics
            on only every pth entry of each tensor. For values > 1 this is an
            approximation. Values ~11 are reasonable (default: 1).

    Example:
        # Basic usage - let Prodigy find the right LR automatically
        optimizer = Prodigy(model.parameters(), lr=1.0)
        
        # With conservative growth for smoother training
        optimizer = Prodigy(
            model.parameters(),
            lr=1.0,
            growth_rate=1.02,  # Gentle warmup-like effect
            safeguard_warmup=True  # Extra safety during early training
        )
    """
    
    def __init__(self, params, lr=1.0,
                 betas=(0.9, 0.999), beta3=None,
                 eps=1e-8, weight_decay=0, decouple=True, 
                 use_bias_correction=False, safeguard_warmup=False,
                 d0=1e-6, d_coef=1.0, growth_rate=float('inf'),
                 fsdp_in_use=False,
                 slice_p=1):
        
        # Parameter validation
        if not 0.0 < d0:
            raise ValueError("Invalid d0 value: {}".format(d0))
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 < eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        if decouple and weight_decay > 0:
            logging.debug("Prodigy: Using decoupled weight decay")

        defaults = dict(lr=lr, betas=betas, beta3=beta3,
                        eps=eps, weight_decay=weight_decay,
                        d=d0, d0=d0, d_max=d0,
                        d_numerator=0.0, d_coef=d_coef,
                        k=0, growth_rate=growth_rate,
                        use_bias_correction=use_bias_correction,
                        decouple=decouple, safeguard_warmup=safeguard_warmup,
                        fsdp_in_use=fsdp_in_use,
                        slice_p=slice_p)
        self.d0 = d0
        super(Prodigy, self).__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return False

    @property
    def supports_flat_params(self):
        return True

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        d_denom = 0.0
        group = self.param_groups[0]
        
        use_bias_correction = group['use_bias_correction']
        beta1, beta2 = group['betas']
        beta3 = group['beta3']
        if beta3 is None:
            beta3 = math.sqrt(beta2)
        k = group['k']

        d = group['d']
        d_max = group['d_max']
        d_coef = group['d_coef']
        lr = max(group['lr'] for group in self.param_groups)

        if use_bias_correction:
            bias_correction = ((1 - beta2**(k+1))**0.5) / (1 - beta1**(k+1))
        else:
            bias_correction = 1

        dlr = d * lr * bias_correction
        
        growth_rate = group['growth_rate']
        decouple = group['decouple']
        fsdp_in_use = group['fsdp_in_use']

        d_numerator = group['d_numerator']
        d_numerator *= beta3
        delta_numerator = 0.0

        for group in self.param_groups:
            decay = group['weight_decay']
            k = group['k']
            eps = group['eps']
            group_lr = group['lr']
            d0 = group['d0']
            safeguard_warmup = group['safeguard_warmup']
            slice_p = group['slice_p']

            if group_lr not in [lr, 0.0]:
                raise RuntimeError(
                    f"Setting different lr values in different parameter groups "
                    f"is only supported for values of 0"
                )

            for p in group['params']:
                if p.grad is None:
                    continue
                    
                # Auto-detect FSDP
                if hasattr(p, "_fsdp_flattened"):
                    fsdp_in_use = True
                
                grad = p.grad.data
                
                # Apply weight decay (coupled variant)
                if decay != 0 and not decouple:
                    grad.add_(p.data, alpha=decay)

                state = self.state[p]

                # State initialization
                if 'step' not in state:
                    state['step'] = 0
                    state['s'] = torch.zeros_like(p.data.flatten()[::slice_p]).detach()

                    if p.any():
                        state['p0'] = p.flatten()[::slice_p].detach().clone()
                    else:
                        # All zeros - save VRAM
                        state['p0'] = torch.tensor(0, device=p.device, dtype=p.dtype)

                    # Exponential moving average of gradients
                    if beta1 > 0:
                        state['exp_avg'] = torch.zeros_like(p.data).detach()
                    # Exponential moving average of squared gradients
                    state['exp_avg_sq'] = torch.zeros_like(p.data).detach()

                exp_avg_sq = state['exp_avg_sq']
                s = state['s']
                p0 = state['p0']

                if group_lr > 0.0:
                    # Use d/d0 normalization to avoid very small values
                    sliced_grad = grad.flatten()[::slice_p]
                    delta_numerator += (d / d0) * dlr * torch.dot(
                        sliced_grad, 
                        p0.data - p.data.flatten()[::slice_p]
                    ).item()

                    # Adam EMA updates
                    if beta1 > 0:
                        exp_avg = state['exp_avg']
                        exp_avg.mul_(beta1).add_(grad, alpha=d * (1-beta1))
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=d * d * (1-beta2))

                    if safeguard_warmup:
                        s.mul_(beta3).add_(sliced_grad, alpha=((d / d0) * d))
                    else:
                        s.mul_(beta3).add_(sliced_grad, alpha=((d / d0) * dlr))
                    d_denom += s.abs().sum().item()

        # Compute D-adaptation estimate
        d_hat = d

        # No progress or no gradients
        if d_denom == 0 and not fsdp_in_use:
            return loss
        
        if lr > 0.0:
            if fsdp_in_use and _has_dist:
                dist_tensor = torch.zeros(2).cuda()
                dist_tensor[0] = delta_numerator
                dist_tensor[1] = d_denom
                torch.distributed.all_reduce(dist_tensor, op=torch.distributed.ReduceOp.SUM)
                global_d_numerator = d_numerator + dist_tensor[0]
                global_d_denom = dist_tensor[1]
            else:
                global_d_numerator = d_numerator + delta_numerator
                global_d_denom = d_denom

            d_hat = d_coef * global_d_numerator / global_d_denom
            
            if d == group['d0']:
                d = max(d, d_hat)
            d_max = max(d_max, d_hat)
            d = min(d_max, d * growth_rate)

        # Update all groups with new D estimate
        for group in self.param_groups:
            group['d_numerator'] = global_d_numerator
            group['d_denom'] = global_d_denom
            group['d'] = d
            group['d_max'] = d_max
            group['d_hat'] = d_hat

            decay = group['weight_decay']
            k = group['k']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                state = self.state[p]
                exp_avg_sq = state['exp_avg_sq']

                state['step'] += 1

                denom = exp_avg_sq.sqrt().add_(d * eps)

                # Apply weight decay (decoupled variant)
                if decay != 0 and decouple:
                    p.data.add_(p.data, alpha=-decay * dlr)

                # Take optimization step
                if beta1 > 0:
                    exp_avg = state['exp_avg']
                    p.data.addcdiv_(exp_avg, denom, value=-dlr)
                else:
                    p.data.addcdiv_(grad, denom, value=-dlr * d)

            group['k'] = k + 1

        return loss

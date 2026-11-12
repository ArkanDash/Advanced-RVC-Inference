"""
Ranger2020 Deep Learning Optimizer
===================================

RAdam + Lookahead + Gradient Centralization, combined into one optimizer.

Credits:
- Gradient Centralization: https://arxiv.org/abs/2004.01461v2
- RAdam: https://github.com/LiyuanLucasLiu/RAdam
- Lookahead: MZhang, G Hinton - https://arxiv.org/abs/1907.08610

Ranger has been used to capture 12+ records on the FastAI leaderboard.

Backported from Codename RVC Fork v4 for Advanced-RVC-Inference.
Optimized for RVC/SVC voice conversion training stability.
"""

import math
import torch
from torch.optim.optimizer import Optimizer


def centralized_gradient(x, use_gc=True, gc_conv_only=False):
    """
    Apply gradient centralization (Yonghongwei et al., 2020).
    
    Centralizes gradients by subtracting the mean from each tensor,
    which can improve training stability and generalization.
    
    Args:
        x: Gradient tensor
        use_gc: Whether to apply gradient centralization
        gc_conv_only: If True, only apply to conv layers (dim > 3)
        
    Returns:
        Centralized gradient tensor
    """
    if use_gc:
        if gc_conv_only:
            if len(list(x.size())) > 3:
                x.add_(-x.mean(dim=tuple(range(1, len(list(x.size())))), keepdim=True))
        else:
            if len(list(x.size())) > 1:
                x.add_(-x.mean(dim=tuple(range(1, len(list(x.size())))), keepdim=True))
    return x


class Ranger2020(Optimizer):
    """
    Ranger2020 optimizer: RAdam + Lookahead + Gradient Centralization.
    
    This optimizer combines three techniques:
    1. RAdam: Rectified Adam - eliminates need for warmup via variance rectification
    2. Lookahead: Slow/fast weight interpolation for better minima
    3. Gradient Centralization: Improves generalization by centering gradients
    
    Particularly effective for:
    - GAN training (like RVC discriminator/generator)
    - Deep voice conversion models
    - Scenarios requiring stable convergence
    
    Args:
        params: Model parameters
        lr: Learning rate (default: 1e-3)
        alpha: Lookahead interpolation rate (default: 0.5)
        k: Lookahead update frequency in steps (default: 6)
        N_sma_threshhold: Threshold for RAdam variance rectification (default: 5)
        betas: Adam betas (default: (0.95, 0.999))
        eps: Epsilon for numerical stability (default: 1e-5)
        weight_decay: Weight decay coefficient (default: 0)
        use_gc: Enable gradient centralization (default: True)
        gc_conv_only: Only apply GC to conv layers (default: False)
        gc_loc: Apply GC before (True) or after (False) RAdam update
        
    Example:
        optimizer = Ranger2020(
            model.parameters(),
            lr=1e-3,
            use_gc=True,
            gc_conv_only=False  # Apply to all layers
        )
    """

    def __init__(self, params, lr=1e-3,
                 alpha=0.5, k=6, N_sma_threshhold=5,
                 betas=(.95, 0.999), eps=1e-5, weight_decay=0,
                 use_gc=True, gc_conv_only=False, gc_loc=True):

        # Parameter validation
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        if not lr > 0:
            raise ValueError(f'Invalid Learning Rate: {lr}')
        if not eps > 0:
            raise ValueError(f'Invalid eps: {eps}')

        # Initialize defaults
        defaults = dict(lr=lr, alpha=alpha, k=k, step_counter=0, betas=betas,
                        N_sma_threshhold=N_sma_threshhold, eps=eps, 
                        weight_decay=weight_decay)
        super(Ranger2020, self).__init__(params, defaults)

        # Adjustable threshold
        self.N_sma_threshhold = N_sma_threshhold

        # Lookahead params
        self.alpha = alpha
        self.k = k

        # RAdam buffer for state
        self.radam_buffer = [[None, None, None] for _ in range(10)]

        # Gradient centralization settings
        self.gc_loc = gc_loc
        self.use_gc = use_gc
        self.gc_conv_only = gc_conv_only

    def __setstate__(self, state):
        super(Ranger2020, self).__setstate__(state)

    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data.float()

                if grad.is_sparse:
                    raise RuntimeError('Ranger2020 optimizer does not support sparse gradients')

                p_data_fp32 = p.data.float()
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                    # Lookahead slow weights storage
                    state['slow_buffer'] = torch.empty_like(p.data)
                    state['slow_buffer'].copy_(p.data)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # Gradient Centralization (before RAdam update if gc_loc=True)
                if self.gc_loc:
                    grad = centralized_gradient(grad, use_gc=self.use_gc, gc_conv_only=self.gc_conv_only)

                state['step'] += 1

                # Compute variance moving average
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute mean moving average
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # RAdam variance rectification check
                buffered = self.radam_buffer[int(state['step'] % 10)]

                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    
                    if N_sma > self.N_sma_threshhold:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * 
                                            (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / \
                                            (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                # Compute RAdam update
                if N_sma > self.N_sma_threshhold:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    G_grad = exp_avg / denom
                else:
                    G_grad = exp_avg

                # Apply weight decay
                if group['weight_decay'] != 0:
                    G_grad.add_(p_data_fp32, alpha=group['weight_decay'])

                # Gradient Centralization (after RAdam update if gc_loc=False)
                if not self.gc_loc:
                    G_grad = centralized_gradient(G_grad, use_gc=self.use_gc, gc_conv_only=self.gc_conv_only)

                # Apply update
                p_data_fp32.add_(G_grad, alpha=-step_size * group['lr'])
                p.data.copy_(p_data_fp32)

                # Integrated Lookahead update
                if state['step'] % group['k'] == 0:
                    slow_p = state['slow_buffer']
                    # (fast weights - slow weights) * alpha
                    slow_p.add_(p.data - slow_p, alpha=self.alpha)
                    # Copy interpolated weights to param tensor
                    p.data.copy_(slow_p)

        return loss

"""
Lookahead Optimizer Wrapper

Paper: "Lookahead Optimizer: k steps forward, 1 step back" (2019)
Reference: https://arxiv.org/abs/1907.08610

Lookahead is a meta-optimizer that wraps any base optimizer. It periodically
synchronizes a set of "slow weights" with a linear combination of the slow
weights and the "fast weights" (the weights of the base optimizer). This
provides stability and improves generalization.

Key characteristics:
- Wrapper around any base optimizer
- Improves stability and convergence
- Reduces hyperparameter sensitivity
- Minimal overhead (one extra set of weights)
"""

import torch
from torch.optim.optimizer import Optimizer


class Lookahead(Optimizer):
    def __init__(
        self,
        base_optimizer,
        k: int = 5,
        alpha: float = 0.5,
        pullback_momentum: str = "none",
    ):
        """
        Args:
            base_optimizer: The optimizer to wrap (e.g., torch.optim.AdamW)
            k: Number of fast weight steps per slow weight update
            alpha: Linear interpolation factor (0 = slow, 1 = fast)
            pullback_momentum: How to handle momentum after sync
                               ('none', 'reset', 'pullback')
        """
        if k < 1:
            raise ValueError(f"Invalid k parameter: {k}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha parameter: {alpha}")

        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.pullback_momentum = pullback_momentum
        self.param_groups = base_optimizer.param_groups

        # Initialize slow weights as copies of fast weights
        self._slow_weights = []
        for group in base_optimizer.param_groups:
            slow_params = {}
            for p in group["params"]:
                slow_params[p] = p.data.clone()
            self._slow_weights.append(slow_params)

        # Track fast step count
        self._step_count = 0
        self.state = base_optimizer.state

    def __getstate__(self):
        return {
            "base_optimizer": self.base_optimizer,
            "k": self.k,
            "alpha": self.alpha,
            "pullback_momentum": self.pullback_momentum,
            "_slow_weights": self._slow_weights,
            "_step_count": self._step_count,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)
        self._step_count += 1

        # Sync slow and fast weights every k steps
        if self._step_count % self.k == 0:
            for group_idx, group in enumerate(self.base_optimizer.param_groups):
                for p in group["params"]:
                    slow_p = self._slow_weights[group_idx][p]
                    # Interpolate: slow = alpha * fast + (1 - alpha) * slow
                    p.data.mul_(self.alpha).add_(slow_p, alpha=1 - self.alpha)
                    # Update slow weights
                    slow_p.copy_(p.data)

        return loss

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        pass

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)

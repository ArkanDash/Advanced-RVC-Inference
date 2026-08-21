import math
import torch

from torch.optim.optimizer import Optimizer

class AdaBeliefV2(Optimizer):
    def __init__(
        self,
        params,
        lr = 1e-4,
        betas = (0.9, 0.999),
        eps = 1e-8,
        weight_decay = 0,
        amsgrad = True,
        foreach = True,
    ):
        if lr <= 0.0:
            raise ValueError
        if eps < 0.0:
            raise ValueError
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError
        if weight_decay < 0:
            raise ValueError

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            foreach=foreach,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", True)
            group.setdefault("amsgrad", False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["foreach"]:
                self._step_foreach(group)
            else:
                self._step_single(group)

        return loss

    def _step_single(self, group):
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        amsgrad = group["amsgrad"]

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError

            if weight_decay != 0:
                p.mul_(1 - lr * weight_decay)

            state = self.state[p]

            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["exp_avg_var"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if amsgrad:
                    state["max_exp_avg_var"] = torch.zeros_like(p, memory_format=torch.preserve_format)

            exp_avg = state["exp_avg"]
            exp_avg_var = state["exp_avg_var"]

            state["step"] += 1
            step = state["step"]

            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            grad_residual = grad - exp_avg
            exp_avg_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)

            if amsgrad:
                max_exp_avg_var = state["max_exp_avg_var"]
                torch.maximum(max_exp_avg_var, exp_avg_var, out=max_exp_avg_var)
                denom = max_exp_avg_var.add(eps).sqrt()
            else:
                denom = exp_avg_var.add(eps).sqrt()

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            
            step_size = lr * math.sqrt(bias_correction2) / bias_correction1
            
            p.addcdiv_(exp_avg, denom, value=-step_size)

    def _step_foreach(self, group):
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        amsgrad = group["amsgrad"]
        params_with_grad, grads, exp_avgs, exp_avg_vars, max_exp_avg_vars = [], [], [], [], []

        for p in group["params"]:
            if p.grad is None:
                continue
            params_with_grad.append(p)
            grads.append(p.grad)
            
            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["exp_avg_var"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if amsgrad:
                    state["max_exp_avg_var"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            
            exp_avgs.append(state["exp_avg"])
            exp_avg_vars.append(state["exp_avg_var"])
            if amsgrad:
                max_exp_avg_vars.append(state["max_exp_avg_var"])

        if not params_with_grad:
            return

        state = self.state[params_with_grad[0]]
        state["step"] += 1
        step = state["step"]
        for p in params_with_grad[1:]:
            self.state[p]["step"] = step

        if weight_decay != 0:
            torch._foreach_mul_(params_with_grad, 1 - lr * weight_decay)

        torch._foreach_mul_(exp_avgs, beta1)
        torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)

        grad_residuals = torch._foreach_sub(grads, exp_avgs)
        torch._foreach_mul_(exp_avg_vars, beta2)
        torch._foreach_addcmul_(exp_avg_vars, grad_residuals, grad_residuals, value=1 - beta2)

        if amsgrad:
            torch._foreach_maximum_(max_exp_avg_vars, exp_avg_vars)
            denom = torch._foreach_add(max_exp_avg_vars, eps)
        else: denom = torch._foreach_add(exp_avg_vars, eps)
        
        torch._foreach_sqrt_(denom)

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        step_size = lr * math.sqrt(bias_correction2) / bias_correction1

        torch._foreach_addcdiv_(params_with_grad, exp_avgs, denom, value=-step_size)


def get_inverse_sqrt_scheduler(optimizer, warmup_epochs=15, last_epoch=-1):
    def lr_lambda(current_step):
        ep = current_step + 1

        if ep < warmup_epochs: return float(ep) / float(max(1, warmup_epochs))
        return (warmup_epochs ** 0.5) / (ep ** 0.5)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)

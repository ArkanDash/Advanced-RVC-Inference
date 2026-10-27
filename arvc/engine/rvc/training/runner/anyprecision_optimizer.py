import torch

from torch.optim.optimizer import Optimizer

class AnyPrecisionAdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, use_kahan_summation=True, momentum_dtype=torch.bfloat16, variance_dtype=torch.bfloat16, compensation_buffer_dtype=torch.bfloat16):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, use_kahan_summation=use_kahan_summation, momentum_dtype=momentum_dtype, variance_dtype=variance_dtype, compensation_buffer_dtype=compensation_buffer_dtype)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            use_kahan_summation = group["use_kahan_summation"]
            momentum_dtype = group["momentum_dtype"]
            variance_dtype = group["variance_dtype"]
            compensation_buffer_dtype = group["compensation_buffer_dtype"]

            for p in group["params"]:
                if p.grad is None: continue
                if p.grad.is_sparse: raise RuntimeError

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)
                    state["exp_avg"] = torch.zeros_like(p, dtype=momentum_dtype)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=variance_dtype)
                    if use_kahan_summation: state["compensation"] = torch.zeros_like(p, dtype=compensation_buffer_dtype)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                grad = p.grad
                if weight_decay: p.data.mul_(1 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                step_size = lr / bias_correction1

                denom_correction = (1 - beta2**step) ** 0.5
                centered_variance = (exp_avg_sq.sqrt() / denom_correction).add_(eps, alpha=1)

                if use_kahan_summation:
                    compensation = state["compensation"]
                    compensation.addcdiv_(exp_avg, centered_variance, value=-step_size)

                    temp_buffer = p.detach().clone()
                    p.data.add_(compensation)
                    compensation.add_(temp_buffer.sub_(p.data))
                else: p.data.addcdiv_(exp_avg, centered_variance, value=-step_size)
import torch
import torch.nn as nn
import torch.nn.init as init

import math

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def tensor_like(x, y):
    return torch.as_tensor(x, dtype=y.dtype, device=y.device)

def exp(x):
    return torch.exp(x) if torch.is_tensor(x) else math.exp(x)

def sqrt(x):
    return torch.sqrt(x) if torch.is_tensor(x) else math.sqrt(x)

def snake_variance(alpha):
    num = 1 + exp(-8 * alpha ** 2) - 2 * exp(-4 * alpha ** 2)
    return 1 + num / (8 * alpha ** 2)

def snake_second_moment(alpha):
    num = 3 + exp(-8 * alpha ** 2) - 4 * exp(-2 * alpha ** 2)
    return 1 + num / (8 * alpha ** 2)

alpha_max_var = 0.5604532115
max_std = sqrt(snake_variance(alpha_max_var))  # 1.0971017221...

alpha_max_second_moment = 0.65797
max_second_moment_sqrt = sqrt(snake_second_moment(alpha_max_second_moment))  # 1.1787158655

def snake_correction(alpha, kind=None):
    if kind == 'std':
        return sqrt(snake_variance(alpha))
    elif kind == 'max':
        return max_std
    else:
        return kind

def snake_gain(x):
    if x == 'approx':
        return 1
    elif x == 'max':
        return 1 / max_second_moment_sqrt
    else:
        return 1 / sqrt(snake_second_moment(x))

# initialization functions for network parameters preceding a Snake non-linearity
# pass alpha as 'kind' to use the exact second moment
# optionally pass the correction
def snake_kaiming_uniform_(tensor, kind='approx', correction=None, mode='fan_in'):
    fan = init._calculate_correct_fan(tensor, mode)
    correction = snake_correction(kind, correction)
    gain = snake_gain(kind)
    gain = correction ** 2 * gain if correction is not None else gain
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)

def snake_kaiming_normal_(tensor, kind='approx', correction=None, mode='fan_in'):
    fan = init._calculate_correct_fan(tensor, mode)
    correction = snake_correction(kind, correction)
    gain = snake_gain(kind)
    gain = correction ** 2 * gain if correction is not None else gain
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0, std)

try:
    import triton
    import triton.language as tl

    @triton.autotune(
            configs=[
                triton.Config({}, num_warps=4),
                triton.Config({}, num_warps=8),
                triton.Config({}, num_warps=16),
            ],
            key=['N'],
    )
    @triton.jit
    def _snake_fwd_triton(X, OUT, ALPHA, CR,
                          X_stride1, X_stride2, X_stride3,
                          OUT_stride1, OUT_stride2, OUT_stride3,
                          A_stride, C_stride, C, N,
                          CORR: tl.constexpr,
                          BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        batch_idx = pid // C
        channel_idx = pid % C
        block_start = tl.program_id(1) * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)

        X = X + batch_idx * X_stride1 + channel_idx * X_stride2
        x = tl.load(X + offsets * X_stride3, mask=offsets < N)
        alpha = tl.load(ALPHA + channel_idx * A_stride)
        sinax = tl.sin(alpha * x)
        out = x + sinax * sinax / alpha

        if CORR:
            cr = tl.load(CR + channel_idx * C_stride)
            out = out / cr

        OUT = OUT + batch_idx * OUT_stride1 + channel_idx * OUT_stride2
        tl.store(OUT + offsets * OUT_stride3, out, mask=offsets < N)

    def snake_fwd(x, alpha, cr=None, out=None):
        if out is None:
            out = torch.empty_like(x)
        B, C, N = x.shape
        cr_ = default(cr, x)
        BLOCK_SIZE = min(triton.next_power_of_2(N), 2 ** 14)
        grid = lambda meta: (B * C, triton.cdiv(N, meta['BLOCK_SIZE']))
        _snake_fwd_triton[grid](x, out, alpha, cr_,
                                x.stride(0), x.stride(1), x.stride(2),
                                out.stride(0), out.stride(1), out.stride(2),
                                alpha.stride(0), cr_.stride(0),
                                C, N, exists(cr), BLOCK_SIZE)
        return out

    @triton.autotune(
            configs=[
                triton.Config({}, num_warps=4),
                triton.Config({}, num_warps=8),
                triton.Config({}, num_warps=16),
            ],
            reset_to_zero=['DYDA', 'DYDC'],
            key=['N'],
    )
    @triton.jit
    def _snake_bwd_triton(X, OUT, ALPHA, CR, GRAD,
                          DYDX, DYDA, DYDC,
                          X_stride1, X_stride2, X_stride3,
                          OUT_stride1, OUT_stride2, OUT_stride3,
                          GRAD_stride1, GRAD_stride2, GRAD_stride3,
                          DYDX_stride1, DYDX_stride2, DYDX_stride3,
                          DYDA_stride, DYDC_stride,
                          ALPHA_stride, CR_stride, C, N,
                          CORR: tl.constexpr,
                          X_NEEDS_GRAD: tl.constexpr,
                          ALPHA_NEEDS_GRAD: tl.constexpr,
                          CR_NEEDS_GRAD: tl.constexpr,
                          BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        batch_idx = pid // C
        channel_idx = pid % C
        block_start = tl.program_id(1) * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)

        GRAD = GRAD + batch_idx * GRAD_stride1 + channel_idx * GRAD_stride2
        grad = tl.load(GRAD + offsets * GRAD_stride3, mask=offsets < N, other=0)

        if CORR:
            cr = tl.load(CR + channel_idx * CR_stride)
        if ALPHA_NEEDS_GRAD | CR_NEEDS_GRAD:
            OUT = OUT + batch_idx * OUT_stride1 + channel_idx * OUT_stride2
            out = tl.load(OUT + offsets * OUT_stride3, mask=offsets < N, other=0)
            outgrad = tl.sum(out * grad, axis=0)
        if X_NEEDS_GRAD | ALPHA_NEEDS_GRAD:
            X = X + batch_idx * X_stride1 + channel_idx * X_stride2
            x = tl.load(X + offsets * X_stride3, mask=offsets < N, other=0)
            alpha = tl.load(ALPHA + channel_idx * ALPHA_stride)
            sin2ax = tl.sin(2 * alpha * x)
            dydx = (sin2ax + 1) * grad
            if CORR:
                dydx = dydx / cr

        if X_NEEDS_GRAD:
            DYDX = DYDX + batch_idx * DYDX_stride1 + channel_idx * DYDX_stride2
            tl.store(DYDX + offsets * DYDX_stride3, dydx, mask=offsets < N)
        if ALPHA_NEEDS_GRAD:
            dyda = (tl.sum(x * dydx, axis=0) - outgrad) / alpha
            tl.atomic_add(DYDA + channel_idx * DYDA_stride, dyda)
        if CR_NEEDS_GRAD:
            dydc = -outgrad / cr
            tl.atomic_add(DYDC + channel_idx * DYDC_stride, dydc)

    def snake_bwd(x, alpha, cr, out, grad,
                  x_needs_grad, alpha_needs_grad, cr_needs_grad):
        B, C, N = x.shape
        dydx = torch.empty_like(x, dtype=grad.dtype) if x_needs_grad else None
        dyda = torch.zeros_like(alpha, dtype=alpha.dtype) if alpha_needs_grad else None
        dydc = torch.zeros_like(cr, dtype=cr.dtype) if cr_needs_grad else None
        dyda_ = default(dyda, dydc)
        dydc_ = default(dydc, dyda)
        if not exists(dyda_) and not exists(dydc_):
            dyda_ = dydc_ = x.new_empty((1,))
        cr_ = default(cr, x)
        BLOCK_SIZE = min(triton.next_power_of_2(N), 2 ** 14)
        grid = lambda meta: (B * C, triton.cdiv(N, meta['BLOCK_SIZE']))
        _snake_bwd_triton[grid](x, out, alpha, cr_, grad, dydx, dyda_, dydc_,
                                x.stride(0), x.stride(1), x.stride(2),
                                out.stride(0), out.stride(1), out.stride(2),
                                grad.stride(0), grad.stride(1), grad.stride(2),
                                dydx.stride(0), dydx.stride(1), dydx.stride(2),
                                dyda_.stride(0), dydc_.stride(0),
                                alpha.stride(0), cr_.stride(0), C, N, exists(cr),
                                x_needs_grad, alpha_needs_grad, cr_needs_grad,
                                BLOCK_SIZE)
        return dydx, dyda, dydc

except ImportError:
    # fall back to torchscript
    # have to break things up like this for torchscript to fuse properly
    @torch.jit.script
    def snake_fwd_jit(x, alpha):
        return x + torch.sin(alpha[..., None] * x) ** 2 * torch.reciprocal(alpha[..., None])

    @torch.jit.script
    def snake_fwd_c_jit(x, alpha, correction):
        return snake_fwd_jit(x, alpha) * torch.reciprocal(correction[..., None])

    @torch.jit.script
    def snake_dydx_bwd_jit(x, alpha, grad_output):
        return (torch.sin(2 * alpha[..., None] * x) + 1) * grad_output

    @torch.jit.script
    def snake_dydx_bwd_c_jit(x, alpha, correction, grad_output):
        return torch.reciprocal(correction[..., None]) * snake_dydx_bwd_jit(x, alpha, grad_output)

    @torch.jit.script
    def snake_dyda_bwd_jit(x, dydx, alpha, out, grad_output):
        return torch.reciprocal(alpha) * torch.sum(x * dydx - out * grad_output, dim=(0, 2))

    @torch.jit.script
    def snake_dydc_bwd_jit(out, correction, grad_output):
        return -torch.reciprocal(correction) * torch.sum(out * grad_output, dim=(0, 2))

    # disable autocast to avoid type promotion
    # to float32 when x is float16
    @torch.cuda.amp.autocast(enabled=False)
    def snake_fwd(x, alpha, cr=None):
        if cr is None:
            return snake_fwd_jit(x, alpha)
        else:
            return snake_fwd_c_jit(x, alpha, cr)

    def snake_bwd(x, alpha, cr, out, grad_output,
                  x_needs_grad, alpha_needs_grad, cr_needs_grad):
        dyda, dydc = None, None
        if x_needs_grad or alpha_needs_grad:
            if cr is None:
                dydx = snake_dydx_bwd_jit(x, alpha, grad_output)
            else:
                dydx = snake_dydx_bwd_c_jit(x, alpha, cr, grad_output)
        if alpha_needs_grad:
            dyda = snake_dyda_bwd_jit(x, dydx, alpha, out, grad_output)
        if cr_needs_grad:
            dydc = snake_dydc_bwd_jit(out, cr, grad_output)
        return dydx, dyda, dydc

class SnakeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, correction=None):
        out = snake_fwd(x, alpha, correction)
        ctx.save_for_backward(x, alpha, correction, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha, cr, out = ctx.saved_tensors
        return snake_bwd(x, alpha, cr, out, grad_output,
                         *ctx.needs_input_grad)

class Snake(nn.Module):
    def __init__(self, num_channels, init=0.5, correction=None):
        super().__init__()
        if init == 'periodic':
            # "for tasks with expected periodicity, larger a, 
            # usually from 5 to 50 tend to work well"
            # => use a gamma distribution with median ~5 and a heavy right tail
            gamma = torch.distributions.Gamma(concentration=1.5, rate=0.1)
            self.alpha = nn.Parameter(gamma.sample((num_channels,)))
        elif callable(init):  # e.g. torch.randn
            self.alpha = nn.Parameter(init(num_channels) * torch.ones(num_channels))
        else:  # assume init is a constant
            self.alpha = nn.Parameter(init * torch.ones(num_channels))
        self.correction = correction

    def forward(self, x):
        correction = snake_correction(self.alpha, kind=self.correction)
        alpha = self.alpha.expand(x.size(1))
        if correction is not None:
            correction = tensor_like(correction, self.alpha)
            correction = correction.expand(x.size(1))
        return SnakeFunction.apply(x, alpha, correction)

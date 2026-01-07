import math
import torch

import torch.nn.functional as F

from torch import nn, einsum
from functools import partial
from einops import rearrange, repeat, pack, unpack

def exists(val):
    return val is not None

def default(value, d):
    return value if exists(value) else d

def empty(tensor):
    return tensor.numel() == 0

def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer(): return False, tensor
    return True, F.pad(tensor, (*((0,) * (-1 - dim) * 2), 0, (math.ceil(m) * multiple - seqlen)), value = value)

def look_around(x, backward = 1, forward = 0, pad_value = -1, dim = 2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value = pad_value)
    return torch.cat([padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)], dim = dim)

def rotate_half(x):
    x1, x2 = rearrange(x, 'b ... (r d) -> b ... r d', r = 2).unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(q, k, freqs, scale = 1):
    q_len = q.shape[-2]
    q_freqs = freqs[..., -q_len:, :]
    inv_scale = scale ** -1
    if scale.ndim == 2: scale = scale[-q_len:, :]
    q = (q * q_freqs.cos() * scale) + (rotate_half(q) * q_freqs.sin() * scale)
    k = (k * freqs.cos() * inv_scale) + (rotate_half(k) * freqs.sin() * inv_scale)

    return q, k

def orthogonal_matrix_chunk(cols, qr_uniform_q=False, device=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    q, r = torch.linalg.qr(unstructured_block.cpu(), mode="reduced")
    q, r = map(lambda t: t.to(device), (q, r))
    if qr_uniform_q:
        d = r.diag(0)
        q *= d.sign()

    return q.t()

def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling=0, qr_uniform_q=False, device=None):
    nb_full_blocks = int(nb_rows / nb_columns)
    block_list = []
    for _ in range(nb_full_blocks):
        block_list.append(orthogonal_matrix_chunk(nb_columns, qr_uniform_q=qr_uniform_q, device=device))

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0: block_list.append(orthogonal_matrix_chunk(nb_columns, qr_uniform_q=qr_uniform_q, device=device)[:remaining_rows])
    if scaling == 0: multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1: multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device=device)
    else: raise ValueError(f"{scaling} != 0, 1")

    return multiplier.diag() @ torch.cat(block_list)

def linear_attention(q, k, v):
    return einsum("...ed,...nd->...ne", k, q) if v is None else einsum("...de,...nd,...n->...ne", einsum("...nd,...ne->...de", k, v), q, 1.0 / (einsum("...nd,...d->...n", q, k.sum(dim=-2).type_as(q)) + 1e-8))

def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device=None):
    b, h, *_ = data.shape
    
    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.0
    ratio = projection_matrix.shape[0] ** -0.5
    data_dash = torch.einsum("...id,...jd->...ij", (data_normalizer * data), repeat(projection_matrix, "j d -> b h j d", b=b, h=h).type_as(data))
    diag_data = (((data**2).sum(dim=-1) / 2.0) * (data_normalizer**2)).unsqueeze(dim=-1)

    return (ratio * ((data_dash - diag_data - data_dash.max(dim=-1, keepdim=True).values).exp() + eps) if is_query else ratio * ((data_dash - diag_data + eps).exp())).type_as(data)

class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim, scale_base = None, use_xpos = False, theta = 10000):
        super().__init__()
        inv_freq = 1. / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.use_xpos = use_xpos
        self.scale_base = scale_base
        assert not (use_xpos and not exists(scale_base))
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer('scale', scale, persistent = False)

    def forward(self, x):
        seq_len, device = x.shape[-2], x.device
        t = torch.arange(seq_len, device = x.device).type_as(self.inv_freq)

        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs =  torch.cat((freqs, freqs), dim = -1)

        if not self.use_xpos: return freqs, torch.ones(1, device = device)

        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')

        return freqs, torch.cat((scale, scale), dim = -1)

class LocalAttention(nn.Module):
    def __init__(self, window_size, causal = False, look_backward = 1, look_forward = None, dropout = 0., shared_qk = False, rel_pos_emb_config = None, dim = None, autopad = False, exact_windowsize = False, scale = None, use_rotary_pos_emb = True, use_xpos = False, xpos_scale_base = None):
        super().__init__()
        look_forward = default(look_forward, 0 if causal else 1)
        assert not (causal and look_forward > 0)
        self.scale = scale
        self.window_size = window_size
        self.autopad = autopad
        self.exact_windowsize = exact_windowsize
        self.causal = causal
        self.look_backward = look_backward
        self.look_forward = look_forward
        self.dropout = nn.Dropout(dropout)
        self.shared_qk = shared_qk
        self.rel_pos = None
        self.use_xpos = use_xpos
        if use_rotary_pos_emb and (exists(rel_pos_emb_config) or exists(dim)): 
            if exists(rel_pos_emb_config): dim = rel_pos_emb_config[0]
            self.rel_pos = SinusoidalEmbeddings(dim, use_xpos = use_xpos, scale_base = default(xpos_scale_base, window_size // 2))

    def forward(self, q, k, v, mask = None, input_mask = None, attn_bias = None, window_size = None):
        mask = default(mask, input_mask)
        assert not (exists(window_size) and not self.use_xpos)

        _, autopad, pad_value, window_size, causal, look_backward, look_forward, shared_qk = q.shape, self.autopad, -1, default(window_size, self.window_size), self.causal, self.look_backward, self.look_forward, self.shared_qk
        (q, packed_shape), (k, _), (v, _) = map(lambda t: pack([t], '* n d'), (q, k, v))

        if autopad:
            orig_seq_len = q.shape[1]
            (_, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, self.window_size, dim = -2), (q, k, v))

        b, n, dim_head, device, dtype = *q.shape, q.device, q.dtype
        scale = default(self.scale, dim_head ** -0.5)

        assert (n % window_size) == 0
        windows = n // window_size

        if shared_qk: k = F.normalize(k, dim = -1).type(k.dtype)

        seq = torch.arange(n, device = device)
        b_t = rearrange(seq, '(w n) -> 1 w n', w = windows, n = window_size)
        bq, bk, bv = map(lambda t: rearrange(t, 'b (w n) d -> b w n d', w = windows), (q, k, v))

        bq = bq * scale
        look_around_kwargs = dict(backward =  look_backward, forward =  look_forward, pad_value = pad_value)

        bk = look_around(bk, **look_around_kwargs)
        bv = look_around(bv, **look_around_kwargs)

        if exists(self.rel_pos):
            pos_emb, xpos_scale = self.rel_pos(bk)
            bq, bk = apply_rotary_pos_emb(bq, bk, pos_emb, scale = xpos_scale)

        bq_t = b_t
        bq_k = look_around(b_t, **look_around_kwargs)
        bq_t = rearrange(bq_t, '... i -> ... i 1')
        bq_k = rearrange(bq_k, '... j -> ... 1 j')

        pad_mask = bq_k == pad_value
        sim = einsum('b h i e, b h j e -> b h i j', bq, bk)

        if exists(attn_bias):
            heads = attn_bias.shape[0]
            assert (b % heads) == 0

            attn_bias = repeat(attn_bias, 'h i j -> (b h) 1 i j', b = b // heads)
            sim = sim + attn_bias

        mask_value = -torch.finfo(sim.dtype).max
        if shared_qk:
            self_mask = bq_t == bq_k
            sim = sim.masked_fill(self_mask, -5e4)
            del self_mask

        if causal:
            causal_mask = bq_t < bq_k
            if self.exact_windowsize: causal_mask = causal_mask | (bq_t > (bq_k + (self.window_size * self.look_backward)))
            sim = sim.masked_fill(causal_mask, mask_value)
            del causal_mask

        sim = sim.masked_fill(((bq_k - (self.window_size * self.look_forward)) > bq_t) | (bq_t > (bq_k + (self.window_size * self.look_backward))) | pad_mask, mask_value) if not causal and self.exact_windowsize else sim.masked_fill(pad_mask, mask_value)

        if exists(mask):
            batch = mask.shape[0]
            assert (b % batch) == 0

            h = b // mask.shape[0]
            if autopad: _, mask = pad_to_multiple(mask, window_size, dim = -1, value = False)

            mask = repeat(rearrange(look_around(rearrange(mask, '... (w n) -> (...) w n', w = windows, n = window_size), **{**look_around_kwargs, 'pad_value': False}), '... j -> ... 1 j'), 'b ... -> (b h) ...', h = h)
            sim = sim.masked_fill(~mask, mask_value)

            del mask

        out = rearrange(einsum('b h i j, b h j e -> b h i e', self.dropout(sim.softmax(dim = -1)), bv), 'b w n d -> b (w n) d')
        if autopad: out = out[:, :orig_seq_len, :]

        out, *_ = unpack(out, packed_shape, '* n d')
        return out
    
class FastAttention(nn.Module):
    def __init__(self, dim_heads, nb_features=None, ortho_scaling=0, causal=False, generalized_attention=False, kernel_fn=nn.ReLU(), qr_uniform_q=False, no_projection=False):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))
        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling
        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows=self.nb_features, nb_columns=dim_heads, scaling=ortho_scaling, qr_uniform_q=qr_uniform_q)
        projection_matrix = self.create_projection()
        self.register_buffer("projection_matrix", projection_matrix)
        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn
        self.no_projection = no_projection
        self.causal = causal

    @torch.no_grad()
    def redraw_projection_matrix(self):
        projections = self.create_projection()
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v):
        if self.no_projection: q, k = q.softmax(dim=-1), (k.exp() if self.causal else k.softmax(dim=-2)) 
        else:
            create_kernel = partial(softmax_kernel, projection_matrix=self.projection_matrix, device=q.device)
            q, k = create_kernel(q, is_query=True), create_kernel(k, is_query=False)

        attn_fn = linear_attention if not self.causal else self.causal_linear_fn
        return attn_fn(q, k, None) if v is None else attn_fn(q, k, v)

class SelfAttention(nn.Module):
    def __init__(self, dim, causal=False, heads=8, dim_head=64, local_heads=0, local_window_size=256, nb_features=None, feature_redraw_interval=1000, generalized_attention=False, kernel_fn=nn.ReLU(), qr_uniform_q=False, dropout=0.0, no_projection=False):
        super().__init__()
        assert dim % heads == 0
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        self.fast_attention = FastAttention(dim_head, nb_features, causal=causal, generalized_attention=generalized_attention, kernel_fn=kernel_fn, qr_uniform_q=qr_uniform_q, no_projection=no_projection)
        self.heads = heads
        self.global_heads = heads - local_heads
        self.local_attn = (LocalAttention(window_size=local_window_size, causal=causal, autopad=True, dropout=dropout, look_forward=int(not causal), rel_pos_emb_config=(dim_head, local_heads)) if local_heads > 0 else None)
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    @torch.no_grad()
    def redraw_projection_matrix(self):
        self.fast_attention.redraw_projection_matrix()

    def forward(self, x, context=None, mask=None, context_mask=None, name=None, inference=False, **kwargs):
        _, _, _, h, gh = *x.shape, self.heads, self.global_heads
        cross_attend = exists(context)
        context = default(context, x)
        context_mask = default(context_mask, mask) if not cross_attend else context_mask

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (self.to_q(x), self.to_k(context), self.to_v(context)))
        (q, lq), (k, lk), (v, lv) = map(lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))

        attn_outs = []

        if not empty(q):
            if exists(context_mask): v.masked_fill_(~context_mask[:, None, :, None], 0.0)
            if cross_attend: pass  
            else: out = self.fast_attention(q, k, v)

            attn_outs.append(out)

        if not empty(lq):
            assert (not cross_attend), "not cross_attend"

            out = self.local_attn(lq, lk, lv, input_mask=mask)
            attn_outs.append(out)

        return self.dropout(self.to_out(rearrange(torch.cat(attn_outs, dim=1), "b h n d -> b n (h d)")))
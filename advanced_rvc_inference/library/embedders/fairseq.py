import re
import sys
import math
import uuid
import torch
import types
import contextlib

import numpy as np
import torch.nn.functional as F

from torch import nn
from omegaconf import DictConfig, open_dict

class Dictionary:
    def __init__(self, *args, **kwargs):
        pass

fairseq = types.ModuleType("fairseq")
fairseq_data = types.ModuleType("fairseq.data")
fairseq_data_dictionary = types.ModuleType("fairseq.data.dictionary")
fairseq_data_dictionary.Dictionary = Dictionary
fairseq.data = fairseq_data
fairseq_data.dictionary = fairseq_data_dictionary
sys.modules["fairseq"] = fairseq
sys.modules["fairseq.data"] = fairseq_data
sys.modules["fairseq.data.dictionary"] = fairseq_data_dictionary

def load_model(filename):
    state = torch.load(filename, map_location="cpu", weights_only=False)

    model = HubertModel(HubertConfig(**state['cfg']['model']), num_classes=int(state['model']['label_embs_concat'].shape[0]))
    model.load_state_dict(state['model'], strict=False)

    return model

def softmax(x, dim, onnx_trace = False):
    return F.softmax(x.float(), dim=dim) if onnx_trace else F.softmax(x, dim=dim, dtype=torch.float32)

def log_softmax(x, dim, onnx_trace = False):
    return F.log_softmax(x.float(), dim=dim) if onnx_trace else F.log_softmax(x, dim=dim, dtype=torch.float32)

def eval_str_dict(x, type=dict):
    if x is None: return None
    if isinstance(x, str): x = eval(x)
    return x

def with_incremental_state(cls):
    cls.__bases__ = (FairseqIncrementalState,) + tuple(b for b in cls.__bases__ if b != FairseqIncrementalState)
    return cls

def quant_noise(module, p, block_size):
    if p <= 0: return module
    assert isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d))
    is_conv = module.weight.ndim == 4
    if not is_conv: assert (module.weight.size(1) % block_size == 0)
    else:
        if module.kernel_size == (1, 1): assert (module.in_channels % block_size == 0)
        else:
            k = module.kernel_size[0] * module.kernel_size[1]
            assert k % block_size == 0

    def _forward_pre_hook(mod, input):
        if mod.training:
            if not is_conv:
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)
                mask = torch.zeros(in_features // block_size * out_features, device=weight.device)
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)
            else:
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels

                if mod.kernel_size == (1, 1):
                    mask = torch.zeros(int(in_channels // block_size * out_channels), device=weight.device)
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                else:
                    mask = torch.zeros(weight.size(0), weight.size(1), device=weight.device)
                    mask.bernoulli_(p)
                    mask = (mask.unsqueeze(2).unsqueeze(3).repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1]))

            mask = mask.to(torch.bool)
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)

    module.register_forward_pre_hook(_forward_pre_hook)
    return module

class FairseqDropout(nn.Module):
    def __init__(self, p, module_name=None):
        super().__init__()
        self.p = p
        self.module_name = module_name
        self.apply_during_inference = False

    def forward(self, x, inplace = False):
        return F.dropout(x, p=self.p, training=True, inplace=inplace) if self.p > 0 and (self.training or self.apply_during_inference) else x

    def make_generation_fast_(self, name, retain_dropout = False, retain_dropout_modules = None, **kwargs):
        if retain_dropout:
            if (retain_dropout_modules is None or self.module_name in retain_dropout_modules): self.apply_during_inference = True

class FairseqIncrementalState(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_incremental_state()

    def init_incremental_state(self):
        self._incremental_state_id = str(uuid.uuid4())

    def _get_full_incremental_state_key(self, key):
        return "{}.{}".format(self._incremental_state_id, key)

    def get_incremental_state(self, incremental_state, key):
        full_key = self._get_full_incremental_state_key(key)
        if incremental_state is None or full_key not in incremental_state: return None
        return incremental_state[full_key]

    def set_incremental_state(self, incremental_state, key, value):
        if incremental_state is not None: incremental_state[self._get_full_incremental_state_key(key)] = value
        return incremental_state

class FairseqDecoder(nn.Module):
    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary
        self.onnx_trace = False
        self.adaptive_softmax = None

    def forward(self, prev_output_tokens, encoder_out=None, **kwargs):
        x, extra = self.extract_features(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return self.output_layer(x), extra

    def extract_features(self, prev_output_tokens, encoder_out=None, **kwargs):
        pass

    def output_layer(self, features, **kwargs):
        pass

    def get_normalized_probs(self, net_output, log_probs, sample = None):
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)

    def get_normalized_probs_scriptable(self, net_output, log_probs, sample = None):
        if hasattr(self, "adaptive_softmax") and self.adaptive_softmax is not None:
            if sample is not None:
                assert "target" in sample
                target = sample["target"]
            else: target = None
            out = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
            return out.exp_() if not log_probs else out

        logits = net_output[0]
        return log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace) if log_probs else softmax(logits, dim=-1, onnx_trace=self.onnx_trace)

    def max_positions(self):
        return 1e6

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

@with_incremental_state
class FairseqIncrementalDecoder(FairseqDecoder):
    def __init__(self, dictionary):
        super().__init__(dictionary)

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):
        pass

    def extract_features(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):
        pass

    def reorder_incremental_state(self, incremental_state, new_order):
        pass

    def reorder_incremental_state_scripting(self, incremental_state, new_order):
        for module in self.modules():
            if hasattr(module, "reorder_incremental_state"):
                result = module.reorder_incremental_state(incremental_state, new_order)
                if result is not None: incremental_state = result

    def set_beam_size(self, beam_size):
        if getattr(self, "_beam_size", -1) != beam_size:
            seen = set()

            def apply_set_beam_size(module):
                if (module != self and hasattr(module, "set_beam_size") and module not in seen):
                    seen.add(module)
                    module.set_beam_size(beam_size)

            self.apply(apply_set_beam_size)
            self._beam_size = beam_size

class MultiheadAttention(FairseqIncrementalDecoder):
    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, self_attention=False, encoder_decoder_attention=False, dictionary=None, q_noise=0.0, qn_block_size=8, xformers_att_config=None, xformers_blocksparse_layout=None, xformers_blocksparse_blocksize=16):
        super().__init__(dictionary)
        xformers_att_config = eval_str_dict(xformers_att_config)
        self.use_xformers = xformers_att_config is not None
        if self.use_xformers: raise ImportError
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.head_dim = embed_dim // num_heads
        assert (self.head_dim * num_heads == self.embed_dim)
        self.scaling = self.head_dim**-0.5
        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert not self.self_attention or self.qkv_same_dim
        self.k_proj = quant_noise(nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.v_proj = quant_noise(nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.q_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.out_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
        if add_bias_kv: self.bias_k, self.bias_v = nn.Parameter(torch.Tensor(1, 1, embed_dim)), nn.Parameter(torch.Tensor(1, 1, embed_dim))
        else: self.bias_k = self.bias_v = None
        self.add_zero_attn = add_zero_attn
        self.beam_size = 1
        self.reset_parameters()
        self.onnx_trace = False
        self.skip_embed_dim_check = False
        self.init_incremental_state()

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None: nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None: nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None: nn.init.xavier_normal_(self.bias_v)

    def _get_reserve_head_index(self, num_heads_to_keep: int):
        k_proj_heads_norm, q_proj_heads_norm, v_proj_heads_norm = [], [], []
        for i in range(self.num_heads):
            start_idx = i * self.head_dim
            end_idx = (i + 1) * self.head_dim
            k_proj_heads_norm.append((self.k_proj.weight[start_idx:end_idx]).abs().sum().tolist() + (self.k_proj.bias[start_idx:end_idx]).abs().sum().tolist())
            q_proj_heads_norm.append((self.q_proj.weight[start_idx:end_idx]).abs().sum().tolist() + (self.q_proj.bias[start_idx:end_idx]).abs().sum().tolist())
            v_proj_heads_norm.append((self.v_proj.weight[start_idx:end_idx]).abs().sum().tolist() + (self.v_proj.bias[start_idx:end_idx]).abs().sum().tolist())

        heads_norm = []
        for i in range(self.num_heads):
            heads_norm.append(k_proj_heads_norm[i] + q_proj_heads_norm[i] + v_proj_heads_norm[i])

        sorted_head_index = sorted(range(self.num_heads), key=lambda k: heads_norm[k], reverse=True)
        reserve_head_index = []
        for i in range(num_heads_to_keep):
            reserve_head_index.append((sorted_head_index[i] * self.head_dim, (sorted_head_index[i] + 1) * self.head_dim))
        return reserve_head_index

    def _adaptive_prune_heads(self, reserve_head_index):
        new_q_weight, new_q_bias, new_k_weight, new_k_bias, new_v_weight, new_v_bias, new_out_proj_weight = [], [], [], [], [], [], []
        for ele in reserve_head_index:
            start_idx, end_idx = ele
            new_q_weight.append(self.q_proj.weight[start_idx:end_idx])
            new_q_bias.append(self.q_proj.bias[start_idx:end_idx])
            new_k_weight.append(self.k_proj.weight[start_idx:end_idx])
            new_k_bias.append(self.k_proj.bias[start_idx:end_idx])
            new_v_weight.append(self.v_proj.weight[start_idx:end_idx])
            new_v_bias.append(self.v_proj.bias[start_idx:end_idx])
            new_out_proj_weight.append(self.out_proj.weight[:, start_idx:end_idx])
        new_q_weight = torch.cat(new_q_weight).detach()
        new_k_weight = torch.cat(new_k_weight).detach()
        new_v_weight = torch.cat(new_v_weight).detach()
        new_out_proj_weight = torch.cat(new_out_proj_weight, dim=-1).detach()
        new_q_weight.requires_grad = True
        new_k_weight.requires_grad = True
        new_v_weight.requires_grad = True
        new_out_proj_weight.requires_grad = True
        new_q_bias = torch.cat(new_q_bias).detach()
        new_q_bias.requires_grad = True
        new_k_bias = torch.cat(new_k_bias).detach()
        new_k_bias.requires_grad = True
        new_v_bias = torch.cat(new_v_bias).detach()
        new_v_bias.requires_grad = True
        self.q_proj.weight = nn.Parameter(new_q_weight)
        self.q_proj.bias = nn.Parameter(new_q_bias)
        self.k_proj.weight = nn.Parameter(new_k_weight)
        self.k_proj.bias = nn.Parameter(new_k_bias)
        self.v_proj.weight = nn.Parameter(new_v_weight)
        self.v_proj.bias = nn.Parameter(new_v_bias)
        self.out_proj.weight = nn.Parameter(new_out_proj_weight)
        self.num_heads = len(reserve_head_index)
        self.embed_dim = self.head_dim * self.num_heads
        self.q_proj.out_features = self.embed_dim
        self.k_proj.out_features = self.embed_dim
        self.v_proj.out_features = self.embed_dim

    def _set_skip_embed_dim_check(self):
        self.skip_embed_dim_check = True

    def _pad_masks(self, key_padding_mask, attn_mask):
        if attn_mask is not None:
            shape = attn_mask.size()[:-1] + torch.Size([1])
            attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(shape)], dim=-1)

        if key_padding_mask is not None:
            shape = key_padding_mask.size()[:-1] + torch.Size([1])
            key_padding_mask = torch.cat([key_padding_mask, key_padding_mask.new_zeros(shape)], dim=-1)

        return key_padding_mask, attn_mask

    def _add_bias(self, k, v, key_padding_mask, attn_mask, bsz):
        assert self.bias_k is not None or self.bias_v is not None
        key_padding_mask, attn_mask = self._pad_masks(key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return torch.cat([k, self.bias_k.repeat(1, bsz, 1)]), torch.cat([v, self.bias_v.repeat(1, bsz, 1)]), key_padding_mask, attn_mask

    def _append_zero_attn(self, k, v, key_padding_mask, attn_mask):
        zero_attn_shape = k.size()[:-2] + torch.Size([1]) + k.size()[-1:]
        key_padding_mask, attn_mask = self._pad_masks(key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=-2), torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=-2), key_padding_mask, attn_mask

    def forward(self, query, key, value, key_padding_mask = None, incremental_state = None, need_weights = True, static_kv = False, attn_mask = None, before_softmax = False, need_head_weights = False):
        if need_head_weights: need_weights = True
        is_tpu = query.device.type == "xla"
        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        if not self.skip_embed_dim_check: assert (embed_dim == self.embed_dim)
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert value is not None
                assert src_len, key_bsz == value.shape[:2]

        if (not self.onnx_trace and not is_tpu and incremental_state is None and not static_kv and not torch.jit.is_scripting() and not self.skip_embed_dim_check):
            assert key is not None and value is not None
            return F.multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads, torch.empty([0]), torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)), self.bias_k, self.bias_v, self.add_zero_attn, self.dropout_module.p, self.out_proj.weight, self.out_proj.bias, self.training or self.dropout_module.apply_during_inference, key_padding_mask.bool() if key_padding_mask is not None else None, need_weights, attn_mask, use_separate_proj_weight=True, q_proj_weight=self.q_proj.weight, k_proj_weight=self.k_proj.weight, v_proj_weight=self.v_proj.weight)

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else: saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                if self.beam_size > 1 and bsz == key.size(1):
                    key = key.view(key.size(0), -1, self.beam_size, key.size(2))[:, :, 0, :]
                    if key_padding_mask is not None: key_padding_mask = key_padding_mask.view(-1, self.beam_size, key_padding_mask.size(1))[:, 0, :]
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

        q *= self.scaling
        if self.bias_k is not None:
            assert self.bias_v is not None
            k, v, attn_mask, key_padding_mask = self._add_bias(k, v, attn_mask, key_padding_mask, bsz)

        q = (q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1))
        kv_bsz = bsz 
        if k is not None:
            kv_bsz = k.size(1)
            k = (k.contiguous().view(-1, kv_bsz * self.num_heads, self.head_dim).transpose(0, 1))

        if v is not None: v = (v.contiguous().view(-1, kv_bsz * self.num_heads, self.head_dim).transpose(0, 1))
        if saved_state is not None:
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None

                kv_bsz = _prev_key.size(0)
                prev_key = _prev_key.view(kv_bsz * self.num_heads, -1, self.head_dim)

                if static_kv: k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
                src_len = k.size(1)

            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None or kv_bsz == _prev_value.size(0)
                prev_value = _prev_value.view(kv_bsz * self.num_heads, -1, self.head_dim)
                if static_kv: v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)

            prev_key_padding_mask = None
            if "prev_key_padding_mask" in saved_state: prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(key_padding_mask=key_padding_mask, prev_key_padding_mask=prev_key_padding_mask, batch_size=kv_bsz, src_len=k.size(1), static_kv=static_kv)
            saved_state["prev_key"] = k.view(kv_bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(kv_bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)

        assert k is not None
        assert k.size(1) == src_len

        if key_padding_mask is not None and key_padding_mask.dim() == 0: key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == kv_bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k, v, key_padding_mask, attn_mask = self._append_zero_attn(k=k, v=v, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        if self.encoder_decoder_attention and bsz != kv_bsz:
            attn_weights = torch.einsum("bxhtd,bhsd->bxhts", q.view((kv_bsz, -1, self.num_heads) + q.size()[1:]), k.view((kv_bsz, self.num_heads) + k.size()[1:]))
            attn_weights = attn_weights.reshape((-1,) + attn_weights.size()[-2:])
        else: attn_weights = q.bmm(k.transpose(1, 2))

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace: attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(kv_bsz, -1, self.num_heads, tgt_len, src_len).masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(torch.bool), float("-inf")) if not is_tpu else attn_weights.transpose(0, 2).masked_fill(key_padding_mask, float("-inf")).transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax: return attn_weights, v
        attn_weights_float = softmax(attn_weights, dim=-1, onnx_trace=self.onnx_trace)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)
        assert v is not None
        attn = None

        if self.encoder_decoder_attention and bsz != kv_bsz:
            attn = torch.einsum("bxhts,bhsd->bxhtd", attn_probs.view((kv_bsz, -1, self.num_heads) + attn_probs.size()[1:]), v.view((kv_bsz, self.num_heads) + v.size()[1:]))
            attn = attn.reshape((-1,) + attn.size()[-2:])
        else: attn = attn_probs.bmm(v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.contiguous().view(tgt_len, bsz, self.embed_dim) if self.onnx_trace and attn.size(1) == 1 else attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        attn = self.out_proj(attn)
        attn_weights = None

        if need_weights:
            attn_weights = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
            if not need_head_weights: attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights

    @staticmethod
    def _append_prev_key_padding_mask(key_padding_mask, prev_key_padding_mask, batch_size, src_len, static_kv):
        if prev_key_padding_mask is not None and static_kv: new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None: new_key_padding_mask = torch.cat([prev_key_padding_mask.float(), key_padding_mask.float()], dim=1)
        elif prev_key_padding_mask is not None:
            if src_len > prev_key_padding_mask.size(1):
                filler = torch.zeros((batch_size, src_len - prev_key_padding_mask.size(1)), device=prev_key_padding_mask.device)
                new_key_padding_mask = torch.cat([prev_key_padding_mask.float(), filler.float()], dim=1)
            else: new_key_padding_mask = prev_key_padding_mask.float()
        elif key_padding_mask is not None:
            if src_len > key_padding_mask.size(1):
                filler = torch.zeros((batch_size, src_len - key_padding_mask.size(1)), device=key_padding_mask.device)
                new_key_padding_mask = torch.cat([filler.float(), key_padding_mask.float()], dim=1)
            else: new_key_padding_mask = key_padding_mask.float()
        else: new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(self, incremental_state, new_order):
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention:
                        if input_buffer_k.size(0) * self.beam_size == new_order.size(0): return incremental_state
                        elif self.beam_size > 1: input_buffer[k] = input_buffer_k.index_select(0, new_order.reshape(-1, self.beam_size)[:, 0] // self.beam_size)
                        else: input_buffer[k] = input_buffer_k.index_select(0, new_order)
                    else: input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def set_beam_size(self, beam_size):
        self.beam_size = beam_size

    def _get_input_buffer(self, incremental_state):
        result = self.get_incremental_state(incremental_state, "attn_state")
        return result if result is not None else {}

    def _set_input_buffer(self, incremental_state, buffer):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add, keys_to_remove = {}, []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]
                keys_to_remove.append(k)
                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][dim : 2 * dim]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]
                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value

def init_bert_params(module):
    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None: module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None: module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)

def make_conv_pos(e, k, g):
    pos_conv = nn.Conv1d(e, e, kernel_size=k, padding=k // 2, groups=g)
    dropout = 0
    nn.init.normal_(pos_conv.weight, mean=0, std=math.sqrt((4 * (1.0 - dropout)) / (k * e)))
    nn.init.constant_(pos_conv.bias, 0)
    return nn.Sequential(nn.utils.parametrizations.weight_norm(pos_conv, name="weight", dim=2), SamePad(k), nn.GELU())

def is_xla_tensor(tensor):
    return torch.is_tensor(tensor) and tensor.device.type == "xla"

def index_put(tensor, indices, value):
    if is_xla_tensor(tensor):
        for _ in range(indices.dim(), tensor.dim()): 
            indices = indices.unsqueeze(-1)

        if indices.size(-1) < tensor.size(-1): indices = indices.expand_as(tensor)
        tensor = tensor.mul(~indices).add(value.mul(indices))
    else: tensor[indices] = value

    return tensor

def pad_to_multiple(x, multiple, dim=-1, value=0):
    if x is None: return None, 0
    tsz = x.size(dim)
    m = tsz / multiple
    remainder = math.ceil(m) * multiple - tsz
    if m.is_integer(): return x, 0
    return F.pad(x, (*((0,) * (-1 - dim) * 2), 0, remainder), value=value), remainder

def compute_mask_indices(shape, padding_mask, mask_prob, mask_length, mask_type = "static", mask_other = 0.0, min_masks = 0, no_overlap = False, min_space = 0, require_same_masks = True, mask_dropout = 0.0, add_masks = False, seed = None, epoch = None, indices = None, idc_select_ver = 1, num_mask_ver = 2):
    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)
    if num_mask_ver == 1: all_num_mask = max(min_masks, int(mask_prob * all_sz / float(mask_length) + np.random.rand()))
    mask_idcs = []

    for i in range(bsz):
        seed_i = int(hash((seed, epoch, indices[i].item())) % 1e6) if seed is not None and epoch is not None and indices is not None else None
        rng = np.random.default_rng(seed_i)

        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            assert sz >= 0, sz
        else: sz = all_sz

        if num_mask_ver == 1: num_mask = max(min_masks, int(mask_prob * sz / float(mask_length) + np.random.rand())) if padding_mask is not None else all_num_mask
        elif num_mask_ver == 2: num_mask = max(min_masks, int(mask_prob * sz / float(mask_length) + rng.random()))
        else: raise ValueError

        if mask_type == "static": lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform": lengths = rng.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal": lengths = [max(1, int(round(x))) for x in rng.normal(mask_length, mask_other, size=num_mask)]
        elif mask_type == "poisson": lengths = [int(round(x)) for x in rng.poisson(mask_length, size=num_mask)]
        else: raise Exception

        if sum(lengths) == 0:
            if mask_type == "static": raise ValueError
            else: lengths = [min(mask_length, sz - 1)]

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = rng.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))
                new_parts = []
                if span_start - s - min_space >= keep_length: new_parts.append((s, span_start - min_space + 1))
                if e - span_start - length - min_space > keep_length: new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter((e - s if e - s >= length + min_space else 0 for s, e in parts), np.int32)
                l_sum = np.sum(lens)
                if l_sum == 0: break
                s, e = parts.pop(rng.choice(len(parts), p=lens / np.sum(lens)))
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            if idc_select_ver == 1:
                min_len = min(lengths)
                if sz - min_len <= num_mask: min_len = sz - num_mask - 1
                mask_idc = rng.choice(sz - min_len, num_mask, replace=False)
            elif idc_select_ver == 2: mask_idc = rng.choice(sz, num_mask, replace=False)
            else: raise ValueError

            mask_idc = np.asarray([mask_idc[j] + offset for j in range(len(mask_idc)) for offset in range(lengths[j])])

        mask_idc = np.unique(mask_idc[mask_idc < sz])
        if len(mask_idc) >= sz: raise ValueError
        mask_idcs.append(mask_idc)

    target_len = None
    if require_same_masks: target_len = max([len(m) for m in mask_idcs]) if add_masks else min([len(m) for m in mask_idcs])

    for i, mask_idc in enumerate(mask_idcs):
        if target_len is not None and len(mask_idc) > target_len: mask_idc = rng.choice(mask_idc, target_len, replace=False)
        mask[i, mask_idc] = True

        if target_len is not None and len(mask_idc) < target_len:
            to_mask = rng.choice(np.flatnonzero(~mask[i]), target_len - len(mask_idc), replace=False)
            mask[i, to_mask] = True

        if mask_dropout > 0:
            masked = np.flatnonzero(mask[i])
            mask[i, rng.choice(masked, np.rint(len(masked) * mask_dropout).astype(int), replace=False)] = False

    return mask

def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):
    return nn.LayerNorm(normalized_shape, eps, elementwise_affine)

def prune_state_dict(state_dict, model_cfg):
    arch = None
    if model_cfg is not None: arch = (model_cfg._name if isinstance(model_cfg, DictConfig) else getattr(model_cfg, "arch", None))
    if not model_cfg or arch is None or arch == "ptt_transformer": return state_dict
    encoder_layers_to_keep = getattr(model_cfg, "encoder_layers_to_keep", None)
    decoder_layers_to_keep = getattr(model_cfg, "decoder_layers_to_keep", None)
    if not encoder_layers_to_keep and not decoder_layers_to_keep: return state_dict

    def create_pruning_pass(layers_to_keep, layer_name):
        keep_layers = sorted(int(layer_string) for layer_string in layers_to_keep.split(","))
        mapping_dict = {}
        for i in range(len(keep_layers)):
            mapping_dict[str(keep_layers[i])] = str(i)

        return {"substitution_regex": re.compile(r"^{layer}.*\.layers\.(\d+)".format(layer=layer_name)), "mapping_dict": mapping_dict}

    pruning_passes, new_state_dict = [], {}
    if encoder_layers_to_keep: pruning_passes.append(create_pruning_pass(encoder_layers_to_keep, "encoder"))
    if decoder_layers_to_keep: pruning_passes.append(create_pruning_pass(decoder_layers_to_keep, "decoder"))

    for layer_name in state_dict.keys():
        match = re.search(r"\.layers\.(\d+)\.", layer_name)
        if not match:
            new_state_dict[layer_name] = state_dict[layer_name]
            continue

        original_layer_number = match.group(1)
        for pruning_pass in pruning_passes:
            if original_layer_number in pruning_pass["mapping_dict"] and pruning_pass["substitution_regex"].search(layer_name):
                substitution_match = pruning_pass["substitution_regex"].search(layer_name)
                new_state_dict[(layer_name[: substitution_match.start(1)] + pruning_pass["mapping_dict"][original_layer_number] + layer_name[substitution_match.end(1) :])] = state_dict[layer_name]

    with open_dict(model_cfg) if isinstance(model_cfg, DictConfig) else contextlib.ExitStack():
        if hasattr(model_cfg, "encoder_layers_to_keep"): model_cfg.encoder_layers_to_keep = None
        if hasattr(model_cfg, "decoder_layers_to_keep"): model_cfg.decoder_layers_to_keep = None

    return new_state_dict

def relu_squared(x):
    return F.relu(x).pow(2)

def get_activation_fn(activation):
    def gelu(x):
        return nn.functional.gelu(x.float()).type_as(x)
    
    def gelu_accurate(x):
        if not hasattr(gelu_accurate, "_a"):
            gelu_accurate._a = math.sqrt(2 / math.pi)
            return (0.5 * x * (1 + (gelu_accurate._a * (x + 0.044715 * x.pow(3))).tanh()))

    if activation == "relu": return F.relu
    elif activation == "relu_squared": return relu_squared
    elif activation == "gelu": return gelu
    elif activation == "gelu_fast": return gelu_accurate
    elif activation == "gelu_accurate": return gelu_accurate
    elif activation == "tanh": return torch.tanh
    elif activation == "linear": return lambda x: x
    elif activation == "swish": return nn.SiLU
    else: raise RuntimeError

class SamePad(nn.Module):
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal: self.remove = kernel_size - 1
        else: self.remove = int(kernel_size % 2 == 0)

    def forward(self, x):
        if self.remove > 0: x = x[:, :, : -self.remove]
        return x

class TransformerSentenceEncoderLayer(nn.Module):
    def __init__(self, embedding_dim = 768, ffn_embedding_dim = 3072, num_attention_heads = 8, dropout = 0.1, attention_dropout = 0.1, activation_dropout = 0.1, activation_fn = "relu", layer_norm_first = False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(self.embedding_dim, num_attention_heads, dropout=attention_dropout, self_attention=True)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.layer_norm_first = layer_norm_first
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)
        self.final_layer_norm = LayerNorm(self.embedding_dim)

    def forward(self, x, self_attn_mask=None, self_attn_padding_mask=None, need_weights=False, att_args=None):
        residual = x
        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(query=x, key=x, value=x, key_padding_mask=self_attn_padding_mask, attn_mask=self_attn_mask, need_weights=False)
            x = residual + self.dropout1(x)
            residual = x
            x = self.fc2(self.dropout2(self.activation_fn(self.fc1(self.final_layer_norm(x)))))
            layer_result = x
            x = residual + self.dropout3(x)
        else:
            x, attn = self.self_attn(query=x, key=x, value=x, key_padding_mask=self_attn_padding_mask, need_weights=False)
            x = self.self_attn_layer_norm(residual + self.dropout1(x))
            residual = x
            x = self.fc2(self.dropout2(self.activation_fn(self.fc1(x))))
            layer_result = x
            x = self.final_layer_norm(residual + self.dropout3(x))

        return x, (attn, layer_result)

class AdapterFast(nn.Module):
    def __init__(self, adapter_num, input_dim, hidden_dim, act_fn):
        super().__init__()
        self.adapter_num = adapter_num
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.W_a = nn.Parameter(torch.empty(adapter_num, hidden_dim, input_dim))
        self.W_b = nn.Parameter(torch.empty(adapter_num, input_dim, hidden_dim))
        self.b_a = nn.Parameter(torch.empty(adapter_num, hidden_dim))
        self.b_b = nn.Parameter(torch.empty(adapter_num, input_dim))
        self.ln_W = nn.Parameter(torch.empty(adapter_num, input_dim))
        self.ln_b = nn.Parameter(torch.empty(adapter_num, input_dim))
        self.act_fn = nn.Identity()
        if act_fn == "relu": self.act_fn = nn.ReLU()
        elif act_fn == "gelu": self.act_fn = nn.GELU()
        elif act_fn == "selu": self.act_fn = nn.SELU()
        else: raise ValueError
        self.input_dim = input_dim
        self.reset_parameters()

    def reset_parameters(self):
        for ii in range(self.adapter_num):
            nn.init.kaiming_uniform_(self.W_a[ii], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.W_b[ii], a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W_a[ii])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.b_a[ii], -bound, bound)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W_b[ii])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.b_b[ii], -bound, bound)
        
        nn.init.ones_(self.ln_W)
        nn.init.zeros_(self.ln_b)

    def forward(self, x, adapter_id):
        ii = adapter_id
        return F.linear(self.act_fn(F.linear(F.layer_norm(x, (self.input_dim, ), self.ln_W[ii], self.ln_b[ii]), self.W_a[ii], self.b_a[ii])), self.W_b[ii], self.b_b[ii])
    
    def extra_repr(self):
        return ('adapter={}, input_dim={}, hidden_dim={}'.format(self.adapter_num, self.input_dim, self.hidden_dim))

class FeedForwardModule(nn.Module):
    def __init__(self, input_feat, hidden_units, dropout1, dropout2, activation_fn="swish", bias=True):
        super(FeedForwardModule, self).__init__()
        self.layer_norm = LayerNorm(input_feat)
        self.w_1 = nn.Linear(input_feat, hidden_units, bias=bias)
        self.w_2 = nn.Linear(hidden_units, input_feat, bias=bias)
        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)
        self.activation = get_activation_fn(activation_fn)(hidden_units)

    def forward(self, x):
        return self.dropout2(self.w_2(self.dropout1(self.activation(self.w_1(self.layer_norm(x))))))

class ConvolutionModule(nn.Module):
    def __init__(self, embed_dim, channels, depthwise_kernel_size, dropout, activation_fn="swish", bias=False, export=False):
        super(ConvolutionModule, self).__init__()
        assert (depthwise_kernel_size - 1) % 2 == 0
        self.layer_norm = LayerNorm(embed_dim, export=export)
        self.pointwise_conv1 = nn.Conv1d(embed_dim, 2 * channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(channels, channels, depthwise_kernel_size, stride=1, padding=(depthwise_kernel_size - 1) // 2, groups=channels, bias=bias)
        self.batch_norm = nn.BatchNorm1d(channels)
        self.activation = get_activation_fn(activation_fn)(channels)
        self.pointwise_conv2 = nn.Conv1d(channels, embed_dim, kernel_size=1, stride=1, padding=0, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.pointwise_conv2(self.activation(self.batch_norm(self.depthwise_conv(self.glu(self.pointwise_conv1(self.layer_norm(x).transpose(1, 2)))))))).transpose(1, 2)

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)

def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos, sin = (cos[offset : q.shape[0] + offset, ...], sin[offset : q.shape[0] + offset, ...])
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, base=10000, precision=torch.half):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = 0
        self.cos_cached = torch.empty(self.seq_len_cached, 1, 1, dim)
        self.sin_cached = torch.empty(self.seq_len_cached, 1, 1, dim)
        self.precision = precision

    def forward(self, x, seq_len = 0):
        if seq_len > self.seq_len_cached:
            self.seq_len_cached = seq_len
            freqs = torch.einsum("i,j->ij", torch.arange(seq_len, device=x.device).type_as(self.inv_freq), self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos().view(emb.size(0), 1, 1, emb.size(1))
            self.sin_cached = emb.sin().view(emb.size(0), 1, 1, emb.size(1))
        return self.cos_cached, self.sin_cached

class ESPNETMultiHeadedAttention(nn.Module):
    def __init__(self, n_feat, n_head, dropout):
        super(ESPNETMultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward_qkv(self, query, key, value, **kwargs):
        n_batch = query.size(0)
        return self.linear_q(query).view(n_batch, -1, self.h, self.d_k).transpose(1, 2), self.linear_k(key).view(n_batch, -1, self.h, self.d_k).transpose(1, 2), self.linear_v(value).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)

    def forward_attention(self, value, scores, mask):
        n_batch = value.size(0)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2).to(bool), float("-inf"))
            self.attn = scores.softmax(dim=-1)
        else: self.attn = scores.softmax(dim=-1)

        return self.linear_out(((self.dropout(self.attn) @ value).transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k))) 

    def forward(self, query, key, value, key_padding_mask=None, **kwargs):
        q, k, v = self.forward_qkv(query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1))
        return self.forward_attention(v, (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k), key_padding_mask).transpose(0, 1), None

class RelPositionMultiHeadedAttention(ESPNETMultiHeadedAttention):
    def __init__(self, n_feat, n_head, dropout, zero_triu=False):
        super().__init__(n_feat, n_head, dropout)
        self.zero_triu = zero_triu
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.zeros(self.h, self.d_k))
        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        x = torch.cat([torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype), x], dim=-1).view(*x.size()[:2], x.size(3) + 1, x.size(2))[:, :, 1:].view_as(x)[:, :, :, : x.size(-1) // 2 + 1]
        if self.zero_triu: x = x * torch.ones((x.size(2), x.size(3)), device=x.device).tril(x.size(3) - x.size(2))[None, None, :, :]
        return x

    def forward(self, query, key, value, pos_emb, key_padding_mask=None, **kwargs):
        pos_emb = pos_emb.transpose(0, 1)
        q, k, v = self.forward_qkv(query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1))
        q = q.transpose(1, 2)

        return self.forward_attention(v, (((q + self.pos_bias_u).transpose(1, 2) @ k.transpose(-2, -1)) + self.rel_shift(((q + self.pos_bias_v).transpose(1, 2) @ self.linear_pos(pos_emb).view(pos_emb.size(0), -1, self.h, self.d_k).transpose(1, 2).transpose(-2, -1)))) / math.sqrt(self.d_k), key_padding_mask).transpose(0, 1), None

class RotaryPositionMultiHeadedAttention(ESPNETMultiHeadedAttention):
    def __init__(self, n_feat, n_head, dropout, precision, rotary_emd_base=10000):
        super().__init__(n_feat, n_head, dropout)
        precision = torch.float
        self.rotary_ndims = self.d_k
        if precision == "fp16": precision = torch.half
        self.rotary_emb = RotaryPositionalEmbedding(self.rotary_ndims, base=rotary_emd_base, precision=precision)

    def forward(self, query, key, value, key_padding_mask=None, **kwargs):
        T, B, C = value.size()
        query = query.view(T, B, self.h, self.d_k)
        key = key.view(T, B, self.h, self.d_k)
        value = value.view(T, B, self.h, self.d_k)
        cos, sin = self.rotary_emb(value, seq_len=T)
        query, key = apply_rotary_pos_emb(query, key, cos, sin, offset=0)
        query = query.view(T, B, self.h * self.d_k)
        key = key.view(T, B, self.h * self.d_k)
        value = value.view(T, B, self.h * self.d_k)
        q, k, v = self.forward_qkv(query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1))
        return self.forward_attention(v, (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k), key_padding_mask).transpose(0, 1), None

class ConformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, attention_heads, dropout, use_fp16, depthwise_conv_kernel_size=31, activation_fn="swish", attn_type=None, pos_enc_type="abs"):
        self.pos_enc_type = pos_enc_type
        super(ConformerEncoderLayer, self).__init__()
        self.ffn1 = FeedForwardModule(embed_dim, ffn_embed_dim, dropout, dropout)
        self.self_attn_layer_norm = LayerNorm(embed_dim, export=False)
        self.self_attn_dropout = nn.Dropout(dropout)
        if attn_type == "espnet":
            if self.pos_enc_type == "rel_pos": self.self_attn = RelPositionMultiHeadedAttention(embed_dim, attention_heads, dropout=dropout)
            elif self.pos_enc_type == "rope": self.self_attn = RotaryPositionMultiHeadedAttention(embed_dim, attention_heads, dropout=dropout, precision=use_fp16)
            elif self.pos_enc_type == "abs": self.self_attn = ESPNETMultiHeadedAttention(embed_dim, attention_heads, dropout=dropout)
            else: raise Exception
        else: self.self_attn = MultiheadAttention(embed_dim, attention_heads, dropout=dropout)
        self.conv_module = ConvolutionModule(embed_dim=embed_dim, channels=embed_dim, depthwise_kernel_size=depthwise_conv_kernel_size, dropout=dropout, activation_fn=activation_fn)
        self.ffn2 = FeedForwardModule(embed_dim, ffn_embed_dim, dropout, dropout, activation_fn=activation_fn)
        self.final_layer_norm = LayerNorm(embed_dim, export=False)

    def forward(self, x, encoder_padding_mask, position_emb = None):
        residual = x
        x = self.ffn1(x) * 0.5 + residual
        residual = x
        x = self.self_attn_layer_norm(x)
        if self.pos_enc_type == "rel_pos": x, attn = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask, pos_emb=position_emb, need_weights=False)
        else: x, attn = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask, need_weights=False)
        x = self.self_attn_dropout(x)
        x = x + residual
        residual = x
        x = residual + self.conv_module(x.transpose(0, 1)).transpose(0, 1)
        residual = x
        x = self.ffn2(x)
        layer_result = x
        x = self.final_layer_norm(x * 0.5 + residual)
        return x, (attn, layer_result)

class ConformerWav2Vec2EncoderLayer(ConformerEncoderLayer):
    def forward(self, x, self_attn_mask=None, self_attn_padding_mask=None, need_weights=False, att_args=None, position_emb=None):
        return super().forward(x, self_attn_padding_mask, position_emb)

class TransformerSentenceEncoderWithAdapterLayer(TransformerSentenceEncoderLayer):
    def __init__(self, embedding_dim = 768, ffn_embedding_dim = 3072, num_attention_heads = 8, dropout = 0.1, attention_dropout = 0.1, activation_dropout = 0.1, activation_fn = "relu", layer_norm_first = False, adapter_num=201, adapter_dim=64, adapter_act_fn="relu"):
        super().__init__(embedding_dim=embedding_dim, ffn_embedding_dim=ffn_embedding_dim, num_attention_heads=num_attention_heads, dropout=dropout, attention_dropout=attention_dropout, activation_dropout=activation_dropout, activation_fn=activation_fn, layer_norm_first=layer_norm_first)
        self.adapter_num = adapter_num
        self.adapter_dim = adapter_dim
        self.adapter_layer = AdapterFast(adapter_num, self.embedding_dim, self.adapter_dim, adapter_act_fn)

    def forward(self, x, self_attn_mask=None, self_attn_padding_mask=None, need_weights=False, att_args=None, corpus_key=None):
        x, (attn, layer_result) = super().forward(x=x, self_attn_mask=self_attn_mask, self_attn_padding_mask=self_attn_padding_mask, need_weights=need_weights, att_args=att_args)
        assert corpus_key is not None
        assert len(set(corpus_key)) == 1
        return x + self.adapter_layer(x, corpus_key[0]), (attn, layer_result)

class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None, tranpose_dim=-2):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx
        self.tranpose_dim = tranpose_dim

    def forward(self, x):
        if self.deconstruct_idx is not None: x = x[self.deconstruct_idx]
        return x.transpose(self.tranpose_dim, -1)

class TransformerEncoder(nn.Module):
    def build_encoder_layer(self, args, **kwargs):
        if args.layer_type == "transformer": layer = TransformerSentenceEncoderLayer(embedding_dim=self.embedding_dim, ffn_embedding_dim=args.encoder_ffn_embed_dim, num_attention_heads=args.encoder_attention_heads, dropout=self.dropout, attention_dropout=args.attention_dropout, activation_dropout=args.activation_dropout, activation_fn=args.activation_fn, layer_norm_first=args.layer_norm_first)
        elif args.layer_type == "conformer": layer = ConformerWav2Vec2EncoderLayer(embed_dim=self.embedding_dim, ffn_embed_dim=args.encoder_ffn_embed_dim, attention_heads=args.encoder_attention_heads, dropout=args.dropout, depthwise_conv_kernel_size=args.depthwise_conv_kernel_size, activation_fn="swish", attn_type=args.attn_type, use_fp16=args.fp16, pos_enc_type="abs")
        elif args.layer_type == "trf_adp":
            use_adp = False
            if args.adp_trf_idx == "all": use_adp = True
            else: 
                if kwargs.get("layer_idx", None) in list(range(*[int(g) for g in args.adp_trf_idx.split(":")])): use_adp = True

            layer = TransformerSentenceEncoderWithAdapterLayer(embedding_dim=self.embedding_dim, ffn_embedding_dim=args.encoder_ffn_embed_dim, num_attention_heads=args.encoder_attention_heads, dropout=self.dropout, attention_dropout=args.attention_dropout, activation_dropout=args.activation_dropout, activation_fn=args.activation_fn, layer_norm_first=args.layer_norm_first, adapter_num=args.adp_num, adapter_dim=args.adp_dim, adapter_act_fn=args.adp_act_fn) if use_adp else TransformerSentenceEncoderLayer(embedding_dim=self.embedding_dim, ffn_embedding_dim=args.encoder_ffn_embed_dim, num_attention_heads=args.encoder_attention_heads, dropout=self.dropout, attention_dropout=args.attention_dropout, activation_dropout=args.activation_dropout, activation_fn=args.activation_fn, layer_norm_first=args.layer_norm_first,)

        return layer

    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim
        self.required_seq_len_multiple = args.required_seq_len_multiple
        pos_conv_depth = getattr(args, "pos_conv_depth", 1)
        if pos_conv_depth > 1:
            num_layers = args.pos_conv_depth
            k = max(3, args.conv_pos // num_layers)

            def make_conv_block(e, k, g, l):
                return nn.Sequential(*[nn.Sequential(nn.Conv1d(e, e, kernel_size=k, padding=k // 2, groups=g), SamePad(k), TransposeLast(), LayerNorm(e, elementwise_affine=False), TransposeLast(), nn.GELU()) for _ in range(l)])

            self.pos_conv = make_conv_block(self.embedding_dim, k, args.conv_pos_groups, num_layers)
        else: self.pos_conv = make_conv_pos(self.embedding_dim, args.conv_pos, args.conv_pos_groups)

        self.layers = nn.ModuleList([self.build_encoder_layer(args, layer_idx=ii) for ii in range(args.encoder_layers)])
        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop
        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None, layer=None, corpus_key=None):
        x, layer_results = self.extract_features(x, padding_mask, layer, corpus_key=corpus_key)
        if self.layer_norm_first and layer is None: x = self.layer_norm(x)
        return x, layer_results

    def extract_features(self, x, padding_mask=None, tgt_layer=None, min_layer=0, corpus_key=None):
        if padding_mask is not None: x = index_put(x, padding_mask, 0)
        x = x + self.pos_conv(x.transpose(1, 2)).transpose(1, 2)
        if not self.layer_norm_first: x = self.layer_norm(x)
        x, pad_length = pad_to_multiple(x, self.required_seq_len_multiple, dim=-2, value=0)
        if pad_length > 0 and padding_mask is None:
            padding_mask = x.new_zeros((x.size(0), x.size(1)), dtype=torch.bool)
            padding_mask[:, -pad_length:] = True
        else: padding_mask, _ = pad_to_multiple(padding_mask, self.required_seq_len_multiple, dim=-1, value=True)
        x = F.dropout(x, p=self.dropout, training=self.training).transpose(0, 1)
        layer_results = []
        r = None

        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random() if self.layerdrop > 0 else 1
            if not self.training or (dropout_probability > self.layerdrop):
                layer_check = layer
                if (corpus_key is None) or (not isinstance(layer_check, (TransformerSentenceEncoderWithAdapterLayer))): x, (z, lr) = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
                else: x, (z, lr) = layer(x, self_attn_padding_mask=padding_mask, need_weights=False, corpus_key=corpus_key)
                if i >= min_layer: layer_results.append((x, z, lr))
            if i == tgt_layer:
                r = x
                break

        if r is not None: x = r
        x = x.transpose(0, 1)

        if pad_length > 0:
            x = x[:, :-pad_length]
            def undo_pad(a, b, c):
                return (a[:-pad_length], b[:-pad_length] if b is not None else b, c[:-pad_length])

            layer_results = [undo_pad(*u) for u in layer_results]

        return x, layer_results

    def max_positions(self):
        return self.args.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict

class Fp32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.group_norm(input.float(), self.num_groups, self.weight.float() if self.weight is not None else None, self.bias.float() if self.bias is not None else None, self.eps)
        return output.type_as(input)

class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(input.float(), self.normalized_shape, self.weight.float() if self.weight is not None else None, self.bias.float() if self.bias is not None else None, self.eps)
        return output.type_as(input)

class ConvFeatureExtractionModel(nn.Module):
    def __init__(self, conv_layers, dropout = 0.0, mode = "default", conv_bias = False):
        super().__init__()
        assert mode in {"default", "layer_norm"}

        def block(n_in, n_out, k, stride, is_layer_norm=False, is_group_norm=False, conv_bias=False):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (is_layer_norm and is_group_norm) == False

            if is_layer_norm: return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.Sequential(TransposeLast(), Fp32LayerNorm(dim, elementwise_affine=True), TransposeLast()), nn.GELU())
            elif is_group_norm: return nn.Sequential(make_conv(), nn.Dropout(p=dropout), Fp32GroupNorm(dim, dim, affine=True), nn.GELU())
            else: return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3
            (dim, k, stride) = cl
            self.conv_layers.append(block(in_d, dim, k, stride, is_layer_norm=mode == "layer_norm", is_group_norm=mode == "default" and i == 0, conv_bias=conv_bias))
            in_d = dim

    def forward(self, x):
        x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)

        return x

class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None

class BaseFairseqModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._is_generation_fast = False

    def get_targets(self, sample, net_output):
        return sample["target"]

    def extract_features(self, *args, **kwargs):
        return self(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, model_cfg = None, args = None):
        self.upgrade_state_dict(state_dict)
        new_state_dict = prune_state_dict(state_dict, model_cfg)
        return super().load_state_dict(new_state_dict, strict)

    def upgrade_state_dict(self, state_dict):
        self.upgrade_state_dict_named(state_dict, "")

    def upgrade_state_dict_named(self, state_dict, name):
        assert state_dict is not None

        def do_upgrade(m, prefix):
            if len(prefix) > 0: prefix += "."
            for n, c in m.named_children():
                name = prefix + n
                if hasattr(c, "upgrade_state_dict_named"): c.upgrade_state_dict_named(state_dict, name)
                elif hasattr(c, "upgrade_state_dict"): c.upgrade_state_dict(state_dict)
                do_upgrade(c, name)

        do_upgrade(self, name)

    def make_generation_fast_(self, **kwargs):
        if self._is_generation_fast: return
        self._is_generation_fast = True

        def apply_remove_weight_norm(module):
            try:
                nn.utils.remove_weight_norm(module)
            except (AttributeError, ValueError):
                return

        self.apply(apply_remove_weight_norm)
        def apply_make_generation_fast_(module, prefix):
            if len(prefix) > 0: prefix += "."

            base_func = BaseFairseqModel.make_generation_fast_
            for n, m in module.named_modules():
                if (m != self and hasattr(m, "make_generation_fast_") and m.make_generation_fast_.__func__ is not base_func): m.make_generation_fast_(name=prefix + n, **kwargs)

        apply_make_generation_fast_(self, "")
        self.eval()

class HubertConfig:
    def __init__(
        self, 
        _name = None, 
        label_rate = 50, 
        encoder_layers_1 = 3, 
        logit_temp_ctr = 0.1, 
        num_negatives = 100, 
        cross_sample_negatives = 0, 
        ctr_layers = [-6],
        crop_seq_to_multiple = 1,
        extractor_mode = "default", 
        encoder_layers = 12, 
        encoder_embed_dim = 768, 
        encoder_ffn_embed_dim = 3072, 
        encoder_attention_heads = 12, 
        activation_fn = "gelu", 
        layer_type = "transformer", 
        dropout = 0.1, 
        attention_dropout = 0.1, 
        activation_dropout = 0.0, 
        encoder_layerdrop = 0.0, 
        dropout_input = 0.0, 
        dropout_features = 0.0, 
        final_dim = 0, 
        untie_final_proj = False, 
        layer_norm_first = False, 
        conv_feature_layers = "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2", 
        conv_bias = False, 
        logit_temp = 0.1, 
        target_glu = False, 
        feature_grad_mult = 1.0, 
        mask_length = 10, 
        mask_prob = 0.65, 
        mask_selection = "static", 
        mask_other = 0.0, 
        no_mask_overlap = False, 
        mask_min_space = 1, 
        mask_channel_length = 10, 
        mask_channel_prob = 0.0, 
        mask_channel_selection = "static", 
        mask_channel_other = 0.0, 
        no_mask_channel_overlap = False, 
        mask_channel_min_space = 1, 
        conv_pos = 128, 
        conv_pos_groups = 16, 
        conv_pos_batch_norm = False, 
        latent_temp = (2, 0.5, 0.999995), 
        skip_masked = False, 
        skip_nomask = False, 
        checkpoint_activations = False, 
        required_seq_len_multiple = 2, 
        depthwise_conv_kernel_size = 31, 
        attn_type = "", 
        pos_enc_type = "abs", 
        fp16 = False
    ):
        self._name = _name
        self.label_rate = label_rate
        self.encoder_layers_1 = encoder_layers_1
        self.logit_temp_ctr = logit_temp_ctr
        self.num_negatives = num_negatives
        self.cross_sample_negatives = cross_sample_negatives
        self.ctr_layers = ctr_layers
        self.crop_seq_to_multiple = crop_seq_to_multiple
        self.extractor_mode = extractor_mode
        self.encoder_layers = encoder_layers
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_ffn_embed_dim = encoder_ffn_embed_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.activation_fn = activation_fn
        self.layer_type = layer_type
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.encoder_layerdrop = encoder_layerdrop
        self.dropout_input = dropout_input
        self.dropout_features = dropout_features
        self.final_dim = final_dim
        self.untie_final_proj = untie_final_proj
        self.layer_norm_first = layer_norm_first
        self.conv_feature_layers = conv_feature_layers
        self.conv_bias = conv_bias
        self.logit_temp = logit_temp
        self.target_glu = target_glu
        self.feature_grad_mult = feature_grad_mult
        self.mask_length = mask_length
        self.mask_prob = mask_prob
        self.mask_selection = mask_selection
        self.mask_other = mask_other
        self.no_mask_overlap = no_mask_overlap
        self.mask_min_space = mask_min_space
        self.mask_channel_length = mask_channel_length
        self.mask_channel_prob = mask_channel_prob
        self.mask_channel_selection = mask_channel_selection
        self.mask_channel_other = mask_channel_other
        self.no_mask_channel_overlap = no_mask_channel_overlap
        self.mask_channel_min_space = mask_channel_min_space
        self.conv_pos = conv_pos
        self.conv_pos_groups = conv_pos_groups
        self.conv_pos_batch_norm = conv_pos_batch_norm
        self.latent_temp = latent_temp
        self.skip_masked = skip_masked
        self.skip_nomask = skip_nomask
        self.checkpoint_activations = checkpoint_activations
        self.required_seq_len_multiple = required_seq_len_multiple
        self.depthwise_conv_kernel_size = depthwise_conv_kernel_size
        self.attn_type = attn_type
        self.pos_enc_type = pos_enc_type
        self.fp16 = fp16

class HubertModel(BaseFairseqModel):
    def __init__(self, cfg, num_classes):
        super().__init__()
        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]
        self.feature_extractor = ConvFeatureExtractionModel(conv_layers=feature_enc_layers, dropout=0.0, mode=cfg.extractor_mode, conv_bias=cfg.conv_bias)
        feature_ds_rate = np.prod([s for _, _, s in feature_enc_layers])
        self.feat2tar_ratio = cfg.label_rate * feature_ds_rate / 16000
        self.post_extract_proj = (nn.Linear(self.embed, cfg.encoder_embed_dim) if self.embed != cfg.encoder_embed_dim else None)
        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space
        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space
        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)
        self.feature_grad_mult = cfg.feature_grad_mult
        self.logit_temp = cfg.logit_temp
        self.skip_masked = cfg.skip_masked
        self.skip_nomask = cfg.skip_nomask
        final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim
        self.mask_emb = nn.Parameter(torch.FloatTensor(cfg.encoder_embed_dim).uniform_())
        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)
        self.target_glu = None
        if cfg.target_glu: self.target_glu = nn.Sequential(nn.Linear(final_dim, final_dim * 2), nn.GLU())
        self.untie_final_proj = cfg.untie_final_proj
        self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)
        self.num_classes = [num_classes]
        self.label_embs_concat = nn.Parameter(torch.FloatTensor(sum(self.num_classes), final_dim))
        nn.init.uniform_(self.label_embs_concat)

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def apply_mask(self, x, padding_mask, target_list):
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = torch.from_numpy(compute_mask_indices((B, T), padding_mask, self.mask_prob, self.mask_length, self.mask_selection, self.mask_other, min_masks=2, no_overlap=self.no_mask_overlap, min_space=self.mask_min_space)).to(x.device)
            x[mask_indices] = self.mask_emb
        else: mask_indices = None

        if self.mask_channel_prob > 0: x[(torch.from_numpy(compute_mask_indices((B, C), None, self.mask_channel_prob, self.mask_channel_length, self.mask_channel_selection, self.mask_channel_other, no_overlap=self.no_mask_channel_overlap, min_space=self.mask_channel_min_space)).to(x.device).unsqueeze(1).expand(-1, T, -1))] = 0
        return x, mask_indices

    def compute_nce(self, x, pos, negs):
        neg_is_pos = (pos == negs).all(-1)
        logits = torch.cosine_similarity(x.float(), torch.cat([pos.unsqueeze(0), negs], dim=0).float(), dim=-1).type_as(x)
        logits /= self.logit_temp
        if neg_is_pos.any(): logits[1:][neg_is_pos] = float("-inf")
        return logits.transpose(0, 1)

    def forward_features(self, source):
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0: features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)
        return features

    def forward_targets(self, features, target_list):
        feat_tsz = features.size(2)
        targ_tsz = min([t.size(1) for t in target_list])
        if self.feat2tar_ratio * feat_tsz > targ_tsz:
            feat_tsz = int(targ_tsz / self.feat2tar_ratio)
            features = features[..., :feat_tsz]

        return features, [t[:, (torch.arange(feat_tsz).float() * self.feat2tar_ratio).long()] for t in target_list]

    def forward_padding_mask(self, features, padding_mask):
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0: padding_mask = padding_mask[:, :-extra]
        return padding_mask.view(padding_mask.size(0), features.size(1), -1).all(-1)

    def forward(self, source, target_list = None, padding_mask = None, mask = True, features_only = False, output_layer = None):
        features = self.forward_features(source)
        if target_list is not None: features, target_list = self.forward_targets(features, target_list)
        features_pen = features.float().pow(2).mean()
        features = self.layer_norm(features.transpose(1, 2))
        unmasked_features = features.clone()
        if padding_mask is not None: padding_mask = self.forward_padding_mask(features, padding_mask)
        if self.post_extract_proj is not None: features = self.post_extract_proj(features)
        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)
        if mask: x, mask_indices = self.apply_mask(features, padding_mask, target_list)
        else: x, mask_indices = features, None
        x, _ = self.encoder(x, padding_mask=padding_mask, layer=None if output_layer is None else output_layer - 1)
        if features_only: return {"x": x, "padding_mask": padding_mask, "features": features}

        def compute_pred(proj_x, target, label_embs):
            y = torch.index_select(label_embs, 0, target.long())
            negs = label_embs.unsqueeze(1).expand(-1, proj_x.size(0), -1)
            if self.target_glu:
                y = self.target_glu(y)
                negs = self.target_glu(negs)

            return self.compute_nce(proj_x, y, negs)

        label_embs_list = self.label_embs_concat.split(self.num_classes, 0)
        if not self.skip_masked:
            masked_indices = torch.logical_and(~padding_mask, mask_indices)
            proj_x_m = self.final_proj(x[masked_indices])
            logit_m_list = [compute_pred(proj_x_m, t[masked_indices], label_embs_list[i]) for i, (proj_x_m, t) in enumerate(zip(proj_x_m.chunk(len(target_list), dim=-1) if self.untie_final_proj else [proj_x_m for _ in range(len(target_list))], target_list))]
        else: logit_m_list = [None for _ in target_list]

        if not self.skip_nomask:
            nomask_indices = torch.logical_and(~padding_mask, ~mask_indices)
            proj_x_u = self.final_proj(x[nomask_indices])
            logit_u_list = [compute_pred(proj_x_u, t[nomask_indices], label_embs_list[i]) for i, (proj_x_u, t) in enumerate(zip(proj_x_u.chunk(len(target_list), dim=-1) if self.untie_final_proj else [proj_x_u for _ in range(len(target_list))], target_list))]
        else: logit_u_list = [None for _ in target_list]

        return {"logit_m_list": logit_m_list, "logit_u_list": logit_u_list, "padding_mask": padding_mask, "features_pen": features_pen}

    def extract_features(self, source, padding_mask = None, mask = False, ret_conv = False, output_layer = None):
        res = self.forward(source, padding_mask=padding_mask, mask=mask, features_only=True, output_layer=output_layer)
        return res["features"] if ret_conv else res["x"], res["padding_mask"]

    def get_logits(self, net_output, is_masked=True):
        return [x.float() for x in (net_output["logit_m_list"] if is_masked else net_output["logit_u_list"]) if x is not None]

    def get_targets(self, net_output, is_masked=True):
        return [x.new_zeros(x.size(0), dtype=torch.long) for x in self.get_logits(net_output, is_masked)]

    def get_extra_losses(self, net_output):
        extra_losses, names = [], []
        if "features_pen" in net_output:
            extra_losses.append(net_output["features_pen"])
            names.append("features_pen")

        return extra_losses, names

    def remove_pretraining_modules(self):
        self.target_glu = None
        self.final_proj = None
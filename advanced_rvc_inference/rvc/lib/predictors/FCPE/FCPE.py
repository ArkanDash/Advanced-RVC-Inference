import os
import sys
import torch

import numpy as np
import torch.nn as nn
import onnxruntime as ort
import torch.nn.functional as F

from einops import rearrange
from torch.nn.utils.parametrizations import weight_norm

sys.path.append(os.getcwd())
os.environ["LRU_CACHE_CAPACITY"] = "3"

from main.library.predictors.FCPE.wav2mel import Wav2Mel
from main.library.predictors.FCPE.encoder import EncoderLayer, ConformerNaiveEncoder
from main.library.predictors.FCPE.utils import l2_regularization, batch_interp_with_replacement_detach, decrypt_model, DotDict

@torch.no_grad()
def cent_to_f0(cent):
    return 10 * 2 ** (cent / 1200)

@torch.no_grad()
def f0_to_cent(f0):
    return 1200 * (f0 / 10).log2()

@torch.no_grad()
def latent2cents_decoder(cent_table, y, threshold = 0.05, mask = True):
    if str(y.device).startswith("privateuseone"): 
        cent_table = cent_table.cpu()
        y = y.cpu()

    B, N, _ = y.size()
    ci = cent_table[None, None, :].expand(B, N, -1)
    rtn = (ci * y).sum(dim=-1, keepdim=True) / y.sum(dim=-1, keepdim=True)  

    if mask:
        confident = y.max(dim=-1, keepdim=True)[0]
        confident_mask = torch.ones_like(confident)
        confident_mask[confident <= threshold] = float("-INF")
        rtn = rtn * confident_mask

    return rtn

@torch.no_grad()
def latent2cents_local_decoder(cent_table, out_dims, y, threshold = 0.05, mask = True):
    if str(y.device).startswith("privateuseone"): 
        cent_table = cent_table.cpu()
        y = y.cpu()

    B, N, _ = y.size()
    ci = cent_table[None, None, :].expand(B, N, -1)
    confident, max_index = y.max(dim=-1, keepdim=True)

    local_argmax_index = torch.arange(0, 9).to(max_index.device) + (max_index - 4)
    local_argmax_index[local_argmax_index < 0] = 0
    local_argmax_index[local_argmax_index >= out_dims] = out_dims - 1

    y_l = y.gather(-1, local_argmax_index)
    rtn = (ci.gather(-1, local_argmax_index) * y_l).sum(dim=-1, keepdim=True) / y_l.sum(dim=-1, keepdim=True) 

    if mask:
        confident_mask = torch.ones_like(confident)
        confident_mask[confident <= threshold] = float("-INF")
        rtn = rtn * confident_mask

    return rtn

def cents_decoder(cent_table, y, confidence, threshold = 0.05, mask=True):
    if str(y.device).startswith("privateuseone"): 
        cent_table = cent_table.cpu()
        y = y.cpu()

    B, N, _ = y.size()
    rtn = (cent_table[None, None, :].expand(B, N, -1) * y).sum(dim=-1, keepdim=True) / y.sum(dim=-1, keepdim=True)

    if mask:
        confident = y.max(dim=-1, keepdim=True)[0]
        confident_mask = torch.ones_like(confident)
        confident_mask[confident <= threshold] = float("-INF")
        rtn = rtn * confident_mask

    return (rtn, confident) if confidence else rtn

def cents_local_decoder(cent_table, y, n_out, confidence, threshold = 0.05, mask=True):
    if str(y.device).startswith("privateuseone"): 
        cent_table = cent_table.cpu()
        y = y.cpu()

    B, N, _ = y.size()
    confident, max_index = y.max(dim=-1, keepdim=True)
    local_argmax_index = (torch.arange(0, 9).to(max_index.device) + (max_index - 4)).clamp(0, n_out - 1)
    y_l = y.gather(-1, local_argmax_index)
    rtn = (cent_table[None, None, :].expand(B, N, -1).gather(-1, local_argmax_index) * y_l).sum(dim=-1, keepdim=True) / y_l.sum(dim=-1, keepdim=True)

    if mask:
        confident_mask = torch.ones_like(confident)
        confident_mask[confident <= threshold] = float("-INF")
        rtn = rtn * confident_mask

    return (rtn, confident) if confidence else rtn

class PCmer(nn.Module):
    def __init__(self, num_layers, num_heads, dim_model, dim_keys, dim_values, residual_dropout, attention_dropout):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.dim_values = dim_values
        self.dim_keys = dim_keys
        self.residual_dropout = residual_dropout
        self.attention_dropout = attention_dropout
        self._layers = nn.ModuleList([EncoderLayer(self) for _ in range(num_layers)])

    def forward(self, phone, mask=None):
        for layer in self._layers:
            phone = layer(phone, mask)

        return phone

class CFNaiveMelPE(nn.Module):
    def __init__(self, input_channels, out_dims, hidden_dims = 512, n_layers = 6, n_heads = 8, f0_max = 1975.5, f0_min = 32.70, use_fa_norm = False, conv_only = False, conv_dropout = 0, atten_dropout = 0, use_harmonic_emb = False):
        super().__init__()
        self.input_channels = input_channels
        self.out_dims = out_dims
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.f0_max = f0_max
        self.f0_min = f0_min
        self.use_fa_norm = use_fa_norm
        self.residual_dropout = 0.1  
        self.attention_dropout = 0.1  
        self.harmonic_emb = nn.Embedding(9, hidden_dims) if use_harmonic_emb else None
        self.input_stack = nn.Sequential(nn.Conv1d(input_channels, hidden_dims, 3, 1, 1), nn.GroupNorm(4, hidden_dims), nn.LeakyReLU(), nn.Conv1d(hidden_dims, hidden_dims, 3, 1, 1))
        self.net = ConformerNaiveEncoder(num_layers=n_layers, num_heads=n_heads, dim_model=hidden_dims, use_norm=use_fa_norm, conv_only=conv_only, conv_dropout=conv_dropout, atten_dropout=atten_dropout)
        self.norm = nn.LayerNorm(hidden_dims)
        self.output_proj = weight_norm(nn.Linear(hidden_dims, out_dims))
        self.cent_table_b = torch.linspace(f0_to_cent(torch.Tensor([f0_min]))[0], f0_to_cent(torch.Tensor([f0_max]))[0], out_dims).detach()
        self.register_buffer("cent_table", self.cent_table_b)
        self.gaussian_blurred_cent_mask_b = (1200 * torch.Tensor([self.f0_max / 10.]).log2())[0].detach()
        self.register_buffer("gaussian_blurred_cent_mask", self.gaussian_blurred_cent_mask_b)

    def forward(self, x, _h_emb=None):
        x = self.input_stack(x.transpose(-1, -2)).transpose(-1, -2)
        if self.harmonic_emb is not None: x = x + self.harmonic_emb(torch.LongTensor([0]).to(x.device)) if _h_emb is None else x + self.harmonic_emb(torch.LongTensor([int(_h_emb)]).to(x.device))
        return self.output_proj(self.norm(self.net(x))).sigmoid()

    @torch.no_grad()
    def infer(self, mel, decoder = "local_argmax", threshold = 0.05):
        latent = self.forward(mel)
        return cent_to_f0(latent2cents_decoder(self.cent_table, latent, threshold=threshold) if decoder == "argmax" else latent2cents_local_decoder(self.cent_table, self.out_dims, latent, threshold=threshold))

class FCPE_LEGACY(nn.Module):
    def __init__(self, input_channel=128, out_dims=360, n_layers=12, n_chans=512, loss_mse_scale=10, loss_l2_regularization=False, loss_l2_regularization_scale=1, loss_grad1_mse=False, loss_grad1_mse_scale=1, f0_max=1975.5, f0_min=32.70, confidence=False, threshold=0.05, use_input_conv=True):
        super().__init__()
        self.loss_mse_scale = loss_mse_scale
        self.loss_l2_regularization = loss_l2_regularization
        self.loss_l2_regularization_scale = loss_l2_regularization_scale
        self.loss_grad1_mse = loss_grad1_mse
        self.loss_grad1_mse_scale = loss_grad1_mse_scale
        self.f0_max = f0_max
        self.f0_min = f0_min
        self.confidence = confidence
        self.threshold = threshold
        self.use_input_conv = use_input_conv
        self.cent_table_b = torch.Tensor(np.linspace(f0_to_cent(torch.Tensor([f0_min]))[0], f0_to_cent(torch.Tensor([f0_max]))[0], out_dims))
        self.register_buffer("cent_table", self.cent_table_b)
        self.stack = nn.Sequential(nn.Conv1d(input_channel, n_chans, 3, 1, 1), nn.GroupNorm(4, n_chans), nn.LeakyReLU(), nn.Conv1d(n_chans, n_chans, 3, 1, 1))
        self.decoder = PCmer(num_layers=n_layers, num_heads=8, dim_model=n_chans, dim_keys=n_chans, dim_values=n_chans, residual_dropout=0.1, attention_dropout=0.1)
        self.norm = nn.LayerNorm(n_chans)
        self.n_out = out_dims
        self.dense_out = weight_norm(nn.Linear(n_chans, self.n_out))

    def forward(self, mel, infer=True, gt_f0=None, return_hz_f0=False, cdecoder="local_argmax", output_interp_target_length=None):
        x = self.dense_out(self.norm(self.decoder((self.stack(mel.transpose(1, 2)).transpose(1, 2) if self.use_input_conv else mel)))).sigmoid()

        if not infer:
            loss_all = self.loss_mse_scale * F.binary_cross_entropy(x, self.gaussian_blurred_cent(f0_to_cent(gt_f0)))
            if self.loss_l2_regularization: loss_all = loss_all + l2_regularization(model=self, l2_alpha=self.loss_l2_regularization_scale)
            x = loss_all
        else:
            x = cent_to_f0(cents_decoder(self.cent_table, x, self.confidence, threshold=self.threshold, mask=True) if cdecoder == "argmax" else cents_local_decoder(self.cent_table, x, self.n_out, self.confidence, threshold=self.threshold, mask=True))
            x = (1 + x / 700).log() if not return_hz_f0 else x

        if output_interp_target_length is not None: 
            x = F.interpolate(torch.where(x == 0, float("nan"), x).transpose(1, 2), size=int(output_interp_target_length), mode="linear").transpose(1, 2)
            x = torch.where(x.isnan(), float(0.0), x)

        return x

    def gaussian_blurred_cent(self, cents):
        B, N, _ = cents.size()
        return (-(self.cent_table[None, None, :].expand(B, N, -1) - cents).square() / 1250).exp() * (cents > 0.1) & (cents < (1200.0 * np.log2(self.f0_max / 10.0))).float()

class InferCFNaiveMelPE(torch.nn.Module):
    def __init__(self, args, state_dict):
        super().__init__()
        self.model = CFNaiveMelPE(input_channels=args.mel.num_mels, out_dims=args.model.out_dims, hidden_dims=args.model.hidden_dims, n_layers=args.model.n_layers, n_heads=args.model.n_heads, f0_max=args.model.f0_max, f0_min=args.model.f0_min, use_fa_norm=args.model.use_fa_norm, conv_only=args.model.conv_only, conv_dropout=args.model.conv_dropout, atten_dropout=args.model.atten_dropout, use_harmonic_emb=False)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.register_buffer("tensor_device_marker", torch.tensor(1.0).float(), persistent=False)

    def forward(self, mel, decoder_mode = "local_argmax", threshold = 0.006):
        with torch.no_grad():
            mels = rearrange(torch.stack([mel], -1), "B T C K -> (B K) T C")
            f0s = rearrange(self.model.infer(mels, decoder=decoder_mode, threshold=threshold), "(B K) T 1 -> B T (K 1)", K=1)

        return f0s 

    def infer(self, mel, decoder_mode = "local_argmax", threshold = 0.006, f0_min = None, f0_max = None, interp_uv = False, output_interp_target_length = None, return_uv = False):
        f0 = self.__call__(mel, decoder_mode, threshold)
        f0_for_uv = f0

        uv = (f0_for_uv < f0_min).type(f0_for_uv.dtype)
        f0 = f0 * (1 - uv)

        if interp_uv: f0 = batch_interp_with_replacement_detach(uv.squeeze(-1).bool(), f0.squeeze(-1)).unsqueeze(-1)
        if f0_max is not None: f0[f0 > f0_max] = f0_max
        if output_interp_target_length is not None: 
            f0 = F.interpolate(torch.where(f0 == 0, float("nan"), f0).transpose(1, 2), size=int(output_interp_target_length), mode="linear").transpose(1, 2)
            f0 = torch.where(f0.isnan(), float(0.0), f0)

        if return_uv: return f0, F.interpolate(uv.transpose(1, 2), size=int(output_interp_target_length), mode="nearest").transpose(1, 2)
        else: return f0

class FCPEInfer_LEGACY:
    def __init__(self, configs, model_path, device=None, dtype=torch.float32, providers=None, onnx=False, f0_min=50, f0_max=1100):
        if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = dtype
        self.onnx = onnx
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.wav2mel = Wav2Mel(device=self.device, dtype=self.dtype)

        if self.onnx:
            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            self.model = ort.InferenceSession(decrypt_model(configs, model_path), sess_options=sess_options, providers=providers)
        else:
            ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
            self.args = DotDict(ckpt["config"])
            model = FCPE_LEGACY(input_channel=self.args.model.input_channel, out_dims=self.args.model.out_dims, n_layers=self.args.model.n_layers, n_chans=self.args.model.n_chans, loss_mse_scale=self.args.loss.loss_mse_scale, loss_l2_regularization=self.args.loss.loss_l2_regularization, loss_l2_regularization_scale=self.args.loss.loss_l2_regularization_scale, loss_grad1_mse=self.args.loss.loss_grad1_mse, loss_grad1_mse_scale=self.args.loss.loss_grad1_mse_scale, f0_max=self.f0_max, f0_min=self.f0_min, confidence=self.args.model.confidence)
            model.to(self.device).to(self.dtype)
            model.load_state_dict(ckpt["model"])
            model.eval()
            self.model = model

    @torch.no_grad()
    def __call__(self, audio, sr, threshold=0.05, p_len=None):
        if not self.onnx: self.model.threshold = threshold
        if not hasattr(self, "numpy_threshold") and self.onnx: self.numpy_threshold = np.array(threshold, dtype=np.float32)

        mel = self.wav2mel(audio=audio[None, :], sample_rate=sr).to(self.dtype)

        if self.onnx:
            return torch.as_tensor(
                self.model.run(
                    [self.model.get_outputs()[0].name], 
                    {
                        self.model.get_inputs()[0].name: mel.detach().cpu().numpy(), 
                        self.model.get_inputs()[1].name: self.numpy_threshold
                    }
                )[0], 
                dtype=self.dtype, 
                device=self.device
            )
        else: 
            return self.model(
                mel=mel, 
                infer=True, 
                return_hz_f0=True, 
                output_interp_target_length=p_len
            )

class FCPEInfer:
    def __init__(self, configs, model_path, device=None, dtype=torch.float32, providers=None, onnx=False, f0_min=50, f0_max=1100):
        if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = dtype
        self.onnx = onnx
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.wav2mel = Wav2Mel(device=self.device, dtype=self.dtype)

        if self.onnx:
            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            self.model = ort.InferenceSession(decrypt_model(configs, model_path), sess_options=sess_options, providers=providers)
        else:
            ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
            ckpt["config_dict"]["model"]["conv_dropout"] = ckpt["config_dict"]["model"]["atten_dropout"] = 0.0
            self.args = DotDict(ckpt["config_dict"])
            model = InferCFNaiveMelPE(self.args, ckpt["model"])
            self.model = model.to(device).to(self.dtype).eval()

    @torch.no_grad()
    def __call__(self, audio, sr, threshold=0.05, p_len=None):
        if not hasattr(self, "numpy_threshold") and self.onnx: self.numpy_threshold = np.array(threshold, dtype=np.float32)
        mel = self.wav2mel(audio=audio[None, :], sample_rate=sr).to(self.dtype)

        if self.onnx:
            return torch.as_tensor(
                self.model.run(
                    [self.model.get_outputs()[0].name], 
                    {
                        self.model.get_inputs()[0].name: mel.detach().cpu().numpy(), 
                        self.model.get_inputs()[1].name: self.numpy_threshold
                    }
                )[0], 
                dtype=self.dtype, 
                device=self.device
            ) 
        else: 
            return self.model.infer(
                mel, 
                threshold=threshold, 
                f0_min=self.f0_min, 
                f0_max=self.f0_max, 
                output_interp_target_length=p_len
            )

class FCPE:
    def __init__(self, configs, model_path, hop_length=512, f0_min=50, f0_max=1100, dtype=torch.float32, device=None, sample_rate=16000, threshold=0.05, providers=None, onnx=False, legacy=False):
        self.model = FCPEInfer_LEGACY if legacy else FCPEInfer
        self.fcpe = self.model(configs, model_path, device=device, dtype=dtype, providers=providers, onnx=onnx, f0_min=f0_min, f0_max=f0_max)
        self.hop_length = hop_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.dtype = dtype
        self.legacy = legacy

    def compute_f0(self, wav, p_len=None):
        x = torch.FloatTensor(wav).to(self.dtype).to(self.device)
        p_len = (x.shape[0] // self.hop_length) if p_len is None else p_len

        f0 = self.fcpe(x, sr=self.sample_rate, threshold=self.threshold, p_len=p_len)
        f0 = f0[:] if f0.dim() == 1 else f0[0, :, 0]

        if torch.all(f0 == 0): return f0.cpu().numpy() if p_len is None else np.zeros(p_len)
        return f0.cpu().numpy()
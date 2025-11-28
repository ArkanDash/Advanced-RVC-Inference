import os
import sys
import math
import torch
import inspect
import functools

sys.path.append(os.getcwd())

from main.library.speaker_diarization.speechbrain import MAIN_PROC_ONLY, is_distributed_initialized, main_process_only

KEYS_MAPPING = {".mutihead_attn": ".multihead_attn",  ".convs_intermedite": ".convs_intermediate"}

def map_old_state_dict_weights(state_dict, mapping):
    for replacement_old, replacement_new in mapping.items():
        for old_key in list(state_dict.keys()):
            if replacement_old in old_key: state_dict[old_key.replace(replacement_old, replacement_new)] = state_dict.pop(old_key)

    return state_dict

def hook_on_loading_state_dict_checkpoint(state_dict):
    return map_old_state_dict_weights(state_dict, KEYS_MAPPING)

def torch_patched_state_dict_load(path, device="cpu"):
    return hook_on_loading_state_dict_checkpoint(torch.load(path, map_location=device, weights_only=False))

@main_process_only
def torch_save(obj, path):
    state_dict = obj.state_dict()
    torch.save(state_dict, path)

def torch_recovery(obj, path, end_of_epoch):
    del end_of_epoch  

    state_dict = torch_patched_state_dict_load(path, "cpu")
    try:
        obj.load_state_dict(state_dict, strict=True)
    except TypeError:
        obj.load_state_dict(state_dict)

def torch_parameter_transfer(obj, path):
    incompatible_keys = obj.load_state_dict(torch_patched_state_dict_load(path, "cpu"), strict=False)

    for missing_key in incompatible_keys.missing_keys:
        pass
    for unexpected_key in incompatible_keys.unexpected_keys:
        pass

WEAKREF_MARKER = "WEAKREF"

def _cycliclrsaver(obj, path):
    state_dict = obj.state_dict()
    if state_dict.get("_scale_fn_ref") is not None: state_dict["_scale_fn_ref"] = WEAKREF_MARKER

    torch.save(state_dict, path)

def _cycliclrloader(obj, path, end_of_epoch):
    del end_of_epoch  

    try:
        obj.load_state_dict(torch.load(path, map_location="cpu", weights_only=False), strict=True)
    except TypeError:
        obj.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))

DEFAULT_LOAD_HOOKS = {torch.nn.Module: torch_recovery, torch.optim.Optimizer: torch_recovery, torch.optim.lr_scheduler.ReduceLROnPlateau: torch_recovery, torch.cuda.amp.grad_scaler.GradScaler: torch_recovery}
DEFAULT_SAVE_HOOKS = { torch.nn.Module: torch_save, torch.optim.Optimizer: torch_save, torch.optim.lr_scheduler.ReduceLROnPlateau: torch_save, torch.cuda.amp.grad_scaler.GradScaler: torch_save}
DEFAULT_LOAD_HOOKS[torch.optim.lr_scheduler.LRScheduler] = torch_recovery
DEFAULT_SAVE_HOOKS[torch.optim.lr_scheduler.LRScheduler] = torch_save
DEFAULT_TRANSFER_HOOKS = {torch.nn.Module: torch_parameter_transfer}
DEFAULT_SAVE_HOOKS[torch.optim.lr_scheduler.CyclicLR] = _cycliclrsaver
DEFAULT_LOAD_HOOKS[torch.optim.lr_scheduler.CyclicLR] = _cycliclrloader

def register_checkpoint_hooks(cls, save_on_main_only=True):
    global DEFAULT_LOAD_HOOKS, DEFAULT_SAVE_HOOKS, DEFAULT_TRANSFER_HOOKS

    for name, method in cls.__dict__.items():
        if hasattr(method, "_speechbrain_saver"): DEFAULT_SAVE_HOOKS[cls] = main_process_only(method) if save_on_main_only else method
        if hasattr(method, "_speechbrain_loader"): DEFAULT_LOAD_HOOKS[cls] = method
        if hasattr(method, "_speechbrain_transfer"): DEFAULT_TRANSFER_HOOKS[cls] = method
        
    return cls

def mark_as_saver(method):
    sig = inspect.signature(method)

    try:
        sig.bind(object(), "testpath")
    except TypeError:
        raise TypeError
    
    method._speechbrain_saver = True
    return method

def mark_as_transfer(method):
    sig = inspect.signature(method)
    
    try:
        sig.bind(object(), "testpath")
    except TypeError:
        raise TypeError
    
    method._speechbrain_transfer = True
    return method

def mark_as_loader(method):
    sig = inspect.signature(method)

    try:
        sig.bind(object(), "testpath", True)
    except TypeError:
        raise TypeError
    
    method._speechbrain_loader = True
    return method

def ddp_all_reduce(communication_object, reduce_op):
    if MAIN_PROC_ONLY >= 1 or not is_distributed_initialized(): return communication_object
    torch.distributed.all_reduce(communication_object, op=reduce_op)

    return communication_object

def fwd_default_precision(fwd = None, cast_inputs = torch.float32):
    if fwd is None: return functools.partial(fwd_default_precision, cast_inputs=cast_inputs)

    wrapped_fwd = torch.cuda.amp.custom_fwd(fwd, cast_inputs=cast_inputs)

    @functools.wraps(fwd)
    def wrapper(*args, force_allow_autocast = False, **kwargs):
        return fwd(*args, **kwargs) if force_allow_autocast else wrapped_fwd(*args, **kwargs)

    return wrapper

def spectral_magnitude(stft, power = 1, log = False, eps = 1e-14):
    spectr = stft.pow(2).sum(-1)

    if power < 1: spectr = spectr + eps
    spectr = spectr.pow(power)

    if log: return (spectr + eps).log()
    return spectr

class Filterbank(torch.nn.Module):
    def __init__(self, n_mels=40, log_mel=True, filter_shape="triangular", f_min=0, f_max=8000, n_fft=400, sample_rate=16000, power_spectrogram=2, amin=1e-10, ref_value=1.0, top_db=80.0, param_change_factor=1.0, param_rand_factor=0.0, freeze=True):
        super().__init__()
        self.n_mels = n_mels
        self.log_mel = log_mel
        self.filter_shape = filter_shape
        self.f_min = f_min
        self.f_max = f_max
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.power_spectrogram = power_spectrogram
        self.amin = amin
        self.ref_value = ref_value
        self.top_db = top_db
        self.freeze = freeze
        self.n_stft = self.n_fft // 2 + 1
        self.db_multiplier = math.log10(max(self.amin, self.ref_value))
        self.device_inp = torch.device("cpu")
        self.param_change_factor = param_change_factor
        self.param_rand_factor = param_rand_factor
        self.multiplier = 10 if self.power_spectrogram == 2 else 20

        hz = self._to_hz(torch.linspace(self._to_mel(self.f_min), self._to_mel(self.f_max), self.n_mels + 2))

        band = hz[1:] - hz[:-1]
        self.band = band[:-1]
        self.f_central = hz[1:-1]

        if not self.freeze:
            self.f_central = torch.nn.Parameter(self.f_central / (self.sample_rate * self.param_change_factor))
            self.band = torch.nn.Parameter(self.band / (self.sample_rate * self.param_change_factor))

        self.all_freqs_mat = torch.linspace(0, self.sample_rate // 2, self.n_stft).repeat(self.f_central.shape[0], 1)

    def forward(self, spectrogram):
        f_central_mat = self.f_central.repeat(self.all_freqs_mat.shape[1], 1).transpose(0, 1)
        band_mat = self.band.repeat(self.all_freqs_mat.shape[1], 1).transpose(0, 1)

        if not self.freeze:
            f_central_mat = f_central_mat * (self.sample_rate * self.param_change_factor * self.param_change_factor)
            band_mat = band_mat * (self.sample_rate * self.param_change_factor * self.param_change_factor)
        elif self.param_rand_factor != 0 and self.training:
            rand_change = (1.0 + torch.rand(2) * 2 * self.param_rand_factor - self.param_rand_factor)
            f_central_mat = f_central_mat * rand_change[0]
            band_mat = band_mat * rand_change[1]

        fbank_matrix = self._create_fbank_matrix(f_central_mat, band_mat).to(spectrogram.device)
        sp_shape = spectrogram.shape
        if len(sp_shape) == 4: spectrogram = spectrogram.permute(0, 3, 1, 2).reshape(sp_shape[0] * sp_shape[3], sp_shape[1], sp_shape[2])

        fbanks = spectrogram @ fbank_matrix
        if self.log_mel: fbanks = self._amplitude_to_DB(fbanks)

        if len(sp_shape) == 4:
            fb_shape = fbanks.shape
            fbanks = fbanks.reshape(sp_shape[0], sp_shape[3], fb_shape[1], fb_shape[2]).permute(0, 2, 3, 1)

        return fbanks

    @staticmethod
    def _to_mel(hz):
        return 2595 * math.log10(1 + hz / 700)

    @staticmethod
    def _to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def _triangular_filters(self, all_freqs, f_central, band):
        slope = (all_freqs - f_central) / band
        return torch.zeros(1, device=self.device_inp).max((slope + 1.0).min(-slope + 1.0)).transpose(0, 1)

    def _rectangular_filters(self, all_freqs, f_central, band):
        left_side = right_size = all_freqs.ge(f_central - band)
        right_size = all_freqs.le(f_central + band)

        return (left_side * right_size).float().transpose(0, 1)

    def _gaussian_filters(self, all_freqs, f_central, band, smooth_factor=torch.tensor(2)):
        return (-0.5 * ((all_freqs - f_central) / (band / smooth_factor)) ** 2).exp().transpose(0, 1)

    def _create_fbank_matrix(self, f_central_mat, band_mat):
        if self.filter_shape == "triangular": fbank_matrix = self._triangular_filters(self.all_freqs_mat, f_central_mat, band_mat)
        elif self.filter_shape == "rectangular": fbank_matrix = self._rectangular_filters(self.all_freqs_mat, f_central_mat, band_mat)
        else: fbank_matrix = self._gaussian_filters(self.all_freqs_mat, f_central_mat, band_mat)

        return fbank_matrix

    def _amplitude_to_DB(self, x):
        x_db = self.multiplier * x.clamp(min=self.amin).log10()
        x_db -= self.multiplier * self.db_multiplier

        return x_db.max((x_db.amax(dim=(-2, -1)) - self.top_db).view(x_db.shape[0], 1, 1))

class ContextWindow(torch.nn.Module):
    def __init__(self, left_frames=0, right_frames=0):
        super().__init__()
        self.left_frames = left_frames
        self.right_frames = right_frames
        self.context_len = self.left_frames + self.right_frames + 1
        self.kernel_len = 2 * max(self.left_frames, self.right_frames) + 1
        self.kernel = torch.eye(self.context_len, self.kernel_len)

        if self.right_frames > self.left_frames: self.kernel = torch.roll(self.kernel, self.right_frames - self.left_frames, 1)
        self.first_call = True

    def forward(self, x):
        x = x.transpose(1, 2)
        if self.first_call:
            self.first_call = False
            self.kernel = (self.kernel.repeat(x.shape[1], 1, 1).view(x.shape[1] * self.context_len, self.kernel_len).unsqueeze(1))

        or_shape = x.shape
        if len(or_shape) == 4: x = x.reshape(or_shape[0] * or_shape[2], or_shape[1], or_shape[3])

        cw_x = torch.nn.functional.conv1d(x, self.kernel.to(x.device), groups=x.shape[1], padding=max(self.left_frames, self.right_frames))
        if len(or_shape) == 4: cw_x = cw_x.reshape(or_shape[0], cw_x.shape[1], or_shape[2], cw_x.shape[-1])

        return cw_x.transpose(1, 2)

class FilterProperties:
    def __init__(self, window_size = 0, stride = 1, dilation = 1, causal = False):
        self.window_size = window_size
        self.stride = stride
        self.dilation = dilation
        self.causal = causal
        
    def __post_init__(self):
        assert self.window_size > 0
        assert self.stride > 0
        assert (self.dilation > 0)

    @staticmethod
    def pointwise_filter():
        return FilterProperties(window_size=1, stride=1)

    def get_effective_size(self):
        return 1 + ((self.window_size - 1) * self.dilation)

    def get_convolution_padding(self):
        if self.window_size % 2 == 0: raise ValueError
        if self.causal: return self.get_effective_size() - 1

        return (self.get_effective_size() - 1) // 2

    def get_noncausal_equivalent(self):
        if not self.causal: return self
        return FilterProperties(window_size=(self.window_size - 1) * 2 + 1, stride=self.stride, dilation=self.dilation, causal=False)

    def with_on_top(self, other, allow_approximate=True):
        self_size = self.window_size

        if other.window_size % 2 == 0:
            if allow_approximate: other_size = other.window_size + 1
            else: raise ValueError
        else: other_size = other.window_size

        if (self.causal or other.causal) and not (self.causal and other.causal):
            if allow_approximate: return self.get_noncausal_equivalent().with_on_top(other.get_noncausal_equivalent())
            else: raise ValueError

        return FilterProperties(self_size + (self.stride * (other_size - 1)), self.stride * other.stride, self.dilation * other.dilation, self.causal)

class STFT(torch.nn.Module):
    def __init__(self, sample_rate, win_length=25, hop_length=10, n_fft=400, window_fn=torch.hamming_window, normalized_stft=False, center=True, pad_mode="constant", onesided=True):
        super().__init__()
        self.sample_rate = sample_rate
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.normalized_stft = normalized_stft
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided
        self.win_length = int(round((self.sample_rate / 1000.0) * self.win_length))
        self.hop_length = int(round((self.sample_rate / 1000.0) * self.hop_length))
        self.window = window_fn(self.win_length)

    def forward(self, x):
        or_shape = x.shape
        if len(or_shape) == 3: x = x.transpose(1, 2).reshape(or_shape[0] * or_shape[2], or_shape[1])
        
        device = x.device
        if str(device) not in ["cuda", "cpu"]: x = x.cpu()

        stft = torch.view_as_real(torch.stft(x, self.n_fft, self.hop_length, self.win_length, self.window.to(x.device), self.center, self.pad_mode, self.normalized_stft, self.onesided, return_complex=True))
        stft = stft.reshape(or_shape[0], or_shape[2], stft.shape[1], stft.shape[2], stft.shape[3]).permute(0, 3, 2, 4, 1) if len(or_shape) == 3 else stft.transpose(2, 1)

        return stft.to(device)

    def get_filter_properties(self):
        if not self.center: raise ValueError
        return FilterProperties(window_size=self.win_length, stride=self.hop_length)

class Deltas(torch.nn.Module):
    def __init__(self, input_size, window_length=5):
        super().__init__()
        self.n = (window_length - 1) // 2
        self.denom = self.n * (self.n + 1) * (2 * self.n + 1) / 3
        self.register_buffer("kernel", torch.arange(-self.n, self.n + 1, dtype=torch.float32).repeat(input_size, 1, 1),)

    def forward(self, x):
        x = x.transpose(1, 2).transpose(2, -1)
        or_shape = x.shape

        if len(or_shape) == 4: x = x.reshape(or_shape[0] * or_shape[2], or_shape[1], or_shape[3])

        x = torch.nn.functional.pad(x, (self.n, self.n), mode="replicate")
        delta_coeff = (torch.nn.functional.conv1d(x, self.kernel.to(x.device), groups=x.shape[1]) / self.denom)

        if len(or_shape) == 4: delta_coeff = delta_coeff.reshape(or_shape[0], or_shape[1], or_shape[2], or_shape[3])
        return delta_coeff.transpose(1, -1).transpose(2, -1)

class Fbank(torch.nn.Module):
    def __init__(self, deltas=False, context=False, requires_grad=False, sample_rate=16000, f_min=0, f_max=None, n_fft=400, n_mels=40, filter_shape="triangular", param_change_factor=1.0, param_rand_factor=0.0, left_frames=5, right_frames=5, win_length=25, hop_length=10):
        super().__init__()
        self.deltas = deltas
        self.context = context
        self.requires_grad = requires_grad
        if f_max is None: f_max = sample_rate / 2
        self.compute_STFT = STFT(sample_rate=sample_rate,n_fft=n_fft,win_length=win_length,hop_length=hop_length)
        self.compute_fbanks = Filterbank(sample_rate=sample_rate,n_fft=n_fft,n_mels=n_mels,f_min=f_min,f_max=f_max,freeze=not requires_grad,filter_shape=filter_shape,param_change_factor=param_change_factor,param_rand_factor=param_rand_factor)
        self.compute_deltas = Deltas(input_size=n_mels)
        self.context_window = ContextWindow(left_frames=left_frames, right_frames=right_frames)

    @fwd_default_precision(cast_inputs=torch.float32)
    def forward(self, wav):
        fbanks = self.compute_fbanks(spectral_magnitude(self.compute_STFT(wav)))
        if self.deltas:
            delta1 = self.compute_deltas(fbanks)
            fbanks = torch.cat([fbanks, delta1, self.compute_deltas(delta1)], dim=2)

        if self.context: fbanks = self.context_window(fbanks)
        return fbanks

    def get_filter_properties(self):
        return self.compute_STFT.get_filter_properties()

@register_checkpoint_hooks
class InputNormalization(torch.nn.Module):
    def __init__(self, mean_norm=True, std_norm=True, norm_type="global", avg_factor=None, requires_grad=False, update_until_epoch=3):
        super().__init__()
        self.mean_norm = mean_norm
        self.std_norm = std_norm
        self.norm_type = norm_type
        self.avg_factor = avg_factor
        self.requires_grad = requires_grad
        self.glob_mean = torch.tensor([0])
        self.glob_std = torch.tensor([0])
        self.spk_dict_mean = {}
        self.spk_dict_std = {}
        self.spk_dict_count = {}
        self.weight = 1.0
        self.count = 0
        self.eps = 1e-10
        self.update_until_epoch = update_until_epoch

    def forward(self, x, lengths, spk_ids = torch.tensor([]), epoch=0):
        N_batches = x.shape[0]
        current_means, current_stds = [], []

        if self.norm_type == "sentence" or self.norm_type == "speaker": out = torch.empty_like(x)

        for snt_id in range(N_batches):
            actual_size = torch.round(lengths[snt_id] * x.shape[1]).int()
            current_mean, current_std = self._compute_current_stats(x[snt_id, 0:actual_size, ...])

            current_means.append(current_mean)
            current_stds.append(current_std)

            if self.norm_type == "sentence": out[snt_id] = (x[snt_id] - current_mean.data) / current_std.data

            if self.norm_type == "speaker":
                spk_id = int(spk_ids[snt_id][0])

                if self.training:
                    if spk_id not in self.spk_dict_mean:
                        self.spk_dict_mean[spk_id] = current_mean
                        self.spk_dict_std[spk_id] = current_std
                        self.spk_dict_count[spk_id] = 1
                    else:
                        self.spk_dict_count[spk_id] = (self.spk_dict_count[spk_id] + 1)
                        self.weight = (1 / self.spk_dict_count[spk_id]) if self.avg_factor is None else self.avg_factor

                        self.spk_dict_mean[spk_id] = (1 - self.weight) * self.spk_dict_mean[spk_id].to(current_mean) + self.weight * current_mean
                        self.spk_dict_std[spk_id] = (1 - self.weight) * self.spk_dict_std[spk_id].to(current_std) + self.weight * current_std

                        self.spk_dict_mean[spk_id].detach()
                        self.spk_dict_std[spk_id].detach()

                    speaker_mean = self.spk_dict_mean[spk_id].data
                    speaker_std = self.spk_dict_std[spk_id].data
                else:
                    if spk_id in self.spk_dict_mean:
                        speaker_mean = self.spk_dict_mean[spk_id].data
                        speaker_std = self.spk_dict_std[spk_id].data
                    else:
                        speaker_mean = current_mean.data
                        speaker_std = current_std.data

                out[snt_id] = (x[snt_id] - speaker_mean) / speaker_std

        if self.norm_type == "batch" or self.norm_type == "global":
            current_mean = ddp_all_reduce(torch.stack(current_means).mean(dim=0), torch.distributed.ReduceOp.AVG)
            current_std = ddp_all_reduce(torch.stack(current_stds).mean(dim=0), torch.distributed.ReduceOp.AVG)

            if self.norm_type == "batch": out = (x - current_mean.data) / (current_std.data)

            if self.norm_type == "global":
                if self.training:
                    if self.count == 0:
                        self.glob_mean = current_mean
                        self.glob_std = current_std
                    elif epoch is None or epoch < self.update_until_epoch:
                        self.weight = (1 / (self.count + 1)) if self.avg_factor is None else self.avg_factor
                        self.glob_mean = (1 - self.weight) * self.glob_mean.to(current_mean) + self.weight * current_mean
                        self.glob_std = (1 - self.weight) * self.glob_std.to(current_std) + self.weight * current_std

                    self.glob_mean.detach()
                    self.glob_std.detach()
                    self.count = self.count + 1

                out = (x - self.glob_mean.data.to(x)) / (self.glob_std.data.to(x))

        return out

    def _compute_current_stats(self, x):
        current_std = x.std(dim=0).detach().data if self.std_norm else torch.tensor([1.0], device=x.device)
        return x.mean(dim=0).detach().data if self.mean_norm else torch.tensor([0.0], device=x.device), torch.max(current_std, self.eps * torch.ones_like(current_std))

    def _statistics_dict(self):
        state = {}
        state["count"] = self.count
        state["glob_mean"] = self.glob_mean
        state["glob_std"] = self.glob_std
        state["spk_dict_mean"] = self.spk_dict_mean
        state["spk_dict_std"] = self.spk_dict_std
        state["spk_dict_count"] = self.spk_dict_count

        return state

    def _load_statistics_dict(self, state):
        self.count = state["count"]

        if isinstance(state["glob_mean"], int):
            self.glob_mean = state["glob_mean"]
            self.glob_std = state["glob_std"]
        else:
            self.glob_mean = state["glob_mean"]  
            self.glob_std = state["glob_std"]  

        self.spk_dict_mean = {}
        for spk in state["spk_dict_mean"]:
            self.spk_dict_mean[spk] = state["spk_dict_mean"][spk]

        self.spk_dict_std = {}
        for spk in state["spk_dict_std"]:
            self.spk_dict_std[spk] = state["spk_dict_std"][spk] 

        self.spk_dict_count = state["spk_dict_count"]
        return state

    def to(self, device):
        self = super(InputNormalization, self).to(device)
        self.glob_mean = self.glob_mean.to(device)
        self.glob_std = self.glob_std.to(device)

        for spk in self.spk_dict_mean:
            self.spk_dict_mean[spk] = self.spk_dict_mean[spk].to(device)
            self.spk_dict_std[spk] = self.spk_dict_std[spk].to(device)

        return self

    @mark_as_saver
    def _save(self, path):
        torch.save(self._statistics_dict(), path)

    @mark_as_transfer
    @mark_as_loader
    def _load(self, path, end_of_epoch=False):
        del end_of_epoch  
        stats = torch.load(path, map_location="cpu", weights_only=False)
        self._load_statistics_dict(stats)
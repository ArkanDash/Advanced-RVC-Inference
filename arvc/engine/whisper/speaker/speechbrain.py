import os
import torch
import torchaudio

from functools import wraps
from types import SimpleNamespace
from torch.nn import SyncBatchNorm
from hyperpyyaml import load_hyperpyyaml

from torch.nn import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP

MAIN_PROC_ONLY = 0

def fetch(filename, source):
    return os.path.abspath(os.path.join(source, filename))

def run_on_main(func, args=None, kwargs=None, post_func=None, post_args=None, post_kwargs=None, run_post_on_main=False):
    if args is None: args = []
    if kwargs is None: kwargs = {}
    if post_args is None: post_args = []
    if post_kwargs is None: post_kwargs = {}

    main_process_only(func)(*args, **kwargs)
    ddp_barrier()

    if post_func is not None:
        if run_post_on_main: post_func(*post_args, **post_kwargs)
        else:
            if not if_main_process(): post_func(*post_args, **post_kwargs)
            ddp_barrier()

def is_distributed_initialized():
    return (torch.distributed.is_available() and torch.distributed.is_initialized())

def if_main_process():
    if is_distributed_initialized(): return torch.distributed.get_rank() == 0
    else: return True

class MainProcessContext:
    def __enter__(self):
        global MAIN_PROC_ONLY

        MAIN_PROC_ONLY += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        global MAIN_PROC_ONLY

        MAIN_PROC_ONLY -= 1

def main_process_only(function):
    @wraps(function)
    def main_proc_wrapped_func(*args, **kwargs):
        with MainProcessContext():
            return function(*args, **kwargs) if if_main_process() else None

    return main_proc_wrapped_func

def ddp_barrier():
    if MAIN_PROC_ONLY >= 1 or not is_distributed_initialized(): return

    if torch.distributed.get_backend() == torch.distributed.Backend.NCCL: torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    else: torch.distributed.barrier()

class Resample(torch.nn.Module):
    def __init__(self, orig_freq=16000, new_freq=16000, *args, **kwargs):
        super().__init__()

        self.orig_freq = orig_freq
        self.new_freq = new_freq
        self.resampler = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=new_freq, *args, **kwargs)

    def forward(self, waveforms):
        if self.orig_freq == self.new_freq: return waveforms

        unsqueezed = False
        if len(waveforms.shape) == 2:
            waveforms = waveforms.unsqueeze(1)
            unsqueezed = True
        elif len(waveforms.shape) == 3: waveforms = waveforms.transpose(1, 2)
        else: raise ValueError

        self.resampler.to(waveforms.device) 
        resampled_waveform = self.resampler(waveforms)
        
        return resampled_waveform.squeeze(1) if unsqueezed else resampled_waveform.transpose(1, 2)

class AudioNormalizer:
    def __init__(self, sample_rate=16000, mix="avg-to-mono"):
        self.sample_rate = sample_rate

        if mix not in ["avg-to-mono", "keep"]: raise ValueError

        self.mix = mix
        self._cached_resamplers = {}

    def __call__(self, audio, sample_rate):
        if sample_rate not in self._cached_resamplers: self._cached_resamplers[sample_rate] = Resample(sample_rate, self.sample_rate)
        return self._mix(self._cached_resamplers[sample_rate](audio.unsqueeze(0)).squeeze(0))

    def _mix(self, audio):
        flat_input = audio.dim() == 1

        if self.mix == "avg-to-mono":
            if flat_input: return audio
            return torch.mean(audio, 1)
        
        if self.mix == "keep": return audio

class Pretrained(torch.nn.Module):
    HPARAMS_NEEDED, MODULES_NEEDED = [], []
    def __init__(self, modules=None, hparams=None, run_opts=None, freeze_params=True):
        super().__init__()

        for arg, default in {"device": "cpu", "data_parallel_count": -1, "data_parallel_backend": False, "distributed_launch": False, "distributed_backend": "nccl", "jit": False, "jit_module_keys": None, "compile": False, "compile_module_keys": None, "compile_mode": "reduce-overhead", "compile_using_fullgraph": False, "compile_using_dynamic_shape_tracing": False}.items():
            if run_opts is not None and arg in run_opts: setattr(self, arg, run_opts[arg])
            elif hparams is not None and arg in hparams: setattr(self, arg, hparams[arg])
            else: setattr(self, arg, default)

        self.mods = torch.nn.ModuleDict(modules)

        for module in self.mods.values():
            if module is not None: module.to(self.device)

        if self.HPARAMS_NEEDED and hparams is None: raise ValueError

        if hparams is not None:
            for hp in self.HPARAMS_NEEDED:
                if hp not in hparams: raise ValueError

            self.hparams = SimpleNamespace(**hparams)

        self._prepare_modules(freeze_params)
        self.audio_normalizer = hparams.get("audio_normalizer", AudioNormalizer())

    def _prepare_modules(self, freeze_params):
        self._compile()
        self._wrap_distributed()

        if freeze_params:
            self.mods.eval()
            for p in self.mods.parameters():
                p.requires_grad = False

    def _compile(self):
        compile_available = hasattr(torch, "compile")
        if not compile_available and self.compile_module_keys is not None: raise ValueError

        compile_module_keys = set()
        if self.compile: compile_module_keys = set(self.mods) if self.compile_module_keys is None else set(self.compile_module_keys)

        jit_module_keys = set()
        if self.jit: jit_module_keys = set(self.mods) if self.jit_module_keys is None else set(self.jit_module_keys)

        for name in compile_module_keys | jit_module_keys:
            if name not in self.mods: raise ValueError

        for name in compile_module_keys:
            try:
                module = torch.compile(self.mods[name], mode=self.compile_mode, fullgraph=self.compile_using_fullgraph, dynamic=self.compile_using_dynamic_shape_tracing)
            except Exception:
                continue

            self.mods[name] = module.to(self.device)
            jit_module_keys.discard(name)

        for name in jit_module_keys:
            module = torch.jit.script(self.mods[name])
            self.mods[name] = module.to(self.device)

    def _compile_jit(self):
        self._compile()

    def _wrap_distributed(self):
        if not self.distributed_launch and not self.data_parallel_backend: return
        elif self.distributed_launch:
            for name, module in self.mods.items():
                if any(p.requires_grad for p in module.parameters()): self.mods[name] = DDP(SyncBatchNorm.convert_sync_batchnorm(module), device_ids=[self.device])
        else:
            for name, module in self.mods.items():
                if any(p.requires_grad for p in module.parameters()): self.mods[name] = DP(module) if self.data_parallel_count == -1 else DP(module, [i for i in range(self.data_parallel_count)])

    @classmethod
    def from_hparams(cls, source, hparams_file="hyperparams.yaml", overrides={}, download_only=False, overrides_must_match=True, **kwargs):
        with open(fetch(filename=hparams_file, source=source)) as fin:
            hparams = load_hyperpyyaml(fin, overrides, overrides_must_match=overrides_must_match)

        pretrainer = hparams.get("pretrainer", None)

        if pretrainer is not None:
            run_on_main(pretrainer.collect_files, kwargs={"default_source": source})
            if not download_only:
                pretrainer.load_collected()
                return cls(hparams["modules"], hparams, **kwargs)
        else: return cls(hparams["modules"], hparams, **kwargs)

class EncoderClassifier(Pretrained):
    MODULES_NEEDED = ["compute_features", "mean_var_norm", "embedding_model", "classifier"]

    def encode_batch(self, wavs, wav_lens=None, normalize=False):
        if len(wavs.shape) == 1: wavs = wavs.unsqueeze(0)
        if wav_lens is None: wav_lens = torch.ones(wavs.shape[0], device=self.device)

        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        wavs = wavs.float()

        embeddings = self.mods.embedding_model(self.mods.mean_var_norm(self.mods.compute_features(wavs), wav_lens), wav_lens)

        if normalize: embeddings = self.hparams.mean_var_norm_emb(embeddings, torch.ones(embeddings.shape[0], device=self.device))
        return embeddings

    def classify_batch(self, wavs, wav_lens=None):
        out_prob = self.mods.classifier(self.encode_batch(wavs, wav_lens)).squeeze(1)
        score, index = out_prob.max(dim=-1)

        return out_prob, score, index, self.hparams.label_encoder.decode_torch(index)

    def forward(self, wavs, wav_lens=None):
        return self.classify_batch(wavs, wav_lens)